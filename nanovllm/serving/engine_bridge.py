"""Bridge between the asyncio HTTP layer and the engine.

The engine is not thread safe, so exactly one thread owns it. Requests are handed over
through a queue; each one gets an asyncio Future that the thread resolves with the
completion token ids. TP drives add_request / step / is_finished directly; DP has no
step loop, so the thread polls the controller's result socket instead -- never the
offline ``res_recv_from_client`` / ``Res_Budget`` channel, which belongs to generate().
"""
import asyncio
import atexit
import os
import queue
import threading
import time

import zmq

from nanovllm.llm import LLM


class EngineError(RuntimeError):
    """A request could not be run (bridge-side failure, mapped to HTTP 5xx)."""


class _Request:
    __slots__ = ("future", "prompt_ids", "sampling_params")

    def __init__(self, future, prompt_ids, sampling_params):
        self.future = future
        self.prompt_ids = prompt_ids
        self.sampling_params = sampling_params


class EngineBridge:

    def __init__(self, model: str, tensor_parallel_size: int, data_parallel_size: int,
                 max_model_len: int, async_scheduling: bool,
                 async_event_loop: bool = False):
        # Only the values the CLI can vary are passed; every other engine setting keeps
        # its Config default, which equals deployment_point.config.
        self._llm = LLM(
            model,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            max_model_len=max_model_len,
            async_scheduling=async_scheduling,
            async_event_loop=async_event_loop,
        )
        self.engine = self._llm._engine
        self.tokenizer = self.engine.tokenizer
        self.config = self.engine.config
        self.served_model = os.path.basename(model.rstrip("/\\"))
        self.eos = self.config.eos
        self.max_model_len = self.config.max_model_len
        self._data_parallel = data_parallel_size > 1
        self._submit: queue.Queue = queue.Queue()
        self._futures: dict[int, _Request] = {}
        self._loop = None
        self._wake = threading.Event()
        self._running = True
        self._alive = True
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="engine-bridge", daemon=True)
        self._thread.start()

    @property
    def alive(self) -> bool:
        return self._alive

    # -- called from the HTTP event loop ------------------------------------

    def submit(self, prompt_ids: list[int], sampling_params) -> asyncio.Future:
        """Queue one request; the Future resolves with the completion token ids."""
        loop = asyncio.get_running_loop()
        self._loop = loop
        future = loop.create_future()
        self._submit.put(_Request(future, prompt_ids, sampling_params))
        self._wake.set()
        return future

    # -- engine thread ------------------------------------------------------

    def _run(self):
        try:
            if self._data_parallel:
                self._run_dp()
            else:
                self._run_tp()
        except Exception as exc:  # noqa: BLE001 - a dead engine must fail requests, not crash
            self._fail_all(exc)
        finally:
            self._alive = False

    def _run_tp(self):
        engine = self.engine
        while self._running:
            self._drain()
            if not self._running:
                break
            if engine.is_finished():
                self._wake.wait(0.1)
                self._wake.clear()
                continue
            outputs, _ = engine.step()
            self._settle(outputs)

    def _run_dp(self):
        result_socket = self.engine.dp_controller.result_socket
        while self._running:
            self._drain()
            if not self._running:
                break
            if result_socket.poll(100):
                while True:
                    try:
                        payload = result_socket.recv_pyobj(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    self._settle(payload)

    def _drain(self):
        while True:
            try:
                request = self._submit.get_nowait()
            except queue.Empty:
                return
            if request.future.cancelled():
                continue        # the client is gone; do not admit the sequence
            try:
                seq_id = self.engine.add_request(request.prompt_ids,
                                                 request.sampling_params)
            except Exception as exc:  # noqa: BLE001
                self._settle_one(request.future, error=EngineError(str(exc)))
                continue
            if seq_id is None:
                reason = self.engine.failed[-1].reason if self.engine.failed else "rejected"
                self._settle_one(request.future, error=EngineError(reason))
                continue
            self._futures[seq_id] = request

    def _settle(self, outputs):
        for seq_id, token_ids in outputs:
            request = self._futures.pop(seq_id, None)
            if request is not None:
                self._settle_one(request.future, result=token_ids)

    def _settle_one(self, future, result=None, error=None):
        loop = self._loop
        if loop is None:
            return

        def _apply():
            if future.done():
                return
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(result)

        try:
            loop.call_soon_threadsafe(_apply)
        except RuntimeError:
            pass    # loop already closed: the server is shutting down, nobody is waiting

    def _fail_all(self, exc):
        error = EngineError(str(exc))
        for seq_id in list(self._futures):
            self._settle_one(self._futures.pop(seq_id).future, error=error)
        while True:
            try:
                request = self._submit.get_nowait()
            except queue.Empty:
                break
            self._settle_one(request.future, error=error)

    # -- shutdown -----------------------------------------------------------

    def shutdown(self):
        """Stop the engine thread first, then release the engine: the runner has a
        single-writer exit protocol, so it must run after the thread has joined.

        Idempotent: the SIGTERM path in api_server can call it before the normal
        ``finally`` does.
        """
        if self._closed:
            return
        self._closed = True
        self._running = False
        self._wake.set()
        self._thread.join()
        if self._data_parallel:
            self._drain_dp_results(timeout=10.0)
        atexit.unregister(self.engine.exit)
        self.engine.exit()

    def _drain_dp_results(self, timeout):
        """Collect results still owed at shutdown.

        A DP client drains its in-flight batch only after the exit command, but by then
        the controller closes the result socket; an unconsumed PUSH send would back off
        forever inside the client. Pull everything the pipeline still produces first.
        """
        result_socket = self.engine.dp_controller.result_socket
        deadline = time.time() + timeout
        while self._futures and time.time() < deadline:
            if not result_socket.poll(100):
                continue
            while True:
                try:
                    payload = result_socket.recv_pyobj(zmq.NOBLOCK)
                except zmq.Again:
                    break
                for seq_id, _ in payload:
                    self._futures.pop(seq_id, None)
