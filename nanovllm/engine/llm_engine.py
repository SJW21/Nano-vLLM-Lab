import atexit
from dataclasses import dataclass, fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import TokenizedInput, SequenceManager
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.data_parallel_controller import DataParallelController
from nanovllm.utils.profiler import Profiler


@dataclass
class FailedRequest:
    prompt_len: int
    max_tokens: int
    max_model_len: int
    reason: str


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self.failed: list[FailedRequest] = []
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, 0, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.profiler = Profiler(config)
        # async scheduling pipeline state: the batch prepared for the next launch, and
        # the launched batch whose sampled tokens are consumed one step later; each
        # carries its is_prefill flag, since both kinds run through the same pipeline
        self._pending: tuple[list[Sequence], bool] | None = None
        self._inflight: tuple[list[Sequence], bool, tuple[int, int]] | None = None
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        prompt_len = len(prompt)
        if prompt_len == 0:
            self.failed.append(FailedRequest(
                prompt_len, sampling_params.max_tokens,
                self.config.max_model_len, "empty prompt"))
            return
        if prompt_len + sampling_params.max_tokens > self.config.max_model_len:
            self.failed.append(FailedRequest(
                prompt_len, sampling_params.max_tokens,
                self.config.max_model_len,
                f"prompt_len ({prompt_len}) + max_tokens ({sampling_params.max_tokens})"
                f" = {prompt_len + sampling_params.max_tokens}"
                f" exceeds max_model_len ({self.config.max_model_len})"))
            return
        seq = Sequence(prompt, next(Sequence.counter), sampling_params)
        self.scheduler.add(seq)

    def step(self):
        if self.config.async_scheduling:
            return self._step_async()
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def _step_async(self):
        """Run-ahead step, uniform over prefill and decode: launch the batch prepared by
        the previous call, then consume the tokens of the step before it. By the time the
        CPU blocks on those tokens the next step is already enqueued, so the GPU never
        idles on the D2H round trip -- and a prefill no longer costs a call either."""
        model_runner = self.model_runner
        outputs = []
        # the batch launched by the previous call is the one whose tokens we collect now
        self._inflight, to_consume = None, self._inflight
        # (1) launch the prepared batch (pure enqueue, never blocks); its token count is
        #     what this call's throughput reports -- read before the consume below zeroes
        #     num_scheduled_tokens
        num_tokens = 0
        if self._pending is not None:
            seqs, is_prefill = self._pending
            num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
            method = "launch_prefill" if is_prefill else "launch_decode"
            self._inflight = (seqs, is_prefill, model_runner.call(method, seqs))
            self._pending = None
        # (2) consume the previous call's launch -- one step behind, and by the time the
        #     CPU blocks here the next step is already enqueued, so the wait overlaps
        #     useful GPU work instead of leaving a bubble
        if to_consume is not None:
            seqs, is_prefill, handle = to_consume
            token_ids = model_runner.wait_sample(handle)
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
            outputs += [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # (3) schedule the next batch; a round in which every sequence is already in
        #     flight returns nothing, and the pipeline above finishes the work
        seqs, is_prefill = self.scheduler.schedule()
        if seqs:
            self._pending = (seqs, is_prefill)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        self.failed = []
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        num_valid = len(prompts) - len(self.failed)
        pbar = tqdm(total=num_valid, desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        with self.profiler:
            while not self.is_finished():
                t = perf_counter()
                output, num_tokens = self.step()
                self.profiler.step()
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
                for seq_id, token_ids in output:
                    outputs[seq_id] = token_ids
                    pbar.update(1)
            # async scheduling: safety net -- is_finished() already counts in-flight
            # batches, so this normally never iterates; if anything is still staged,
            # drain it so every sampled token is back before the outputs are collected
            while self._pending is not None or self._inflight is not None:
                output, _ = self.step()
                self.profiler.step()
                for seq_id, token_ids in output:
                    outputs[seq_id] = token_ids
                    pbar.update(1)
        pbar.close()
        if self.failed:
            print(f"\n[WARNING] {len(self.failed)}/{len(prompts)} request(s) rejected:")
            for i, f in enumerate(self.failed):
                print(f"  [{i+1}] {f.reason}")
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs


class DPLLMEngine:
    """
    DP LLM Engine offline inference
    """
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self.failed: list[FailedRequest] = []
        self.dp_controller = DataParallelController(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        self.seqm = SequenceManager()
        config.eos = self.tokenizer.eos_token_id
        atexit.register(self.exit)
    
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        prompt_len = len(prompt)
        if prompt_len == 0:
            self.failed.append(FailedRequest(
                prompt_len, sampling_params.max_tokens,
                self.config.max_model_len, "empty prompt"))
            return
        if prompt_len + sampling_params.max_tokens > self.config.max_model_len:
            self.failed.append(FailedRequest(
                prompt_len, sampling_params.max_tokens,
                self.config.max_model_len,
                f"prompt_len ({prompt_len}) + max_tokens ({sampling_params.max_tokens})"
                f" = {prompt_len + sampling_params.max_tokens}"
                f" exceeds max_model_len ({self.config.max_model_len})"))
            return
        seq = TokenizedInput(prompt, self.seqm.get_id, sampling_params)
        self.dp_controller.dispatcher(seq)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        self.failed = []
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        num_valid = len(prompts) - len(self.failed)
        pbar = tqdm(total=num_valid, desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        outputs = {}
        self.dp_controller.Res_Budget = outputs
        num_token_finished = 0
        num_req_finished = 0
        output_throughput = 0.
        t0 = perf_counter()
        while len(outputs) < num_valid:
            num_tokens = self.dp_controller.res_recv_from_client()
            num_token_finished += num_tokens

            if num_tokens > 0:
                output_throughput = num_token_finished / (perf_counter() - t0)
                pbar.set_postfix({
                    "Throughput": f"{int(output_throughput)}tok/s"
                })
                pbar.update(len(outputs) - num_req_finished)
            num_req_finished = len(outputs)
        
        pbar.close()
        if self.failed:
            print(f"\n[WARNING] {len(self.failed)}/{len(prompts)} request(s) rejected:")
            for i, f in enumerate(self.failed):
                print(f"  [{i+1}] {f.reason}")
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]

        return outputs

    def exit(self):
        self.dp_controller.call("exit")
        del self.dp_controller