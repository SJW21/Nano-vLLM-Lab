"""OpenAI-compatible + native HTTP serving entry point.

Run::

    python -m nanovllm.serving.api_server --model <path> --tp N --dp N \
        --port N --max-model-len 4096 [--async-scheduling]

The engine is constructed *before* uvicorn binds (A-construct-before-listen), so the
first successful /health connection already implies a usable engine; 503 is reserved for
the engine-thread-death path.

Streaming is a single-delta SSE shell: the engine has no incremental output, so the
full generation is awaited once and emitted as role chunk + one content delta + finish
chunk (+ usage chunk), then ``data: [DONE]``.
"""
import argparse
import asyncio
import json
import os
import signal
import time
import uuid

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from nanovllm.sampling_params import SamplingParams
from nanovllm.serving.engine_bridge import EngineBridge, EngineError
from nanovllm.serving.protocol import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    GenerateRequest,
    ModelCard,
    ModelList,
    ResponseMessage,
    Usage,
)
from nanovllm.serving.translation import (
    TranslationError,
    finalize_result,
    frame_max_tokens,
    map_temperature,
    validate_n,
)


# ------------------------------------------------------------------ helpers

def _error_body(message: str, param: str, code: int, kind: str) -> dict:
    return {"error": {"message": message, "type": kind, "param": param, "code": code}}


def _error_response(message: str, param: str, code: int = 400,
                    kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(status_code=code, content=_error_body(message, param, code, kind))


def _sse(payload) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _resolve_sampling(bridge: EngineBridge, prompt_ids: list[int],
                      max_tokens: int | None, temperature: float | None,
                      ignore_eos: bool = False) -> SamplingParams:
    return SamplingParams(
        temperature=map_temperature(temperature),
        max_tokens=frame_max_tokens(len(prompt_ids), max_tokens, bridge.max_model_len),
        ignore_eos=ignore_eos,
    )


async def _await_completion(bridge: EngineBridge, prompt_ids: list[int], sampling: SamplingParams):
    future = bridge.submit(prompt_ids, sampling)
    try:
        return await future
    except asyncio.CancelledError:
        future.cancel()
        raise


def _chat_prompt(bridge: EngineBridge, request: ChatCompletionRequest) -> list[int]:
    if not any(message.content.strip() for message in request.messages):
        raise TranslationError("prompt is empty: every message has empty content",
                               "messages")
    messages = [{"role": message.role, "content": message.content}
                for message in request.messages]
    try:
        text = bridge.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception as exc:  # noqa: BLE001 - a template failure is a client-visible 400
        raise TranslationError(f"chat template failed: {exc}", "messages") from exc
    prompt_ids = bridge.tokenizer.encode(text)
    if not prompt_ids:
        raise TranslationError("prompt is empty after the chat template", "messages")
    return prompt_ids


def _text_prompt(bridge: EngineBridge, prompt: str) -> list[int]:
    if not prompt:
        raise TranslationError("prompt must be a non-empty string", "prompt")
    prompt_ids = bridge.tokenizer.encode(prompt)
    if not prompt_ids:
        raise TranslationError("prompt is empty after tokenization", "prompt")
    return prompt_ids


def _chat_chunk(stream_id: str, created: int, model: str, delta: dict,
                finish_reason: str | None) -> dict:
    return {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _completion_chunk(stream_id: str, created: int, model: str, text: str,
                      finish_reason: str | None) -> dict:
    return {
        "id": stream_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": text, "logprobs": None,
                     "finish_reason": finish_reason}],
    }


def _sse_response(generator) -> StreamingResponse:
    return StreamingResponse(generator, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


async def _chat_stream(bridge: EngineBridge, prompt_ids: list[int],
                       sampling: SamplingParams, include_usage: bool):
    created = int(time.time())
    stream_id = "chatcmpl-" + uuid.uuid4().hex
    future = bridge.submit(prompt_ids, sampling)
    try:
        token_ids = await future
    except asyncio.CancelledError:
        future.cancel()
        raise
    except EngineError as exc:
        yield _sse(_error_body(str(exc), "prompt", 500, "server_error"))
        yield "data: [DONE]\n\n"
        return
    result = finalize_result(prompt_ids, token_ids, bridge.eos, bridge.tokenizer.decode,
                             sampling.ignore_eos)
    model = bridge.served_model
    yield _sse(_chat_chunk(stream_id, created, model,
                           {"role": "assistant", "content": ""}, None))
    yield _sse(_chat_chunk(stream_id, created, model,
                           {"content": result["text"]}, None))
    yield _sse(_chat_chunk(stream_id, created, model, {}, result["finish_reason"]))
    if include_usage:
        yield _sse({"id": stream_id, "object": "chat.completion.chunk",
                    "created": created, "model": model, "choices": [],
                    "usage": result["usage"]})
    yield "data: [DONE]\n\n"


async def _completion_stream(bridge: EngineBridge, prompt_ids: list[int],
                             sampling: SamplingParams):
    created = int(time.time())
    stream_id = "cmpl-" + uuid.uuid4().hex
    future = bridge.submit(prompt_ids, sampling)
    try:
        token_ids = await future
    except asyncio.CancelledError:
        future.cancel()
        raise
    except EngineError as exc:
        yield _sse(_error_body(str(exc), "prompt", 500, "server_error"))
        yield "data: [DONE]\n\n"
        return
    result = finalize_result(prompt_ids, token_ids, bridge.eos, bridge.tokenizer.decode,
                             sampling.ignore_eos)
    model = bridge.served_model
    yield _sse(_completion_chunk(stream_id, created, model, result["text"], None))
    yield _sse(_completion_chunk(stream_id, created, model, "",
                                 result["finish_reason"]))
    yield "data: [DONE]\n\n"


async def _generate_stream(bridge: EngineBridge, prompt_ids: list[int],
                           sampling: SamplingParams):
    future = bridge.submit(prompt_ids, sampling)
    try:
        token_ids = await future
    except asyncio.CancelledError:
        future.cancel()
        raise
    except EngineError as exc:
        yield _sse(_error_body(str(exc), "prompt", 500, "server_error"))
        yield "data: [DONE]\n\n"
        return
    result = finalize_result(prompt_ids, token_ids, bridge.eos, bridge.tokenizer.decode,
                             sampling.ignore_eos)
    yield _sse(result)
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------- app

def create_app(bridge: EngineBridge) -> FastAPI:
    app = FastAPI(title="nano-vllm serving")

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request, exc):
        errors = exc.errors() or [{}]
        first = errors[0]
        location = first.get("loc") or ("body",)
        return _error_response(first.get("msg") or "invalid request",
                               str(location[-1]))

    @app.exception_handler(TranslationError)
    async def _on_translation_error(request, exc):
        return _error_response(str(exc), exc.param)

    @app.exception_handler(StarletteHTTPException)
    async def _on_http_error(request, exc):
        return _error_response(str(exc.detail), "body", exc.status_code,
                               "server_error" if exc.status_code >= 500
                               else "invalid_request_error")

    @app.exception_handler(Exception)
    async def _on_internal_error(request, exc):
        return _error_response(str(exc) or "internal error", "body", 500, "server_error")

    @app.get("/health")
    async def health():
        return Response(status_code=200 if bridge.alive else 503)

    @app.get("/v1/models")
    async def models():
        card = ModelCard(id=bridge.served_model, created=int(time.time()))
        return ModelList(data=[card]).model_dump()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        prompt_ids = _chat_prompt(bridge, request)
        max_tokens = (request.max_completion_tokens
                      if request.max_completion_tokens is not None
                      else request.max_tokens)
        validate_n(request.n)
        sampling = _resolve_sampling(bridge, prompt_ids, max_tokens,
                                     request.temperature)
        if request.stream:
            include_usage = bool(request.stream_options
                                 and request.stream_options.include_usage)
            return _sse_response(_chat_stream(bridge, prompt_ids, sampling,
                                              include_usage))
        try:
            token_ids = await _await_completion(bridge, prompt_ids, sampling)
        except EngineError as exc:
            return _error_response(str(exc), "prompt", 500, "server_error")
        result = finalize_result(prompt_ids, token_ids, bridge.eos,
                                 bridge.tokenizer.decode, sampling.ignore_eos)
        choice = ChatChoice(index=0, message=ResponseMessage(content=result["text"]),
                            finish_reason=result["finish_reason"])
        return ChatCompletionResponse(
            id="chatcmpl-" + uuid.uuid4().hex,
            created=int(time.time()),
            model=bridge.served_model,
            choices=[choice],
            usage=Usage(**result["usage"]),
        ).model_dump()

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        prompt_ids = _text_prompt(bridge, request.prompt)
        validate_n(request.n)
        sampling = _resolve_sampling(bridge, prompt_ids, request.max_tokens,
                                     request.temperature)
        if request.stream:
            return _sse_response(_completion_stream(bridge, prompt_ids, sampling))
        try:
            token_ids = await _await_completion(bridge, prompt_ids, sampling)
        except EngineError as exc:
            return _error_response(str(exc), "prompt", 500, "server_error")
        result = finalize_result(prompt_ids, token_ids, bridge.eos,
                                 bridge.tokenizer.decode, sampling.ignore_eos)
        choice = CompletionChoice(index=0, text=result["text"],
                                  finish_reason=result["finish_reason"])
        return CompletionResponse(
            id="cmpl-" + uuid.uuid4().hex,
            created=int(time.time()),
            model=bridge.served_model,
            choices=[choice],
            usage=Usage(**result["usage"]),
        ).model_dump()

    @app.post("/generate")
    async def generate(request: GenerateRequest):
        prompt_ids = _text_prompt(bridge, request.prompt)
        params = request.sampling_params or {}
        validate_n(params.get("n"))
        try:
            sampling = _resolve_sampling(bridge, prompt_ids, params.get("max_tokens"),
                                         params.get("temperature"),
                                         bool(params.get("ignore_eos", False)))
        except TranslationError:
            raise
        except (TypeError, ValueError) as exc:
            raise TranslationError(f"invalid sampling_params: {exc}",
                                   "sampling_params") from exc
        if request.stream:
            return _sse_response(_generate_stream(bridge, prompt_ids, sampling))
        try:
            token_ids = await _await_completion(bridge, prompt_ids, sampling)
        except EngineError as exc:
            return _error_response(str(exc), "prompt", 500, "server_error")
        return finalize_result(prompt_ids, token_ids, bridge.eos,
                               bridge.tokenizer.decode, sampling.ignore_eos)

    return app


# --------------------------------------------------------------------- main

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="nano-vllm HTTP serving")
    parser.add_argument("--model", required=True, help="checkpoint path")
    parser.add_argument("--tp", type=int, required=True, help="tensor parallel size")
    parser.add_argument("--dp", type=int, required=True, help="data parallel size")
    parser.add_argument("--port", type=int, required=True, help="listen port (127.0.0.1)")
    parser.add_argument("--max-model-len", dest="max_model_len", type=int, required=True)
    parser.add_argument("--async-scheduling", dest="async_scheduling",
                        action="store_true")
    return parser.parse_args(argv)


def run_server(args):
    # M7 guard: the DP sync event loop busy-spins (no idle poll), so online DP is
    # accepted only in its async form. Fail before any engine work or socket bind.
    if args.dp > 1:
        assert args.async_scheduling, (
            "online serving with data_parallel_size>1 requires async_scheduling=True: "
            "the DP sync event loop busy-spins without an idle poll")
    print(f"[serving] model={args.model} tp={args.tp} dp={args.dp} "
          f"port={args.port} async={args.async_scheduling}", flush=True)

    # uvicorn re-raises the captured signal after its graceful shutdown, before
    # `uvicorn.run` returns, so the code below the call never sees SIGTERM. The
    # handler we install *is* the "previous handler" uvicorn restores and re-raises:
    # it shuts the engine down, and the process then unwinds normally.
    state = {}

    def _on_sigterm(signum, frame):
        if state.get("bridge") is None:
            os._exit(1)     # interrupted mid-construction: nothing to release cleanly
        state["bridge"].shutdown()

    signal.signal(signal.SIGTERM, _on_sigterm)
    bridge = EngineBridge(args.model, args.tp, args.dp, args.max_model_len,
                          args.async_scheduling)
    state["bridge"] = bridge
    app = create_app(bridge)
    try:
        uvicorn.run(app, host="127.0.0.1", port=args.port)
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    run_server(parse_args())
