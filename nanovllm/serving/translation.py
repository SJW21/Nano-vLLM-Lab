"""Request semantics shared by the OpenAI-compatible endpoints and native /generate.

Pure stdlib on purpose: the local CPU test loads this file by path, so it must not
import ``nanovllm`` or any third-party package.

Error convention: success returns a bare value (or dict); a client mistake raises
``TranslationError`` carrying the offending request field, which the HTTP layer turns
into ``400 {error: {message, type, param, code}}``.
"""


class TranslationError(ValueError):
    """A client-facing 400: message plus the request field it refers to."""

    def __init__(self, message: str, param: str):
        super().__init__(message)
        self.param = param


def frame_max_tokens(prompt_len: int, max_tokens: int | None, max_model_len: int) -> int:
    """Resolve max_tokens against the remaining context window.

    ``None`` means "the rest of the window". An explicit value must be positive and leave
    room for the prompt: the engine drops an over-window request silently (it only records
    it in ``failed``), so the HTTP layer must reject it instead.
    """
    remaining = max_model_len - prompt_len
    if remaining <= 0:
        raise TranslationError(
            f"prompt length ({prompt_len}) leaves no room in max_model_len "
            f"({max_model_len})", "max_tokens")
    if max_tokens is None:
        return remaining
    if max_tokens <= 0:
        raise TranslationError("max_tokens must be positive", "max_tokens")
    if prompt_len + max_tokens > max_model_len:
        raise TranslationError(
            f"prompt_len ({prompt_len}) + max_tokens ({max_tokens}) exceeds "
            f"max_model_len ({max_model_len})", "max_tokens")
    return max_tokens


def map_temperature(temperature: float | None) -> float:
    """Map OpenAI's temperature onto the engine's domain.

    ``None`` means the engine default (1.0). Temperature 0 asks for greedy decoding,
    which this engine forbids (it asserts ``temperature > 1e-10``), so zero and tiny
    values are clamped to 1e-6; negatives are a client error.
    """
    if temperature is None:
        return 1.0
    if temperature < 0:
        raise TranslationError("temperature must be non-negative", "temperature")
    if temperature <= 1e-10:
        return 1e-6
    return temperature


def validate_n(n: int | None) -> int:
    """Only n=1 is supported; fan-out is rejected rather than silently dropped."""
    if n is None or n == 1:
        return 1
    raise TranslationError("only n=1 is supported", "n")


def finalize_result(prompt_ids: list[int], completion_ids: list[int], eos: int,
                    decode, ignore_eos: bool = False) -> dict:
    """Turn raw engine output into the native /generate response shape.

    The scheduler appends the sampled token *before* checking EOS, so a stop leaves the
    EOS id at the tail; strip it so text and usage do not count it. With ``ignore_eos``
    the sequence cannot stop on EOS, so a trailing id equal to eos is a normal token.
    """
    raw = list(completion_ids)
    hit_eos = (not ignore_eos) and bool(raw) and raw[-1] == eos
    token_ids = raw[:-1] if hit_eos else raw
    return {
        "text": decode(token_ids),
        "token_ids": token_ids,
        "finish_reason": "stop" if hit_eos else "length",
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(token_ids),
            "total_tokens": len(prompt_ids) + len(token_ids),
        },
    }
