import torch
from torch import nn

# torch._dynamo.mark_dynamic has to be applied to the *argument*, from outside the
# compiled region -- calling it inside is rejected ("Attempt to trace forbidden
# callable"). Left static, the token count is a specialization: every prefill
# length, chunk boundary and decode batch size mints another graph of the very same
# function until dynamo's per-function cache (config.cache_size_limit, 8) is full,
# at which point it warns and stops compiling. Only dim 0 is marked: the hidden
# size and the head dims stay static, and those are what the kernels use.
# None on torch < 2, where there is no dynamo to mark for.
_MARK_DYNAMIC = getattr(getattr(torch, "_dynamo", None), "mark_dynamic", None)


def _mark_tokens_dynamic(x: torch.Tensor) -> None:
    if _MARK_DYNAMIC is not None:
        _MARK_DYNAMIC(x, 0)


def _apply_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    orig_dtype: torch.dtype,
    zero_centred: bool,
) -> torch.Tensor:
    """Scale the normalised activations, which arrive here in fp32."""
    if zero_centred:
        # Qwen3.5 stores a zero-centred weight, so the scale is `1 + w` and the
        # cast happens after the multiply (HF's Qwen3_5RMSNorm: "Llama does
        # x.to(float16) * w whilst Qwen3_5 is (x * w).to(float16)"). The
        # `.float()` matters -- the weight arrives in the checkpoint dtype,
        # where `1.0 + weight` would keep only a few significant digits.
        return (x * (1.0 + weight.float())).to(orig_dtype)
    # Qwen3 stores the scale itself, and casts before multiplying.
    return x.to(orig_dtype).mul_(weight)


# Compiled at module level, with the weight *passed in* instead of read off
# `self`. `torch.compile` guards on its arguments, and a bound method drags the
# instance into those guards -- every layer builds its own norm, so a method is
# recompiled once per layer (24 layers, 24 graphs of the same shape, well past
# `dynamo.config.cache_size_limit = 8`, after which dynamo warns and gives up on
# compiling). As an argument the weight is guarded by shape/dtype only, so every
# layer shares one graph.
@torch.compile
def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    zero_centred: bool,
) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    return _apply_scale(x, weight, orig_dtype, zero_centred)


@torch.compile
def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    zero_centred: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    x = x.float().add_(residual.float())
    residual = x.to(orig_dtype)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    return _apply_scale(x, weight, orig_dtype, zero_centred), residual


class RMSNorm(nn.Module):

    # Qwen3 stores the scale itself; Qwen3.5 stores a zero-centred weight.
    zero_centred = False

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        _mark_tokens_dynamic(x)
        if residual is None:
            return rms_norm(x, self.weight, self.eps, self.zero_centred)
        else:
            _mark_tokens_dynamic(residual)
            return add_rms_norm(x, residual, self.weight, self.eps, self.zero_centred)


class Qwen3p5RMSNorm(RMSNorm):
    """Qwen3.5's plain (non-gated) RMSNorm -- the weight is zero-centred.

    The checkpoint stores `weight ~ 0` and the scale is `1 + weight` (HF's
    `Qwen3_5RMSNorm`; vLLM gets the same result by aliasing its `GemmaRMSNorm` as
    `Qwen3NextRMSNorm`). Qwen3 stores the scale itself, so the plain `RMSNorm`
    stays as it is -- and so does the GDN's gated norm below, which is
    unit-centred even in Qwen3.5.

    Scaling by `weight` instead of `1 + weight` multiplies every layer output by
    ~0 with no shape error anywhere: the model still runs and just degenerates
    into a repeated token, which is what `tests/test_zero_centered_norm.py`
    guards.
    """

    zero_centred = True

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__(hidden_size, eps)
        self.weight = nn.Parameter(torch.zeros(hidden_size))


class Qwen3p5RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.act = nn.functional.silu

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * self.act(gate.to(torch.float32))

        return hidden_states.to(input_dtype)
