import torch
from torch import nn

# mark_dynamic must be applied to the argument from *outside* the compiled region (inside, dynamo
# rejects it), and only dim 0: the token count is what varies, hidden/head dims stay static.
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
        # scale is `1 + w`, the cast comes after the multiply, and the weight is upcast first
        # (HF's Qwen3_5RMSNorm) -- at checkpoint dtype `1 + w` keeps only a few significant digits
        return (x * (1.0 + weight.float())).to(orig_dtype)
    return x.to(orig_dtype).mul_(weight)      # Qwen3 stores the scale itself


# Module level, weight *passed in*: a bound method drags its instance into dynamo's guards,
# which costs one graph per layer (24 layers, all past `cache_size_limit`)
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

    The checkpoint stores `weight ~ 0` and the scale is `1 + weight` (HF's `Qwen3_5RMSNorm`;
    vLLM reaches the same result by aliasing its `GemmaRMSNorm`). Scaling by `weight` instead
    still runs -- it just multiplies every layer output by ~0 and degenerates into a repeated
    token, which is what `tests/test_zero_centered_norm.py` guards. The GDN's gated norm below
    is unit-centred even in Qwen3.5.
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
