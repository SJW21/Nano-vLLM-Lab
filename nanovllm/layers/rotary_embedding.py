from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int | None = None,
) -> torch.Tensor:
    """Rotate the first ``rotary_dim`` dims of each head, pass the tail through.

    Qwen3.5 rotates only 25% of the head (partial rotary); the remaining dims ride
    along untouched, laid out as in transformers' ``apply_rotary_pos_emb``.
    """
    if rotary_dim is None or rotary_dim == x.shape[-1]:
        x_rot, x_pass = x.float(), None
    else:
        x_rot, x_pass = x.float()[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = torch.chunk(x_rot, 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    out = torch.cat((y1, y2), dim=-1)
    if x_pass is not None:
        out = torch.cat((out, x_pass), dim=-1)
    return out.to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert 0 < rotary_dim <= head_size, "rotary_dim must fit inside the head"
        assert rotary_dim % 2 == 0, "rotary_dim must be even (pairs are rotated)"
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin, self.rotary_dim)
        key = apply_rotary_emb(key, cos, sin, self.rotary_dim)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
