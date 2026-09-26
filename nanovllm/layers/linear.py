import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.distributed.parallel_state import get_tp_group


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = get_tp_group().local_rank
        self.tp_size = get_tp_group().world_size
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = get_tp_group().world_size
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def segmented_weight_loader(spec: list[tuple[int, int, bool]], tp_size: int, tp_rank: int):
    """Shard a checkpoint tensor whose dim 0 concatenates the `spec` segments.

    Each spec entry is ``(full_dim, extra, duplicate_groups)``, the shape of the
    segment before sharding -- the same convention as vLLM's
    ``mamba_v2_sharded_weight_loader``, which its GDN layers use for the depthwise
    conv weight and for A_log / dt_bias (``extra`` > 0 / ``duplicate_groups`` are
    only for replicated groups, unused here).

    Every segment contributes its own ``full_dim // tp_size`` rows to every rank,
    so the slice is taken from *inside* each segment rather than across the whole
    tensor. That is what keeps a rank's q / k / v channels -- and its A_log heads
    -- contiguous and head-aligned, which the kernels rely on: they index A_log by
    the local value-head number, and the conv state's row stride is the local
    conv width.
    """
    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        boundary = loaded_boundary = 0
        for full_dim, extra, duplicate_groups in spec:
            shard_size = full_dim // tp_size
            rank = 0 if duplicate_groups else tp_rank
            start = loaded_boundary + rank * shard_size
            take = min(shard_size, full_dim - extra - rank * shard_size)
            param.data[boundary:boundary + take, ...] = loaded_weight[start:start + take]
            boundary += shard_size
            loaded_boundary += full_dim - extra
    return loader


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      loaded_shard_id: int | tuple[int, ...] = None):
        """Load one segment -- or several consecutive ones at once.

        A tuple shard id means the checkpoint tensor spans several fused segments:
        Qwen3.5 ships the GDN's `in_proj_qkv` as one tensor while we fuse q|k|v|z
        (same convention as vLLM's `validate_shard_id`). The checkpoint tensor is
        cut per segment with the *full* widths first -- a plain chunk by tp would
        hand a rank a slice straddling two segments (rank 0 of tp=2 would get all
        of q plus the first half of k), which loads without complaint and is wrong.
        """
        shard_ids = loaded_shard_id if isinstance(loaded_shard_id, tuple) else (loaded_shard_id,)
        for shard_id in shard_ids:
            assert 0 <= shard_id < len(self.output_sizes), \
                f"shard id {shard_id} out of range for {self.output_sizes}"
            assert self.output_sizes[shard_id] % self.tp_size == 0, \
                f"segment {shard_id} of {self.output_sizes} does not divide by tp={self.tp_size}"
        if isinstance(loaded_shard_id, tuple):
            offset = 0
            for shard_id in shard_ids:
                size = self.output_sizes[shard_id]
                self.weight_loader(param, loaded_weight.narrow(self.tp_dim, offset, size), shard_id)
                offset += size
            return
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        attn_output_gate: bool = False,
    ):
        tp_size = get_tp_group().world_size
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        # with an output gate each head carries head_size query dims *and*
        # head_size gate dims, both inside the checkpoint's q_proj
        self.attn_output_gate = attn_output_gate
        output_size = (total_num_heads * (1 + int(attn_output_gate))
                       + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        q_size = self.num_heads * self.head_size * (1 + int(self.attn_output_gate))
        if loaded_shard_id == "q":
            shard_size = q_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = q_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = q_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = get_tp_group().world_size
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            get_tp_group().all_reduce(y)
        return y
