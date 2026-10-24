"""Mixture-of-Experts layer (`moe_tp`).

Every rank holds all experts and one 1/EP slice of the intermediate
dimension. Experts are evaluated with grouped GEMMs and combined with one
all-reduce over the EP group; there is no all-to-all, because tokens are
replicated across TP ranks.

Eager and CUDA-graph execution share one path: `moe_forward` allocates its
workspace per call, and inside `torch.cuda.graph(pool=...)` those allocations
come from the graph's private pool.
"""

import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.layers.moe_gemm import MoEWeights, moe_forward
from nanovllm.distributed.parallel_state import get_ep_group, get_tp_group


class MoELayer(nn.Module):
    """MoE MLP: all experts local, intermediate dimension sharded by EP.

    Weights use the kernel's terminal K-major layout:
      - stacked_gate_tp / stacked_up_tp: [E, H, M/EP]
      - stacked_down_tp:                  [E, M/EP, H]
    Sharding the intermediate dimension keeps Qwen3-30B-A3B at about
    14.5GB/rank for EP=4; a full copy would not fit a 32GB card.

    With DP=1, routing is identical on every rank (tokens are replicated
    across TP), so each rank computes its M-shard partial and one all-reduce
    over EP produces the full output.
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
        ep_size: int,
        ep_rank: int,
        hidden_act: str = "silu",
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        assert hidden_act == "silu", "MoE experts support only SiLU."
        assert moe_intermediate_size % ep_size == 0, (
            f"moe_intermediate_size ({moe_intermediate_size}) must be "
            f"divisible by ep_size ({ep_size})")
        M_EP = moe_intermediate_size // ep_size
        # down_proj has K = M/EP; EVEN_K requires a multiple of 64.
        assert M_EP % 64 == 0, (
            f"M/EP = {M_EP} must be a multiple of 64 (ep_size={ep_size})")

        tp_size = get_tp_group().world_size
        if ep_size > tp_size:
            raise NotImplementedError(
                f"EP ({ep_size}) > TP ({tp_size}) would require data "
                "parallelism; DP>1 is not implemented")

        self.top_k = top_k
        self.ep_size = ep_size
        self.norm_topk_prob = norm_topk_prob

        # Router is replicated; named "gate" to match the HF checkpoint.
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)

        # Expert weights in the kernel's terminal layout, so finalize only
        # has to concatenate gate|up (no per-expert transpose copy).
        self.stacked_gate_tp = nn.Parameter(
            torch.empty(num_experts, hidden_size, M_EP))
        self.stacked_up_tp = nn.Parameter(
            torch.empty(num_experts, hidden_size, M_EP))
        self.stacked_down_tp = nn.Parameter(
            torch.empty(num_experts, M_EP, hidden_size))

        # HF layouts: gate/up [M, D] (M in dim 0), down [D, M] (M in dim 1).
        # Narrow the EP slice, then transpose into K-major.
        m_slice = ep_rank * M_EP

        def _make_tp_loader_m_dim0():
            def loader(param, loaded_weight, expert_id=0):
                w = loaded_weight.narrow(0, m_slice, M_EP)   # [M_EP, D]
                param.data[expert_id].copy_(w.t())           # [D, M_EP]
            return loader

        def _make_tp_loader_m_dim1():
            def loader(param, loaded_weight, expert_id=0):
                w = loaded_weight.narrow(1, m_slice, M_EP)   # [D, M_EP]
                param.data[expert_id].copy_(w.t())           # [M_EP, D]
            return loader

        self.stacked_gate_tp.weight_loader = _make_tp_loader_m_dim0()
        self.stacked_up_tp.weight_loader = _make_tp_loader_m_dim0()
        self.stacked_down_tp.weight_loader = _make_tp_loader_m_dim1()

        self._moe_weights = None

    @property
    def moe_finalized(self) -> bool:
        return self._moe_weights is not None

    def finalize_moe(self):
        """Build `MoEWeights` and free the HF-layout source parameters.

        Called once after loading and before warmup. Concatenate and release
        layer by layer: collecting every layer's `w13` before freeing would
        add about 9.7GB of peak memory.
        """
        if self._moe_weights is not None:
            return
        w13 = torch.cat(
            [self.stacked_gate_tp.data, self.stacked_up_tp.data], dim=2)
        down = self.stacked_down_tp.data
        self._moe_weights = MoEWeights(w13, down)
        del self.stacked_gate_tp, self.stacked_up_tp, self.stacked_down_tp

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MoE layer."""
        if self._moe_weights is None:
            raise RuntimeError(
                "MoE weights are not finalized; call finalize_moe() after "
                "loading and before warmup")

        scores = self.gate(hidden_states)               # [N, num_experts]
        weights, indices = self._topk_routing(scores)   # [N, top_k] each
        out = moe_forward(hidden_states, weights, indices, self._moe_weights)

        if self.ep_size > 1:
            # Every EP rank must join, including ranks with no tokens.
            get_ep_group().all_reduce(out)
        return out

    def _topk_routing(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Softmax + top-k routing.

        Returns (weights [N, top_k], indices [N, top_k]).
        """
        orig_dtype = scores.dtype
        scores = F.softmax(scores, dim=-1, dtype=torch.float32)
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights.to(orig_dtype), indices
