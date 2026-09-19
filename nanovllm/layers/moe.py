"""MoE 层（moe_tp）：全专家 × 1/EP intermediate 分片 + AllReduce[EP]。

计算走 `moe_gemm` 的 grouped GEMM（jagged M、无固定容量、virtual-SM 调度）；
通信为纯集合通信（无 a2a）：每 rank 只计算自己 M 分片的 partial，AR[EP] 在
全 EP 组上求和得到完整输出。这是 AgRs 形态（all-gather dispatch +
reduce-scatter combine）在当前布局下的退化实现——token 在 TP 组内一致时
dispatch 是恒等操作，combine 退化为一次 all_reduce（DP>1 的 AG[DP] + RS
见 plan S6）。

CUDA graph：中间张量必须预分配、地址固定（见 moe_gemm.MoEGraphBuffers），
因此 `finalize_moe()` 在加载后一次成型；层内按 capture 状态三分支，见
`_moe_compute`。
"""

import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.layers.moe_gemm import (
    MoEWeights, MoEGraphBuffers, moe_forward, moe_forward_graph,
    moe_forward_graph_captured)
from nanovllm.distributed.parallel_state import get_ep_group, get_tp_group


class MoELayer(nn.Module):
    """Mixture-of-Experts layer（moe_tp）。

    权重布局：全专家本地持有，每个专家沿 intermediate（M）维切 EP 份
      - stacked_gate_tp / stacked_up_tp: [E, M/EP, D]
      - stacked_down_tp:                  [E, D, M/EP]
    （EP>1 是 32GB 显存的硬约束：全量专家权重的完整副本 58GB/rank 不可行，
    M 维切分后约 14.5GB/rank。）

    数据流（DP=1）：
        gate → topk_routing → grouped GEMM（本 rank 的 M 分片 → partial）
        → AR[EP] 求和 → 输出
    路由每 rank 独立计算：token 在 TP 组内一致（attention 沿 head 维切分 +
    AR），各 rank 的 topk 结果因此一致，partial 可按 token 直接求和。

    旧 moe_ep（All2All/fixbuf）路径已移除（git 历史可查）；moe_mode 保留为
    单值开关，传入旧值会得到明确报错。
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
        ep_size: int,
        ep_rank: int,
        moe_mode: str = "tp",
        hidden_act: str = "silu",
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        assert hidden_act == "silu", "Only SiLU activation is supported for MoE experts."
        assert moe_mode == "tp", (
            f"moe_mode={moe_mode!r} 不受支持：moe_ep（All2All 后端）已移除，"
            f"当前仅支持 'tp'（AllReduce 后端）")
        assert num_experts % ep_size == 0, \
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"
        assert moe_intermediate_size % ep_size == 0, \
            f"moe_intermediate_size ({moe_intermediate_size}) must be divisible by ep_size ({ep_size})"
        M_EP = moe_intermediate_size // ep_size
        # M/EP 必须是 64 的倍数：down 阶段 K = M/EP，K % 64 != 0 会使 grouped
        # GEMM 的 EVEN_K 失效（退化为逐元素 mask 路径）。A3B（I=768）下
        # EP ∈ {1,2,4}，EP=8（M/EP=96）被此断言拒绝。
        assert M_EP % 64 == 0, \
            f"M/EP = {M_EP} must be a multiple of 64 (EP={ep_size} 使 down 的 EVEN_K 失效)"

        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.moe_mode = moe_mode
        self.norm_topk_prob = norm_topk_prob
        self.tp_size = get_tp_group().world_size
        self.dp_size = ep_size // self.tp_size
        if self.dp_size > 1:
            raise NotImplementedError(
                "DP>1 暂不支持（plan S6 预留 AG[DP] + RS/AR 骨架）；"
                "当前实现要求 EP == TP（DP=1）")

        # Router (gate): fully replicated, small weight footprint
        # Named "gate" to match HF checkpoint key: model.layers.X.mlp.gate.weight
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)

        # --- Stacked expert weights: ALL experts × M/EP slice ---
        self.stacked_gate_tp = nn.Parameter(torch.empty(num_experts, M_EP, hidden_size))
        self.stacked_up_tp = nn.Parameter(torch.empty(num_experts, M_EP, hidden_size))
        self.stacked_down_tp = nn.Parameter(torch.empty(num_experts, hidden_size, M_EP))
        # M 维 narrow loader：切片索引 = rank 在 EP group 内位置（EP group = 全体
        # rank → ep_rank）。HF loaded_weight 布局：gate/up = [M, D]（M 在 dim0）；
        # down = [D, M]（M 在 dim1）。
        m_slice = self.ep_rank * M_EP
        def _make_tp_loader_m_dim0():      # gate/up：loaded [M, D] 的 M 维 = dim0
            def loader(param, loaded_weight, expert_id=0):
                w = loaded_weight.narrow(0, m_slice, M_EP)
                param.data[expert_id].copy_(w)
            return loader
        def _make_tp_loader_m_dim1():      # down：loaded [D, M] 的 M 维 = dim1
            def loader(param, loaded_weight, expert_id=0):
                w = loaded_weight.narrow(1, m_slice, M_EP)
                param.data[expert_id].copy_(w)
            return loader
        self.stacked_gate_tp.weight_loader = _make_tp_loader_m_dim0()
        self.stacked_up_tp.weight_loader   = _make_tp_loader_m_dim0()
        self.stacked_down_tp.weight_loader = _make_tp_loader_m_dim1()

        # moe_gemm 计算路径：finalize_moe() 之后就绪。
        # _moe_weights 持有转置后的专家权重（MoEWeights 自己的拷贝），
        # _moe_bufs 持有 graph 兼容的固定 buffer（pool 共享时只建 stage 表）。
        self._moe_weights = None
        self._moe_bufs = None

    # ------------------------------------------------------------------
    # 计算路径（moe_gemm grouped GEMM）
    # ------------------------------------------------------------------

    @property
    def moe_finalized(self) -> bool:
        """finalize_moe() 是否已完成（幂等入口判断用）。"""
        return self._moe_weights is not None

    def finalize_moe(self, pool=None, max_tokens=None):
        """加载完成后、warmup 之前调用一次（幂等）。

        1. stacked_gate_tp/up_tp → w1 = [E, 2*M_EP, H]（gate 在前、up 在后，
           匹配 fused kernel 在 half = N//2 处的 gate/up 切分）→ MoEWeights；
        2. 释放 HF 布局的源参数——EP=4 时 302MB/层 × 48 ≈ 14.5GB，权重出现
           第二份副本会直接 OOM（32GB 卡）；
        3. 构造 MoEGraphBuffers（传 pool 时跨层共享 buffer，只建本层 stage 表）。

        注意：_GroupedGemmStage 会把权重与 buffer 的 data_ptr 烘进表里，
        finalize 之后不得再移动/重建权重或 buffer。
        """
        if self._moe_weights is not None:
            return
        w1 = torch.cat([self.stacked_gate_tp.data, self.stacked_up_tp.data], dim=1)
        down = self.stacked_down_tp.data
        self._moe_weights = MoEWeights(w1, down)
        del self.stacked_gate_tp, self.stacked_up_tp, self.stacked_down_tp
        del w1, down
        self._moe_bufs = MoEGraphBuffers(self._moe_weights, max_tokens=max_tokens,
                                         top_k=self.top_k, pool=pool)

    def _moe_compute(self, hidden_states, topk_weights, topk_ids):
        """moe_gemm 计算路径三分支：图内 / staged / eager。

        与 model_runner 的 eager/graph 分流方向一致：
          - capture 区内 → 图内入口（输入来自图私有内存池，地址跨 replay 稳定）；
          - 图外且装得下 pool 容量 → staged（set_inputs + 固定 buffer）；
          - 图外且超容量（大 batch prefill）→ eager 动态分配。

        三条路径共用同一 kernel 与同一段计算逻辑，差别只在数据组织方式。
        """
        if self._moe_bufs is None:
            raise RuntimeError(
                "MoE 计算路径需要先调用 finalize_moe()（加载完成后、warmup 前）")
        bufs = self._moe_bufs
        if torch.cuda.is_current_stream_capturing():
            return moe_forward_graph_captured(
                bufs, hidden_states, topk_weights, topk_ids)
        if hidden_states.size(0) <= bufs.max_tokens:
            h, w, ids = bufs.set_inputs(hidden_states, topk_weights, topk_ids)
            return moe_forward_graph(bufs, h, w, ids)
        return moe_forward(hidden_states, topk_weights, topk_ids, self._moe_weights)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for MoE layer.

        Args:
            hidden_states: [N, hidden_size] input token representations.

        Returns:
            [N, hidden_size] output after expert computation and merge.
        """
        scores = self.gate(hidden_states)               # [N, num_experts]
        weights, indices = self._topk_routing(scores)   # [N, top_k] each

        out = self._moe_compute(hidden_states, weights, indices)
        if self.ep_size > 1:
            # partial 求和（全组必须参与，含空输入——否则其他 rank 死锁）
            get_ep_group().all_reduce(out)
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _topk_routing(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Softmax + top-k routing.

        Args:
            scores: [N, num_experts] raw router logits.

        Returns:
            weights: [N, top_k] normalized routing weights.
            indices: [N, top_k] selected expert indices (long).
        """
        orig_dtype = scores.dtype
        scores = F.softmax(scores, dim=-1, dtype=torch.float32)
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights.to(orig_dtype), indices
