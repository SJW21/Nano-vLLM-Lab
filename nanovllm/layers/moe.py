import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.moe_kernels import BLOCK_M, moe_align, moe_ffn_pair
from nanovllm.distributed.parallel_state import get_ep_group, get_tp_group, get_dp_group

# Module-level cache for alignment buffers (shared across all MoE layers
# because all layers have the same D and N → identical T_MAX).
_moe_buf_cache: dict[tuple, dict[str, torch.Tensor]] = {}


def _cap_from_expected(expected, bound):
    """CAP 计算：3σ Chernoff 期望值对齐 BLOCK_M，夹紧到 bound 对齐值。

    moe_ffn_pair 要求 chunk 不跨块（CAP % BLOCK_M == 0）且 T_ALIGN 为 BLOCK_M
    倍数，故下取整到 BLOCK_M；NK 本身是可行上界（每个 pair 至多占一行）。
    """
    cap = max(((expected + BLOCK_M - 1) // BLOCK_M) * BLOCK_M, BLOCK_M)
    bound_aligned = max(((bound + BLOCK_M - 1) // BLOCK_M) * BLOCK_M, BLOCK_M)
    return min(cap, bound_aligned)


def get_or_create_moe_buffers(mode: str, device, dtype, N, D, T_MAX, num_blocks, CAP):
    """跨层共享的 MoE 对齐/通信 buffer（各层 D、T_MAX 相同 → O(1) 份）。

    mode: "fixbuf"（moe_ep decode 固定容量 a2a）或 "moe_tp"（129 块布局）。
    cache key 含 mode，两种模式互不别名。block_id=0 保留为 null 块，共
    num_blocks 个有效块 → cnt 长度 num_blocks+1，T_MAX = (num_blocks+1)*CAP。
    fixbuf 模式额外提供 dispatch 用 send/recv buffer（§5.1），含随行传送
    路由权重的 w 通道 send_w_buf/recv_w_buf。
    e_align 在创建时填充哨兵（moe_ffn_pair 读 e_max-1 定位专家权重），fixbuf
    compute 与 moe_tp 共用统一约定：哨兵 = block 号（1..num_blocks），block 0
    保持 0（null 块）；专家 e 在 block e+1 → 哨兵 e+1 → kernel e0 = e。
    （e_align 仅 compute/moe_tp 条目消费；send_e_buf 每轮 scatter，不走 e_align。）
    """
    key = (mode, device.index if device.index is not None else device, dtype, N, D, T_MAX, num_blocks, CAP)
    if key not in _moe_buf_cache:
        bufs = {
            "x_align":   torch.empty(T_MAX, D, dtype=dtype, device=device),
            "w_align":   torch.empty(T_MAX,    dtype=dtype, device=device),
            "e_align":   torch.zeros(T_MAX,    dtype=torch.int64, device=device),
            "cnt":       torch.zeros(num_blocks + 1, dtype=torch.int32, device=device),
            "out_align": torch.empty(T_MAX, D, dtype=dtype, device=device),
            "partial":   torch.empty(N, D,     dtype=dtype, device=device),
        }
        if mode == "fixbuf":
            # dispatch 阶段：send_buf [(EP+1)*CAP, D]；a2a ×3（token + expert_id + w 通道）
            bufs.update(
                send_buf=torch.empty(T_MAX, D, dtype=dtype, device=device),
                send_e_buf=torch.zeros(T_MAX, dtype=torch.int64, device=device),
                send_w_buf=torch.empty(T_MAX, dtype=dtype, device=device),
                recv_buf=torch.empty(T_MAX, D, dtype=dtype, device=device),
                recv_e_buf=torch.zeros(T_MAX, dtype=torch.int64, device=device),
                recv_w_buf=torch.empty(T_MAX, dtype=dtype, device=device),
            )
        # e_align 哨兵（创建时一次填充，之后每轮不重置）：哨兵 = block 号（1..num_blocks），
        # block 0 保持 0 = null 块。moe_ffn_pair 读 e_max-1 定位专家权重：专家 e 在 block e+1
        # → 哨兵 e+1 → e0 = e。fixbuf compute 与 moe_tp 共用同一约定。
        for b in range(1, num_blocks + 1):
            bufs["e_align"][b * CAP:(b + 1) * CAP] = b
        _moe_buf_cache[key] = bufs
    return _moe_buf_cache[key]


class ExpertFFN(nn.Module):
    """Individual expert FFN with full (unsharded) weights.

    In EP mode, each rank holds a subset of experts with complete weights.
    A single expert has small weight footprint (3 × hidden × moe_intermediate),
    so there is no need for TP sharding inside the expert.
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(moe_intermediate_size, hidden_size, bias=False)
        self.act = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, hidden_size]
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # SiluAndMul expects concatenated [gate, up] along last dim
        y = self.act(torch.cat([gate, up], dim=-1))
        return self.down_proj(y)


class MoELayer(nn.Module):
    """Mixture-of-Experts layer supporting two parallel modes.

    EP mode (moe_mode="ep", default):
        All2All backend.  Experts are sharded by expert ID: each rank owns
        E/EP complete experts, and tokens are dispatched to their target
        experts via all-to-all.  Internally, decode uses the fixbuf
        fixed-capacity a2a path (_forward_ep_a2a_fixbuf) and prefill uses
        the dynamic a2a path (_forward_ep_a2a_eager); the routing threshold
        is N_local × K ≤ 2048 / TP (§2.2).

    TP mode (moe_mode="tp"):
        AllReduce backend.  Expert weights are sharded along the Hidden/M
        dimension; each rank computes a partial result for all tokens,
        combined via AllReduce over the full EP group
        (_forward_tp, §2.1).

    The router (gate) is always fully replicated (no sharding) since it is
    small: hidden_size × num_experts ≈ 2048 × 128 ≈ 262K parameters.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
        ep_size: int,
        ep_rank: int,
        moe_mode: str = "ep",
        hidden_act: str = "silu",
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        assert hidden_act == "silu", "Only SiLU activation is supported for MoE experts."
        assert moe_mode in ("ep", "tp"), f"Invalid moe_mode: {moe_mode}"
        assert num_experts % ep_size == 0, \
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"

        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.moe_mode = moe_mode
        self.norm_topk_prob = norm_topk_prob
        self.experts_per_rank = num_experts // ep_size
        # EP group = 全体 rank → dp_size = ep_size // tp_size（§2.1/§2.2）
        self.tp_size = get_tp_group().world_size

        # Router (gate): fully replicated, small weight footprint
        # Named "gate" to match HF checkpoint key: model.layers.X.mlp.gate.weight
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)

        # Expert weights: when EP > 1, store as stacked tensors directly
        # (ready for bmm, no runtime copies).  For ep_size == 1 (128 experts
        # → would be ~3.6 GB/layer), keep individual ExpertFFN modules and
        # stack only active experts on-the-fly in _compute_local.
        if moe_mode == "tp" and ep_size > 1:
            # --- Stacked parameters (TP mode: ALL experts × M/EP slice, §1) ---
            # gate/up: [E, M/EP, D]；down: [E, D, M/EP]，E = num_experts（全专家本地持有）
            assert moe_intermediate_size % ep_size == 0, \
                f"moe_intermediate_size ({moe_intermediate_size}) 必须能被 ep_size ({ep_size}) 整除（R8）"
            M_EP = moe_intermediate_size // ep_size
            self.stacked_gate_tp = nn.Parameter(torch.empty(num_experts, M_EP, hidden_size))
            self.stacked_up_tp = nn.Parameter(torch.empty(num_experts, M_EP, hidden_size))
            self.stacked_down_tp = nn.Parameter(torch.empty(num_experts, hidden_size, M_EP))
            # M 维 narrow loader：份数 = EP，切片索引 = rank 在 EP group 内位置（EP group = 全体 rank → ep_rank）。
            # HF loaded_weight 布局：gate/up = [M, D]（M 在 dim0）；down = [D, M]（M 在 dim1）。
            # 切片后形状与 param slot 对齐：gate/up [M_EP, D] ← [M_EP, D]；down [D, M_EP] ← [D, M_EP]。
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
            self.experts = nn.ModuleList()      # API 兼容
        elif ep_size > 1:
            # --- Stacked parameters (EP mode, ~32 experts/rank) ---
            # Shape follows nn.Linear convention [out_features, in_features]:
            #   gate/up: [moe_intermediate, hidden]   = [M, D]
            #   down:    [hidden, moe_intermediate]    = [D, M]
            # bmm uses .transpose(1,2) to get [E, D, M] — a view, zero copy.
            E = self.experts_per_rank
            self.stacked_gate = nn.Parameter(
                torch.empty(E, moe_intermediate_size, hidden_size))
            self.stacked_up = nn.Parameter(
                torch.empty(E, moe_intermediate_size, hidden_size))
            self.stacked_down = nn.Parameter(
                torch.empty(E, hidden_size, moe_intermediate_size))

            # Custom weight-loaders: write directly into expert slot.
            def _make_loader():
                def loader(param, loaded_weight, expert_id=0):
                    param.data[expert_id].copy_(loaded_weight)
                return loader
            self.stacked_gate.weight_loader = _make_loader()
            self.stacked_up.weight_loader   = _make_loader()
            self.stacked_down.weight_loader = _make_loader()

            # Keep an empty experts list for API compatibility with tests.
            self.experts = nn.ModuleList()
        else:
            # --- Individual experts (ep_size=1, 128 experts/rank) ---
            self.experts = nn.ModuleList([
                ExpertFFN(hidden_size, moe_intermediate_size)
                for _ in range(self.experts_per_rank)
            ])

        # moe_ep 且 TP>1 时为 True，供 decoder layer 决定 o_proj 的 RS 输出与
        # residual 位置（§5.1）。
        self.use_rs_wrap = (moe_mode == "ep") and (get_tp_group().world_size > 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for MoE layer.

        Args:
            hidden_states: [N, hidden_size] input token representations.

        Returns:
            [N, hidden_size] output after expert computation and merge.
        """
        scores = self.gate(hidden_states)               # [N, num_experts]
        weights, indices = self._topk_routing(scores)   # [N, top_k] each

        if self.ep_size == 1:
            # ep_size==1 无通信：两模式退化为全专家本地全量计算（数学等价）；
            # 该配置构造走个体专家分支（无 stacked_*_tp），统一走 local 路径。
            return self._forward_ep_local(hidden_states, weights, indices)
        if self.moe_mode == "ep":
            # §2.2 分流：输入为 RS 后 [N/TP, D]（N 指全局 batch）→ 局部换算
            # N_local × K ≤ 2048/TP 走 fixbuf（decode 主路径），否则走动态 a2a（prefill）。
            if hidden_states.size(0) * self.top_k <= 2048 // self.tp_size:
                return self._forward_ep_a2a_fixbuf(hidden_states, weights, indices)
            return self._forward_ep_a2a_eager(hidden_states, weights, indices)
        # moe_mode == "tp"（此处 ep_size > 1，stacked_*_tp 恒存在）：
        # §2.1 AllReduce 后端，全 token 本地计算 + AR[EP]（+ AG[DP]/slice）
        return self._forward_tp(hidden_states, weights, indices)

    # ------------------------------------------------------------------
    # EP mode (All2All, eager / prefill): dispatch → compute → combine
    # ------------------------------------------------------------------

    def _forward_ep_a2a_eager(
        self,
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        N, D = hidden_states.shape
        K = self.top_k
        ep_group = get_ep_group() if self.ep_size > 1 else None

        # --- Phase 0: route + EP mapping ---
        # ep_rank_target: which EP rank owns the expert for each (token, top_k) pair
        ep_rank_target = indices // self.experts_per_rank      # [N, K], range [0, ep_size)
        local_expert_id = indices % self.experts_per_rank       # [N, K], range [0, experts_per_rank)

        # --- Phase 0b: metadata exchange (skip if ep_size == 1) ---
        if self.ep_size > 1:
            send_counts = torch.bincount(
                ep_rank_target.reshape(-1), minlength=self.ep_size
            ).to(torch.int64)
            # Exchange split sizes (ep_size int64 values ~ 64 bytes total)
            recv_counts = torch.empty(self.ep_size, dtype=torch.int64, device=hidden_states.device)
            torch.distributed.all_to_all_single(
                recv_counts, send_counts,
                output_split_sizes=[1] * self.ep_size,
                input_split_sizes=[1] * self.ep_size,
                group=ep_group.device_group,
            )
            recv_counts = recv_counts.tolist()
            total_recv = sum(recv_counts)
        else:
            send_counts = [N * K]
            recv_counts = [N * K]
            total_recv = N * K
            # ep_rank_target is all zeros when ep_size=1

        # --- Phase 1: prepare dispatch tensors ---
        # Expand and flatten: [N, D] → [N, K, D] → [N*K, D]
        tokens_flat = hidden_states.unsqueeze(1).expand(-1, K, -1).reshape(-1, D)
        weights_flat = weights.reshape(-1)                       # [N*K]
        local_exp_flat = local_expert_id.reshape(-1)             # [N*K]
        ep_rank_flat = ep_rank_target.reshape(-1)                # [N*K]

        # --- Phase 2: sort + dispatch all-to-all ---
        if self.ep_size > 1:
            # Sort by target EP rank so tokens to the same destination are contiguous
            sort_idx = torch.argsort(ep_rank_flat)
            unsort_idx = torch.argsort(sort_idx)

            tokens_sorted = tokens_flat[sort_idx]
            weights_sorted = weights_flat[sort_idx]
            local_exp_sorted = local_exp_flat[sort_idx]

            # Dispatch tokens, weights, expert ids
            dispatched_tokens = ep_group.all_to_all(
                tokens_sorted, output_split_sizes=recv_counts, input_split_sizes=send_counts.tolist()
            )
            dispatched_weights = ep_group.all_to_all(
                weights_sorted, output_split_sizes=recv_counts, input_split_sizes=send_counts.tolist()
            )
            dispatched_exp_ids = ep_group.all_to_all(
                local_exp_sorted, output_split_sizes=recv_counts, input_split_sizes=send_counts.tolist()
            )
        else:
            dispatched_tokens = tokens_flat
            dispatched_weights = weights_flat
            dispatched_exp_ids = local_exp_flat
            unsort_idx = torch.arange(N * K, device=hidden_states.device)

        # --- Phase 3: local expert compute ---
        expert_outputs = self._compute_local(dispatched_tokens, dispatched_exp_ids)

        # --- Phase 4: combine all-to-all (reverse dispatch) ---
        if self.ep_size > 1:
            # Reverse: send split = recv_counts, receive split = send_counts
            combined_outputs = ep_group.all_to_all(
                expert_outputs, output_split_sizes=send_counts.tolist(), input_split_sizes=recv_counts
            )
            combined_weights = ep_group.all_to_all(
                dispatched_weights, output_split_sizes=send_counts.tolist(), input_split_sizes=recv_counts
            )
        else:
            combined_outputs = expert_outputs
            combined_weights = dispatched_weights

        # --- Phase 5: unsort + weighted merge ---
        combined_outputs = combined_outputs[unsort_idx]     # [N*K, D]
        combined_weights = combined_weights[unsort_idx]     # [N*K]

        output_3d = combined_outputs.reshape(N, K, D)
        weights_3d = combined_weights.reshape(N, K)
        output = (output_3d * weights_3d.unsqueeze(-1)).sum(dim=1)  # [N, D]

        return output

    def _forward_ep_a2a_fixbuf(self, hidden_states, weights, indices):
        """§2.2 moe_ep decode 主路径：RS 后 [N/TP, D] → 固定容量 a2a 两跳。

        数据流（行序保持链，全程固定 shape + 固定 split，无 .tolist()/.item()）：
          1. dispatch 对齐（block_id=target_rank+1）→ send_buf 块 i（1-based）= 发往
             rank i-1 的 pair；send_e_buf 同时 scatter 哨兵 e+1（无效 pair 落 null 块）；
          2. 正向 a2a ×3（token / expert_id / 路由权重三通道，split=[send_CAP]*EP）；
          3. compute 对齐：recv 行序天然按 source_rank 分块，block_id=local_exp+1
             （与 moe_tp 统一约定：专家 e → block e+1）→ x_align/w_align；
          4. moe_ffn_pair（全量权重，kernel 内乘路由权重 w）；
          5. gather-back：out_align[recv_pos] 恢复 recv 行序 → 复用 send_buf[send_CAP:]；
          6. 逆向 a2a：结果按 source_rank 分块回源（result 复用 recv_buf）；
          7. 源端 send_pos（含 null 段偏移）直接归组 → 直接 sum 不除 Σw。
        """
        N, D = hidden_states.shape
        K = self.top_k
        E = self.ep_size
        device, dtype = hidden_states.device, hidden_states.dtype
        ep_group = get_ep_group()

        if N == 0:
            return hidden_states

        # ---- expand + EP mapping ----
        NK = N * K
        x_flat = hidden_states.unsqueeze(1).expand(-1, K, -1).reshape(NK, D)
        w_flat = weights.reshape(NK)
        target_rank = indices.reshape(NK) // self.experts_per_rank   # [NK] 0..EP-1
        local_exp = indices.reshape(NK) % self.experts_per_rank       # [NK] 0..E_local-1

        # ---- CAP（3σ Chernoff，对齐 BLOCK_M，上界夹紧到 NK）----
        send_CAP = _cap_from_expected(3 * NK // E, NK)
        recv_CAP = _cap_from_expected(3 * NK // self.num_experts, NK)

        # ---- 取缓存（dispatch / compute 两个条目）+ 本轮参与 buffer 清零 ----
        bufs_send = get_or_create_moe_buffers(
            "fixbuf", device, dtype, N, D,
            T_MAX=(E + 1) * send_CAP, num_blocks=E, CAP=send_CAP)
        bufs_compute = get_or_create_moe_buffers(
            "fixbuf", device, dtype, N, D,
            T_MAX=(self.experts_per_rank + 1) * recv_CAP,
            num_blocks=self.experts_per_rank, CAP=recv_CAP)
        # 对完整 buffer zero_（[send_CAP:] 是 view，自动覆盖）；e_align 为静态哨兵不参与
        for buf in (bufs_send["send_buf"], bufs_send["send_e_buf"], bufs_send["send_w_buf"],
                    bufs_send["recv_buf"], bufs_send["recv_e_buf"], bufs_send["recv_w_buf"],
                    bufs_send["x_align"], bufs_send["w_align"], bufs_send["cnt"],
                    bufs_compute["x_align"], bufs_compute["w_align"], bufs_compute["cnt"]):
            buf.zero_()

        # ---- dispatch 对齐：block_id = target_rank+1，块 1..EP = 目标 rank 0..EP-1 ----
        send_pos, send_keep = moe_align(
            x_flat, w_flat,
            valid_mask=torch.ones(NK, dtype=torch.int32, device=device),
            block_ids=(target_rank + 1).to(torch.int64),
            x_align=bufs_send["send_buf"], w_align=bufs_send["send_w_buf"],
            cnt=bufs_send["cnt"],
            NUM_BLOCKS=E, CAP=send_CAP)

        # ---- send_e_buf 填充（保持门控：无效 pair 落 null 块行 0，防污染 recv 端）----
        safe_pos = torch.where(send_keep.bool(), send_pos, torch.zeros_like(send_pos))
        bufs_send["send_e_buf"].scatter_(
            0, safe_pos,
            torch.where(send_keep.bool(), local_exp + 1, torch.zeros_like(local_exp)))

        # ---- 正向 a2a ×3（token / expert_id / 路由权重，固定 split → graph 可捕获）----
        # 有效区 = [send_CAP, (E+1)*send_CAP) 共 E*send_CAP 行（null 块占 [0, send_CAP)，
        # CAP 行宽而非 1 行，不参与 a2a）
        split = [send_CAP] * E
        dist.all_to_all_single(bufs_send["recv_buf"][send_CAP:], bufs_send["send_buf"][send_CAP:],
                               output_split_sizes=split, input_split_sizes=split,
                               group=ep_group.device_group)
        dist.all_to_all_single(bufs_send["recv_e_buf"][send_CAP:], bufs_send["send_e_buf"][send_CAP:],
                               output_split_sizes=split, input_split_sizes=split,
                               group=ep_group.device_group)
        dist.all_to_all_single(bufs_send["recv_w_buf"][send_CAP:], bufs_send["send_w_buf"][send_CAP:],
                               output_split_sizes=split, input_split_sizes=split,
                               group=ep_group.device_group)

        # ---- compute 对齐：输入行序 = recv 行序（按 source_rank 分块）----
        # 统一 block_id 约定（§5.2）：专家 e → block e+1（block 0 = null）。
        # recv_e_buf 哨兵 = local_exp+1，无效行 0 → clamp(min=1)-1 得裸 local_exp
        # （无效行 → 0；valid_mask=0 时 kernel 不读 block_id），再 +1 对齐到 block 号。
        recv_e_flat = bufs_send["recv_e_buf"][send_CAP:]
        recv_valid = (recv_e_flat > 0).to(torch.int32)
        recv_local_exp = recv_e_flat.clamp(min=1) - 1          # 无效行 → 0（valid_mask=0 时不读 block_id）
        recv_pos, recv_keep = moe_align(
            bufs_send["recv_buf"][send_CAP:], bufs_send["recv_w_buf"][send_CAP:],
            valid_mask=recv_valid, block_ids=recv_local_exp + 1,
            x_align=bufs_compute["x_align"], w_align=bufs_compute["w_align"],
            cnt=bufs_compute["cnt"],
            NUM_BLOCKS=self.experts_per_rank, CAP=recv_CAP)

        # ---- fused compute（全量权重；kernel 内乘路由权重 w，源端不再乘）----
        moe_ffn_pair(bufs_compute["x_align"], self.stacked_gate, self.stacked_up,
                     self.stacked_down, bufs_compute["w_align"], bufs_compute["e_align"],
                     bufs_compute["out_align"], M=self.moe_intermediate_size, D=D)

        # ---- gather-back：out_align[recv_pos] 恢复 recv 行序 → 复用 send_buf[send_CAP:] ----
        out_flat = bufs_compute["out_align"][recv_pos] * recv_keep.to(dtype).unsqueeze(-1)
        bufs_send["send_buf"][send_CAP:].copy_(out_flat)

        # ---- 逆向 a2a：结果按 source_rank 分块回源；result 复用 recv_buf[send_CAP:] ----
        dist.all_to_all_single(bufs_send["recv_buf"][send_CAP:], bufs_send["send_buf"][send_CAP:],
                               output_split_sizes=split, input_split_sizes=split,
                               group=ep_group.device_group)

        # ---- 源端归组：send_pos 含 null 段偏移，直接索引；直接 sum（w 已在 kernel 内乘）。
        # fp32 累加后一次性舍入 bf16（省两次中间舍入，数值贴近参考实现，R3）----
        result = bufs_send["recv_buf"][send_pos].float() \
            * send_keep.float().unsqueeze(-1)                                          # [NK, D] fp32
        bufs_compute["partial"].zero_()
        bufs_compute["partial"].copy_(result.reshape(N, K, D).sum(dim=1).to(dtype))
        return bufs_compute["partial"]

    # ------------------------------------------------------------------
    # Shared local path (ep_size=1): no communication needed
    # ------------------------------------------------------------------

    def _forward_ep_local(
        self,
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """EP forward when ep_size=1 — pure local computation, no collectives."""
        N, D = hidden_states.shape
        K = self.top_k

        tokens_flat = hidden_states.unsqueeze(1).expand(-1, K, -1).reshape(-1, D)
        weights_flat = weights.reshape(-1)
        expert_ids_flat = indices.reshape(-1)

        expert_outputs = self._compute_local(tokens_flat, expert_ids_flat)

        output_3d = expert_outputs.reshape(N, K, D)
        weights_3d = weights_flat.reshape(N, K)
        output = (output_3d * weights_3d.unsqueeze(-1)).sum(dim=1)
        return output

    def _forward_tp(
        self,
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """§2.1 moe_tp（AllReduce 后端）：全 token 本地计算 + AR[EP]（+ AG[DP]/slice）。

        数据流：
          DP>1 时 AG[DP] 前置（AG 顺序 = dp 升序 = slice 块序，R6），在 N_total 上重路由；
          权重沿 M 维切 EP 份（stacked_*_tp，M/EP）→ moe_align（mask=全1,
          block_id=全局 expert+1, 129 块）→ moe_ffn_pair（M/EP）→
          partial [N_total, D]（每 rank 只含其唯一 M 切片的贡献）→ AR[EP] 全组归约
          （AR[TP] 只补 TP 组内切片，缺其他 DP 组的切片，语义错误）→
          DP>1 时本地 slice 取回本 DP 行块（零通信）。
        """
        N, D = hidden_states.shape
        K = self.top_k
        device, dtype = hidden_states.device, hidden_states.dtype
        ep_group = get_ep_group()
        dp_size = self.ep_size // self.tp_size               # EP group = 全体 rank

        if dp_size > 1:
            all_tokens = get_dp_group().all_gather(hidden_states, dim=0)   # AG[DP] 前置
        else:
            all_tokens = hidden_states
        N_total = all_tokens.size(0)
        if N_total == 0:
            return hidden_states

        all_weights, all_indices = self._topk_routing(self.gate(all_tokens))   # N_total 上重路由
        NK = N_total * K
        x_flat = all_tokens.unsqueeze(1).expand(-1, K, -1).reshape(NK, D)
        w_flat = all_weights.reshape(NK)
        e_flat = all_indices.reshape(NK)                     # 全局 expert id

        tp_CAP = _cap_from_expected(3 * NK // self.num_experts, NK)
        bufs = get_or_create_moe_buffers(
            "moe_tp", device, dtype, N_total, D,
            T_MAX=(self.num_experts + 1) * tp_CAP, num_blocks=self.num_experts, CAP=tp_CAP)
        bufs["x_align"].zero_(); bufs["w_align"].zero_(); bufs["cnt"].zero_()

        pos, keep = moe_align(
            x_flat, w_flat,
            valid_mask=torch.ones(NK, dtype=torch.int32, device=device),
            block_ids=(e_flat + 1).to(torch.int64),
            x_align=bufs["x_align"], w_align=bufs["w_align"], cnt=bufs["cnt"],
            NUM_BLOCKS=self.num_experts, CAP=tp_CAP)

        moe_ffn_pair(bufs["x_align"], self.stacked_gate_tp, self.stacked_up_tp,
                     self.stacked_down_tp, bufs["w_align"], bufs["e_align"],
                     bufs["out_align"], M=self.moe_intermediate_size // self.ep_size, D=D)

        # fp32 累加后一次性舍入 bf16（与 fixbuf 路径同一数值约定，R3）
        out_flat = bufs["out_align"][pos].float() * keep.float().unsqueeze(-1)
        bufs["partial"].zero_()
        bufs["partial"].copy_(out_flat.reshape(N_total, K, D).sum(dim=1).to(dtype))
        ep_group.all_reduce(bufs["partial"])                 # AR[EP] 全体 rank（必须全组）

        if dp_size > 1:
            j = get_dp_group().rank_in_group                 # R6：AG 顺序 = dp 升序 = slice 块序
            return bufs["partial"].narrow(0, j * N, N)
        return bufs["partial"]

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

    def _compute_local(
        self,
        tokens: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Per-expert FFN compute — memory-efficient, no padding.

        Groups tokens by expert then iterates over active experts with
        individual matmuls.  Uses stacked weights (EP>1) for fast lookup;
        CUDA Graph eliminates per-expert launch overhead during decode.

        Args:
            tokens: [total_tokens, hidden_size] dispatched tokens.
            expert_ids: [total_tokens] local expert index (0..experts_per_rank-1).

        Returns:
            [total_tokens, hidden_size] expert outputs in original order.
        """
        total_tokens = tokens.size(0)
        if total_tokens == 0:
            return tokens

        D = self.hidden_size

        # --- Group tokens by local expert ---
        sort_idx = torch.argsort(expert_ids)
        tokens_sorted = tokens[sort_idx]
        ids_sorted = expert_ids[sort_idx]

        unique_exp, counts = torch.unique_consecutive(ids_sorted, return_counts=True)

        output_sorted = tokens_sorted.new_empty(total_tokens, D)

        if self.ep_size > 1:
            # EP mode: use stacked [E, out, in] parameters.
            ptr = 0
            for e, count in zip(unique_exp.tolist(), counts.tolist()):
                x = tokens_sorted[ptr:ptr + count]                # [cnt, D]
                gate = torch.mm(x, self.stacked_gate[e].t())      # [cnt, M]
                up   = torch.mm(x, self.stacked_up[e].t())        # [cnt, M]
                out  = torch.mm(F.silu(gate) * up,
                                self.stacked_down[e].t())         # [cnt, D]
                output_sorted[ptr:ptr + count] = out
                ptr += count
        else:
            # ep_size=1: use individual ExpertFFN modules.
            ptr = 0
            for e, count in zip(unique_exp.tolist(), counts.tolist()):
                x = tokens_sorted[ptr:ptr + count]
                output_sorted[ptr:ptr + count] = self.experts[e](x)
                ptr += count

        # --- Restore original order ---
        unsort_idx = torch.argsort(sort_idx)
        return output_sorted[unsort_idx]
