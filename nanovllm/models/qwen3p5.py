import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen3_5TextConfig

from nanovllm.models.qwen3 import Qwen3Attention
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.gdn_ops import (FLA_CHUNK_SIZE,
                                     causal_conv1d_prefill,
                                     causal_conv1d_update,
                                     chunk_gated_delta_rule,
                                     fused_recurrent_gdn_decode,
                                     l2norm_fwd,
                                     prepare_chunk_indices,
                                     prepare_chunk_offsets)
from nanovllm.layers.layernorm import Qwen3p5RMSNorm, Qwen3p5RMSNormGated
from nanovllm.layers.linear import (MergedColumnParallelLinear, RowParallelLinear,
                                    segmented_weight_loader)
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.distributed.parallel_state import get_tp_group
from nanovllm.utils.context import get_context


class Qwen3p5Attention(Qwen3Attention):

    # Qwen3.5 folds a +1 into every plain norm, q/k included (the GDN's gated norm is
    # unit-centred and keeps its own class)
    norm_cls = Qwen3p5RMSNorm

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_idx: int
    ) -> None:
        # rotary settings live in `rope_parameters` here, and only a fraction of each head rotates
        rope_params = getattr(config, "rope_parameters", None) or {}
        head_dim = (getattr(config, "head_dim", None)
                    or config.hidden_size // config.num_attention_heads)
        super().__init__(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=head_dim,
            rope_theta=rope_params.get("rope_theta",
                                       getattr(config, "rope_theta", 10000.0)),
            rope_scaling=getattr(config, "rope_scaling", None),
            rotary_dim=int(head_dim * rope_params.get("partial_rotary_factor", 1.0)),
            attn_output_gate=True,
        )
        self.layer_idx = layer_idx

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        # per head: 2*head_dim in q_proj, query first then gate (transformers' layout)
        q_gate, k, v = qkv.split([2 * self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, gate = q_gate.view(-1, self.num_heads, 2 * self.head_dim).chunk(2, dim=-1)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        # prefill o is [T, H, D], decode o is [B, 1, H, D]: flatten both sides first, or the
        # 3-D gate broadcasts the seqlen axis into the feature axis (or silently mixes
        # sequences where a batch of one happens to line up)
        o = o.flatten(1, -1) * gate.sigmoid().flatten(1, -1)
        output = self.o_proj(o)
        return output


class Qwen3p5GDNState(nn.Module):
    """Per-layer linear-attention state holder, bound to the shared ``MambaPool`` by
    ``ModelRunner.allocate_kv_cache``. Stays empty until then (warmup runs before it):
    the GDN layer falls back to a private scratch pool, see ``_state_buffers``.
    """

    def __init__(self):
        super().__init__()
        self.conv_cache = self.temporal_cache = torch.tensor([])


class Qwen3p5GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().local_rank
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        # head-parallel sharding (vLLM's design): a rank owns heads [r*H/tp, (r+1)*H/tp) and
        # keeps state for only those. Runtime code uses the `local_*` sizes below; the layers
        # constructed here keep the global ones, since they divide by tp themselves.
        self.local_num_k_heads = self.num_k_heads // tp_size
        self.local_num_v_heads = self.num_v_heads // tp_size
        self.local_key_dim = self.head_k_dim * self.local_num_k_heads
        self.local_value_dim = self.head_v_dim * self.local_num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.local_conv_dim = self.conv_dim // tp_size
        self.conv1d = nn.Conv1d(
            in_channels=self.local_conv_dim,
            out_channels=self.local_conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.local_conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        # checkpoint stores one [conv_dim, 1, W] depthwise conv over [q | k | v]; sharded
        # per segment so a rank's channels stay inside one of them (see segmented_weight_loader)
        self.conv1d.weight.weight_loader = segmented_weight_loader(
            [(self.key_dim, 0, False), (self.key_dim, 0, False), (self.value_dim, 0, False)],
            tp_size, tp_rank,
        )

        # four segments rather than two, so a rank's slice of the fused tensor lands
        # inside q / k / v / z instead of straddling two of them
        self.in_proj_qkvz = MergedColumnParallelLinear(
            self.hidden_size,
            [self.key_dim, self.key_dim, self.value_dim, self.value_dim],
            bias=False,
        )
        self.in_proj_ba = MergedColumnParallelLinear(self.hidden_size,
                                                     [self.num_v_heads] * 2,
                                                     bias=False)

        # time step projection (discretization); the kernels index A_log / dt_bias by the
        # *local* value-head number, so a full-size copy would silently use the wrong half
        self.dt_bias = nn.Parameter(torch.ones(self.local_num_v_heads))

        # lower bound kept away from 0 so log(A) never becomes -inf
        A = torch.empty(self.local_num_v_heads).uniform_(0.01, 16)
        self.A_log = nn.Parameter(torch.log(A))
        head_loader = segmented_weight_loader([(self.num_v_heads, 0, False)], tp_size, tp_rank)
        self.A_log.weight_loader = head_loader
        self.dt_bias.weight_loader = head_loader

        self.norm = Qwen3p5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = RowParallelLinear(self.value_dim, self.hidden_size, bias=False)

        # state cache: ModelRunner binds these to the shared MambaPool later on
        self.attn = Qwen3p5GDNState()
        self._scratch_conv = None
        self._scratch_temporal = None

    def _state_buffers(self, state_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The slot-indexed (conv, ssm) state pool, or a private scratch pool during
        warmup -- which runs before ``ModelRunner.allocate_kv_cache`` binds it."""
        conv, temporal = self.attn.conv_cache, self.attn.temporal_cache
        if conv.numel():
            return conv, temporal
        num_states = int(state_ids.max().item()) + 1
        if self._scratch_conv is None or self._scratch_conv.shape[0] < num_states:
            device = self.out_proj.weight.device
            self._scratch_conv = torch.zeros(
                num_states, self.local_conv_dim, self.conv_kernel_size - 1,
                dtype=self.conv1d.weight.dtype, device=device,
            )
            self._scratch_temporal = torch.zeros(
                num_states, self.local_num_v_heads, self.head_v_dim, self.head_k_dim,
                dtype=torch.float32, device=device,
            )
        return self._scratch_conv, self._scratch_temporal

    @staticmethod
    def _state_ids(context, num_seqs: int, device) -> torch.Tensor:
        """State slot each sequence writes to (decode updates it in place)."""
        state_ids = context.state_ids
        if state_ids is None or state_ids.numel() != num_seqs:
            # warmup / profiling pass: no state indices were prepared
            return torch.arange(num_seqs, dtype=torch.int32, device=device)
        return state_ids

    def _write_snapshots(self, context, conv_input, conv_state, temporal_state,
                         state_ids, h, cu_seqlens, chunk_offsets):
        """Copy this step's designated prefix states into their cache slots.

        The scheduler picks one boundary per step (``Scheduler.plan_snapshot``); the state
        there already exists -- the step's final state, or the per-chunk prefix state ``h``
        the chunked scan computed -- so this only scatters it.
        """
        snapshot_ids = context.snapshot_ids
        snapshot_pos = context.snapshot_pos
        if snapshot_ids is None or snapshot_pos is None:
            return
        # one device sync per layer, only on steps that actually snapshot something
        idx = (snapshot_ids >= 0).nonzero(as_tuple=True)[0]
        slots = snapshot_ids[idx].long()
        pos = snapshot_pos[idx].long()
        bos = cu_seqlens[idx].long()
        eos = cu_seqlens[idx + 1].long()

        # boundary at the step's end: that state is the one just written back
        end = pos >= eos
        if bool(end.any()):
            temporal_state[slots[end]] = temporal_state[state_ids[idx[end]].long()]
        # boundary inside the step: the state *before* the chunk starting there, which the
        # chunked scan already materialised for every chunk
        mid = ~end
        if bool(mid.any()):
            gidx = chunk_offsets[idx[mid]].long() + (pos[mid] - bos[mid]) // FLA_CHUNK_SIZE
            temporal_state[slots[mid]] = h[0, gidx].to(temporal_state.dtype)

        # same window either way: the W-1 raw inputs before the boundary, which the
        # scheduler guarantees are part of this step's input
        S = self.conv_kernel_size - 1
        rows = pos[:, None] - S + torch.arange(S, device=conv_input.device)
        windows = conv_input[rows]                       # [K, S, conv_dim]
        conv_state[slots] = windows.permute(0, 2, 1)     # pool layout is [D, S]

    @staticmethod
    def _chunk_metadata(context, cu_seqlens: torch.Tensor):
        """chunk_indices / chunk_offsets for the chunked kernels, built once per step."""
        if context.chunk_indices is None:
            context.chunk_indices = prepare_chunk_indices(cu_seqlens, FLA_CHUNK_SIZE)
            context.chunk_offsets = prepare_chunk_offsets(cu_seqlens, FLA_CHUNK_SIZE)
        return context.chunk_indices, context.chunk_offsets

    def _forward_prefill(self, mixed_qkv, b, a, context, conv_weight, conv_bias):
        cu_seqlens = context.cu_seqlens_q
        num_seqs = cu_seqlens.numel() - 1
        chunk_indices, chunk_offsets = self._chunk_metadata(context, cu_seqlens)
        # cached_states[i]: the slot this sequence resumes from (-1 => zero). A chunked
        # continuation re-reads its own working slot, arranged by ``carry_state``.
        cached_states = context.cached_states
        if cached_states is None or cached_states.numel() != num_seqs:
            # warmup / profiling pass: no state bookkeeping was prepared at all
            cached_states = cu_seqlens.new_full((num_seqs,), -1)
        state_ids = self._state_ids(context, num_seqs, cu_seqlens.device)
        conv_state, temporal_state = self._state_buffers(state_ids)

        conv_input = mixed_qkv    # raw conv input, still needed for the snapshots
        mixed_qkv = causal_conv1d_prefill(
            conv_input, conv_weight, conv_bias, cu_seqlens, chunk_indices,
            conv_state, cached_states, state_ids,
        )
        # q/k keep their own head count: the kernels map each value head to its key
        # head (GQA), so no repeat_interleave is needed
        q, k, v = mixed_qkv.split([self.local_key_dim, self.local_key_dim,
                                   self.local_value_dim], dim=-1)
        q = l2norm_fwd(q.reshape(-1, self.local_num_k_heads, self.head_k_dim)).unsqueeze(0)
        k = l2norm_fwd(k.reshape(-1, self.local_num_k_heads, self.head_k_dim)).unsqueeze(0)
        v = v.reshape(-1, self.local_num_v_heads, self.head_v_dim).contiguous().unsqueeze(0)

        beta = b.float().sigmoid().contiguous().unsqueeze(0)
        # fp32 A_log: at fp16 the .exp() can underflow to -inf
        g = (-self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float()))
        g = g.contiguous().unsqueeze(0)

        # a sequence resuming a cached prefix starts from that state, everything else from zero
        initial_state = torch.zeros(
            num_seqs, self.local_num_v_heads, self.head_v_dim, self.head_k_dim,
            dtype=torch.float32, device=mixed_qkv.device,
        )
        hit = cached_states >= 0
        initial_state[hit] = temporal_state[cached_states[hit].long()].float()

        out, final_state, h = chunk_gated_delta_rule(
            q, k, v, g, beta, initial_state, True, cu_seqlens, chunk_indices, chunk_offsets,
            return_h=True,
        )
        # a negative slot would wrap to the last pool row instead of being a no-op
        live = state_ids >= 0
        temporal_state[state_ids[live].long()] = final_state[live].to(temporal_state.dtype)
        if context.has_snapshot:
            self._write_snapshots(context, conv_input, conv_state, temporal_state,
                                  state_ids, h, cu_seqlens, chunk_offsets)
        return out[0]

    def _forward_decode(self, mixed_qkv, b, a, context, conv_weight, conv_bias):
        num_seqs = mixed_qkv.shape[0]
        state_ids = self._state_ids(context, num_seqs, mixed_qkv.device)
        conv_state, temporal_state = self._state_buffers(state_ids)

        mixed_qkv = causal_conv1d_update(
            mixed_qkv, conv_weight, conv_bias, conv_state, state_ids
        )
        # kernel takes (a, b) -- decay then write strength, as in prefill. Swapping them
        # only shows up as drift, hence the state-level check in tests/test_gdn.py.
        out = fused_recurrent_gdn_decode(
            mixed_qkv, a, b, self.A_log, self.dt_bias, temporal_state, state_ids,
            scale=self.head_k_dim ** -0.5,
        )
        # decode snapshots are opt-in and eager-only (a captured graph cannot carry a
        # data dependent write); the state wanted is this step's own working slot
        if context.has_snapshot:
            snap = context.snapshot_ids >= 0
            idx = snap.nonzero(as_tuple=True)[0]
            dst, src = context.snapshot_ids[idx].long(), state_ids[idx].long()
            temporal_state[dst] = temporal_state[src]
            conv_state[dst] = conv_state[src]
        return out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        conv_weight = self.conv1d.weight.squeeze(1)

        projected_qkvz = self.in_proj_qkvz(hidden_states)
        projected_ba = self.in_proj_ba(hidden_states)
        # checkpoint order is q/k/v and [b, a], so both fused projections split straight
        # down the last dim (no GQA interleaving)
        mixed_qkv, z = projected_qkvz.split(
            [self.local_key_dim * 2 + self.local_value_dim, self.local_value_dim], dim=-1
        )
        b, a = projected_ba.chunk(2, dim=-1)
        z = z.reshape(z.shape[0], -1, self.head_v_dim)

        if context.is_prefill:
            core_attn_out = self._forward_prefill(
                mixed_qkv, b, a, context, conv_weight, self.conv1d.bias
            )
        else:
            core_attn_out = self._forward_decode(
                mixed_qkv, b, a, context, conv_weight, self.conv1d.bias
            )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        return self.out_proj(core_attn_out.reshape(z_shape_og[0], -1))


class Qwen3p5MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3p5DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_idx: int
    ) -> None:
        super().__init__()
        # token mixer
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3p5GatedDeltaNet(config, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3p5Attention(config, layer_idx)

        self.mlp = Qwen3p5MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = Qwen3p5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3p5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Token Mixer
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        elif self.block_type == "full_attention":
            hidden_states = self.self_attn(positions, hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3p5Model(nn.Module):

    def __init__(
        self,
        config: Qwen3_5TextConfig,
    ) -> None:
        super().__init__()
        # normalise the layout once: a VL checkpoint nests the text config, a text-only
        # one *is* it (every layer below reads its fields off this)
        config = getattr(config, "text_config", config)
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3p5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = Qwen3p5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3p5ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        # the checkpoint's in_proj_qkv spans our first three segments (q|k|v) -> tuple shard id
        "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
        "in_proj_z": ("in_proj_qkvz", 3),
        "in_proj_b": ("in_proj_ba", 0),
        "in_proj_a": ("in_proj_ba", 1)
    }

    def __init__(
        self,
        config: Qwen3_5TextConfig
    ) -> None:
        super().__init__()
        self.model = Qwen3p5Model(config)
        text_config = getattr(config, "text_config", config)
        self.lm_head = ParallelLMHead(text_config.vocab_size, text_config.hidden_size)
        # (`tie_word_embeddings` sits on the top-level config in both layouts)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)