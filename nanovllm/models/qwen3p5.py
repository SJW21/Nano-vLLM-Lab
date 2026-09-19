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
from nanovllm.layers.layernorm import RMSNorm, Qwen3p5RMSNormGated
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.distributed.parallel_state import get_tp_group
from nanovllm.utils.context import get_context
from contextlib import nullcontext


class Qwen3p5Attention(Qwen3Attention):

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_idx: int
    ) -> None:
        # Qwen3.5 keeps its rotary settings in `rope_parameters` (the text config
        # has no `rope_theta`) and rotates only a fraction of each head
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
        # q_proj carries a per-head query *and* gate: 2*head_dim per head, query
        # first (transformers' layout), and the output is scaled by sigmoid(gate)
        q_gate, k, v = qkv.split([2 * self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, gate = q_gate.view(-1, self.num_heads, 2 * self.head_dim).chunk(2, dim=-1)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        o = o * gate.sigmoid()
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3p5GDNState(nn.Module):
    """Per-layer linear-attention state holder.

    ``ModelRunner.allocate_kv_cache`` binds ``conv_cache`` / ``temporal_cache`` to
    this layer's slice of the shared ``MambaPool`` once the pool size is known.
    Before that (the warmup pass) they stay empty and the GDN layer falls back to
    a private scratch pool -- see ``Qwen3p5GatedDeltaNet._state_buffers``.
    """

    def __init__(self):
        super().__init__()
        self.conv_cache = self.temporal_cache = torch.tensor([])


class Qwen3p5GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # projection of the input hidden states
        projection_size_qkv = self.key_dim * 2 + self.value_dim
        self.in_proj_qkvz = MergedColumnParallelLinear(self.hidden_size, 
                                                       [projection_size_qkv, self.value_dim], 
                                                       bias=False)
        self.in_proj_ba = MergedColumnParallelLinear(self.hidden_size, 
                                                     [self.num_v_heads] * 2, 
                                                     bias=False)

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        # Lower bound kept away from 0 so log(A) never becomes -inf
        A = torch.empty(self.num_v_heads).uniform_(0.01, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = Qwen3p5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = RowParallelLinear(self.value_dim, self.hidden_size, bias=False)

        self.layer_type = config.layer_types[layer_idx]

        # state cache: ModelRunner binds these to the shared MambaPool later on
        self.attn = Qwen3p5GDNState()
        self._scratch_conv = None
        self._scratch_temporal = None

    def _state_buffers(self, state_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The slot-indexed (conv, ssm) state pool.

        Falls back to a private scratch pool during warmup, which runs *before*
        ``ModelRunner.allocate_kv_cache`` binds the shared ``MambaPool``.
        """
        conv, temporal = self.attn.conv_cache, self.attn.temporal_cache
        if conv.numel():
            return conv, temporal
        num_states = int(state_ids.max().item()) + 1
        if self._scratch_conv is None or self._scratch_conv.shape[0] < num_states:
            device = self.out_proj.weight.device
            self._scratch_conv = torch.zeros(
                num_states, self.conv_dim, self.conv_kernel_size - 1,
                dtype=self.conv1d.weight.dtype, device=device,
            )
            self._scratch_temporal = torch.zeros(
                num_states, self.num_v_heads, self.head_v_dim, self.head_k_dim,
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

        The prefix cache is only useful if a state exists at the block boundary
        the KV cache resumes from, so the scheduler asks for one snapshot per step
        at the last complete boundary (see ``Scheduler.plan_snapshot``). The state
        at that boundary is either the final state of the step (the boundary *is*
        the step's end) or the per-chunk prefix state ``h`` the chunked scan
        already computed -- either way the data already exists, we only scatter it.
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

        # (a) boundary sits at the end of this step's tokens: what the initial
        #     state gathered earlier plus this step's tokens add up to is exactly
        #     the final state that was just written back
        end = pos >= eos
        if bool(end.any()):
            temporal_state[slots[end]] = temporal_state[state_ids[idx[end]].long()]
        # (b) boundary inside the step: the state *before* the chunk starting at
        #     it, which the chunked scan materialised for every chunk
        mid = ~end
        if bool(mid.any()):
            gidx = chunk_offsets[idx[mid]].long() + (pos[mid] - bos[mid]) // FLA_CHUNK_SIZE
            temporal_state[slots[mid]] = h[0, gidx].to(temporal_state.dtype)

        # conv window is the same in both cases: the W-1 raw inputs before the
        # boundary. The scheduler keeps the boundary at least a block into the
        # step, so those tokens are always part of this step's input.
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
        # Contract: cached_states[i] is the slot holding the state this sequence
        # resumes from (-1 => start from zero). A chunked-prefill continuation
        # re-reads its own working slot, which Scheduler.postprocess arranges via
        # ``carry_state`` once the first chunk is done.
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
        # q/k keep their own head count: the kernels index the shared key head
        # for each value head (GQA), no repeat_interleave needed.
        q, k, v = mixed_qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = l2norm_fwd(q.reshape(-1, self.num_k_heads, self.head_k_dim)).unsqueeze(0)
        k = l2norm_fwd(k.reshape(-1, self.num_k_heads, self.head_k_dim)).unsqueeze(0)
        v = v.reshape(-1, self.num_v_heads, self.head_v_dim).contiguous().unsqueeze(0)

        beta = b.float().sigmoid().contiguous().unsqueeze(0)
        # fp32 A_log: with fp16 weights, .exp() on the raw dtype can underflow to -inf
        g = (-self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float()))
        g = g.contiguous().unsqueeze(0)

        # sequences resuming a cached prefix continue from that state (and conv
        # window, handled above); everything else starts from zero
        initial_state = torch.zeros(
            num_seqs, self.num_v_heads, self.head_v_dim, self.head_k_dim,
            dtype=torch.float32, device=mixed_qkv.device,
        )
        hit = cached_states >= 0
        initial_state[hit] = temporal_state[cached_states[hit].long()].float()

        out, final_state, h = chunk_gated_delta_rule(
            q, k, v, g, beta, initial_state, True, cu_seqlens, chunk_indices, chunk_offsets,
            return_h=True,
        )
        # guard against a negative slot reaching the scatter (it would wrap to the
        # last pool row rather than being a no-op)
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
        out = fused_recurrent_gdn_decode(
            mixed_qkv, b, a, self.A_log, self.dt_bias, temporal_state, state_ids,
            scale=self.head_k_dim ** -0.5,
        )
        # decode snapshots are opt-in (config.enable_decode_snapshot, and eager
        # only -- a captured graph cannot carry a data dependent write). The state
        # they want is this step's working slot, so it is a pair of copies; like
        # prefill, every rank writes the slot its own sequence named.
        if context.has_snapshot and context.snapshot_ids is not None:
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
        # Qwen3.5 stores q/k/v and [b, a] in checkpoint order, so both fused
        # projections split straight down the last dim (no GQA interleaving).
        mixed_qkv, z = projected_qkvz.split(
            [self.key_dim * 2 + self.value_dim, self.value_dim], dim=-1
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
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


def maybe_autocast(
    device_type: str,
    dtype: torch.dtype | None = None,
    enabled: bool = True,
    cache_enabled: bool | None = None,
):
    """
    Context manager that only autocasts if:

    - `autocast` is already enabled in this context
    - Or this call to `maybe_autocast` has `enabled=True`

    This prevents `autocast` being added to the graph when it is effectively a no-op.
    Which makes graph splitting in `torch.compile` more flexible as it removes the
    requirement that partition IDs be monotonically increasing.
    """
    # version >= 2.5.0
    import torch

    if device_type == "meta":
        return nullcontext()
    if torch.is_autocast_enabled(device_type) or enabled:
        return torch.autocast(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
    else:
        return nullcontext()


class Qwen3_5TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn = self.compute_default_rope_parameters
        # if self.rope_type != "default":
        #     rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    def compute_default_rope_parameters(
        config: Qwen3_5TextConfig, device=None, **kwargs
    ) -> tuple[torch.Tensor, float]:
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE
        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        return inv_freq.to(device), attention_factor

    def forward(self, x, position_ids):
        # In contrast to other models, Qwen3VLText has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            cos = freqs.cos() * self.attention_scaling
            sin = freqs.sin() * self.attention_scaling

        sin = self.recomposition_frequencies(sin)
        cos = self.recomposition_frequencies(cos)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def recomposition_frequencies(self, freq):
        """
        Recompose the frequencies into the final spatial layout used per each grid.
        """
        freqs_thw = freq[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_thw[..., idx] = freq[dim, ..., idx]
        return torch.cat((freqs_thw, freqs_thw), dim=-1)



class Qwen3p5Model(nn.Module):

    def __init__(
        self,
        config: Qwen3_5TextConfig,
    ) -> None:
        super().__init__()
        # Every layer reads its fields (hidden_size, layer_types, linear_*, ...)
        # off the text config, so normalise once here: a VL checkpoint nests it
        # under `config.text_config`, a text-only one *is* it.
        config = getattr(config, "text_config", config)
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3p5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        # GDN fused projections.
        "in_proj_qkv": ("in_proj_qkvz", 0),
        "in_proj_z": ("in_proj_qkvz", 1),
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
        # tie_word_embeddings sits on the top-level config in both layouts
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