import torch
from torch import nn
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.moe import MoELayer
from nanovllm.distributed.parallel_state import get_tp_group
from nanovllm.utils.context import get_context, set_context


class Qwen3MoEAttention(nn.Module):
    """Attention layer — identical to Qwen3Attention, reused for MoE model."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
        output_comm: str = "all_reduce",
    ) -> None:
        super().__init__()
        tp_size = get_tp_group().world_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            output_comm=output_comm,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MoEDecoderLayer(nn.Module):
    """Decoder layer with MoE MLP instead of dense MLP.

    Attention layer is unchanged from Qwen3 dense.  Only the MLP is
    replaced by MoELayer (expert parallelism / tensor parallelism).
    """

    def __init__(
        self,
        config: Qwen3Config,
        ep_size: int = 1,
        ep_rank: int = 0,
        moe_mode: str = "ep",
    ) -> None:
        super().__init__()
        # moe_ep + TP>1 时 o_proj 用 RS 输出（§2.2 核心包装）；此处先按公式预置，
        # mlp 构造后以 MoELayer.use_rs_wrap（§4 同公式）覆盖确认。
        self.use_rs_wrap = (moe_mode == "ep") and (get_tp_group().world_size > 1)
        self.self_attn = Qwen3MoEAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            output_comm="reduce_scatter" if self.use_rs_wrap else "all_reduce",
        )
        self.mlp = MoELayer(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            moe_intermediate_size=config.moe_intermediate_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            ep_size=ep_size,
            ep_rank=ep_rank,
            moe_mode=moe_mode,
            hidden_act=config.hidden_act,
            norm_topk_prob=getattr(config, "norm_topk_prob", True),
        )
        self.use_rs_wrap = self.mlp.use_rs_wrap
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # R2 RS 均匀性（moe_ep + TP>1）：RS 发生在 attention 内部的 o_proj
        # （RowParallelLinear reduce_scatter），N % TP != 0 必须在此前 pad 到
        # TP 倍数（vLLM SP 同款手法）。pad 行 = 末尾追加全零行：attention
        # causal 下真实行不读 pad 行的 k/v（pad 在序列末尾），MLP 段零输入
        # 恒零输出 → pad 行只污染自身输出（slice 丢弃）。context
        # （cu_seqlens / context_lens / block_tables / slot_mapping）同步扩展，
        # 返回前恢复原值——各层独立 pad，互不累积。graph 档位 bs 已对齐 TP
        # （capture_cudagraph），capture 时 pad_n==0 → 分支整体跳过，图兼容。
        pad_n = 0
        orig_ctx = None
        if self.use_rs_wrap:
            tp_size = get_tp_group().world_size
            N_full = hidden_states.size(0)
            pad_n = (-N_full) % tp_size
            if pad_n:
                zero = hidden_states.new_zeros(pad_n, hidden_states.size(1))
                hidden_states = torch.cat([hidden_states, zero], dim=0)
                if residual is not None:
                    residual = torch.cat([residual, zero], dim=0)
                # positions 同步扩展：attention 内部 rotary 按 position 查
                # cos/sin，pad 行位置延续最后一个值（仅影响 pad 行自身输出）
                positions = torch.cat([positions, positions[-1:].expand(pad_n)])
                orig_ctx = get_context()
                if orig_ctx.is_prefill:
                    # cu_seqlens 末尾追加 dummy 段 [S, S+pad]；slot_mapping 补
                    # -1（store_kvcache kernel 对 slot==-1 跳过）
                    cq, ck = orig_ctx.cu_seqlens_q, orig_ctx.cu_seqlens_k
                    set_context(True,
                                torch.cat([cq, cq[-1:] + pad_n]),
                                torch.cat([ck, ck[-1:] + pad_n]),
                                orig_ctx.max_seqlen_q, orig_ctx.max_seqlen_k,
                                torch.cat([orig_ctx.slot_mapping,
                                           orig_ctx.slot_mapping.new_full((pad_n,), -1)]),
                                orig_ctx.context_lens, orig_ctx.block_tables)
                else:
                    # decode：context_lens 补 1（pad 行只见 block 0 的 1 个
                    # slot，garbage 只进 pad 行自身输出）；block_tables 补全 0
                    # 行（block 0 合法）；slot_mapping 补 -1
                    set_context(False,
                                None, None, 0, 0,
                                torch.cat([orig_ctx.slot_mapping,
                                           orig_ctx.slot_mapping.new_full((pad_n,), -1)]),
                                torch.cat([orig_ctx.context_lens,
                                           orig_ctx.context_lens.new_ones(pad_n)]),
                                torch.cat([orig_ctx.block_tables,
                                           orig_ctx.block_tables.new_zeros(pad_n,
                                                                           orig_ctx.block_tables.size(1))]))
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)

        if self.use_rs_wrap:
            # --- moe_ep + TP>1：RS/AG 包装（§2.2）---
            # o_proj 输出为 RS 后的 [N/TP, D]（每 rank 持互不重复的 token 行块）。
            # 残差分支 (A + R) 无法在全宽形成（A 已 RS 切分），因此将残差行块
            # R_r 逐行并入 post_attention_layernorm（add_rms_forward），与 MLP
            # 输出本地求和后单次 AG 还原完整 block 输出 M + A + R（§5.4 "residual
            # 移到 AG 之后" 的实现语义：残差加法在 MLP 之后、AG 边界完成；数值等
            # 价于 AG(mlp_out) + (A + R)，R3 覆盖加序差异 < 1e-2）。
            tp_group = get_tp_group()
            N_local = hidden_states.size(0)                  # = (N_full+pad_n)/TP
            r_slice = residual.narrow(0, tp_group.rank_in_group * N_local, N_local)
            hidden_states, res_slice = self.post_attention_layernorm(hidden_states, r_slice)
            hidden_states = self.mlp(hidden_states)          # [N/TP, D]
            block_out = hidden_states + res_slice            # M + (A_r + R_r) 本地求和
            hidden_states = tp_group.all_gather(block_out, dim=0)   # [N, D] 全量恢复
            if pad_n:
                hidden_states = hidden_states[:N_full]       # slice 回真实行
                set_context(orig_ctx.is_prefill, orig_ctx.cu_seqlens_q,
                            orig_ctx.cu_seqlens_k, orig_ctx.max_seqlen_q,
                            orig_ctx.max_seqlen_k, orig_ctx.slot_mapping,
                            orig_ctx.context_lens, orig_ctx.block_tables)
            return hidden_states, None                       # 残差已在层内吸收
        else:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
            return hidden_states, residual


class Qwen3MoEModel(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        ep_size: int = 1,
        ep_rank: int = 0,
        moe_mode: str = "ep",
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3MoEDecoderLayer(config, ep_size, ep_rank, moe_mode)
            for _ in range(config.num_hidden_layers)
        ])
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
        if residual is None:
            # RS/AG 包装（moe_ep + TP>1）的层内残差吸收语义：末层已返回
            # residual=None，final norm 只需 rms（残差已在层内并入）。
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoEForCausalLM(nn.Module):
    """Qwen3-MoE top-level model.

    packed_modules_mapping does NOT include gate_proj/up_proj because
    MoE experts use separate Linear layers (not MergedColumnParallelLinear).
    Expert weights are loaded via the EP-aware remapping in loader.py.
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(
        self,
        config: Qwen3Config,
        ep_size: int = 1,
        ep_rank: int = 0,
        moe_mode: str = "ep",
    ) -> None:
        super().__init__()
        self.model = Qwen3MoEModel(config, ep_size, ep_rank, moe_mode)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
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
