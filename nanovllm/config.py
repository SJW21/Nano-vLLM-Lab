import os
from dataclasses import dataclass
from transformers import AutoConfig

# Opt-in support list: a checkpoint is built as Qwen3.5 (the hybrid GDN stack) iff its
# `architectures` names one of these. Everything else takes the Qwen3 path.
hybrid_model_list = {
    "Qwen3_5ForConditionalGeneration"
}


def is_hybrid_config(hf_config) -> bool:
    """Is this checkpoint one we run on the hybrid (linear-attention) path?
       See hybrid_model_list for hybrid models have supported.
    """
    architectures = getattr(hf_config, "architectures", None) or []
    return any(a in hybrid_model_list for a in architectures)


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    dplb_method: str = "ROUND_ROBIN" # Option["LEAST_REQUESTS", "ROUND_ROBIN"]
    async_event_loop: bool = False
    enable_profiler: bool = False
    profiler_output: str = "./profiler_logs"
    profiler_warmup_steps: int = 3 # skip num_steps before profile
    profiler_active_steps: int = 0 # num_steps to profile
    linear_cache_ratio: float = 0.2 # default ratio of linear_cache
    # derived fields: __post_init__ and ModelRunner assign them, so they must be
    # declared -- dataclass(slots=True) instances have no __dict__
    hybrid_model: bool = False
    # Qwen3.5 ships in two layouts (VL nests the text config, text-only *is* it); every
    # reader of text fields goes through this one, resolved once in __post_init__
    text_config: AutoConfig | None = None
    num_linear_layers: int = 0
    num_full_layers: int = 0
    num_linear_states: int = -1 # linear state slots, written back by allocate_kv_cache
    # linear-state prefix snapshots; decode is off by default (one per block boundary is too
    # costly for long generations) and coarser when enabled, in units of blocks
    enable_decode_snapshot: bool = False
    decode_snapshot_interval: int = 16

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert 1 <= self.data_parallel_size <= 8
        assert 1 <= self.data_parallel_size * self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # resolve the text half of the checkpoint once (Qwen3.5 VL configs nest it)
        self.text_config = getattr(self.hf_config, "text_config", self.hf_config)
        self.max_model_len = min(self.max_model_len, self.text_config.max_position_embeddings)
        self.hybrid_model = is_hybrid_config(self.hf_config)
        if self.hybrid_model:
            self.num_linear_layers = self.text_config.layer_types.count('linear_attention')
            self.num_full_layers = self.text_config.num_hidden_layers - self.num_linear_layers
        else:
            self.num_full_layers, self.num_linear_layers = self.text_config.num_hidden_layers, 0
