import os
from dataclasses import dataclass
from transformers import AutoConfig

hybrid_model_list = {
    "Qwen3_5ForConditionalGeneration"
}


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

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert 1 <= self.data_parallel_size <= 8
        assert 1 <= self.data_parallel_size * self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        self.hybrid_model = any(a in hybrid_model_list for a in self.hf_config.architectures)
        if self.hybrid_model:
            self.num_linear_layers = self.hf_config.layer_types.count('linear_attention')
            self.num_full_layers = self.hf_config.num_hidden_layers - self.num_linear_layers
        else:
            self.num_full_layers, self.num_linear_layers = self.hf_config.num_hidden_layers, 0
