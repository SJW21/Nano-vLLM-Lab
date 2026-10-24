import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    dplb_method: str = "ROUND_ROBIN" # Option["LEAST_REQUESTS", "ROUND_ROBIN"]
    async_event_loop: bool = False
    # ----- profiler -----
    enable_profiler: bool = False
    profiler_output: str = "./profiler_logs"
    profiler_warmup_steps: int = 3 # skip num_steps before profile
    profiler_active_steps: int = 0 # num_steps to profile

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert 1 <= self.data_parallel_size <= 8
        assert 1 <= self.data_parallel_size * self.tensor_parallel_size <= 8
        assert self.expert_parallel_size in (1, self.data_parallel_size * self.tensor_parallel_size), \
            "EP size must be 1 (disabled) or equal to dp_size * tp_size"
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.hf_config.model_type == "qwen3_moe":
            m_inter = self.hf_config.moe_intermediate_size
            assert m_inter % self.expert_parallel_size == 0, (
                f"moe_intermediate_size ({m_inter}) must be divisible by "
                f"expert_parallel_size ({self.expert_parallel_size})")
            m_ep = m_inter // self.expert_parallel_size
            # down_proj uses K = M/EP; EVEN_K requires a multiple of 64.
            assert m_ep % 64 == 0, (
                f"M/EP = {m_ep} must be a multiple of 64 (A3B/I=768 supports "
                f"EP in {{1, 2, 4}})")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
