import pickle
import torch
import torch._dynamo            # for reset(), see the device flip in __init__
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from torch.profiler import record_function
from nanovllm.utils.loader import load_model
from nanovllm.utils.profiler import Profiler
from nanovllm.distributed.parallel_state import (init_distributed_environment,
                                                 initialize_model_parallel,
                                                 get_tp_group)


class ModelRunner:

    def __init__(self, config: Config, rank: int, tar_rank: int, event: Event | list[Event]):
        self.config = config
        # text half of the checkpoint: for Qwen3.5 VL configs the text fields
        # (dtype, hidden_size, ...) live under text_config, not at the top level
        hf_config = config.text_config
        self.block_size = config.kvcache_block_size
        # every rank computes slot_mapping from Sequence.num_blocks / last_block_num_tokens,
        # and spawn'ed workers only ever see the class default (LLMEngine sets it in the
        # parent process, which does not survive the spawn) -- so set it per rank here
        Sequence.block_size = self.block_size
        self.enforce_eager = config.enforce_eager
        self.tp_size = config.tensor_parallel_size
        self.dp_size = config.data_parallel_size
        self.world_size = self.dp_size * self.tp_size
        self.rank = rank
        self.target_rank = tar_rank
        self.event = event
        self.is_hybrid = config.hybrid_model
        self.mamba_pool = None

        init_distributed_environment(self.world_size, self.rank, backend="nccl")
        initialize_model_parallel(self.world_size, tensor_parallel_size=self.tp_size,
                                  data_parallel_size=self.dp_size)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        if self.is_hybrid:
            from nanovllm.models.qwen3p5 import Qwen3p5ForCausalLM
            self.model = Qwen3p5ForCausalLM(hf_config)
        else:
            from nanovllm.models.qwen3 import Qwen3ForCausalLM
            self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        # Everything compiled above was traced while "cuda" was the ambient default,
        # and torch.compile guards every graph on torch.utils._device.CURRENT_DEVICE
        # -- so those graphs are dead from here on ("torch._dynamo hit config.cache_size_limit"). 
        # Drop them: the kernels they loaded stay alive for the graphs already captured, and the 
        # serving paths recompile under the cpu default with the whole budget free.
        torch._dynamo.reset()
        torch.set_default_dtype(default_dtype)

        if self.tp_size > 1:
            dp_id = self.target_rank // self.tp_size
            if rank == self.target_rank:
                self.shm = SharedMemory(name=f"nanovllm_dp_{dp_id}", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=f"nanovllm_dp_{dp_id}")
                self.loop()

    def exit(self):
        if self.tp_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == self.target_rank:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        profiler = Profiler(self.config, rank=self.rank)
        with profiler:
            while True:
                method_name, args = self.read_shm()
                self.call(method_name, *args)
                if method_name == "exit":
                    break
                profiler.step()

    def read_shm(self):
        assert self.tp_size > 1 and self.rank != self.target_rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.tp_size > 1 and self.rank == self.target_rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.tp_size > 1 and self.rank == self.target_rank:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len, -1) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.text_config
        full_ratio = 1. - config.linear_cache_ratio if self.is_hybrid else 1.
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.tp_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * config.num_full_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        avaliable_mem = total * config.gpu_memory_utilization - used - peak + current
        # sentinel: all tp ranks should have the same num of kv slots
        if self.tp_size > 1:
            budget = torch.tensor([int(avaliable_mem)], dtype=torch.int64, device="cuda")
            dist.all_reduce(budget, op=dist.ReduceOp.MIN, group=get_tp_group().device_group)
            avaliable_mem = float(budget.item())
        config.num_kvcache_blocks = int(avaliable_mem*full_ratio) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, config.num_full_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)

        # linear state cache init
        if self.is_hybrid:
            from nanovllm.engine.block_manager import MambaPool
            self.mamba_pool = MambaPool(config, int(avaliable_mem*config.linear_cache_ratio),
                                        config.num_linear_layers, self.kv_cache.device)
            mamba_pool = self.mamba_pool
            # state ids index pool rows directly: hand the slot count to the
            # scheduler (built after this runner in both engines)
            config.num_linear_states = mamba_pool.num_slots
            # sentinel: the ids the scheduler hands out must fit this rank's pool
            assert config.num_linear_states <= mamba_pool.num_slots

        li_layer_id = kv_layer_id = 0
        for layer in self.model.model.layers:
            if not hasattr(layer, "self_attn") and hasattr(layer, "linear_attn"):
                attn = layer.linear_attn.attn
                attn.conv_cache = mamba_pool.conv_state[li_layer_id]
                attn.temporal_cache = mamba_pool.temporal_state[li_layer_id]
                li_layer_id += 1
            elif hasattr(layer, "self_attn"):
                attn = layer.self_attn.attn
                attn.k_cache = self.kv_cache[0, kv_layer_id]
                attn.v_cache = self.kv_cache[1, kv_layer_id]
                kv_layer_id += 1
            else:
                ValueError("Unsupport Attn Type in Model!")
        assert li_layer_id == config.num_linear_layers, f"bound {li_layer_id} != {config.num_linear_layers}"
        assert kv_layer_id == config.num_full_layers, f"bound {kv_layer_id} != {config.num_full_layers}"

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        state_ids = [] if self.is_hybrid else None
        cached_states = [] if self.is_hybrid else None
        snapshot_ids = [] if self.is_hybrid else None
        snapshot_pos = [] if self.is_hybrid else None
        has_snapshot = False
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            bos = cu_seqlens_q[-1]      # this sequence's first index in the flat batch
            cu_seqlens_q.append(bos + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
            if self.is_hybrid:
                state_ids.append(seq.mamba_state)
                cached_states.append(seq.cached_state)
                if seq.snapshot_state >= 0:
                    # where the snapshot sits, as an index into the flattened batch
                    # (bos, not cu_seqlens_q[-1]: that is this sequence's *end*)
                    flat = bos + (seq.snapshot_pos - start)
                    assert bos <= flat <= bos + seqlen_q, \
                        f"snapshot position {flat} left sequence [{bos}, {bos + seqlen_q})"
                    snapshot_ids.append(seq.snapshot_state)
                    snapshot_pos.append(flat)
                    has_snapshot = True
                else:
                    snapshot_ids.append(-1)
                    snapshot_pos.append(-1)
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        if self.is_hybrid:
            state_ids = torch.tensor(state_ids, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            cached_states = torch.tensor(cached_states, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            snapshot_ids = torch.tensor(snapshot_ids, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            snapshot_pos = torch.tensor(snapshot_pos, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables,
                    state_ids, cached_states, snapshot_ids, snapshot_pos, has_snapshot)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        state_ids = [] if self.is_hybrid else None
        snapshot_ids = [] if self.is_hybrid else None
        has_snapshot = False
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
            if self.is_hybrid:
                state_ids.append(seq.mamba_state)
                # a decode snapshot is a copy of the working slot after this step,
                # so it needs no position (only ever set when the scheduler opted in)
                snapshot_ids.append(seq.snapshot_state)
                has_snapshot = has_snapshot or seq.snapshot_state >= 0
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        if self.is_hybrid:
            state_ids = torch.tensor(state_ids, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            snapshot_ids = torch.tensor(snapshot_ids, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables,
                    state_ids=state_ids, snapshot_ids=snapshot_ids, has_snapshot=has_snapshot)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            if self.is_hybrid:
                graph_vars["state_ids"].fill_(-1)
                graph_vars["state_ids"][:bs] = context.state_ids
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        with record_function("prepare_inputs"):
            input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
            temperatures = self.prepare_sample(seqs) if self.rank == self.target_rank else None
        with record_function("model"):
            logits = self.run_model(input_ids, positions, is_prefill)
        with record_function("sample"):
            token_ids = self.sampler(logits, temperatures).tolist() if self.rank == self.target_rank else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.text_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        state_ids = torch.zeros(max_bs, dtype=torch.int32) if self.is_hybrid else None
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs],
                        state_ids=state_ids[:bs] if self.is_hybrid else None)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
            state_ids=state_ids
        )
