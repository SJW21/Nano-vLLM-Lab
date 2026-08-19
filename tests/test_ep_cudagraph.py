"""
CUDA Graph capture verification for MoE EP (all_gather mode).

Run with torchrun:
    torchrun --nproc_per_node=4 tests/test_ep_cudagraph.py

Validates that the all_gather MoE forward can be captured and replayed
inside a torch.cuda.CUDAGraph without errors or deadlocks.
"""

import os
import sys
import torch
import torch.nn.init as init
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.layers.moe import MoELayer
from nanovllm.distributed.parallel_state import (
    initialize_model_parallel,
    get_ep_group,
    reset_parallel_groups,
)

# --- config ---
HIDDEN = 128
MOE_INTERMEDIATE = 64
NUM_EXPERTS = 8
TOP_K = 2
EP_SIZE = 4


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank,
                            device_id=torch.device("cuda", local_rank))
    initialize_model_parallel(world_size, tensor_parallel_size=1,
                              data_parallel_size=world_size,
                              expert_parallel_size=EP_SIZE,
                              backend="nccl")
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)

    # Build MoELayer with all_gather comm mode
    moe = MoELayer(HIDDEN, HIDDEN, MOE_INTERMEDIATE, NUM_EXPERTS, TOP_K,
                   ep_size=EP_SIZE, ep_rank=rank, comm_mode="all_gather")

    # Init weights for reproducibility (stacked params)
    init.kaiming_uniform_(moe.stacked_gate)
    init.kaiming_uniform_(moe.stacked_up)
    init.kaiming_uniform_(moe.stacked_down)

    # Init gate (ReplicatedLinear base weight)
    if hasattr(moe.gate, 'weight'):
        init.kaiming_uniform_(moe.gate.weight)

    # --- Test 1: eager forward runs without error ---
    N = 32
    x = torch.randn(N, HIDDEN)

    eager_out = moe(x)
    assert not torch.isnan(eager_out).any(), "NaN in eager output"
    assert eager_out.shape == (N, HIDDEN), f"Shape mismatch: {eager_out.shape}"

    if rank == 0:
        print(f"[Test 1] Eager forward: PASSED (output shape {eager_out.shape})")
    dist.barrier()

    # --- Test 2: CUDA Graph capture ---
    graph_in = torch.zeros(N, HIDDEN)
    graph_in.copy_(x)
    graph_out = torch.zeros(N, HIDDEN)

    # Warmup
    with torch.no_grad():
        graph_out.copy_(moe(graph_in))
    torch.cuda.synchronize()

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g):
            graph_out.copy_(moe(graph_in))
    torch.cuda.synchronize()

    if rank == 0:
        print("[Test 2] CUDA Graph capture: PASSED")
    dist.barrier()

    # --- Test 3: Replay with same input → outputs match eager ---
    with torch.no_grad():
        graph_in.copy_(x)
        g.replay()

    torch.cuda.synchronize()

    max_diff = (graph_out - eager_out).abs().max().item()
    if max_diff > 1e-3:
        raise AssertionError(
            f"[Rank {rank}] Graph replay mismatch: max diff {max_diff:.6f}")

    if rank == 0:
        print(f"[Test 3] Graph replay matched eager: PASSED (max diff {max_diff:.6f})")
    dist.barrier()

    # --- Test 4: Replay with different input → no deadlock, valid output ---
    x2 = torch.randn(N, HIDDEN)
    eager_out2 = moe(x2)

    with torch.no_grad():
        graph_in.copy_(x2)
        g.replay()

    torch.cuda.synchronize()

    max_diff2 = (graph_out - eager_out2).abs().max().item()
    if max_diff2 > 1e-3:
        raise AssertionError(
            f"[Rank {rank}] Graph replay 2 mismatch: max diff {max_diff2:.6f}")

    if rank == 0:
        print(f"[Test 4] Graph replay with new input: PASSED (max diff {max_diff2:.6f})")
    dist.barrier()

    # --- Test 5: Replay N times in a loop → stable, no memory leak ---
    for i in range(10):
        noise = torch.randn(N, HIDDEN)
        with torch.no_grad():
            graph_in.copy_(noise)
            g.replay()
        torch.cuda.synchronize()

    if rank == 0:
        print("[Test 5] 10x graph replay: PASSED")
    dist.barrier()

    del g
    torch.cuda.empty_cache()
    dist.destroy_process_group()

    if rank == 0:
        print("\n=== All CUDA Graph tests PASSED ===")


if __name__ == "__main__":
    main()
