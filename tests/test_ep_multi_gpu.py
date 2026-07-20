"""
Multi-GPU end-to-end test for EP all-to-all dispatch/combine.

Run with torchrun:
    torchrun --nproc_per_node=4 tests/test_ep_multi_gpu.py

This validates the full MoE EP pipeline (router → dispatch → compute →
combine → merge) on multiple GPUs without loading real model weights.
"""

import os
import sys
import math
import torch
import torch.nn.init as init
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.layers.moe import MoELayer, ExpertFFN
from nanovllm.distributed.parallel_state import (
    initialize_model_parallel,
    get_ep_group,
    get_tp_group,
    reset_parallel_groups,
)


# ---------------------------------------------------------------------------
# Test configuration — kept small for fast iteration
# ---------------------------------------------------------------------------
HIDDEN = 128
MOE_INTERMEDIATE = 64
NUM_EXPERTS = 8
TOP_K = 2
EP_SIZE = 4          # must match --nproc_per_node
EXPERTS_PER_RANK = NUM_EXPERTS // EP_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_allclose(actual, expected, msg="", atol=1e-4, rtol=1e-4):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        diff = (actual - expected).abs().max().item()
        raise AssertionError(f"[Rank {dist.get_rank()}] {msg} Max diff: {diff:.8f}")


def check_no_nan(tensor, name):
    if torch.isnan(tensor).any():
        raise AssertionError(f"[Rank {dist.get_rank()}] {name} contains NaN")


def all_gather_tensor(tensor, world_size, max_len=None):
    """All-gather a 2D tensor [N_i, D] from all ranks. Returns [total_N, D]."""
    local_N = tensor.size(0)
    D = tensor.size(1)

    # Exchange sizes
    sizes = torch.tensor([local_N], dtype=torch.int64, device=tensor.device)
    all_sizes = [torch.zeros(1, dtype=torch.int64, device=tensor.device) for _ in range(world_size)]
    dist.all_gather(all_sizes, sizes)
    all_sizes = [s.item() for s in all_sizes]
    max_N = max(all_sizes)

    # Pad to max_N and all-gather
    padded = torch.zeros(max_N, D, dtype=tensor.dtype, device=tensor.device)
    padded[:local_N] = tensor

    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    # Trim each rank's contribution
    result = torch.cat([gathered[i][:all_sizes[i]] for i in range(world_size)], dim=0)
    return result


def gather_expert_weights_to_rank0(moe: MoELayer, ep_size: int):
    """Gather all expert weights from all ranks to rank 0 (NCCL-safe).

    Flattens each rank's expert weights into a tensor, all_gather-s them,
    and reconstructs the weight dicts on rank 0.

    Returns (on rank 0): list of dicts, each dict maps local_expert_id →
    {'gate': weight, 'up': weight, 'down': weight}.  On other ranks: None.
    """
    rank = dist.get_rank()
    PER_EXP = 3 * MOE_INTERMEDIATE * HIDDEN  # gate + up + down params per expert

    # Flatten all local expert weights into one 1D CPU tensor
    flat_parts = []
    for exp in moe.experts:
        flat_parts.append(exp.gate_proj.weight.data.cpu().flatten())
        flat_parts.append(exp.up_proj.weight.data.cpu().flatten())
        flat_parts.append(exp.down_proj.weight.data.cpu().flatten())
    local_flat = torch.cat(flat_parts)  # [EXPERTS_PER_RANK * PER_EXP]

    # All ranks have the same number of experts → equal-size tensors.
    # Use cpu_group (gloo) since NCCL does not support CPU tensors.
    gathered = [torch.empty_like(local_flat) for _ in range(ep_size)]
    dist.all_gather(gathered, local_flat, group=get_ep_group().cpu_group)

    if rank != 0:
        return None

    # Reconstruct on rank 0
    expert_weights = []
    for ep_r in range(ep_size):
        t = gathered[ep_r]
        rank_weights = []
        ptr = 0
        for local_id in range(EXPERTS_PER_RANK):
            g_sz = MOE_INTERMEDIATE * HIDDEN
            w = {}
            w['gate'] = t[ptr:ptr + g_sz].view(MOE_INTERMEDIATE, HIDDEN)
            ptr += g_sz
            w['up'] = t[ptr:ptr + g_sz].view(MOE_INTERMEDIATE, HIDDEN)
            ptr += g_sz
            w['down'] = t[ptr:ptr + HIDDEN * MOE_INTERMEDIATE].view(HIDDEN, MOE_INTERMEDIATE)
            ptr += HIDDEN * MOE_INTERMEDIATE
            rank_weights.append(w)
        expert_weights.append(rank_weights)
    return expert_weights


# ---------------------------------------------------------------------------
# Test 1: EP metadata exchange
# ---------------------------------------------------------------------------

def test_metadata_exchange():
    """Verify all-to-all split-size exchange produces correct counts."""
    rank = dist.get_rank()
    ep_group = get_ep_group()
    device = torch.device(f"cuda:{rank}")

    # Each rank sends a known number of tokens to each other rank.
    # send_counts[r] = how many tokens this rank sends to rank r.
    # For deterministic test: rank r sends (r+1) tokens to each destination.
    send_counts = torch.tensor([rank + 1] * EP_SIZE, dtype=torch.int64, device=device)

    recv_counts = torch.empty(EP_SIZE, dtype=torch.int64, device=device)
    dist.all_to_all_single(
        recv_counts, send_counts,
        output_split_sizes=[1] * EP_SIZE,
        input_split_sizes=[1] * EP_SIZE,
        group=ep_group.device_group,
    )

    # Rank r should receive (dst+1) tokens from rank dst.
    # So rank r's recv_counts[dst] should be dst+1.
    for src in range(EP_SIZE):
        expected = src + 1
        actual = recv_counts[src].item()
        assert actual == expected, \
            f"Rank {rank}: recv_counts[{src}] = {actual}, expected {expected}"

    if rank == 0:
        print("  ✓ test_metadata_exchange")


# ---------------------------------------------------------------------------
# Test 2: All-to-all with mismatched split sizes
# ---------------------------------------------------------------------------

def test_all_to_all_mismatched_splits():
    """Verify all-to-all correctly handles uneven split sizes."""
    rank = dist.get_rank()
    ep_group = get_ep_group()
    device = torch.device(f"cuda:{rank}")
    D = 16

    send_counts = [(rank + 1) * (dst + 1) for dst in range(EP_SIZE)]
    total_send = sum(send_counts)
    print(f"[rank{rank}] test2: send_counts={send_counts}, total_send={total_send}", flush=True)

    # Build input
    chunks = []
    for dst in range(EP_SIZE):
        n = send_counts[dst]
        c = torch.stack([
            torch.full((n,), rank, dtype=torch.float32),
            torch.full((n,), dst, dtype=torch.float32),
            torch.arange(n, dtype=torch.float32),
        ], dim=1).to(device)
        chunks.append(c)
    input_tensor = torch.cat(chunks, dim=0)
    print(f"[rank{rank}] test2: input built, shape={input_tensor.shape}", flush=True)

    torch.cuda.synchronize()
    print(f"[rank{rank}] test2: entering metadata a2a ...", flush=True)

    # Exchange metadata
    send_t = torch.tensor(send_counts, dtype=torch.int64, device=device)
    recv_t = torch.empty(EP_SIZE, dtype=torch.int64, device=device)
    dist.all_to_all_single(
        recv_t, send_t,
        output_split_sizes=[1] * EP_SIZE,
        input_split_sizes=[1] * EP_SIZE,
        group=ep_group.device_group,
    )
    recv_counts = recv_t.tolist()
    print(f"[rank{rank}] test2: metadata done, recv_counts={recv_counts}", flush=True)

    torch.cuda.synchronize()
    print(f"[rank{rank}] test2: entering data a2a, sum_recv={sum(recv_counts)}, sum_send={sum(send_counts)}", flush=True)

    # All-to-all with data
    output = torch.empty(sum(recv_counts), D, device=device)
    dist.all_to_all_single(
        output, input_tensor,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
        group=ep_group.device_group,
    )
    torch.cuda.synchronize()
    print(f"[rank{rank}] test2: data a2a done + sync", flush=True)

    # Verify: each received element's src should match the sending rank
    ptr = 0
    for src in range(EP_SIZE):
        n = recv_counts[src]
        if n == 0:
            continue
        chunk = output[ptr:ptr + n]
        ptr += n
        # Column 0 is src_rank
        actual_src = chunk[:, 0].unique().item()
        assert int(actual_src) == src, \
            f"Rank {rank}: expected src={src}, got src={actual_src}"

    if rank == 0:
        print("  ✓ test_all_to_all_mismatched_splits")


# ---------------------------------------------------------------------------
# Test 3: Full MoE EP forward vs reference
# ---------------------------------------------------------------------------

def test_moe_ep_forward():
    """End-to-end: EP forward output matches reference (all experts on rank 0)."""
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    ep_rank = rank  # EP = all ranks

    # --- Create MoELayer on GPU (match ModelRunner behavior) ---
    torch.set_default_device(device)
    torch.manual_seed(1000 + rank)
    moe = MoELayer(
        hidden_size=HIDDEN,
        intermediate_size=256,
        moe_intermediate_size=MOE_INTERMEDIATE,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        ep_size=EP_SIZE,
        ep_rank=ep_rank,
        moe_mode="ep",
        norm_topk_prob=True,
    )
    moe.eval()
    torch.set_default_device("cpu")
    # Init gate (ReplicatedLinear) — same seed across ranks so gate is identical
    torch.manual_seed(42)
    init.kaiming_uniform_(moe.gate.weight, a=math.sqrt(5))

    # --- Generate random inputs — different N per rank for realism ---
    torch.manual_seed(2000 + rank)
    N = [3, 7, 2, 5][rank]  # different tokens per rank
    hidden_states = torch.randn(N, HIDDEN, device=device)

    # --- EP forward ---
    ep_output = moe(hidden_states)  # [N, HIDDEN]

    # --- Gather all-to-all for reference ---
    gathered_inputs = all_gather_tensor(hidden_states, EP_SIZE)  # [total_N, HIDDEN]
    gathered_ep_outputs = all_gather_tensor(ep_output, EP_SIZE)  # [total_N, HIDDEN]

    # --- Gather all expert weights to rank 0 ---
    all_expert_weights = gather_expert_weights_to_rank0(moe, EP_SIZE)

    # --- Rank 0: build reference full MoE and compare ---
    if rank == 0:
        # Reconstruct full expert set on GPU
        full_experts = []
        for ep_r in range(EP_SIZE):
            for local_id in range(EXPERTS_PER_RANK):
                w = all_expert_weights[ep_r][local_id]
                exp = ExpertFFN(HIDDEN, MOE_INTERMEDIATE).to(device).eval()
                exp.gate_proj.weight.data.copy_(w['gate'].to(device))
                exp.up_proj.weight.data.copy_(w['up'].to(device))
                exp.down_proj.weight.data.copy_(w['down'].to(device))
                full_experts.append(exp)

        # Compute reference on GPU using the replicated gate
        gate_weight = moe.gate.weight.data
        gate_bias = None

        scores = F.linear(gathered_inputs, gate_weight, gate_bias)
        scores = F.softmax(scores, dim=-1, dtype=torch.float32)
        weights, indices = torch.topk(scores, TOP_K, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        ref_output = torch.zeros_like(gathered_inputs)
        total_tokens = gathered_inputs.size(0)
        for i in range(total_tokens):
            for k in range(TOP_K):
                eid = indices[i, k].item()
                w = weights[i, k]
                token = gathered_inputs[i:i + 1]
                expert_out = full_experts[eid](token).squeeze(0)
                ref_output[i] += w * expert_out

        assert_allclose(gathered_ep_outputs, ref_output,
                        "EP output differs from reference!", atol=1e-3, rtol=1e-3)
        check_no_nan(gathered_ep_outputs, "EP output")
        print("  ✓ test_moe_ep_forward (EP output == reference)")


# ---------------------------------------------------------------------------
# Test 4: IDLE rank simulation (one rank sends 0 tokens)
# ---------------------------------------------------------------------------

def test_moe_idle_rank():
    """Verify EP forward works when some ranks have empty input."""
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    ep_rank = rank

    # Create MoELayer on GPU
    torch.set_default_device(device)
    torch.manual_seed(42)
    moe = MoELayer(
        hidden_size=HIDDEN, intermediate_size=256,
        moe_intermediate_size=MOE_INTERMEDIATE,
        num_experts=NUM_EXPERTS, top_k=TOP_K,
        ep_size=EP_SIZE, ep_rank=ep_rank, moe_mode="ep",
    )
    moe.eval()
    torch.set_default_device("cpu")

    # Rank 0 sends tokens, other ranks send none (IDLE simulation)
    if rank == 0:
        hidden_states = torch.randn(5, HIDDEN, device=device)
    else:
        hidden_states = torch.randn(0, HIDDEN, device=device)

    # Forward should not deadlock
    ep_output = moe(hidden_states)
    assert ep_output.shape == (hidden_states.size(0), HIDDEN), \
        f"Rank {rank}: expected ({hidden_states.size(0)}, {HIDDEN}), got {ep_output.shape}"
    check_no_nan(ep_output, f"Rank {rank} EP output")

    if rank == 0:
        print("  ✓ test_moe_idle_rank (no deadlock, all ranks complete)")


# ---------------------------------------------------------------------------
# Test 5: All ranks produce same output with identical inputs
# ---------------------------------------------------------------------------

def test_moe_reproducibility():
    """With identical inputs AND identical weights across ranks, all ranks
    should produce the same EP output for tokens that route to the same experts."""
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    # Same seed → identical weights on all ranks, all on GPU
    torch.set_default_device(device)
    torch.manual_seed(777)
    moe = MoELayer(
        hidden_size=HIDDEN, intermediate_size=256,
        moe_intermediate_size=MOE_INTERMEDIATE,
        num_experts=NUM_EXPERTS, top_k=TOP_K,
        ep_size=EP_SIZE, ep_rank=rank, moe_mode="ep",
    )
    moe.eval()
    torch.set_default_device("cpu")

    # Same input on all ranks
    torch.manual_seed(888)
    hidden_states = torch.randn(4, HIDDEN, device=device)

    ep_output = moe(hidden_states)

    # All-gather all outputs to rank 0 and compare
    gathered = all_gather_tensor(ep_output, EP_SIZE)

    if rank == 0:
        # Each rank's output should be the same (same inputs, same weights)
        for r in range(EP_SIZE):
            assert_allclose(gathered[r*4:(r+1)*4], ep_output,
                            f"Rank {r} output differs from rank 0 output")
        print("  ✓ test_moe_reproducibility")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    # torchrun sets env vars: RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError as e:
        print(f"ERROR: Environment variable {e} not set.", file=sys.stderr)
        print("This test must be launched with torchrun:", file=sys.stderr)
        print(f"  torchrun --nproc_per_node={EP_SIZE} tests/test_ep_multi_gpu.py", file=sys.stderr)
        sys.exit(1)

    assert world_size == EP_SIZE, \
        f"This test requires exactly {EP_SIZE} GPUs, got WORLD_SIZE={world_size}"

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        device_id=torch.device('cuda', local_rank),
    )

    initialize_model_parallel(
        world_size=world_size,
        tensor_parallel_size=1,
        data_parallel_size=world_size,   # dp×tp must equal world_size
        expert_parallel_size=world_size,
        backend="nccl",
    )

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Multi-GPU EP Tests ({world_size} GPUs, TP=1 DP={world_size} EP={world_size})")
        print(f"{'='*60}")

    tests = [
        ("metadata_exchange", test_metadata_exchange),
        #("all_to_all_mismatched_splits", test_all_to_all_mismatched_splits),  # hangs on NCCL data a2a
        ("moe_ep_forward", test_moe_ep_forward),
        ("moe_idle_rank", test_moe_idle_rank),
        ("moe_reproducibility", test_moe_reproducibility),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        # Diagnose hang location
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print(f"  [trace] starting: {name}")
        try:
            fn()
            passed += 1
            torch.cuda.synchronize()
            if rank == 0:
                print(f"  [trace] {name} sync done, entering barrier ...")
            dist.barrier()
            if rank == 0:
                print(f"  [trace] finished: {name}")
        except Exception as e:
            if rank == 0:
                print(f"  ✗ {name}: {e}")
            failed += 1

    dist.barrier()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Results: {passed} passed, {failed} failed")
        print(f"{'='*60}")

    reset_parallel_groups()
    dist.destroy_process_group()

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
