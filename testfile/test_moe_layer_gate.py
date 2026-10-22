"""Single-GPU MoE layer gate with REAL checkpoint weights.

`testfile/test_moe_gemm.py` validates the grouped-GEMM kernel with random weights.
This test validates everything between the checkpoint and that kernel:

    HF key "model.layers.L.mlp.experts.E.gate_proj.weight"
        -> _remap_expert_weight -> stacked_gate_tp (M-dim narrow loader)
        -> w13 = cat([gate, up], dim=2)         (gate first, up second, K-major)
        -> MoEWeights (transpose to K-major)
        -> router (softmax fp32 + top-k + norm_topk_prob)
        -> grouped GEMM + unpermute

against an fp32 reference computed from the same checkpoint tensors.

EP sharding is NOT exercised here: MoELayer's DP guard rejects ep_size > TP,
so a single process can only run EP=1. The 4-GPU e2e gate
(testfile/test_moe_tp_gate.py) covers the M-dim shard + AllReduce path.

Memory: one layer of Qwen3-30B-A3B is ~1.2GB bf16 (128 experts x 3 x 768 x
2048); the fp32 reference stacks add ~2.4GB. Fine on one 32GB card.

Usage (server):

    CUDA_VISIBLE_DEVICES=0 python testfile/test_moe_layer_gate.py \
        --model <model-dir> --layer 0

Exit code 0 iff every token count passes. A JSON report is written next to the
script output (--report).
"""

import argparse
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _metrics(a, b):
    import torch.nn.functional as F

    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    diff = (a - b).abs()
    return {
        "rel_l2": (diff.norm() / b.norm().clamp_min(1e-12)).item(),
        "mean_abs": diff.mean().item(),
        "max_abs": diff.max().item(),
        "cosine": F.cosine_similarity(a, b, dim=0).item(),
    }


def load_layer_weights(mlp, model_path, layer_idx):
    """Load one layer's gate + expert weights through the production remap.

    Returns the CPU bf16 tensors keyed by their original HF names, for the
    reference implementation.
    """
    from safetensors import safe_open

    from nanovllm.utils.loader import _remap_expert_weight, default_weight_loader

    prefix = f"model.layers.{layer_idx}.mlp."
    ref = {}
    n_loaded = 0
    for file in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as f:
            for hf_name in f.keys():
                if not hf_name.startswith(prefix):
                    continue
                # clone: get_tensor may be a zero-copy view into the mmap, and
                # the reference tensors outlive the safe_open() context
                tensor = f.get_tensor(hf_name).clone()
                ref[hf_name] = tensor

                param_name, expert_id = _remap_expert_weight(hf_name)
                short = param_name[len(prefix):]
                try:
                    param = mlp.get_parameter(short)
                except AttributeError as e:
                    raise RuntimeError(
                        f"unsupported checkpoint layout: {hf_name!r} maps to "
                        f"{short!r}, which does not exist on MoELayer") from e
                loader = getattr(param, "weight_loader", default_weight_loader)
                if expert_id is None:
                    loader(param, tensor)
                else:
                    loader(param, tensor, expert_id)
                n_loaded += 1

    assert n_loaded > 0, f"no {prefix}* tensors found in {model_path}"
    return ref, n_loaded


def build_reference(ref, prefix, num_experts, device):
    """Stack the checkpoint tensors as fp32 on `device` (the fp32 truth)."""
    import torch

    gate = ref[f"{prefix}gate.weight"].to(device=device, dtype=torch.float32)
    gate_stack, up_stack, down_stack = [], [], []
    for e in range(num_experts):
        gate_stack.append(ref[f"{prefix}experts.{e}.gate_proj.weight"])
        up_stack.append(ref[f"{prefix}experts.{e}.up_proj.weight"])
        down_stack.append(ref[f"{prefix}experts.{e}.down_proj.weight"])
    to_f32 = lambda xs: torch.stack(xs).to(device=device, dtype=torch.float32)
    return gate, to_f32(gate_stack), to_f32(up_stack), to_f32(down_stack)


def reference_experts(hidden_f32, gate_w, up_w, down_w, topk_weights, topk_ids):
    """fp32 per-expert reference, using the routing handed in (nano's routing).

    hidden_f32: [T, H]; gate_w/up_w: [E, I, H]; down_w: [E, H, I]
    """
    import torch
    import torch.nn.functional as F

    out = torch.zeros_like(hidden_f32)
    for e in range(gate_w.shape[0]):
        mask = topk_ids == e
        if not bool(mask.any()):
            continue
        tok, slot = mask.nonzero(as_tuple=True)
        x = hidden_f32[tok]                                  # [t, H]
        g = x @ gate_w[e].T                                  # [t, I]
        u = x @ up_w[e].T
        h = F.silu(g) * u
        y = h @ down_w[e].T                                  # [t, H]
        out.index_add_(0, tok, y * topk_weights[tok, slot].unsqueeze(-1))
    return out


def _topk_ids(scores, top_k):
    import torch

    probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
    return torch.topk(probs, top_k, dim=-1).indices


def reference_routing(hidden_f32, gate_w, top_k):
    """Return (scores_fp32, ids_bf16, ids_fp32).

    Top-k is a discrete criterion, so its oracle must share the production
    precision semantics.  Nano's router is a bf16 ``F.linear``: the GEMM
    accumulates in fp32 but its *output* is rounded to bf16 before the fp32
    softmax/top-k.  ``ids_bf16`` models exactly that one extra rounding step
    by quantizing the fp32 logits to bf16 first; ``ids_fp32`` is the
    unquantized diagnostic.  Comparing raw fp32 logits against nano's ids
    makes near-tied 8th/9th experts flip and understates a correct router.
    """
    import torch

    scores = hidden_f32 @ gate_w.T
    ids_fp32 = _topk_ids(scores, top_k)
    ids_bf16 = _topk_ids(scores.to(torch.bfloat16).float(), top_k)
    return scores, ids_bf16, ids_fp32


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", default="1,17,64",
                        help="comma-separated token counts to test")
    parser.add_argument("--rel-l2-max", type=float, default=1e-2)
    parser.add_argument("--router-rel-l2-max", type=float, default=5e-3,
                        help="fp32 logits vs nano's bf16 logits, relative L2")
    parser.add_argument("--routing-min", type=float, default=0.99,
                        help="top-k agreement against the bf16-quantized oracle")
    parser.add_argument("--report", default=None,
                        help="JSON report path (default: <script-dir>/layer_gate_report.json)")
    args = parser.parse_args()

    import torch
    from transformers import AutoConfig

    from nanovllm.distributed.parallel_state import (
        init_distributed_environment, initialize_model_parallel)
    from nanovllm.layers.moe import MoELayer

    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    # the framework runs under @torch.inference_mode(); match that
    torch.set_grad_enabled(False)
    init_distributed_environment(1, 0, backend="nccl")
    initialize_model_parallel(world_size=1, tensor_parallel_size=1,
                              data_parallel_size=1, expert_parallel_size=1)

    hf = AutoConfig.from_pretrained(args.model)
    assert hf.model_type == "qwen3_moe", f"expected qwen3_moe, got {hf.model_type}"
    hidden_size = hf.hidden_size
    inter = hf.moe_intermediate_size
    num_experts = hf.num_experts
    top_k = hf.num_experts_per_tok
    norm_topk_prob = bool(getattr(hf, "norm_topk_prob", True))
    print(f"config: H={hidden_size} I={inter} E={num_experts} k={top_k} "
          f"norm_topk_prob={norm_topk_prob} layer={args.layer}")

    # Production dtype: the checkpoint and the framework both run bf16.
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        mlp = MoELayer(
            hidden_size=hidden_size,
            moe_intermediate_size=inter,
            num_experts=num_experts,
            top_k=top_k,
            ep_size=1,
            ep_rank=0,
            hidden_act=hf.hidden_act,
            norm_topk_prob=norm_topk_prob,
        )
    finally:
        torch.set_default_dtype(old_dtype)
    mlp = mlp.to(device).eval()

    prefix = f"model.layers.{args.layer}.mlp."
    ref, n_loaded = load_layer_weights(mlp, args.model, args.layer)
    print(f"loaded {n_loaded} tensors ({prefix}*) through _remap_expert_weight")
    assert n_loaded == 1 + 3 * num_experts, (
        f"expected {1 + 3 * num_experts} tensors for one layer, got {n_loaded}")

    token_counts = [int(x) for x in args.tokens.split(",") if x.strip()]
    mlp.finalize_moe()

    gate_w, gate_stack, up_stack, down_stack = build_reference(
        ref, prefix, num_experts, device)

    results = []
    gen = torch.Generator(device=device).manual_seed(1234)
    for t in token_counts:
        # bf16 activations, matching the framework's runtime dtype
        hidden = torch.randn(t, hidden_size, device=device,
                             generator=gen).to(torch.bfloat16)

        # --- nano: single compute path (workspace allocated per call) ---
        out = mlp(hidden)

        # The router is recomputed only for the cross-check below; mlp(hidden)
        # already ran the same routing internally.
        scores = mlp.gate(hidden)
        topk_weights, topk_ids = mlp._topk_routing(scores)

        # --- fp32 truth: expert math with nano's routing ---
        ref_out = reference_experts(hidden.float(), gate_stack, up_stack,
                                    down_stack, topk_weights.float(), topk_ids)

        # --- router cross-check ---
        # Hard criterion: logits (nano's bf16 gate GEMM vs fp32 matmul).
        # Discrete criterion: ids against the bf16-quantized oracle, which
        # models production's "GEMM output stored as bf16" step.  The raw fp32
        # oracle is reported as a diagnostic only (see reference_routing).
        scores_ref, ids_ref, ids_ref_fp32 = reference_routing(
            hidden.float(), gate_w, top_k)
        m_router = _metrics(scores, scores_ref)
        routing_agree = (ids_ref == topk_ids).float().mean().item()
        routing_agree_fp32 = (ids_ref_fp32 == topk_ids).float().mean().item()

        m_forward = _metrics(out, ref_out)

        row = {
            "tokens": t,
            "forward_vs_fp32": m_forward,
            "router_vs_fp32": m_router,
            "routing_agreement": routing_agree,
            "routing_agreement_fp32_oracle": routing_agree_fp32,
        }
        results.append(row)

        print(f"\n[T={t}]")
        print(f"  forward vs fp32 : relL2={m_forward['rel_l2']:.6f} "
              f"mean|d|={m_forward['mean_abs']:.6f} max|d|={m_forward['max_abs']:.6f} "
              f"cos={m_forward['cosine']:.6f}")
        print(f"  router vs fp32 : relL2={m_router['rel_l2']:.6f} "
              f"max|d|={m_router['max_abs']:.6f}")
        print(f"  routing agreement (bf16 oracle): {routing_agree * 100:.2f}%")
        print(f"  routing agreement (fp32 oracle, diagnostic): "
              f"{routing_agree_fp32 * 100:.2f}%")

    passed = all(
        r["forward_vs_fp32"]["rel_l2"] <= args.rel_l2_max
        and r["router_vs_fp32"]["rel_l2"] <= args.router_rel_l2_max
        and r["routing_agreement"] >= args.routing_min
        for r in results
    )

    report = {
        "model": args.model,
        "layer": args.layer,
        "config": {
            "hidden_size": hidden_size,
            "moe_intermediate_size": inter,
            "num_experts": num_experts,
            "top_k": top_k,
            "norm_topk_prob": norm_topk_prob,
        },
        "thresholds": {
            "rel_l2_max": args.rel_l2_max,
            "router_rel_l2_max": args.router_rel_l2_max,
            "routing_min": args.routing_min,
            "routing_oracle": "bf16-quantized logits",
        },
        "results": results,
        "passed": passed,
    }
    report_path = args.report or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "layer_gate_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    print(f"report -> {report_path}")

    import torch.distributed as dist
    dist.destroy_process_group()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
