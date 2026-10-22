"""moe_gemm 模块级测试（单卡，无需分布式）。

覆盖重构后的单路径：
  1. moe_forward vs 逐专家 fp32 参考 —— I=768（EP=1）与 I=192（EP=4 的 M 分片）
     两种形状，后者同时覆盖 fused kernel 的 EVEN_N=False 编译分支；
  2. M 分片 partial 求和 == 全量输出（moe_tp 的数学）；
  3. capture 内 torch.empty 的 workspace：capture/replay vs eager、两次 replay
     位级一致、换输入 replay（输入用固定 buffer 原地 copy）；
  4. 共享 graph pool：两套权重交替 replay、多档位 T=1/16/512 乱序 replay
     （验证图私有池跨图复用时的地址稳定性）；
  5. 空专家/退化路由不越界。

需要 Phase 2 的 stacked `MoEWeights`（`w13 [E,H,2I]` / `w2 [E,I,H]`）；
在 Phase 1 的逐 expert 权重布局上运行会在 `_to_weights` 处断言失败。

用法：python -m pytest testfile/test_moe_gemm.py -v   （需要 GPU + triton）
"""

import os
import sys

import pytest

pytest.importorskip("triton")

import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("test_moe_gemm requires a CUDA device", allow_module_level=True)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nanovllm.layers.moe_gemm import MoEWeights, moe_forward

DEVICE = torch.device("cuda")
H, TOP_K, E = 2048, 8, 8          # 小 E 以降低显存/时间，形状语义与 A3B 相同


def _make_inputs_k(T, E_, top_k, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    hidden = torch.randn(T, H, device=DEVICE, dtype=torch.bfloat16, generator=g) * 0.1
    logits = torch.randn(T, E_, device=DEVICE, generator=g)
    scores = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(scores, top_k, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).to(torch.bfloat16)
    return hidden, topk_weights, topk_ids


def _make_inputs(T, E_, seed=0):
    return _make_inputs_k(T, E_, TOP_K, seed)


def _make_weights(I_local, E_=E, seed=1):
    """nn.Linear layout: w1 (E, 2I, H), w2 (E, H, I)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    w1 = (torch.randn(E_, 2 * I_local, H, device=DEVICE,
                      dtype=torch.bfloat16, generator=g) * 0.05)
    w2 = (torch.randn(E_, H, I_local, device=DEVICE,
                      dtype=torch.bfloat16, generator=g) * 0.05)
    return w1, w2


def _to_weights(w1, w2):
    """nn.Linear (E, out, in) -> the kernel's stacked K-major (E, K, N)."""
    return MoEWeights(w1.transpose(1, 2).contiguous(),
                      w2.transpose(1, 2).contiguous())


def reference_moe(hidden, topk_weights, topk_ids, w1, w2):
    """逐专家 fp32 参考（nn.Linear 布局：w1 (E,2I,H)、w2 (E,H,I)）。"""
    T = hidden.shape[0]
    out = torch.zeros(T, H, dtype=torch.float32, device=DEVICE)
    h32 = hidden.float()
    for e in range(w1.shape[0]):
        mask = (topk_ids == e)
        if not mask.any():
            continue
        rows, ks = mask.nonzero(as_tuple=True)
        x = h32[rows]
        gate_up = x @ w1[e].float().t()
        I = gate_up.shape[1] // 2
        act = F.silu(gate_up[:, :I]) * gate_up[:, I:]
        y = act @ w2[e].float().t()
        out.index_add_(0, rows, y * topk_weights[rows, ks].float().unsqueeze(-1))
    return out


def _max_rel(a, b):
    return ((a.float() - b.float()).abs().max()
            / b.float().abs().max().clamp_min(1e-6)).item()


def _buffered_inputs(T, seed=0):
    hidden, topk_w, topk_ids = _make_inputs(T, E, seed)
    return hidden.clone(), topk_w.clone(), topk_ids.clone()


def _capture_tier(weights, T, pool=None, seed=0):
    """Capture one T tier; returns (graph, output, input buffers, eager output).

    The inputs are fixed buffers copied into outside capture; the op allocates
    its own workspace inside capture, so the graph owns those addresses.
    """
    hidden_buf, w_buf, ids_buf = _buffered_inputs(T, seed)
    eager = moe_forward(hidden_buf, w_buf, ids_buf, weights).float().clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool):
        out = moe_forward(hidden_buf, w_buf, ids_buf, weights)
    return graph, out, (hidden_buf, w_buf, ids_buf), eager


@pytest.mark.parametrize("I_local", [768, 192])
def test_moe_forward_vs_reference(I_local):
    """1. 单路径 vs fp32 参考（192 覆盖 EVEN_N=False 分支）。"""
    T = 128
    hidden, topk_w, topk_ids = _make_inputs(T, E)
    w1, w2 = _make_weights(I_local)
    weights = _to_weights(w1, w2)

    out = moe_forward(hidden, topk_w, topk_ids, weights)
    ref = reference_moe(hidden, topk_w, topk_ids, w1, w2)

    rel = _max_rel(out, ref)
    assert rel < 1e-2, f"I_local={I_local}: max_rel {rel:.3e}"


def test_m_dim_sharding_sums_to_full():
    """2. EP=4 的 M 分片语义：4 份 partial 求和 == 全量输出。"""
    T, I_full, EP = 128, 768, 4
    I_local = I_full // EP
    hidden, topk_w, topk_ids = _make_inputs(T, E)
    w1, w2 = _make_weights(I_full)
    ref = reference_moe(hidden, topk_w, topk_ids, w1, w2)

    partial_sum = torch.zeros(T, H, dtype=torch.float32, device=DEVICE)
    for r in range(EP):
        w1_r = torch.cat([w1[:, r * I_local:(r + 1) * I_local, :],
                          w1[:, I_full + r * I_local:I_full + (r + 1) * I_local, :]],
                         dim=1)
        w2_r = w2[:, :, r * I_local:(r + 1) * I_local]
        out_r = moe_forward(hidden, topk_w, topk_ids, _to_weights(w1_r, w2_r))
        partial_sum += out_r.float()

    rel = _max_rel(partial_sum, ref)
    assert rel < 1e-2, f"partial sum: max_rel {rel:.3e}"


def test_capture_replay_vs_eager():
    """3. capture 内分配 + replay vs eager：位级一致、换输入 replay。"""
    T = 64
    w1, w2 = _make_weights(192, seed=2)
    weights = _to_weights(w1, w2)
    graph, out, buffers, eager = _capture_tier(weights, T)
    hidden_buf, w_buf, ids_buf = buffers

    graph.replay(); torch.cuda.synchronize()
    out1 = out.float().clone()
    graph.replay(); torch.cuda.synchronize()
    out2 = out.float().clone()
    assert torch.equal(out1, out2), "两次 replay 非位级一致"
    assert torch.equal(out1, eager), "capture/replay 与 eager 非位级一致"

    # Swap inputs in place (the graph reads these fixed addresses), then replay.
    hidden2, w2_, ids2 = _make_inputs(T, E, seed=7)
    hidden_buf.copy_(hidden2)
    w_buf.copy_(w2_)
    ids_buf.copy_(ids2)
    graph.replay(); torch.cuda.synchronize()
    ref2 = moe_forward(hidden_buf, w_buf, ids_buf, weights).float()
    rel = _max_rel(out, ref2)
    assert rel < 1e-2, f"replay with new inputs: max_rel {rel:.3e}"


def test_shared_pool_two_weights_alternate():
    """4a. 两张图共享一个 graph pool、两套权重交替 replay。"""
    T = 32
    wa = _to_weights(*_make_weights(192, seed=2))
    wb = _to_weights(*_make_weights(192, seed=3))
    ga, out_a, bufs_a, eager_a = _capture_tier(wa, T, pool=None, seed=0)
    gb, out_b, bufs_b, eager_b = _capture_tier(wb, T, pool=ga.pool(), seed=4)

    for _ in range(2):
        ga.replay(); torch.cuda.synchronize()
        assert torch.equal(out_a.float(), eager_a), "A 图 replay 与 eager 不一致"
        gb.replay(); torch.cuda.synchronize()
        assert torch.equal(out_b.float(), eager_b), "B 图 replay 与 eager 不一致"


def test_multi_tier_alternate_replay():
    """4b. 多档位共享 pool，乱序 replay；每档 vs eager 位级一致。

    This is the direct check for the graph-pool replay-order risk: tiers are
    captured small->large and replayed in a different order.
    """
    w1, w2 = _make_weights(192, seed=5)
    weights = _to_weights(w1, w2)
    tiers = {}
    pool = None
    for T in (1, 8, 64):
        graph, out, buffers, eager = _capture_tier(weights, T, pool=pool, seed=10 + T)
        tiers[T] = (graph, out, buffers, eager)
        pool = graph.pool()

    for T in (64, 1, 8, 64, 1):
        graph, out, _, eager = tiers[T]
        graph.replay(); torch.cuda.synchronize()
        assert torch.equal(out.float(), eager), f"T={T} replay 与 eager 不一致"


def test_empty_experts():
    """5. 只命中少量专家（其余 counts=0）不越界。"""
    T = 32
    hidden, topk_w, topk_ids = _make_inputs_k(T, E, top_k=2, seed=3)
    # Force all routes to experts 0 and 1 so experts 2..E-1 stay empty.
    topk_ids = topk_ids % 2
    w1, w2 = _make_weights(192, seed=6)
    weights = _to_weights(w1, w2)

    out = moe_forward(hidden, topk_w, topk_ids, weights)
    ref = reference_moe(hidden, topk_w, topk_ids, w1, w2)
    rel = _max_rel(out, ref)
    assert rel < 1e-2, f"empty experts: max_rel {rel:.3e}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
