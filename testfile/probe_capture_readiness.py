"""S0 穿刺验证：CUDA graph capture 对 MoE 图路径的兼容性（实施计划 S0）。

两个阻塞项（见 plan「S0 — 穿刺验证」）：

  [1] torch.argsort(stable=True) 在 capture 区内、n=8 与 n=4096（bs=512 档
      的 n = bs*top_k 上限）两个尺寸下可捕获。moe_gemm._permute_into 的
      全流程依赖它——这是本项目首次在 capture 内使用 sort。
      失败缓解（按代价递增）：
        a. 图档上限降到 bs=384（n=3072）；
        b. 改用复合键 flat*NK + arange(NK) 的 argsort（仍走小尺寸稳定路径）。

  [2] moe_gemm 两个 autotune kernel（grouped_matmul_kernel_masked /
      grouped_matmul_silu_kernel_masked）在 capture 外预热后，capture 内
      不再触发 autotune benchmark（autotune 内含 host sync/events，会 abort
      capture）。预热由本脚本的 eager 段完成。
      此项需要 Phase 2 的 stacked `MoEWeights` 布局。

另附 [3]：图内 index_select / index_add_ / scatter_add_ / cumsum 的捕获
（_permute_into / _unpermute_into 的全部 device op），一次 capture 全量覆盖。

用法（服务器，单卡即可；不必 4 卡）：
    python testfile/probe_capture_readiness.py

退出码 0 = 全部通过；非 0 = 有阻塞项，按输出的「缓解」提示处理。
"""

import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEVICE = torch.device("cuda")
E, H, I_LOCAL, TOP_K = 32, 2048, 192, 8   # EP=4 的 M 分片形状（I=768/4）


def _report(name, ok, detail="", remedy=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok and remedy:
        print(f"       缓解：{remedy}")
    return ok


def probe_argsort(n, device=DEVICE):
    """[1] capture 内 argsort(stable) + index_select 的可捕获性与正确性。"""
    torch.manual_seed(0)
    flat = torch.randint(0, 128, (n,), device=device, dtype=torch.int64)
    src = torch.randn(n, H, device=device, dtype=torch.bfloat16)
    ref_order = torch.argsort(flat, stable=True)
    ref = src.index_select(0, ref_order)

    order_buf = torch.empty_like(flat)          # 固定输出（图内写入）
    out = torch.empty_like(src)

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):                   # capture 外 warmup（sort 首次调用）
        for _ in range(3):
            order_buf.copy_(torch.argsort(flat, stable=True))
    torch.cuda.current_stream().wait_stream(s)

    try:
        with torch.cuda.graph(g):
            order_buf.copy_(torch.argsort(flat, stable=True))
            out.copy_(src.index_select(0, order_buf))
    except Exception as exc:                     # cudaErrorStreamCaptureUnsupported 等
        return _report(f"argsort(stable) capture n={n}", False, repr(exc),
                       "降档到 bs=384（n=3072）或改复合键 argsort（见文件头）")

    g.replay()
    torch.cuda.synchronize()
    ok = torch.equal(out, ref)
    return _report(f"argsort(stable) capture n={n}", ok,
                   "" if ok else "replay 结果与 eager 不一致")


def probe_moe_device_ops(device=DEVICE):
    """[3] _permute_into/_unpermute_into 的全部 device op 一次 capture 覆盖。"""
    torch.manual_seed(0)
    T, E_ = 64, E
    hidden = torch.randn(T, H, device=device, dtype=torch.bfloat16)
    topk_ids = torch.randint(0, E_, (T, TOP_K), device=device, dtype=torch.int64)
    topk_w = torch.rand(T, TOP_K, device=device, dtype=torch.bfloat16)
    rows = T * TOP_K
    token_idx = torch.empty(rows, device=device, dtype=torch.int64)
    sorted_w = torch.empty(rows, device=device, dtype=torch.bfloat16)
    permuted = torch.empty(rows, H, device=device, dtype=torch.bfloat16)
    counts = torch.empty(E_, device=device, dtype=torch.int32)
    row_off = torch.empty(E_, device=device, dtype=torch.int32)
    count_f = torch.zeros(E_, device=device, dtype=torch.float32)
    ones_f = torch.ones(rows, device=device, dtype=torch.float32)
    scratch = torch.empty(rows, H, device=device, dtype=torch.float32)
    acc = torch.empty(T, H, device=device, dtype=torch.float32)
    final = torch.empty(T, H, device=device, dtype=torch.bfloat16)

    def _body():
        flat = topk_ids.reshape(-1)
        order = torch.argsort(flat, stable=True)
        token_idx.copy_(torch.div(order, TOP_K, rounding_mode="floor"))
        sorted_w.copy_(topk_w.reshape(-1).index_select(0, order))
        torch.index_select(hidden, 0, token_idx, out=permuted)
        count_f.zero_()
        count_f.scatter_add_(0, flat, ones_f)
        counts.copy_(count_f)
        row_off.copy_(torch.cumsum(counts, 0) - counts)
        w = sorted_w.to(torch.float32).unsqueeze(-1)
        torch.mul(permuted.view(rows, H)[:, :H], w, out=scratch)
        acc.zero_()
        acc.index_add_(0, token_idx, scratch)
        final.copy_(acc)

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _body()
    torch.cuda.current_stream().wait_stream(s)
    try:
        with torch.cuda.graph(g):
            _body()
    except Exception as exc:
        return _report("moe permute/unpermute device ops capture", False, repr(exc),
                       "定位到具体 op（逐个 capture），必要时 device 化替换")
    g.replay()
    torch.cuda.synchronize()
    return _report("moe permute/unpermute device ops capture", True)


def probe_moe_graph(device=DEVICE):
    """[2] moe_gemm autotune 预热后，op 内分配 + capture/replay 正确。"""
    try:
        from nanovllm.layers.moe_gemm import MoEWeights, moe_forward
    except Exception as exc:
        return _report("import moe_gemm", False, repr(exc))

    torch.manual_seed(0)
    T = 64
    w1 = (torch.randn(E, 2 * I_LOCAL, H, device=device, dtype=torch.bfloat16) * 0.02)
    w2 = (torch.randn(E, H, I_LOCAL, device=device, dtype=torch.bfloat16) * 0.02)
    weights = MoEWeights(w1.transpose(1, 2).contiguous(),
                         w2.transpose(1, 2).contiguous())

    # Fixed input buffers: the graph records their addresses, so replay must
    # read these objects after an in-place copy.
    hidden = torch.randn(T, H, device=device, dtype=torch.bfloat16)
    topk_ids = torch.randint(0, E, (T, TOP_K), device=device, dtype=torch.int64)
    topk_w = torch.softmax(torch.rand(T, TOP_K, device=device), dim=-1).to(torch.bfloat16)

    # eager 参考——同时完成两个 kernel 的 autotune（务必在 capture 之前）
    ref = moe_forward(hidden, topk_w, topk_ids, weights).float().clone()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = moe_forward(hidden, topk_w, topk_ids, weights)
    g.replay()
    torch.cuda.synchronize()
    diff = (out.float() - ref).abs().max().item()

    # 换输入 replay：原地 copy 进固定 buffer，再 replay
    hidden2 = torch.randn(T, H, device=device, dtype=torch.bfloat16)
    ids2 = torch.randint(0, E, (T, TOP_K), device=device, dtype=torch.int64)
    w2_ = torch.softmax(torch.rand(T, TOP_K, device=device), dim=-1).to(torch.bfloat16)
    ref2 = moe_forward(hidden2, w2_, ids2, weights).float().clone()
    hidden.copy_(hidden2)
    topk_w.copy_(w2_)
    topk_ids.copy_(ids2)
    g.replay()
    torch.cuda.synchronize()
    diff2 = (out.float() - ref2).abs().max().item()

    ok = diff < 1e-2 and diff2 < 1e-2
    return _report("moe_gemm graph capture/replay", ok,
                   f"max|diff| replay1={diff:.3e} replay2={diff2:.3e}",
                   "检查 autotune 是否在 capture 内触发（本脚本应先跑 eager 段）")


def main():
    if not torch.cuda.is_available():
        print("需要 CUDA 设备")
        return 2
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    results = [
        probe_argsort(8),
        probe_argsort(4096),
        probe_moe_device_ops(),
        probe_moe_graph(),
    ]
    print(f"\n{sum(results)}/{len(results)} PASS")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
