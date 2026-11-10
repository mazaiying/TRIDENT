#!/usr/bin/env python3
"""
PLAN_V10 shared helpers.
IMPORTANT: call setup_env(gpu_index) BEFORE importing torch / faiss / phase4a_suite.
"""
import csv
import os
import statistics
import sys
import time

TRIDENT_SCRIPTS = os.path.expanduser("~/trident/scripts")
if TRIDENT_SCRIPTS not in sys.path:
    sys.path.insert(0, TRIDENT_SCRIPTS)

DATA_ROOT = os.path.expanduser("~/trident/data")
RES_DIR = os.path.expanduser("~/trident/results/phase10")
NCU_BIN = "/usr/local/cuda-13/bin/ncu"
VENV_PY = "/home/mazaiying/trident/.venv/bin/python"


def setup_env(gpu_index: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    os.environ["CUDA_HOME"] = "/usr/local/cuda-13"
    os.environ["PATH"] = "/usr/local/cuda-13/bin:" + os.environ.get("PATH", "")
    # one JIT build that runs on both the A100 (sm_80) and the RTX 4080 (sm_89)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.9")
    os.environ.setdefault("TRIDENT_UB_SEED", "0")
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(k, None)
    os.makedirs(RES_DIR, exist_ok=True)


_CPP_SRC = r"""
#include <torch/extension.h>
#include <omp.h>
#include <cstdint>
#include <limits>
#include <algorithm>

// Exact FP32 re-ranking of candidate lists (CPU, 24 threads), float32 base vectors in ORIGINAL id order.
void refine_topk_f32(torch::Tensor xq, torch::Tensor xb, torch::Tensor cand,
                     torch::Tensor out_d, torch::Tensor out_i) {
    const int64_t B = xq.size(0), D = xq.size(1), N = xb.size(0), M = cand.size(1), K = out_d.size(1);
    const float* qp = xq.data_ptr<float>();
    const float* bp = xb.data_ptr<float>();
    const int64_t* cp = cand.data_ptr<int64_t>();
    float* od = out_d.data_ptr<float>();
    int64_t* oi = out_i.data_ptr<int64_t>();
    #pragma omp parallel for schedule(dynamic, 4) num_threads(24)
    for (int64_t q = 0; q < B; ++q) {
        const float* qr = qp + q * D;
        float* td = od + q * K;
        int64_t* ti = oi + q * K;
        for (int64_t j = 0; j < K; ++j) { td[j] = std::numeric_limits<float>::infinity(); ti[j] = -1; }
        for (int64_t m = 0; m < M; ++m) {
            const int64_t id = cp[q * M + m];
            if (id < 0 || id >= N) continue;
            const float* br = bp + id * D;
            float s = 0.0f;
            #pragma omp simd reduction(+:s)
            for (int64_t d = 0; d < D; ++d) { const float df = qr[d] - br[d]; s += df * df; }
            if (s < td[K - 1] || (s == td[K - 1] && id < ti[K - 1])) {
                td[K - 1] = s; ti[K - 1] = id;
                for (int64_t j = K - 1; j > 0; --j) {
                    if (td[j] < td[j - 1] || (td[j] == td[j - 1] && ti[j] < ti[j - 1])) {
                        std::swap(td[j], td[j - 1]); std::swap(ti[j], ti[j - 1]);
                    } else break;
                }
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("refine_topk_f32", &refine_topk_f32);
}
"""

_OMP_EXT = None


def get_omp_ext():
    global _OMP_EXT
    if _OMP_EXT is None:
        from torch.utils.cpp_extension import load_inline
        _OMP_EXT = load_inline(
            name="trident_v10_omp_ext",
            cpp_sources=_CPP_SRC,
            extra_cflags=["-O3", "-fopenmp", "-march=native"],
            extra_ldflags=["-fopenmp"],
            verbose=False,
        )
    return _OMP_EXT


def build_faiss_cpu_ivfflat_from_sorted(centroids, xb_sorted_f32, orig_ids_sorted, vec_off):
    """
    Real faiss.IndexIVFFlat (CPU) with the SAME centroids and SAME list assignment as TRIDENT/CM,
    filled directly through invlists.add_entries (no re-assignment, no k-means).
    xb_sorted_f32 : float32 [N, D] C-contiguous, rows in cluster order
    orig_ids_sorted: int64 [N], original id of each sorted row
    vec_off       : int32/int64 [nlist + 1]
    Returns (index, quantizer); keep BOTH references alive while searching.
    """
    import numpy as np
    import faiss
    nlist, d = centroids.shape
    quant = faiss.IndexFlatL2(d)
    quant.add(np.ascontiguousarray(centroids, dtype=np.float32))
    index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_L2)
    index.is_trained = True
    invl = index.invlists
    ids_all = np.ascontiguousarray(orig_ids_sorted, dtype=np.int64)
    assert xb_sorted_f32.dtype == np.float32 and xb_sorted_f32.flags["C_CONTIGUOUS"]
    for c in range(nlist):
        s, e = int(vec_off[c]), int(vec_off[c + 1])
        if e <= s:
            continue
        codes = xb_sorted_f32[s:e].reshape(-1).view(np.uint8)
        invl.add_entries(c, e - s, faiss.swig_ptr(ids_all[s:e]), faiss.swig_ptr(codes))
    index.ntotal = int(vec_off[-1])
    return index, quant


def measure_runs(fn, warmup=12, iters=10, b_size=1024, rounds=5):
    """Same methodology as the main table (phase4a measure_latency_qps: warmup 12, then back-to-back calls with
    a single synchronize at the end). Done in `rounds` rounds of `iters` calls -> mean / std over rounds.
    (`iters` keeps its old meaning for callers that pass a small value for slow CPU baselines.)"""
    import torch
    has_cuda = torch.cuda.is_available()
    if iters < 10:
        # slow baselines (seconds per call): per-call sync overhead is negligible; keep the old call budget
        rounds, iters = iters, 1
    for _ in range(warmup):
        fn()
    if has_cuda:
        torch.cuda.synchronize()
    qps = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        if has_cuda:
            torch.cuda.synchronize()
        qps.append(b_size * iters / max(time.perf_counter() - t0, 1e-9))
    return statistics.mean(qps), (statistics.stdev(qps) if len(qps) > 1 else 0.0)


def parse_ncu_csv(text, metric="lts__t_sectors_aperture_sysmem.sum"):
    """Return [(kernel_name, value)] in launch (ID) order for the given metric."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if start is None:
        return []
    out = []
    for r in csv.DictReader(lines[start:]):
        if r.get("Metric Name") != metric:
            continue
        try:
            v = float(str(r["Metric Value"]).replace(",", ""))
        except ValueError:
            continue
        out.append((int(r["ID"]), r["Kernel Name"], v))
    out.sort(key=lambda t: t[0])
    return [(k, v) for _, k, v in out]


def sectors_to_mb_per_1024q(sectors, batch):
    return round(sectors * 32.0 * (1024.0 / float(batch)) / (1024.0 ** 2), 2)


def write_rows(path, header, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)
    print(f"[csv] +{len(rows)} rows -> {path}", flush=True)
