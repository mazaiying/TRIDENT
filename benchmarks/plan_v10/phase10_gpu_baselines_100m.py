#!/usr/bin/env python3
"""
PLAN_V10 Step E3 — *real* GPU baselines on SIFT100M on the 16 GB RTX 4080 (GPU 0), long time limits.
Every attempt runs in its own subprocess with a hard timeout; the parent records exactly what happened
(OK / OOM / ERROR / TIMEOUT + elapsed seconds + exception text). Nothing is ever filled in by hand.

Modes (default order; pass --modes to choose):
  faiss_ivfflat_device   : faiss.GpuIndexIVFFlat, device memory, copyFrom(CPU IndexIVFFlat with identical lists)
  faiss_ivfflat_uvm      : same, GpuIndexIVFFlatConfig.memorySpace = MemorySpace_Unified (CUDA managed memory)
  cagra_device_ivfpq     : cuVS CAGRA build_algo="ivf_pq" (host dataset), default device memory resource
  cagra_uvm_ivfpq        : same, rmm ManagedMemoryResource (UVM)
  cagra_uvm_nn_descent   : build_algo="nn_descent", rmm ManagedMemoryResource (UVM)
If a build succeeds, the index is searched (Recall@10 vs gt, strict overlap vs exact IVF top-10 from
cm_ref_100m.npz written by phase10_ooc_100m.py --part main, QPS at B=1024).

MUST be launched through run_rapids_env.sh (sets LD_LIBRARY_PATH for librmm/libcuvs):
  nohup bash ~/trident/scripts/plan_v10/run_rapids_env.sh ~/trident/scripts/plan_v10/phase10_gpu_baselines_100m.py \
      > ~/trident/results/phase10/gpu_baselines_100m.log 2>&1 &
Output: ~/trident/results/phase10/gpu_baselines_100m_RTX4080.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase10_common import setup_env, RES_DIR  # noqa: E402

TIMEOUTS = {
    "faiss_ivfflat_device": 3600,
    "faiss_ivfflat_uvm": 3 * 3600,
    "cagra_device_ivfpq": 3 * 3600,
    "cagra_uvm_ivfpq": 6 * 3600,
    "cagra_uvm_nn_descent": 6 * 3600,
}
OUT = os.path.join(RES_DIR, "gpu_baselines_100m_RTX4080.jsonl")


def emit(rec):
    rec["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("[RESULT]", json.dumps(rec), flush=True)


def child(mode):
    setup_env(0)
    import numpy as np
    import torch  # import torch FIRST: it loads the NCCL/cuBLAS libs that cuVS needs
    from phase4a_suite import eval_accuracy_v4
    from phase8_ooc_100m_4080 import load_cached_100m_artifacts, get_helper_ext, N_100M, D_100M, NLIST_100M, DATA_ROOT
    from phase10_common import build_faiss_cpu_ivfflat_from_sorted

    ds_dir = os.path.join(DATA_ROOT, "sift100m")
    xq = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy")).astype(np.float32))
    gt10 = np.load(os.path.join(ds_dir, "gt.npy"))[:1000, :10]
    xq_b = np.ascontiguousarray(np.vstack([xq[:1000], xq[:24]]), dtype=np.float32)
    ref = np.load(os.path.join(RES_DIR, "cm_ref_100m.npz"))
    cm = {0.95: (ref["d95"], ref["i95"]), 0.99: (ref["d99"], ref["i99"])}
    rec = {"mode": mode, "dataset": "sift100m", "gpu": torch.cuda.get_device_name(0)}
    t0 = time.perf_counter()

    if mode.startswith("faiss_ivfflat"):
        import faiss
        helper = get_helper_ext()
        (centroids, _, _, sort_perm, vec_off, _, _, _, _, _, xb_sorted_u8_t) = load_cached_100m_artifacts()
        xb_f32 = torch.empty((N_100M, D_100M), dtype=torch.float32)
        helper.u8_to_f32_omp(xb_sorted_u8_t, xb_f32)
        cpu_idx, quant = build_faiss_cpu_ivfflat_from_sorted(centroids, xb_f32.numpy(), sort_perm.astype(np.int64), vec_off)
        del xb_f32
        res = faiss.StandardGpuResources()
        res.setTempMemory(512 * 1024 * 1024)
        cfg = faiss.GpuIndexIVFFlatConfig()
        cfg.device = 0
        if hasattr(cfg, "use_cuvs"):
            cfg.use_cuvs = False
        if mode == "faiss_ivfflat_uvm":
            cfg.memorySpace = getattr(faiss, "MemorySpace_Unified", getattr(faiss, "Unified", 2))
        rec["memory_space"] = "unified" if mode == "faiss_ivfflat_uvm" else "device"
        try:
            gidx = faiss.GpuIndexIVFFlat(res, D_100M, NLIST_100M, faiss.METRIC_L2, cfg)
            gidx.copyFrom(cpu_idx)
            rec["build_s"] = round(time.perf_counter() - t0, 1)
        except Exception as e:
            rec.update(status="OOM" if "alloc" in str(e).lower() or "memory" in str(e).lower() else "ERROR",
                       error=f"{type(e).__name__}: {str(e).splitlines()[0][:160]}",
                       elapsed_s=round(time.perf_counter() - t0, 1))
            emit(rec)
            return
        del cpu_idx
        ps = faiss.GpuParameterSpace()
        rec["results"] = []
        for R, npb in ((0.95, 48), (0.99, 160)):
            ps.set_index_parameter(gidx, "nprobe", npb)
            dd, ii = gidx.search(xq_b[:1000], 10)
            r10, ov, _ = eval_accuracy_v4(ii, dd, cm[R][1], cm[R][0], gt10)
            gidx.search(xq_b, 10)
            ts = []
            for _ in range(3):
                t1 = time.perf_counter()
                gidx.search(xq_b, 10)
                ts.append(1024 / (time.perf_counter() - t1))
            rec["results"].append({"recall_target": R, "nprobe": npb, "qps_mean": round(float(np.mean(ts)), 1),
                                   "qps_std": round(float(np.std(ts, ddof=1)), 1), "recall_at_10": round(r10, 4),
                                   "strict_overlap_pct": round(ov, 2)})
        rec.update(status="OK", elapsed_s=round(time.perf_counter() - t0, 1))
        emit(rec)
        return

    # ---------------- cuVS CAGRA ----------------
    import rmm
    from cuvs.neighbors import cagra
    algo = "nn_descent" if mode.endswith("nn_descent") else "ivf_pq"
    if "_uvm_" in mode:
        rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
        rec["memory_resource"] = "ManagedMemoryResource (UVM)"
    else:
        rec["memory_resource"] = "default device"
    rec.update(build_algo=algo, graph_degree=32, intermediate_graph_degree=64)
    xb_u8 = np.memmap(os.path.join(ds_dir, "base.100M.u8bin"), dtype=np.uint8, mode="r", offset=8, shape=(N_100M, D_100M))
    try:
        ip = cagra.IndexParams(metric="sqeuclidean", graph_degree=32, intermediate_graph_degree=64, build_algo=algo)
        idx = cagra.build(ip, xb_u8)
        rec["build_s"] = round(time.perf_counter() - t0, 1)
    except Exception as e:
        msg = str(e).splitlines()[0][:160] if str(e) else type(e).__name__
        rec.update(status="OOM" if "alloc" in msg.lower() or "memory" in msg.lower() else "ERROR",
                   error=f"{type(e).__name__}: {msg}", elapsed_s=round(time.perf_counter() - t0, 1))
        emit(rec)
        return
    q_dev = torch.from_numpy(np.clip(np.rint(xq_b), 0, 255).astype(np.uint8)).cuda()
    rec["results"] = []
    for itopk in (128, 256, 512):
        try:
            sp = cagra.SearchParams(itopk_size=itopk)
            d, n = cagra.search(sp, idx, q_dev[:1000], 10)
            ii = torch.as_tensor(n, device="cuda").cpu().numpy().astype(np.int64)
            dd = torch.as_tensor(d, device="cuda").cpu().numpy().astype(np.float32)
            r10, ov, _ = eval_accuracy_v4(ii, dd, cm[0.99][1], cm[0.99][0], gt10)
            cagra.search(sp, idx, q_dev, 10)
            torch.cuda.synchronize()
            ts = []
            for _ in range(5):
                t1 = time.perf_counter()
                cagra.search(sp, idx, q_dev, 10)
                torch.cuda.synchronize()
                ts.append(1024 / (time.perf_counter() - t1))
            rec["results"].append({"itopk": itopk, "qps_mean": round(float(np.mean(ts)), 1),
                                   "qps_std": round(float(np.std(ts, ddof=1)), 1), "recall_at_10": round(r10, 4),
                                   "strict_overlap_vs_exact_ivf_np160_pct": round(ov, 2)})
        except Exception as e:
            rec["results"].append({"itopk": itopk, "error": f"{type(e).__name__}: {str(e)[:120]}"})
    rec.update(status="OK", elapsed_s=round(time.perf_counter() - t0, 1))
    emit(rec)


def parent(modes):
    os.makedirs(RES_DIR, exist_ok=True)
    for m in modes:
        t0 = time.perf_counter()
        print(f"=== {m} (timeout {TIMEOUTS[m]} s) ===", flush=True)
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", m],
                               timeout=TIMEOUTS[m], capture_output=True, text=True, env=os.environ.copy())
            print(p.stdout[-4000:], flush=True)
            if "[RESULT]" not in p.stdout:
                tail = (p.stderr or "").strip().splitlines()[-3:]
                emit({"mode": m, "dataset": "sift100m", "status": "CRASH", "returncode": p.returncode,
                      "error": " | ".join(tail)[:300], "elapsed_s": round(time.perf_counter() - t0, 1)})
        except subprocess.TimeoutExpired:
            emit({"mode": m, "dataset": "sift100m", "status": "TIMEOUT", "timeout_s": TIMEOUTS[m],
                  "elapsed_s": round(time.perf_counter() - t0, 1)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", type=str, default="")
    ap.add_argument("--modes", type=str, default=",".join(TIMEOUTS.keys()))
    a = ap.parse_args()
    if a.child:
        child(a.child)
    else:
        parent([m for m in a.modes.split(",") if m])
