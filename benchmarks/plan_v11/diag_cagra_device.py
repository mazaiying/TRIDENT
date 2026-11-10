#!/usr/bin/env python3
"""
PLAN_V11 Task A — Diagnose cuVS CAGRA-Device on SIFT100M (16 GB RTX 4080, GPU 0).
Launched via:
  bash ~/trident/scripts/plan_v10/run_rapids_env.sh ~/trident/scripts/plan_v11/diag_cagra_device.py \
    > ~/trident/results/phase11/cagra_device_diag.log 2>&1
"""
import json
import os
import subprocess
import sys
import threading
import time

# Enable verbose logging in RAFT / cuVS before importing
os.environ.setdefault("RAFT_LOG_LEVEL", "DEBUG")
os.environ.setdefault("CUVS_LOG_LEVEL", "DEBUG")
os.environ.setdefault("SPDLOG_LEVEL", "debug")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V10_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "plan_v10")
sys.path.insert(0, V10_DIR)
from phase10_common import setup_env, RES_DIR  # noqa: E402

RES_V11 = os.path.join(os.path.dirname(RES_DIR), "phase11")
os.makedirs(RES_V11, exist_ok=True)


def inspect_index(idx, label):
    attrs = {}
    for k in dir(idx):
        if k.startswith("_"):
            continue
        try:
            v = getattr(idx, k)
            if not callable(v):
                attrs[k] = repr(v)
        except Exception as e:
            attrs[k] = f"<err: {e}>"
    print(f"[{label}] dir(idx) non-callable attributes: {json.dumps(attrs)}", flush=True)
    return attrs


def get_gpu0_vram_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        used, total = [int(x.strip()) for x in out.split(",")]
        return used, total
    except Exception:
        return -1, -1


def main():
    setup_env(0)
    import numpy as np
    import torch
    from cuvs.neighbors import cagra
    from phase8_ooc_100m_4080 import N_100M, D_100M, DATA_ROOT

    ds_dir = os.path.join(DATA_ROOT, "sift100m")
    xq = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy")).astype(np.float32))
    xq_b = np.ascontiguousarray(np.vstack([xq[:1000], xq[:24]]), dtype=np.float32)
    xb_u8 = np.memmap(
        os.path.join(ds_dir, "base.100M.u8bin"), dtype=np.uint8, mode="r", offset=8, shape=(N_100M, D_100M)
    )
    q_dev = torch.from_numpy(np.clip(np.rint(xq_b), 0, 255).astype(np.uint8)).cuda()

    print("=== STEP A.2.1: 1M Small-Scale Self-Check ===", flush=True)
    t0 = time.perf_counter()
    xb_1m = xb_u8[:1_000_000]
    ip_1m = cagra.IndexParams(
        metric="sqeuclidean", graph_degree=32, intermediate_graph_degree=64, build_algo="ivf_pq"
    )
    idx_1m = cagra.build(ip_1m, xb_1m)
    build_1m_s = round(time.perf_counter() - t0, 2)
    attrs_1m = inspect_index(idx_1m, "1M")
    used_1m, total_1m = get_gpu0_vram_mib()
    print(f"[1M] build_s={build_1m_s}s, vram={used_1m}/{total_1m} MiB", flush=True)

    sp = cagra.SearchParams(itopk_size=128)
    d_1m, n_1m = cagra.search(sp, idx_1m, q_dev[:1000], 10)
    ii_1m = torch.as_tensor(n_1m, device="cuda").cpu().numpy()
    dd_1m = torch.as_tensor(d_1m, device="cuda").cpu().numpy()
    print(f"[1M] search succeeded! neighbors.shape={ii_1m.shape}, distances.shape={dd_1m.shape}", flush=True)
    Step1_OK = (ii_1m.shape == (1000, 10))
    del idx_1m, d_1m, n_1m
    torch.cuda.empty_cache()

    print("\n=== STEP A.2.2: 100M Reproduction on 16 GB RTX 4080 ===", flush=True)
    peak_vram = [0]
    total_vram = [0]
    stop_mon = threading.Event()

    def monitor_vram():
        while not stop_mon.is_set():
            u, tot = get_gpu0_vram_mib()
            if u > peak_vram[0]:
                peak_vram[0] = u
            if tot > 0:
                total_vram[0] = tot
            time.sleep(0.5)

    mon_thread = threading.Thread(target=monitor_vram, daemon=True)
    mon_thread.start()

    t1 = time.perf_counter()
    ip_100m = cagra.IndexParams(
        metric="sqeuclidean", graph_degree=32, intermediate_graph_degree=64, build_algo="ivf_pq"
    )
    idx_100m = cagra.build(ip_100m, xb_u8)
    build_100m_s = round(time.perf_counter() - t1, 1)
    attrs_100m = inspect_index(idx_100m, "100M")
    used_after_build, tot_after_build = get_gpu0_vram_mib()
    print(
        f"[100M] build_s={build_100m_s}s, vram_after_build={used_after_build}/{tot_after_build} MiB, "
        f"peak_vram_during_build={peak_vram[0]}/{total_vram[0]} MiB",
        flush=True,
    )

    search_err_full = None
    try:
        d_100m, n_100m = cagra.search(sp, idx_100m, q_dev[:1000], 10)
        torch.cuda.synchronize()
        print("[100M] Unexpectedly succeeded search!", flush=True)
    except Exception as e:
        search_err_full = f"{type(e).__name__}: {str(e).strip()}"
        print(f"[100M] Search exception (full): {search_err_full}", flush=True)

    stop_mon.set()
    mon_thread.join(timeout=2.0)

    dim_1m = int(attrs_1m.get("dim", "-1"))
    dim_100m = int(attrs_100m.get("dim", "-1"))
    print(
        f"\n=== STEP A.3 DECISION SUMMARY ===\n"
        f"1M self-check OK: {Step1_OK} (idx_1m.dim={dim_1m})\n"
        f"100M idx_100m.dim: {dim_100m}\n"
        f"100M VRAM after build: {used_after_build}/{tot_after_build} MiB (peak {peak_vram[0]} MiB)\n"
        f"100M full search error: {search_err_full}",
        flush=True,
    )

    if Step1_OK and dim_100m == 0:
        branch = "Branch 1 (Hypothesis Confirmed: OOM during dataset attach; idx.dim=0)"
        status = "OOM"
        reason = (
            f"1M self-check succeeded (idx.dim={dim_1m}, output shape (1000,10)); "
            f"on 100M, graph (100M*32*4B = 12.80 GB) + dataset (100M*128*1B = 12.80 GB) = 25.60 GB > 16 GB VRAM "
            f"(used {used_after_build}/{tot_after_build} MiB after build, peak {peak_vram[0]} MiB), "
            f"so cuVS drops the dataset attachment (idx.dim={dim_100m}) and fails at search with: {search_err_full}"
        )
    else:
        branch = "Branch 3 (Other error)"
        status = "ERROR"
        reason = search_err_full or "Unknown"

    rec = {
        "mode": "cagra_device_ivfpq_v11",
        "supersedes": "cagra_device_ivfpq",
        "dataset": "sift100m",
        "gpu": torch.cuda.get_device_name(0),
        "memory_resource": "default device",
        "build_algo": "ivf_pq",
        "graph_degree": 32,
        "intermediate_graph_degree": 64,
        "self_check_1m_ok": Step1_OK,
        "idx_dim_1m": dim_1m,
        "idx_dim_100m": dim_100m,
        "vram_after_build_mib": used_after_build,
        "vram_total_mib": tot_after_build,
        "peak_vram_mib": peak_vram[0],
        "build_s": build_100m_s,
        "status": status,
        "decision_branch": branch,
        "error": reason,
        "raw_search_exception": search_err_full,
        "elapsed_s": round(time.perf_counter() - t1, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    p10_jsonl = os.path.join(RES_DIR, "gpu_baselines_100m_RTX4080.jsonl")
    p11_jsonl = os.path.join(RES_V11, "gpu_baselines_100m_RTX4080_v11.jsonl")
    with open(p10_jsonl, "a") as f:
        f.write(json.dumps(rec) + "\n")
    with open(p11_jsonl, "w") as f:
        f.write(json.dumps(rec) + "\n")
    print("[RESULT_V11]", json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
