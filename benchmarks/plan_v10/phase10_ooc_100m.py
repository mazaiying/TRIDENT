#!/usr/bin/env python3
"""
PLAN_V10 Steps E2 / N2  —  SIFT100M (51.2 GB FP32) on the 16 GB RTX 4080 (physical GPU index 0), no VRAM cap.

  --part main : TRIDENT-OOC (ub 0/1/2) + CM-FP32-OOC (same run), real Faiss-CPU-IVFFlat (24T, faiss.IndexIVFFlat),
                IVFPQ(M=16, GPU) + exact CPU refine sweep over candidate counts
  --part ncu  : Nsight Compute per-pass PCIe sysmem bytes for ub 0/1/2 and CM at nprobe 48 and 160
Reuses the cached 100M index artifacts built in PLAN_V9 (~/trident/cache_v3/sift100m_nlist16384_s0/).
Outputs (~/trident/results/phase10/): ooc_100m_RTX4080_v10.csv, ncu_perpass_100m_RTX4080.csv, cm_ref_100m.npz
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase10_common import setup_env  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--part", choices=["main", "ncu"], default="main")
ap.add_argument("--ncu-probe", action="store_true")
args = ap.parse_args()
setup_env(0)

import gc  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import faiss  # noqa: E402

from phase10_common import (RES_DIR, NCU_BIN, build_faiss_cpu_ivfflat_from_sorted, measure_runs,  # noqa: E402
                            parse_ncu_csv, sectors_to_mb_per_1024q, write_rows)
from phase4a_suite import eval_accuracy_v4  # noqa: E402
from phase8_ooc_100m_4080 import (load_cached_100m_artifacts, get_helper_ext, Trident100MOOCEngine,  # noqa: E402
                                  N_100M, D_100M, NLIST_100M, DATA_ROOT)

DEV = torch.device("cuda:0")
NP = {0.95: 48, 0.99: 160}
UBS = [0, 1, 2]
M_CANDS = [100, 200, 400, 800, 1600, 2048]
B_EVAL, B = 1000, 1024
HDR = ["dataset", "gpu", "recall_target", "nprobe", "batch", "method", "config", "ub_seed", "m_cand", "qps_mean",
       "qps_std", "recall_at_10", "strict_overlap_pct", "surv_pct", "dims_pruned_pct", "refined_per_query",
       "peak_alloc_gib", "status"]
OUT = os.path.join(RES_DIR, "ooc_100m_RTX4080_v10.csv")


def queries():
    ds_dir = os.path.join(DATA_ROOT, "sift100m")
    xq = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy")).astype(np.float32))
    gt = np.load(os.path.join(ds_dir, "gt.npy"))
    xq_b = np.ascontiguousarray(np.vstack([xq[:1000], xq[:24]]), dtype=np.float32)
    return xq_b, gt[:B_EVAL, :10]


def make_engine(max_batch=1024):
    helper = get_helper_ext()
    (centroids, mean64, R64, sort_perm, vec_off, tile_off, vmin_np, delta_np,
     sq8_t, oids_t, xb_sorted_u8_t) = load_cached_100m_artifacts()
    xb_pin = torch.empty((N_100M, D_100M), dtype=torch.float32, pin_memory=True)
    helper.u8_to_f32_omp(xb_sorted_u8_t, xb_pin)
    oid_pin = torch.empty(N_100M, dtype=torch.int32, pin_memory=True)
    oid_pin.copy_(torch.from_numpy(sort_perm))
    eng = Trident100MOOCEngine(centroids, mean64, R64, vec_off, tile_off, vmin_np, delta_np, sq8_t, oids_t,
                               xb_pin, oid_pin, max_batch=max_batch, max_nprobe=512, dev=DEV)
    return eng, xb_pin, oid_pin, centroids, sort_perm, vec_off


def search(eng, xq_gpu, npb, mode, ub=0, st=None):
    os.environ["TRIDENT_UB_SEED"] = str(ub if mode == 2 else 0)
    return eng.search(xq_gpu, npb, mode=mode, stats_buf=st)


def part_main():
    xq_b, gt10 = queries()
    xq_b_gpu = torch.from_numpy(xq_b).to(DEV).contiguous()
    eng, xb_pin, oid_pin, centroids, sort_perm, vec_off = make_engine()
    torch.cuda.reset_peak_memory_stats()

    cm_ref = {}
    for R, npb in NP.items():
        d, i = search(eng, xq_b_gpu[:B_EVAL], npb, 10)
        torch.cuda.synchronize()
        cm_ref[R] = (d.cpu().numpy().copy(), i.cpu().numpy().copy())
    np.savez(os.path.join(RES_DIR, "cm_ref_100m.npz"), d95=cm_ref[0.95][0], i95=cm_ref[0.95][1],
             d99=cm_ref[0.99][0], i99=cm_ref[0.99][1])

    rows = []
    D_pad = D_100M
    for ub in UBS:
        for R, npb in NP.items():
            st = torch.zeros(4, dtype=torch.int64, device=DEV)
            d, i = search(eng, xq_b_gpu[:B_EVAL], npb, 2, ub, st)
            torch.cuda.synchronize()
            tot, dims, surv, _ = [int(x) for x in st.tolist()]
            rec, ov, _ = eval_accuracy_v4(i.cpu().numpy(), d.cpu().numpy(), cm_ref[R][1], cm_ref[R][0], gt10)
            q, s = measure_runs(lambda: search(eng, xq_b_gpu, npb, 2, ub), warmup=12, iters=10, b_size=B)
            surv_pct = 100.0 * surv / max(tot, 1)
            dpr = 100.0 * (1.0 - dims / max(tot * D_pad, 1))
            print(f"[100M R={R} np={npb} ub={ub}] TRIDENT-OOC {q:.1f}±{s:.1f} QPS Rec={rec:.4f} Ov={ov:.2f}% "
                  f"surv={surv_pct:.4f}% dimsPruned={dpr:.1f}%", flush=True)
            rows.append(["sift100m", "RTX4080_16GB_NoCap", R, npb, B, "TRIDENT-OOC", f"ub_seed={ub}", ub, "",
                         round(q, 1), round(s, 1), round(rec, 4), round(ov, 2), round(surv_pct, 4), round(dpr, 2),
                         round(surv / B_EVAL, 2), "", "OK"])
    for R, npb in NP.items():
        q, s = measure_runs(lambda: search(eng, xq_b_gpu, npb, 10), warmup=1, iters=3, b_size=B)
        print(f"[100M R={R} np={npb}] CM-FP32-OOC {q:.1f}±{s:.1f} QPS", flush=True)
        rows.append(["sift100m", "RTX4080_16GB_NoCap", R, npb, B, "CM-FP32-OOC", "exact", 0, "", round(q, 1),
                     round(s, 1), "", 100.0, 100.0, 0.0, "", "", "OK"])
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    for r in rows:
        r[16] = round(peak, 3)
    write_rows(OUT, HDR, rows)
    rows = []
    del eng
    gc.collect()
    torch.cuda.empty_cache()

    # ---- real Faiss CPU IVF-Flat (faiss.IndexIVFFlat, same centroids & lists), 24 threads ----
    faiss.omp_set_num_threads(24)
    try:
        cpu_idx, quant = build_faiss_cpu_ivfflat_from_sorted(centroids, xb_pin.numpy(), sort_perm.astype(np.int64), vec_off)
        for R, npb in NP.items():
            cpu_idx.nprobe = npb
            dd, ii = cpu_idx.search(xq_b[:B_EVAL], 10)
            rec, ov, _ = eval_accuracy_v4(ii, dd, cm_ref[R][1], cm_ref[R][0], gt10)
            q, s = measure_runs(lambda: cpu_idx.search(xq_b, 10), warmup=1, iters=3, b_size=B)
            print(f"[100M R={R} np={npb}] Faiss-CPU-IVFFlat(24T) {q:.1f}±{s:.1f} QPS Rec={rec:.4f} Ov={ov:.2f}%", flush=True)
            rows.append(["sift100m", "host_RAM", R, npb, B, "Faiss-CPU-IVFFlat-24T", "faiss.IndexIVFFlat", "", "",
                         round(q, 1), round(s, 1), round(rec, 4), round(ov, 2), 100.0, 0.0, "", 0.0, "OK"])
        del cpu_idx, quant
    except Exception as e:
        rows.append(["sift100m", "host_RAM", "", "", B, "Faiss-CPU-IVFFlat-24T", "faiss.IndexIVFFlat", "", "", "", "",
                     "", "", "", "", "", "", f"ERROR {type(e).__name__}: {str(e)[:80]}"])
    write_rows(OUT, HDR, rows)
    rows = []
    del xb_pin, oid_pin
    gc.collect()

    # ---- IVFPQ (M=16, GPU) + exact CPU refine on the raw uint8 vectors, candidate sweep ----
    helper = get_helper_ext()
    ds_dir = os.path.join(DATA_ROOT, "sift100m")
    xb_u8 = np.memmap(os.path.join(ds_dir, "base.100M.u8bin"), dtype=np.uint8, mode="r", offset=8, shape=(N_100M, D_100M))
    try:
        quant_pq = faiss.IndexFlatL2(D_100M)
        quant_pq.add(centroids)
        cpu_pq = faiss.IndexIVFPQ(quant_pq, D_100M, NLIST_100M, 16, 8)
        cpu_pq.by_residual = True
        rng = np.random.default_rng(42)
        tr = np.sort(rng.choice(N_100M, 131072, replace=False))
        cpu_pq.train(np.ascontiguousarray(xb_u8[tr], dtype=np.float32))
        res = faiss.StandardGpuResources()
        res.setTempMemory(256 * 1024 * 1024)
        gpu_pq = faiss.index_cpu_to_gpu(res, 0, cpu_pq)
        del cpu_pq
        for s_i in range(0, N_100M, 5_000_000):
            gpu_pq.add(np.ascontiguousarray(xb_u8[s_i:s_i + 5_000_000], dtype=np.float32))
        ps = faiss.GpuParameterSpace()
        xb_u8_t = torch.from_numpy(np.asarray(xb_u8))
        xq_t = torch.from_numpy(xq_b)
        for R, npb in NP.items():
            ps.set_index_parameter(gpu_pq, "nprobe", npb)
            for m in M_CANDS:
                def run(n_q, m=m):
                    od = torch.empty((n_q, 10), dtype=torch.float32)
                    oi = torch.empty((n_q, 10), dtype=torch.int64)
                    _, cand = gpu_pq.search(xq_b[:n_q], m)
                    helper.omp_refine_topk_u8(xq_t[:n_q].contiguous(), xb_u8_t, torch.from_numpy(cand), od, oi)
                    return od, oi
                try:
                    od, oi = run(B_EVAL)
                    rec, ov, _ = eval_accuracy_v4(oi.numpy(), od.numpy(), cm_ref[R][1], cm_ref[R][0], gt10)
                    q, s = measure_runs(lambda: run(B), warmup=1, iters=5, b_size=B)
                    st = "OK"
                except Exception as e:
                    rec = ov = q = s = float("nan")
                    st = f"ERROR {type(e).__name__}: {str(e)[:60]}"
                print(f"[100M R={R} np={npb}] IVFPQ(M=16)+Refine(m={m}) {q:.1f} QPS Rec={rec:.4f} Ov={ov:.2f}% {st}", flush=True)
                rows.append(["sift100m", "RTX4080_PQ+host_u8", R, npb, B, "IVFPQ+CPU-Refine", "M=16", "", m,
                             round(q, 1), round(s, 1), round(rec, 4), round(ov, 2), "", "", m, "", st])
        del gpu_pq, res
    except Exception as e:
        rows.append(["sift100m", "RTX4080_PQ+host_u8", "", "", B, "IVFPQ+CPU-Refine", "M=16", "", "", "", "", "", "",
                     "", "", "", "", f"ERROR {type(e).__name__}: {str(e)[:80]}"])
    write_rows(OUT, HDR, rows)
    torch.cuda.empty_cache()


def ncu_probe():
    xq_b, _ = queries()
    xq_gpu = torch.from_numpy(xq_b).to(DEV).contiguous()
    eng, xb_pin, oid_pin, *_ = make_engine(max_batch=256)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for R, npb in NP.items():
        for ub in UBS:
            search(eng, xq_gpu[:256], npb, 2, ub)
            torch.cuda.synchronize()
        search(eng, xq_gpu[:64], npb, 10)
        torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def part_ncu():
    out = os.path.join(RES_DIR, "ncu_perpass_100m_RTX4080.csv")
    hdr = ["dataset", "recall_target", "nprobe", "method", "ub_seed", "pass1_mb_per_1024q", "pass2_mb_per_1024q",
           "total_mb_per_1024q", "status"]
    cmd = [NCU_BIN, "--profile-from-start", "off",
           "--kernel-name", "regex:trident_cluster_major_scan_kernel|trident_cm_fp32_scan_kernel",
           "--metrics", "lts__t_sectors_aperture_sysmem.sum", "--csv",
           sys.executable, os.path.abspath(__file__), "--ncu-probe"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=os.environ.copy())
    with open(os.path.join(RES_DIR, "ncu_raw_sift100m.txt"), "w") as f:
        f.write(p.stdout + "\n=== STDERR ===\n" + p.stderr)
    seq = parse_ncu_csv(p.stdout)
    rows = []
    # expected order per nprobe: 3 x (pass1, pass2) cluster_major launches, then 1 cm_fp32 launch
    pos = 0
    ok = len(seq) == 2 * (2 * len(UBS) + 1)
    for R, npb in NP.items():
        if not ok:
            break
        for ub in UBS:
            (k1, v1), (k2, v2) = seq[pos], seq[pos + 1]
            pos += 2
            if "cluster_major" not in k1 or "cluster_major" not in k2:
                ok = False
                break
            p1, p2 = sectors_to_mb_per_1024q(v1, 256), sectors_to_mb_per_1024q(v2, 256)
            rows.append(["sift100m", R, npb, "TRIDENT-OOC", ub, p1, p2, round(p1 + p2, 2), "OK"])
            print(f"[NCU 100M R={R} ub={ub}] pass1={p1} pass2={p2} total={p1 + p2:.2f} MB/1024q", flush=True)
        if not ok:
            break
        kc, vc = seq[pos]
        pos += 1
        if "cm_fp32" not in kc:
            ok = False
            break
        rows.append(["sift100m", R, npb, "CM-FP32-OOC", 0, "", "", sectors_to_mb_per_1024q(vc, 64), "OK"])
    if not ok:
        rows = [["sift100m", "", "", "ALL", "", "", "", "", f"UNEXPECTED kernel sequence len={len(seq)}; see ncu_raw_sift100m.txt"]]
    write_rows(out, hdr, rows)


if __name__ == "__main__":
    if args.ncu_probe:
        ncu_probe()
    elif args.part == "main":
        part_main()
    else:
        part_ncu()
