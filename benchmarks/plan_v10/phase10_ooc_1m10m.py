#!/usr/bin/env python3
"""
PLAN_V10 Steps E0 / E1 / N1  (NVIDIA A100 80GB PCIe = physical GPU index 1 on iclr-a100)

  --part lossless : E0  in-VRAM losslessness + speed of TRIDENT_UB_SEED in {0,1,2} on 6 datasets (R=0.99)
  --part ooc      : E1  OOC (pinned host FP32) on GIST1M / BGE1024 / SIFT10M / Deep10M:
                        TRIDENT-OOC (ub 0/1/2), CM-FP32-OOC, real Faiss-CPU-IVFFlat (24T),
                        IVFPQ + exact CPU refine sweep over candidate counts (equal-accuracy comparison)
  --part ncu      : N1  Nsight Compute per-pass PCIe sysmem bytes (Pass-1 warm-up vs Pass-2), ub 0/1/2, CM
  (--ncu-probe DS is the internal child entry used by --part ncu)

Outputs (~/trident/results/phase10/):
  lossless_A100.csv, ooc_1m10m_A100.csv, ncu_perpass_1m10m_A100.csv, ncu_raw_<ds>.txt
Run:
  cd ~/trident/scripts/plan_v10 && /home/mazaiying/trident/.venv/bin/python phase10_ooc_1m10m.py --gpu 1 --part lossless
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase10_common import setup_env  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=1, help="physical GPU index (A100 = 1 on iclr-a100)")
ap.add_argument("--part", choices=["lossless", "ooc", "ncu"], default="ooc")
ap.add_argument("--datasets", type=str, default="")
ap.add_argument("--ncu-probe", type=str, default="")
args = ap.parse_args()
setup_env(args.gpu)

import gc  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import faiss  # noqa: E402

from phase10_common import (DATA_ROOT, RES_DIR, NCU_BIN, get_omp_ext, build_faiss_cpu_ivfflat_from_sorted,  # noqa: E402
                            measure_runs, parse_ncu_csv, sectors_to_mb_per_1024q, write_rows)
from phase4a_suite import (SharedClusterStorage, TridentV4Index, choose_tile, eval_accuracy_v4,  # noqa: E402
                           load_or_train_clustering, prepare_query_tensors)
from phase4b_suite import load_or_train_clustering_10m  # noqa: E402

CFG = {
    "gist1m":   dict(nlist=1024, np={0.95: 56,  0.99: 128}, mpq=48, big=False, cap=2.2),
    "bge1024":  dict(nlist=1024, np={0.95: 14,  0.99: 128}, mpq=32, big=False, cap=2.5),
    "sift10m":  dict(nlist=4096, np={0.95: 32,  0.99: 96},  mpq=16, big=True,  cap=3.0),
    "deep10m":  dict(nlist=4096, np={0.95: 32,  0.99: 96},  mpq=16, big=True,  cap=2.5),
    "sift1m":   dict(nlist=1024, np={0.95: 24,  0.99: 48},  mpq=16, big=False, cap=None),
    "glove100": dict(nlist=1024, np={0.95: 128, 0.99: 384}, mpq=20, big=False, cap=None),
    "msong":    dict(nlist=1024, np={0.95: 10,  0.99: 24},  mpq=20, big=False, cap=None),
    "nuswide":  dict(nlist=1024, np={0.95: 3,   0.99: 8},   mpq=20, big=False, cap=None),
}
UBS = [0, 1, 2]
M_CANDS = [100, 200, 400, 800, 1600, 2048]
DEV = torch.device("cuda:0")


def load_ds(ds):
    d = os.path.join(DATA_ROOT, ds)
    xb = np.ascontiguousarray(np.load(os.path.join(d, "base.npy")).astype(np.float32))
    xq = np.ascontiguousarray(np.load(os.path.join(d, "query.npy")).astype(np.float32))
    gt = np.load(os.path.join(d, "gt.npy"))
    return xb, xq, gt


def build(ds, xb):
    c = CFG[ds]
    if c["big"]:
        centroids, list_ids, r_vec, mean64, R64, _ = load_or_train_clustering_10m(ds, xb, c["nlist"], 0)
    else:
        centroids, list_ids, r_vec, mean64, R64, _ = load_or_train_clustering(ds, xb, c["nlist"], 0)
    storage = SharedClusterStorage(xb, list_ids, r_vec, c["nlist"], device="cuda:0")
    t = choose_tile(xb.shape[1])
    idx = TridentV4Index(storage, centroids, mean64, R64, tile_size=t, per_cluster_sq8=1, use_pca=1,
                         var_block=0, max_batch=1024, device="cuda:0")
    return centroids, storage, idx, t


def tr_search(idx, xq, nprobe, t, ub, stats=None):
    os.environ["TRIDENT_UB_SEED"] = str(ub)
    d, i, _ = idx.search(xq, nprobe, mode=2, use_l1=0, rerank_m=0, tile_override=t, warp_refine=1, stats_buf=stats)
    return d, i


def cm_search(idx, xq, nprobe, t):
    os.environ["TRIDENT_UB_SEED"] = "0"
    d, i, _ = idx.search(xq, nprobe, mode=10, use_l1=0, rerank_m=0, tile_override=t)
    return d, i


def run_tr_rows(ds, idx, t, xq_eval, xq_b, b_max, cm_ref, gt10, D, placement, ubs=UBS, rtags=(0.95, 0.99)):
    rows = []
    D_pad = ((D + 31) // 32) * 32
    Q = xq_eval.shape[0]
    for ub in ubs:
        for R in rtags:
            npb = CFG[ds]["np"][R]
            st = torch.zeros(4, dtype=torch.int64, device=DEV)
            d, i = tr_search(idx, xq_eval, npb, t, ub, st)
            torch.cuda.synchronize()
            tot, dims, surv, ins = [int(x) for x in st.tolist()]
            cm_d, cm_i = cm_ref[R]
            pi, pd = i.cpu().numpy(), d.cpu().numpy()
            rec, ov, _ = eval_accuracy_v4(pi, pd, cm_i, cm_d, gt10)
            same = (pi == cm_i)
            max_dd = float(np.max(np.abs(pd[same] - cm_d[same]))) if same.any() else -1.0
            q, s = measure_runs(lambda: tr_search(idx, xq_b, npb, t, ub), warmup=12, iters=10, b_size=b_max)
            surv_pct = 100.0 * surv / max(tot, 1)
            dims_pr = 100.0 * (1.0 - dims / max(tot * D_pad, 1))
            print(f"[{ds} {placement} R={R} np={npb} ub={ub}] TRIDENT {q:.1f}±{s:.1f} QPS  Rec={rec:.4f} Ov={ov:.2f}% "
                  f"surv={surv_pct:.3f}% refined/q={surv / Q:.1f} dimsPruned={dims_pr:.1f}% maxΔd={max_dd:.2e}", flush=True)
            rows.append([ds, placement, R, npb, b_max, "TRIDENT", f"ub_seed={ub}", ub, "", round(q, 1), round(s, 1),
                         round(rec, 4), round(ov, 2), round(surv_pct, 4), round(dims_pr, 2), round(surv / Q, 2),
                         f"{max_dd:.3e}"])
    return rows


HDR_LOSSLESS = ["dataset", "placement", "recall_target", "nprobe", "batch", "method", "config", "ub_seed", "m_cand",
                "qps_mean", "qps_std", "recall_at_10", "strict_overlap_pct", "surv_pct", "dims_pruned_pct",
                "refined_per_query", "max_abs_dist_diff_vs_cm"]


def part_lossless(dss):
    out = os.path.join(RES_DIR, "lossless_A100.csv")
    for ds in dss:
        xb, xq, gt = load_ds(ds)
        centroids, storage, idx, t = build(ds, xb)
        Q_eval, xq_eval, b_max, _, xq_batch = prepare_query_tensors(xq, ds, DEV)
        gt10 = gt[:Q_eval, :10]
        cm_ref = {}
        for R in (0.99,):
            d, i = cm_search(idx, xq_eval, CFG[ds]["np"][R], t)
            torch.cuda.synchronize()
            cm_ref[R] = (d.cpu().numpy().copy(), i.cpu().numpy().copy())
        rows = run_tr_rows(ds, idx, t, xq_eval, xq_batch[:b_max].contiguous(), b_max, cm_ref, gt10, xb.shape[1],
                           "InVRAM", rtags=(0.99,))
        write_rows(out, HDR_LOSSLESS, rows)
        del idx, storage, xb
        gc.collect()
        torch.cuda.empty_cache()


HDR_OOC = HDR_LOSSLESS + ["peak_alloc_gb", "cap_gb", "within_cap", "status"]


def part_ooc(dss):
    out = os.path.join(RES_DIR, "ooc_1m10m_A100.csv")
    ext = get_omp_ext()
    faiss.omp_set_num_threads(24)
    for ds in dss:
        c = CFG[ds]
        xb, xq, gt = load_ds(ds)
        N, D = xb.shape
        centroids, storage, idx, t = build(ds, xb)
        Q_eval, xq_eval, b_max, _, xq_batch = prepare_query_tensors(xq, ds, DEV)
        xq_b = xq_batch[:b_max].contiguous()
        gt10 = gt[:Q_eval, :10]
        cm_ref = {}
        for R, npb in c["np"].items():
            d, i = cm_search(idx, xq_eval, npb, t)
            torch.cuda.synchronize()
            cm_ref[R] = (d.cpu().numpy().copy(), i.cpu().numpy().copy())

        xb_sorted_host = np.ascontiguousarray(storage.xb_sorted.cpu().numpy())
        orig_sorted = storage.sort_perm.astype(np.int64)
        vec_off = storage.cluster_vec_off.cpu().numpy()

        # ---- switch TRIDENT / CM to OOC: FP32 rows live in pinned host RAM, read over PCIe (zero-copy) ----
        pinned = torch.from_numpy(xb_sorted_host).pin_memory()
        storage.xb_sorted = None
        gc.collect()
        torch.cuda.empty_cache()
        storage.xb_sorted = pinned
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        rows = run_tr_rows(ds, idx, t, xq_eval, xq_b, b_max, cm_ref, gt10, D, "OOC_pinned")
        for R, npb in c["np"].items():
            q, s = measure_runs(lambda: cm_search(idx, xq_b, npb, t), warmup=1, iters=3, b_size=b_max)
            print(f"[{ds} OOC R={R} np={npb}] CM-FP32-OOC {q:.1f}±{s:.1f} QPS", flush=True)
            rows.append([ds, "OOC_pinned", R, npb, b_max, "CM-FP32-OOC", "exact", 0, "", round(q, 1), round(s, 1),
                         "", 100.0, 100.0, 0.0, "", ""])
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        for r in rows:
            r += [round(peak, 3), c["cap"], int(peak <= c["cap"]), "OK"]
        write_rows(out, HDR_OOC, rows)
        rows = []

        del idx, storage, pinned
        gc.collect()
        torch.cuda.empty_cache()

        # ---- real Faiss CPU IVF-Flat (same centroids, same lists), 24 threads ----
        try:
            cpu_idx, quant = build_faiss_cpu_ivfflat_from_sorted(centroids, xb_sorted_host, orig_sorted, vec_off)
            xq_e_np = np.ascontiguousarray(xq_eval.cpu().numpy())
            xq_b_np = np.ascontiguousarray(xq_b.cpu().numpy())
            for R, npb in c["np"].items():
                cpu_idx.nprobe = npb
                dd, ii = cpu_idx.search(xq_e_np, 10)
                rec, ov, _ = eval_accuracy_v4(ii, dd, cm_ref[R][1], cm_ref[R][0], gt10)
                q, s = measure_runs(lambda: cpu_idx.search(xq_b_np, 10), warmup=1, iters=3, b_size=b_max)
                print(f"[{ds} R={R} np={npb}] Faiss-CPU-IVFFlat(24T) {q:.1f}±{s:.1f} QPS Rec={rec:.4f} Ov={ov:.2f}%", flush=True)
                rows.append([ds, "host_RAM", R, npb, b_max, "Faiss-CPU-IVFFlat-24T", "faiss.IndexIVFFlat", "", "",
                             round(q, 1), round(s, 1), round(rec, 4), round(ov, 2), 100.0, 0.0, "", "",
                             0.0, c["cap"], 1, "OK"])
            del cpu_idx, quant
        except Exception as e:  # report, never invent
            rows.append([ds, "host_RAM", "", "", b_max, "Faiss-CPU-IVFFlat-24T", "faiss.IndexIVFFlat", "", "",
                         "", "", "", "", "", "", "", "", "", c["cap"], "", f"ERROR {type(e).__name__}: {str(e)[:80]}"])
        gc.collect()

        # ---- IVFPQ (GPU) + exact FP32 CPU refine, sweep candidate count (equal-accuracy comparison) ----
        try:
            quant_pq = faiss.IndexFlatL2(D)
            quant_pq.add(centroids)
            cpu_pq = faiss.IndexIVFPQ(quant_pq, D, c["nlist"], c["mpq"], 8)
            cpu_pq.by_residual = True
            rng = np.random.default_rng(42)
            tr_idx = np.sort(rng.choice(N, min(N, 131072), replace=False))
            cpu_pq.train(np.ascontiguousarray(xb[tr_idx]))
            cpu_pq.add(xb)
            res = faiss.StandardGpuResources()
            res.setTempMemory(512 * 1024 * 1024)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = bool(c["mpq"] > 32)
            gpu_pq = faiss.index_cpu_to_gpu(res, 0, cpu_pq, co)
            ps = faiss.GpuParameterSpace()
            xb_t = torch.from_numpy(xb)
            xq_e_np = np.ascontiguousarray(xq_eval.cpu().numpy())
            xq_b_np = np.ascontiguousarray(xq_b.cpu().numpy())
            for R, npb in c["np"].items():
                ps.set_index_parameter(gpu_pq, "nprobe", npb)
                for m in M_CANDS:
                    def run(xq_np, m=m):
                        od = torch.empty((xq_np.shape[0], 10), dtype=torch.float32)
                        oi = torch.empty((xq_np.shape[0], 10), dtype=torch.int64)
                        _, cand = gpu_pq.search(xq_np, m)
                        ext.refine_topk_f32(torch.from_numpy(xq_np), xb_t, torch.from_numpy(cand), od, oi)
                        return od, oi
                    try:
                        od, oi = run(xq_e_np)
                        rec, ov, _ = eval_accuracy_v4(oi.numpy(), od.numpy(), cm_ref[R][1], cm_ref[R][0], gt10)
                        q, s = measure_runs(lambda: run(xq_b_np), warmup=1, iters=5, b_size=b_max)
                        st = "OK"
                    except Exception as e:
                        rec = ov = q = s = float("nan")
                        st = f"ERROR {type(e).__name__}: {str(e)[:60]}"
                    print(f"[{ds} R={R} np={npb}] IVFPQ(M={c['mpq']})+Refine(m={m}) {q:.1f} QPS Rec={rec:.4f} Ov={ov:.2f}% {st}", flush=True)
                    rows.append([ds, "GPU_PQ+host_FP32", R, npb, b_max, "IVFPQ+CPU-Refine", f"M={c['mpq']}", "", m,
                                 round(q, 1), round(s, 1), round(rec, 4), round(ov, 2), "", "", m, "",
                                 "", c["cap"], "", st])
            del gpu_pq, cpu_pq, res
        except Exception as e:
            rows.append([ds, "GPU_PQ+host_FP32", "", "", b_max, "IVFPQ+CPU-Refine", f"M={c['mpq']}", "", "",
                         "", "", "", "", "", "", "", "", "", c["cap"], "", f"ERROR {type(e).__name__}: {str(e)[:80]}"])
        write_rows(out, HDR_OOC, rows)
        del xb, xb_sorted_host
        gc.collect()
        torch.cuda.empty_cache()


def ncu_probe(ds):
    """Child process run under ncu: OOC searches ub0, ub1, ub2 (B=256) then CM (B=64) at R=0.99 nprobe."""
    xb, xq, gt = load_ds(ds)
    centroids, storage, idx, t = build(ds, xb)
    _, _, b_max, _, xq_batch = prepare_query_tensors(xq, ds, DEV)
    pinned = storage.xb_sorted.cpu().pin_memory()
    storage.xb_sorted = None
    gc.collect()
    torch.cuda.empty_cache()
    storage.xb_sorted = pinned
    npb = CFG[ds]["np"][0.99]
    b_tr = min(256, b_max)
    b_cm = min(64, b_max)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for ub in UBS:
        tr_search(idx, xq_batch[:b_tr].contiguous(), npb, t, ub)
        torch.cuda.synchronize()
    cm_search(idx, xq_batch[:b_cm].contiguous(), npb, t)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def part_ncu(dss):
    out = os.path.join(RES_DIR, "ncu_perpass_1m10m_A100.csv")
    hdr = ["dataset", "nprobe", "method", "ub_seed", "pass1_mb_per_1024q", "pass2_mb_per_1024q", "total_mb_per_1024q",
           "n_kernel_records", "status"]
    for ds in dss:
        npb = CFG[ds]["np"][0.99]
        cmd = [NCU_BIN, "--profile-from-start", "off",
               "--kernel-name", "regex:trident_cluster_major_scan_kernel|trident_cm_fp32_scan_kernel",
               "--metrics", "lts__t_sectors_aperture_sysmem.sum", "--csv",
               sys.executable, os.path.abspath(__file__), "--gpu", str(args.gpu), "--ncu-probe", ds]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=os.environ.copy())
        with open(os.path.join(RES_DIR, f"ncu_raw_{ds}.txt"), "w") as f:
            f.write(p.stdout + "\n=== STDERR ===\n" + p.stderr)
        seq = parse_ncu_csv(p.stdout)
        tr = [v for k, v in seq if "cluster_major" in k]
        cm = [v for k, v in seq if "cm_fp32" in k]
        rows = []
        if len(tr) != 2 * len(UBS) or len(cm) != 1:
            rows.append([ds, npb, "ALL", "", "", "", "", len(seq),
                         f"UNEXPECTED kernel count tr={len(tr)} cm={len(cm)} (see ncu_raw_{ds}.txt)"])
        else:
            for j, ub in enumerate(UBS):
                p1 = sectors_to_mb_per_1024q(tr[2 * j], 256)
                p2 = sectors_to_mb_per_1024q(tr[2 * j + 1], 256)
                rows.append([ds, npb, "TRIDENT-OOC", ub, p1, p2, round(p1 + p2, 2), len(seq), "OK"])
                print(f"[NCU {ds} ub={ub}] pass1={p1} MB pass2={p2} MB total={p1 + p2:.2f} MB /1024q", flush=True)
            cmb = sectors_to_mb_per_1024q(cm[0], 64)
            rows.append([ds, npb, "CM-FP32-OOC", 0, "", "", cmb, len(seq), "OK"])
        write_rows(out, hdr, rows)


if __name__ == "__main__":
    if args.ncu_probe:
        ncu_probe(args.ncu_probe)
        sys.exit(0)
    if args.part == "lossless":
        dss = args.datasets.split(",") if args.datasets else ["sift1m", "glove100", "msong", "nuswide", "gist1m", "bge1024"]
        part_lossless(dss)
    elif args.part == "ooc":
        dss = args.datasets.split(",") if args.datasets else ["gist1m", "bge1024", "sift10m", "deep10m"]
        part_ooc(dss)
    elif args.part == "ncu":
        dss = args.datasets.split(",") if args.datasets else ["gist1m", "bge1024", "sift10m", "deep10m"]
        part_ncu(dss)
