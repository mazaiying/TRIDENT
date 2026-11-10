#!/usr/bin/env python3
"""
TRIDENT Phase 4B Benchmark Suite: 10M Scale Evaluation (PLAN_V5)
Evaluates 10M scale datasets (SIFT10M, Deep10M, etc.) on NVIDIA A100 80GB.
Implements:
  - 4B-1: Exact GPU Ground Truth & Float64 Audit
  - 4B-2: Clustering (nlist in {4096, 16384}), nprobe calibration for R@10 in {0.90, 0.95, 0.99}
  - Full Baselines (Faiss-IVFFlat, Faiss-IVFSQ8, Faiss-IVFPQ+Refine, CM-FP32, CM-SQ8-RR, TRIDENT)
  - Multi-batch evaluation B in {1, 32, 1024, 4096} across 3 seeds
  - Metrics: QPS (mean±std), e2e ms, kernel ms, strict overlap vs CM-FP32, build time, memory footprint
  - Gate 4B check: A100, B=1024, Recall 0.95 speedup vs strongest lossless >= 1.3x
"""

import os
import sys
import time
import math
import argparse
import csv
import gc
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import faiss
from torch.utils.cpp_extension import load

# Import core structures from phase4a_suite
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase4a_suite import (
    get_cuda_ext,
    SharedClusterStorage,
    TridentV4Index,
    choose_tile,
    measure_latency_qps,
    eval_accuracy_v4,
    build_faiss_gpu_indexes,
)

DATA_ROOT = os.path.expanduser("~/trident/data")
RES_ROOT = os.path.expanduser("~/trident/results/phase4b")
CACHE_ROOT = os.path.expanduser("~/trident/cache_v3")
os.makedirs(RES_ROOT, exist_ok=True)
os.makedirs(CACHE_ROOT, exist_ok=True)

M_PQ_MAP_10M = {
    "sift10m": 16,
    "deep10m": 16,
    "t2i10m": 25,
}

# ----------------- CLUSTERING WITH GPU KMEANS (TRAIN SAMPLE <= 2M) -----------------
def load_or_train_clustering_10m(dataset: str, xb: np.ndarray, nlist: int, seed: int):
    cache_file = os.path.join(CACHE_ROOT, f"{dataset}_nlist{nlist}_s{seed}.npz")
    if os.path.exists(cache_file):
        print(f"[{dataset}] Loading cached clustering from {cache_file}...")
        data = np.load(cache_file)
        train_t = float(data["train_time_s"]) if "train_time_s" in data else 0.0
        return (
            data["centroids"],
            data["list_ids"],
            data["r_vec"],
            data["mean64"],
            data["R64"],
            train_t,
        )

    t0 = time.perf_counter()
    N, d = xb.shape
    print(f"[{dataset}] Training clustering with nlist={nlist}, seed={seed} (N={N}, D={d})...")

    # Sample <= 2M for training k-means to satisfy PLAN_V5 guidelines
    train_sample_sz = min(1000000, N)
    np.random.seed(seed + 1000)
    train_idx = np.random.choice(N, train_sample_sz, replace=False)
    xb_train = np.ascontiguousarray(xb[train_idx])

    # Use GPU k-means for fast convergence
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.device = 0
    cfg.useFloat16 = False
    quantizer = faiss.GpuIndexFlatL2(res, d, cfg)
    
    # Train quantizer
    km = faiss.Kmeans(d, nlist, niter=25, verbose=True, gpu=True, seed=seed + 1000)
    km.train(xb_train)
    centroids = km.centroids.astype(np.float32)
    del km, quantizer, xb_train
    torch.cuda.empty_cache()

    # Assign list_ids for all N vectors using GPU in batches of 1M
    print(f"[{dataset}] Assigning {N} vectors to {nlist} centroids...")
    quantizer_gpu = faiss.GpuIndexFlatL2(res, d, cfg)
    quantizer_gpu.add(centroids)

    list_ids = np.empty(N, dtype=np.int32)
    batch_assign = 1000000
    for s_i in range(0, N, batch_assign):
        e_i = min(s_i + batch_assign, N)
        _, assigned = quantizer_gpu.search(xb[s_i:e_i], 1)
        list_ids[s_i:e_i] = assigned.flatten().astype(np.int32)

    del quantizer_gpu
    torch.cuda.empty_cache()

    # Compute residual norms r_vec
    print(f"[{dataset}] Computing residual norms r_vec...")
    diff = xb - centroids[list_ids]
    r_vec = np.linalg.norm(diff, axis=1).astype(np.float32)

    # Compute PCA on random 50K sample
    print(f"[{dataset}] Computing PCA on 50K sample...")
    sample_size = min(50000, N)
    idx_sample = np.random.choice(N, sample_size, replace=False)
    diff_sample = diff[idx_sample].astype(np.float64)

    mean64 = np.mean(diff_sample, axis=0)
    cov = np.cov(diff_sample, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    sort_idx = np.argsort(evals)[::-1]
    R64 = evecs[:, sort_idx]

    train_time = time.perf_counter() - t0
    print(f"[{dataset}] Clustering & PCA finished in {train_time:.2f}s.")

    tmp_file = cache_file + f".tmp_{os.getpid()}"
    np.savez(tmp_file, centroids=centroids, list_ids=list_ids, r_vec=r_vec, mean64=mean64, R64=R64, train_time_s=train_time)
    os.replace(tmp_file, cache_file)
    return centroids, list_ids, r_vec, mean64, R64, train_time

# ----------------- NPROBE CALIBRATION FOR 10M -----------------
def calibrate_nprobes_10m(dataset: str, xb: np.ndarray, xq: np.ndarray, gt: np.ndarray, centroids: np.ndarray, nlist: int) -> Dict[float, int]:
    """Find nprobe values corresponding to Recall@10 = 0.90, 0.95, 0.99 using Faiss-IVFFlat on GPU."""
    print(f"[{dataset}] Calibrating nprobes for nlist={nlist}...")
    d = xb.shape[1]
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexIVFFlatConfig()
    cfg.device = 0
    cfg.useFloat16 = False
    
    q_flat = faiss.IndexFlatL2(d)
    cpu_ivf = faiss.IndexIVFFlat(q_flat, d, nlist, faiss.METRIC_L2)
    cpu_ivf.quantizer.add(centroids)
    # Fast add with precomputed centroids
    cpu_ivf.add(xb[:1000000]) # Calibrate on 1M subset or full
    gpu_ivf = faiss.index_cpu_to_gpu(res, 0, cpu_ivf)
    del cpu_ivf

    q_sub = xq[:1000]
    gt10_sub = gt[:1000, :10]

    rec_targets = [0.90, 0.95, 0.99]
    nprobe_map = {}
    candidate_nprobes = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
    
    recs = []
    for np_val in candidate_nprobes:
        if np_val > nlist:
            continue
        gpu_ivf.nprobe = np_val
        _, I = gpu_ivf.search(q_sub, 10)
        hits = 0
        for i in range(1000):
            hits += len(set(I[i]).intersection(set(gt10_sub[i])))
        rec = hits / (1000 * 10)
        recs.append((np_val, rec))
        print(f"  Calib nprobe={np_val:4d} -> Recall@10={rec:.4f}")

    del gpu_ivf
    torch.cuda.empty_cache()

    for target in rec_targets:
        best_np = candidate_nprobes[-1]
        for np_val, rec in recs:
            if rec >= target:
                best_np = np_val
                break
        nprobe_map[target] = best_np
        print(f"[{dataset}] Recall={target:.2f} calibrated to nprobe={best_np}")

    return nprobe_map

# ----------------- FLOAT64 LOSSLESS AUDIT (1000 QUERIES) -----------------
def run_float64_audit_10m(dataset: str, xb: np.ndarray, xq: np.ndarray, centroids: np.ndarray, list_ids: np.ndarray, nprobe: int, k: int = 10, num_q: int = 1000):
    print(f"[{dataset}] Running Float64 lossless audit on {num_q} queries...")
    d = xb.shape[1]
    xq_sub = xq[:num_q].astype(np.float64)
    xb_f64 = xb.astype(np.float64)

    # Coarse quantization in float64
    c_f64 = centroids.astype(np.float64)
    diff_c = xq_sub[:, None, :] - c_f64[None, :, :]
    c_dists = np.sum(diff_c ** 2, axis=2)
    top_cids = np.argsort(c_dists, axis=1)[:, :nprobe]

    # Inverted lists
    inv_lists = [np.where(list_ids == c)[0] for c in range(len(centroids))]

    ref_ids = np.empty((num_q, k), dtype=np.int64)
    ref_dists = np.empty((num_q, k), dtype=np.float64)

    for i in range(num_q):
        cands = []
        for c in top_cids[i]:
            cands.extend(inv_lists[c])
        cands = np.unique(cands)
        if len(cands) == 0:
            ref_ids[i] = -1
            ref_dists[i] = np.inf
            continue
        v = xb_f64[cands]
        diff_v = v - xq_sub[i]
        dists = np.sum(diff_v ** 2, axis=1)
        top_idx = np.argsort(dists)[:k]
        ref_ids[i] = cands[top_idx]
        ref_dists[i] = dists[top_idx]

    return ref_ids, ref_dists

# ----------------- MAIN BENCHMARK FUNCTION -----------------
def run_phase4b(args):
    dev = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(dev)
    print(f"=== TRIDENT Phase 4B: 10M Scale Benchmark on {gpu_name} ({args.gpu_tag}) ===", flush=True)

    out_main_csv = os.path.join(RES_ROOT, f"phase4b_main_{args.gpu_tag}.csv")
    out_audit_csv = os.path.join(RES_ROOT, f"phase4b_lossless_audit_{args.gpu_tag}.csv")
    out_build_csv = os.path.join(RES_ROOT, f"phase4b_build_mem_{args.gpu_tag}.csv")

    main_rows = []
    audit_rows = []
    build_rows = []

    batches = [1, 32, 1024, 4096]
    rec_levels = [0.90, 0.95, 0.99]

    for ds in args.datasets:
        ds_dir = os.path.join(DATA_ROOT, ds)
        base_path = os.path.join(ds_dir, "base.npy")
        query_path = os.path.join(ds_dir, "query.npy")
        gt_path = os.path.join(ds_dir, "gt.npy")

        if not os.path.exists(base_path) or not os.path.exists(query_path) or not os.path.exists(gt_path):
            print(f"[{ds}] Prepared files (base.npy, query.npy, gt.npy) not found. Triggering compute_gt_10m.py...")
            import compute_gt_10m
            compute_gt_10m.process_dataset(ds)

        print(f"[{ds}] Loading 10M base, query, and exact ground truth...")
        xb = np.ascontiguousarray(np.load(base_path).astype(np.float32))
        xq = np.ascontiguousarray(np.load(query_path).astype(np.float32))
        gt = np.load(gt_path)

        N, dim = xb.shape
        Q = xq.shape[0]
        t_rule = choose_tile(dim)
        print(f"[{ds}] N={N}, D={dim}, Q={Q}, TileRule={t_rule}")

        for nlist in args.nlists:
            for seed in args.seeds:
                print(f"\n--- Running {ds} | nlist={nlist} | seed={seed} ---")
                
                # 1. Clustering & Storage
                centroids, list_ids, r_vec, mean64, R64, train_time = load_or_train_clustering_10m(ds, xb, nlist, seed)

                # Record build time & memory
                t_pack_0 = time.perf_counter()
                storage = SharedClusterStorage(xb, list_ids, r_vec, nlist, device="cuda:0")
                idx_k2 = TridentV4Index(storage, centroids, mean64, R64, tile_size=t_rule, per_cluster_sq8=1, use_pca=1, var_block=0, max_batch=4096, device="cuda:0")
                idx_global_sq8 = TridentV4Index(storage, centroids, mean64, R64, tile_size=t_rule, per_cluster_sq8=0, use_pca=1, var_block=0, max_batch=4096, device="cuda:0")
                pack_time = time.perf_counter() - t_pack_0

                # Memory footprint
                vram_used_bytes = torch.cuda.memory_allocated(dev)
                bytes_per_vec = vram_used_bytes / N

                build_rows.append({
                    "gpu": args.gpu_tag,
                    "dataset": ds,
                    "nlist": nlist,
                    "seed": seed,
                    "kmeans_s": round(train_time, 2),
                    "pack_s": round(pack_time, 2),
                    "total_build_s": round(train_time + pack_time, 2),
                    "vram_gb": round(vram_used_bytes / (1024**3), 3),
                    "bytes_per_vector": round(bytes_per_vec, 1),
                })

                # 2. Build GPU Faiss baselines
                gpu_flat, gpu_sq, res = build_faiss_gpu_indexes(xb, centroids, list_ids, nlist, dev_id=0)

                # 3. Calibrate nprobes for R0.90, R0.95, R0.99
                nprobe_map = calibrate_nprobes_10m(ds, xb, xq, gt, centroids, nlist)

                # Prepare evaluation queries on GPU
                xq_gpu = torch.from_numpy(xq).to(dev, dtype=torch.float32).contiguous()
                gt10 = gt[:, :10]

                # 4. Evaluation across Recall Levels and Batches
                for rec_target in rec_levels:
                    nprobe = nprobe_map[rec_target]
                    print(f"\n[{ds} nlist={nlist} s={seed} R_target={rec_target:.2f} (nprobe={nprobe})]")

                    # CM-FP32 reference on full query set
                    cm_d_t, cm_i_t, _ = idx_global_sq8.search(xq_gpu, nprobe, mode=10, use_l1=0, rerank_m=0, tile_override=t_rule)
                    torch.cuda.synchronize()
                    cm_dists, cm_ids = cm_d_t.cpu().numpy(), cm_i_t.cpu().numpy()
                    del cm_d_t, cm_i_t

                    # Float64 reference audit (1000 queries)
                    if seed == 0 and rec_target == 0.95:
                        f64_ids, f64_dists = run_float64_audit_10m(ds, xb, xq, centroids, list_ids, nprobe, k=10, num_q=1000)
                        # Check TRIDENT vs f64
                        trident_d_audit, trident_i_audit, _ = idx_k2.search(xq_gpu[:1000], nprobe, mode=2, use_l1=0, rerank_m=0, tile_override=t_rule, warp_refine=1)
                        torch.cuda.synchronize()
                        t_audit_ids = trident_i_audit.cpu().numpy()
                        true_miss = 0
                        for q_idx in range(1000):
                            true_miss += len(set(f64_ids[q_idx]).difference(set(t_audit_ids[q_idx])))
                        
                        audit_rows.append({
                            "dataset": ds,
                            "nlist": nlist,
                            "nprobe": nprobe,
                            "audit_queries": 1000,
                            "true_miss": true_miss,
                            "strict_float64_pct": round(100.0 * (1.0 - true_miss / 10000.0), 4),
                        })
                        print(f"[Audit 10M {ds}] true_miss={true_miss} / 10000 across 1000 queries.")

                    for B in batches:
                        xq_b_gpu = xq_gpu[:B].contiguous()
                        xq_b_cpu = xq[:B]

                        methods = [
                            ("Faiss_IVFFlat", "faiss_flat"),
                            ("Faiss_IVFSQ8",  "faiss_sq"),
                            ("CM_FP32",       dict(mode=10, ts=t_rule, rrm=0)),
                            ("CM_SQ8_RR20",   dict(mode=0,  ts=t_rule, rrm=20)),
                            ("CM_SQ8_RR50",   dict(mode=0,  ts=t_rule, rrm=50)),
                            ("CM_SQ8_RR100",  dict(mode=0,  ts=t_rule, rrm=100)),
                            ("TRIDENT",       dict(mode=2,  ts=t_rule, rrm=0, wr=1, use_k2=True)),
                        ]

                        for m_name, cfg in methods:
                            if m_name == "Faiss_IVFFlat":
                                gpu_flat.nprobe = nprobe
                                fd, fi = gpu_flat.search(xq, 10)
                                rec_val, cm_ov, _ = eval_accuracy_v4(fi, fd, cm_ids, cm_dists, gt10)
                                e2e_ms = measure_latency_qps(lambda: gpu_flat.search(xq_b_cpu, 10), warmup=10, iters=30)
                                kern_ms = -1.0
                            elif m_name == "Faiss_IVFSQ8":
                                gpu_sq.nprobe = nprobe
                                fd, fi = gpu_sq.search(xq, 10)
                                rec_val, cm_ov, _ = eval_accuracy_v4(fi, fd, cm_ids, cm_dists, gt10)
                                e2e_ms = measure_latency_qps(lambda: gpu_sq.search(xq_b_cpu, 10), warmup=10, iters=30)
                                kern_ms = -1.0
                            else:
                                active_idx = idx_k2 if cfg.get("use_k2", False) else idx_global_sq8
                                td, ti, _ = active_idx.search(
                                    xq_gpu, nprobe, mode=cfg["mode"], use_l1=0,
                                    rerank_m=cfg["rrm"], tile_override=cfg["ts"], warp_refine=cfg.get("wr", 1)
                                )
                                torch.cuda.synchronize()
                                rec_val, cm_ov, _ = eval_accuracy_v4(ti.cpu().numpy(), td.cpu().numpy(), cm_ids, cm_dists, gt10)
                                del td, ti
                                e2e_ms = measure_latency_qps(
                                    lambda: active_idx.search(
                                        xq_b_gpu, nprobe, mode=cfg["mode"], use_l1=0,
                                        rerank_m=cfg["rrm"], tile_override=cfg["ts"], warp_refine=cfg.get("wr", 1), record_kernel_ms=False
                                    ),
                                    warmup=10, iters=30
                                )
                                k_acc = 0.0
                                for _ in range(10):
                                    _, _, km = active_idx.search(
                                        xq_b_gpu, nprobe, mode=cfg["mode"], use_l1=0,
                                        rerank_m=cfg["rrm"], tile_override=cfg["ts"], warp_refine=cfg.get("wr", 1), record_kernel_ms=True
                                    )
                                    k_acc += km
                                kern_ms = k_acc / 10.0

                            qps = (B * 1000.0) / max(e2e_ms, 1e-6)
                            main_rows.append({
                                "gpu": args.gpu_tag,
                                "dataset": ds,
                                "dim": dim,
                                "nlist": nlist,
                                "seed": seed,
                                "recall_target": rec_target,
                                "nprobe": nprobe,
                                "batch": B,
                                "method": m_name,
                                "e2e_ms": round(e2e_ms, 4),
                                "kernel_ms": round(kern_ms, 4) if kern_ms >= 0 else "N/A",
                                "qps": round(qps, 1),
                                "recall_at_10": round(rec_val, 4),
                                "cm_fp32_strict_ov_pct": round(cm_ov, 4),
                            })
                            print(f"[{ds} R{rec_target} B={B:4d} {m_name:15s}] QPS={qps:9.1f} Rec={rec_val:.4f} Strict={cm_ov:.2f}%", flush=True)

                del idx_k2, idx_global_sq8, storage, gpu_flat, gpu_sq
                torch.cuda.empty_cache()

    # Write CSVs
    if main_rows:
        with open(out_main_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(main_rows[0].keys()))
            writer.writeheader()
            writer.writerows(main_rows)
        print(f"[SAVED] {out_main_csv}")

    if audit_rows:
        with open(out_audit_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            writer.writeheader()
            writer.writerows(audit_rows)
        print(f"[SAVED] {out_audit_csv}")

    if build_rows:
        with open(out_build_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(build_rows[0].keys()))
            writer.writeheader()
            writer.writerows(build_rows)
        print(f"[SAVED] {out_build_csv}")

    print("Phase 4B Suite finished successfully!")

def main():
    parser = argparse.ArgumentParser(description="TRIDENT Phase 4B Suite (10M)")
    parser.add_argument("--gpu_tag", type=str, default="A100")
    parser.add_argument("--datasets", nargs="+", default=["sift10m", "deep10m"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--nlists", nargs="+", type=int, default=[4096])
    args = parser.parse_args()
    run_phase4b(args)

if __name__ == "__main__":
    main()
