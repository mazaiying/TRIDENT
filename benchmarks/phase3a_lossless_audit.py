#!/usr/bin/env python3
"""
Phase 3A.3: Rigorous Float64 Lossless Audit across all 6 datasets.
Compares on the EXACT SAME IVF clustering and nprobe cluster lists:
  1. Ref-F64: Exact float64 L2 distance sum((x_f64 - q_f64)^2) over all vectors in the query's nprobe clusters.
  2. Ref-F32 (CM-FP32): Exact float32 direct L2 distance sum((x_f32 - q_f32)^2) over the same nprobe clusters.
  3. Faiss-GPU IVFFlat: Standard Faiss-GPU IndexIVFFlat (uses decomposed L2 ||x||^2 + ||q||^2 - 2<x,q>).
  4. TRIDENT-K2 (T=256 and T=64): Real GPU kernel with PCA + variable-block SQ8 early-stop + FP32 refinement.
Categorizes every single ID difference vs Ref-F64 into:
  - exact_tie: d_f64(q, pred_id) == d_f64(q, ref_id) (duplicate vectors or exact distance tie at rank 10)
  - f32_ulps:  |d_f64(q, pred_id) - d_f64(q, ref_id)| / d_f64 <= 1e-5 (IEEE-754 float32 vs float64 rounding at rank 10/11 boundary)
  - true_miss: |d_f64(q, pred_id) - d_f64(q, ref_id)| / d_f64 > 1e-5  (genuine false negative / missed neighbor)
"""
import os, sys, csv
import numpy as np
import torch
import faiss
import faiss.contrib.torch_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2b_benchmark import (
    DATA_ROOT, NPROBE_MAP_095, train_shared_ivf_centroids,
    build_faiss_gpu_indexes, compute_pca, TridentGPUIndex
)

RES_DIR = os.path.expanduser("~/trident/results/phase3a")
os.makedirs(RES_DIR, exist_ok=True)


def compute_ref_f64_and_f32_direct(xb: np.ndarray, xq: np.ndarray, list_ids: np.ndarray, probe_cids: np.ndarray, dev: torch.device):
    """
    For each query q in 0..Q-1, gathers all base vectors in probe_cids[q, :nprobe],
    computes exact float64 L2 and exact float32 direct L2 on GPU, and returns top-10 IDs & float64 distances.
    """
    Q, nprobe = probe_cids.shape
    N, D = xb.shape
    nlist = int(list_ids.max()) + 1
    order = np.argsort(list_ids, kind="mergesort")
    counts = np.bincount(list_ids[order], minlength=nlist)
    offsets = np.zeros(nlist + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    xb_g = torch.from_numpy(xb).to(dev)
    xq_g = torch.from_numpy(xq).to(dev)

    ref_f64_ids = np.empty((Q, 10), dtype=np.int64)
    ref_f64_dists = np.empty((Q, 10), dtype=np.float64)
    ref_f32_ids = np.empty((Q, 10), dtype=np.int64)
    ref_f32_dists = np.empty((Q, 10), dtype=np.float64)

    for q in range(Q):
        cids = probe_cids[q]
        segs = [order[offsets[c]:offsets[c + 1]] for c in cids if offsets[c + 1] > offsets[c]]
        cand_ids = np.concatenate(segs).astype(np.int64)
        cand_ids_g = torch.from_numpy(cand_ids).to(dev)
        cand_x = xb_g[cand_ids_g]  # [M, D] float32

        # Exact float64 L2
        diff64 = cand_x.to(torch.float64) - xq_g[q].to(torch.float64).unsqueeze(0)
        d64 = (diff64 * diff64).sum(dim=1).cpu().numpy()
        # Sort by (d64, cand_ids) ascending
        idx64 = np.lexsort((cand_ids, d64))[:10]
        ref_f64_ids[q] = cand_ids[idx64]
        ref_f64_dists[q] = d64[idx64]

        # Exact float32 direct L2
        diff32 = cand_x - xq_g[q].unsqueeze(0)
        d32 = (diff32 * diff32).sum(dim=1).cpu().numpy()
        idx32 = np.lexsort((cand_ids, d32))[:10]
        ref_f32_ids[q] = cand_ids[idx32]
        ref_f32_dists[q] = d64[idx32]  # evaluate true float64 distance of chosen IDs

    return ref_f64_ids, ref_f64_dists, ref_f32_ids, ref_f32_dists


def audit_predictions(name: str, pred_ids: np.ndarray, ref_ids: np.ndarray, ref_d64: np.ndarray,
                      xb: np.ndarray, xq: np.ndarray):
    Q = len(ref_ids)
    total_slots = Q * 10
    strict_id_hits = 0
    exact_ties = 0
    f32_ulps = 0
    true_misses = 0
    max_rel_err = 0.0
    queries_strict_10 = 0
    queries_no_miss = 0

    for q in range(Q):
        s_pred = set(pred_ids[q].tolist())
        s_ref = set(ref_ids[q].tolist())
        ov = len(s_pred.intersection(s_ref))
        strict_id_hits += ov
        if ov == 10:
            queries_strict_10 += 1
            queries_no_miss += 1
            continue

        # Inspect the missing/extra IDs in float64
        missing_ref = sorted(list(s_ref - s_pred), key=lambda i: float(np.sum((xb[i].astype(np.float64) - xq[q].astype(np.float64)) ** 2)))
        extra_pred = sorted(list(s_pred - s_ref), key=lambda i: float(np.sum((xb[i].astype(np.float64) - xq[q].astype(np.float64)) ** 2)))
        q_has_true_miss = False
        for r_id, p_id in zip(missing_ref, extra_pred):
            dr = float(np.sum((xb[r_id].astype(np.float64) - xq[q].astype(np.float64)) ** 2))
            dp = float(np.sum((xb[p_id].astype(np.float64) - xq[q].astype(np.float64)) ** 2))
            rel = abs(dp - dr) / max(dr, 1e-12)
            if rel > max_rel_err:
                max_rel_err = rel
            if abs(dp - dr) <= 1e-12 * max(1.0, dr):
                exact_ties += 1
            elif rel <= 1e-5:
                f32_ulps += 1
            else:
                true_misses += 1
                q_has_true_miss = True
        if not q_has_true_miss:
            queries_no_miss += 1

    return {
        "method": name,
        "total_slots": total_slots,
        "strict_id_overlap_pct": round(strict_id_hits / total_slots * 100.0, 4),
        "queries_strict_100_pct": round(queries_strict_10 / Q * 100.0, 2),
        "diff_slots_total": total_slots - strict_id_hits,
        "diff_exact_tie": exact_ties,
        "diff_f32_rounding_le_1e5": f32_ulps,
        "diff_true_miss_gt_1e5": true_misses,
        "lossless_no_miss_pct": round(queries_no_miss / Q * 100.0, 4),
        "max_rel_dist_diff": f"{max_rel_err:.2e}",
    }


def main():
    dev = torch.device("cuda:0")
    datasets = ["gist1m", "bge1024", "sift1m", "glove100", "msong", "nuswide"]
    rows = []
    for ds in datasets:
        ds_dir = os.path.join(DATA_ROOT, ds)
        xb = np.ascontiguousarray(np.load(os.path.join(ds_dir, "base.npy")).astype(np.float32))
        xq_raw = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy")).astype(np.float32))
        Q = min(len(xq_raw), 1000)
        xq = xq_raw[:Q]
        xq_gpu = torch.from_numpy(xq).to(dev)
        nprobe = NPROBE_MAP_095[ds]

        centroids = train_shared_ivf_centroids(xb, nlist=1024, seed=0)
        res_f, gpu_ivfflat, gpu_ivfsq8, list_ids, r_vec = build_faiss_gpu_indexes(xb, centroids, 1024)
        gpu_ivfflat.nprobe = nprobe

        # Also check coarse quantizer probe_cids from Faiss quantizer vs torch.addmm
        quant_cpu = faiss.IndexFlatL2(xb.shape[1])
        quant_cpu.add(centroids)
        _, faiss_probe_cids = quant_cpu.search(xq, nprobe)

        mean64, R64 = compute_pca(xb, seed=0)
        tri_t256 = TridentGPUIndex(xb, centroids, list_ids, r_vec, mean64, R64, tile_size=256, max_batch=1024, device="cuda:0")
        tri_t64  = TridentGPUIndex(xb, centroids, list_ids, r_vec, mean64, R64, tile_size=64,  max_batch=1024, device="cuda:0", shared_xb_gpu=tri_t256.xb_orig)

        # Compute exact coarse probe_cids as used by TridentGPUIndex
        q_norm_sq = (xq_gpu ** 2).sum(dim=1, keepdim=True)
        cdists_sq = torch.addmm(tri_t256.c_norm_sq.unsqueeze(0), xq_gpu, tri_t256.centroids_gpu.t(), beta=1.0, alpha=-2.0)
        cdists_sq.add_(q_norm_sq).clamp_(min=0.0)
        _, top_c_ids = torch.topk(cdists_sq, k=nprobe, dim=1, largest=False, sorted=True)
        tri_probe_cids = top_c_ids.cpu().numpy()

        c_diff_queries = sum(1 for q in range(Q) if set(faiss_probe_cids[q]) != set(tri_probe_cids[q]))

        ref64_ids, ref64_d, ref32_ids, ref32_d = compute_ref_f64_and_f32_direct(xb, xq, list_ids, tri_probe_cids, dev)

        _, f_ids_t = gpu_ivfflat.search(xq_gpu, 10)
        f_ids = f_ids_t.cpu().numpy()

        _, k2_256_i, _ = tri_t256.search(xq_gpu, nprobe, mode=2, use_l1=1)
        k2_256_ids = k2_256_i.cpu().numpy()

        _, k2_64_i, _ = tri_t64.search(xq_gpu, nprobe, mode=2, use_l1=1)
        k2_64_ids = k2_64_i.cpu().numpy()

        print(f"\n=== Dataset: {ds} (Q={Q}, nprobe={nprobe}, coarse_probe_diff_queries={c_diff_queries}) ===", flush=True)
        for m_name, pids in [
            ("Faiss_IVFFlat_vs_RefF64", f_ids),
            ("CM_FP32_Direct_vs_RefF64", ref32_ids),
            ("TRIDENT_K2_T256_vs_RefF64", k2_256_ids),
            ("TRIDENT_K2_T64_vs_RefF64", k2_64_ids),
            ("TRIDENT_K2_T256_vs_CM_FP32", k2_256_ids),
        ]:
            target_ref_ids = ref32_ids if m_name.endswith("CM_FP32") else ref64_ids
            r = audit_predictions(m_name, pids, target_ref_ids, ref64_d, xb, xq)
            r["dataset"] = ds
            r["coarse_diff_queries"] = c_diff_queries
            rows.append(r)
            print(f"  {m_name:<28} | StrictOv={r['strict_id_overlap_pct']:>7.4f}% | "
                  f"Diffs={r['diff_slots_total']:>3} (Tie={r['diff_exact_tie']}, F32_ULP={r['diff_f32_rounding_le_1e5']}, TrueMiss={r['diff_true_miss_gt_1e5']}) | "
                  f"MaxRelErr={r['max_rel_dist_diff']}", flush=True)

        del tri_t256, tri_t64, gpu_ivfflat, gpu_ivfsq8, res_f
        torch.cuda.empty_cache()

    out_csv = os.path.join(RES_DIR, "lossless_audit.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSAVED {out_csv}", flush=True)


if __name__ == "__main__":
    main()
