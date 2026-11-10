#!/usr/bin/env python3
"""
Phase 2A: Audited GPU-Vectorized Pruning & Consistency Analysis for TRIDENT (PLAN_V3 §3).
- 6 datasets (sift1m, nuswide, glove100, msong, gist1m, bge1024) x seeds {0, 1, 2}
- All queries up to 1000 (1000 for 5 datasets, 200 for nuswide)
- nlist = 1024, Recall@10 in {0.95, 0.99}, tile_size in {256, 64}
- 4 configs: orig_b32, var_b32, pca_b32, pca_varblk (16, 32, 32, ...) x L1 in {0, 1}
- Re-clusters after orthogonal PCA rotation; recalibrates nprobe with exact FP32 IVF-Flat top-10 verification.
- Uses conservative SQ8 coordinate error bound (including float64->float32 rotation error slack)
  and verifies per-query top-10 consistency against exact FP32 IVF-Flat (must be 100.00%).
"""
import os, sys, time, math, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.cpp_extension import load
import faiss

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13")
os.environ["PATH"] = "/usr/local/cuda-13/bin:" + os.environ.get("PATH", "")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.expanduser("~/trident/data")
RES_ROOT = os.path.expanduser("~/trident/results/phase2a")
os.makedirs(RES_ROOT, exist_ok=True)

print("[JIT] Compiling phase2a_cuda.cu ...", flush=True)
p2a_ext = load(
    name="phase2a_cuda_ext",
    sources=[os.path.join(SCRIPT_DIR, "phase2a_cuda.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)
print("[JIT] phase2a_cuda_ext compiled successfully.", flush=True)


def train_ivf_and_sort(xb_space: np.ndarray, nlist: int, seed: int):
    N, D = xb_space.shape
    clus = faiss.Clustering(D, nlist)
    clus.niter = 20
    clus.seed = seed
    clus.max_points_per_centroid = 256
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.device = 0
    flat_index = faiss.GpuIndexFlatL2(res, D, cfg)
    clus.train(xb_space, flat_index)
    centroids = faiss.vector_float_to_array(clus.centroids).reshape(nlist, D).copy()
    dists_sq, list_ids = flat_index.search(xb_space, 1)
    list_ids = list_ids.reshape(-1)
    r_vec = np.sqrt(np.maximum(dists_sq.reshape(-1), 0.0)).astype(np.float32)

    # Sort within each cluster by d(x, c) ascending
    order_global = np.lexsort((r_vec, list_ids)).astype(np.int64)
    sorted_cids = list_ids[order_global]
    counts = np.bincount(sorted_cids, minlength=nlist)
    offsets = np.zeros(nlist + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    r_sorted = r_vec[order_global]
    return centroids, order_global, offsets, r_sorted


def compute_pca_rotation(xb: np.ndarray, seed: int, sample_n: int = 100000):
    N, D = xb.shape
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, size=min(N, sample_n), replace=False) if N > sample_n else np.arange(N)
    sub = xb[idx].astype(np.float64)
    mean64 = sub.mean(axis=0)
    centered = sub - mean64
    cov = (centered.T @ centered) / max(len(sub) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    R64 = eigvecs[:, ::-1].copy()
    ortho_err = np.max(np.abs(R64.T @ R64 - np.eye(D)))
    assert ortho_err < 1e-10, f"PCA orthogonality error too high: {ortho_err}"
    return mean64, R64


def transform_and_quantize(
    xb_orig_gpu: torch.Tensor,
    xq_orig_gpu: torch.Tensor,
    order_gpu: torch.Tensor,
    mode: str,
    perm_var_gpu: torch.Tensor = None,
    mean64_gpu: torch.Tensor = None,
    R64_gpu: torch.Tensor = None,
):
    """
    Returns (sq8_sorted_gpu [N, D] uint8, xq_rot_gpu [Q, D] float32,
             vmin_gpu [D] float32, delta_gpu [D] float32, eps_dim_gpu [D] float32,
             first64_var_pct float)
    """
    N, D = xb_orig_gpu.shape
    dev = xb_orig_gpu.device

    # Compute variance and SQ8 quantization in chunks on GPU to keep memory low and speed high
    vmin = torch.full((D,), float("inf"), device=dev, dtype=torch.float32)
    vmax = torch.full((D,), float("-inf"), device=dev, dtype=torch.float32)
    sum_x = torch.zeros((D,), device=dev, dtype=torch.float64)
    sum_x2 = torch.zeros((D,), device=dev, dtype=torch.float64)

    chunk = 200000
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        blk = xb_orig_gpu[order_gpu[s:e]]
        if mode == "orig":
            blk_f64 = blk.to(torch.float64)
        elif mode == "var":
            blk_f64 = blk[:, perm_var_gpu].to(torch.float64)
        elif mode == "pca":
            blk_f64 = torch.matmul(blk.to(torch.float64) - mean64_gpu, R64_gpu)
        else:
            raise ValueError(mode)
        blk_f32 = blk_f64.to(torch.float32)
        vmin = torch.minimum(vmin, blk_f32.min(dim=0).values)
        vmax = torch.maximum(vmax, blk_f32.max(dim=0).values)
        sum_x += blk_f64.sum(dim=0)
        sum_x2 += (blk_f64 ** 2).sum(dim=0)

    var_dim = torch.clamp(sum_x2 / N - (sum_x / N) ** 2, min=0.0)
    k_var = min(64, D)
    first64_var_pct = float((var_dim[:k_var].sum() / torch.clamp(var_dim.sum(), min=1e-12)).item() * 100.0)

    delta = (vmax - vmin) / 255.0
    delta = torch.where(delta > 0.0, delta, torch.ones_like(delta))

    # Transform queries in float64 first to bound query float64->float32 rounding error!
    if mode == "orig":
        xq_f64 = xq_orig_gpu.to(torch.float64)
    elif mode == "var":
        xq_f64 = xq_orig_gpu[:, perm_var_gpu].to(torch.float64)
    elif mode == "pca":
        xq_f64 = torch.matmul(xq_orig_gpu.to(torch.float64) - mean64_gpu, R64_gpu)
    xq_rot_f32 = xq_f64.to(torch.float32)
    q_round_err = (torch.abs(xq_f64 - xq_rot_f32.to(torch.float64)).max(dim=0).values).to(torch.float32)

    sq8_sorted = torch.empty((N, D), device=dev, dtype=torch.uint8)
    max_quant_err = torch.zeros((D,), device=dev, dtype=torch.float32)

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        blk = xb_orig_gpu[order_gpu[s:e]]
        if mode == "orig":
            blk_f64 = blk.to(torch.float64)
        elif mode == "var":
            blk_f64 = blk[:, perm_var_gpu].to(torch.float64)
        elif mode == "pca":
            blk_f64 = torch.matmul(blk.to(torch.float64) - mean64_gpu, R64_gpu)
        blk_f32 = blk_f64.to(torch.float32)
        codes = torch.clamp(torch.round((blk_f32 - vmin) / delta), 0.0, 255.0).to(torch.uint8)
        sq8_sorted[s:e] = codes
        # Reconstruct in exact same float32 expression as CUDA kernel: vmin + float(code) * delta
        x_hat_f32 = vmin + codes.to(torch.float32) * delta
        err = torch.abs(blk_f64 - x_hat_f32.to(torch.float64)).max(dim=0).values.to(torch.float32)
        max_quant_err = torch.maximum(max_quant_err, err)

    # Conservative coordinate error bound covering SQ8 quantization + f64->f32 rounding + 1e-6 safety
    eps_dim = (max_quant_err + q_round_err) * (1.0 + 1e-5) + 1e-6
    return sq8_sorted, xq_rot_f32.contiguous(), vmin.contiguous(), delta.contiguous(), eps_dim.contiguous(), first64_var_pct


def make_block_bounds(D: int, varblk: bool, device):
    if not varblk:
        bounds = list(range(0, D, 32)) + [D]
    else:
        # Strict refinement of 32-dim blocks: (16, 16, 32, 32, ...) -> [0, 16, 32, 64, 96, ..., D]
        bounds = [0]
        for b in (16, 32):
            if b < D:
                bounds.append(b)
        cur = 32
        while cur + 32 < D:
            cur += 32
            bounds.append(cur)
        if bounds[-1] != D:
            bounds.append(D)
    return torch.tensor(bounds, dtype=torch.int32, device=device)


def calibrate_nprobes_exact(
    xb_orig_gpu: torch.Tensor,
    xq_orig_gpu: torch.Tensor,
    gt10_np: np.ndarray,
    centroids_space: np.ndarray,
    xq_space: np.ndarray,
    order_gpu: torch.Tensor,
    offsets_gpu: torch.Tensor,
    r_sorted_gpu: torch.Tensor,
    target_recalls=(0.95, 0.99),
):
    """
    Finds smallest nprobe in grid reaching target_recalls using EXACT FP32 IVF-Flat top-10!
    Also returns the exact IVF-Flat top-10 IDs & dists for each selected nprobe.
    """
    dev = xb_orig_gpu.device
    Q, D = xq_orig_gpu.shape
    nlist = centroids_space.shape[0]

    xq_sp_t = torch.from_numpy(xq_space).to(dev, dtype=torch.float32)
    c_sp_t = torch.from_numpy(centroids_space).to(dev, dtype=torch.float32)
    cdists = torch.cdist(xq_sp_t, c_sp_t, p=2.0)  # Euclidean distance [Q, nlist]
    cdists_sorted, c_order = torch.sort(cdists, dim=1, descending=False)
    c_order_i32 = c_order.to(torch.int32).contiguous()
    cdists_sorted_f32 = cdists_sorted.to(torch.float32).contiguous()

    # Fast rank pre-estimate using gt10 cluster membership, then exact FP32 IVF-Flat verification!
    order_np = order_gpu.cpu().numpy()
    offsets_np = offsets_gpu.cpu().numpy()
    vec_cid = np.empty(xb_orig_gpu.shape[0], dtype=np.int32)
    for c in range(nlist):
        vec_cid[order_np[offsets_np[c]:offsets_np[c+1]]] = c
    c_order_np = c_order_i32.cpu().numpy()
    gt_cids = vec_cid[gt10_np]
    ranks = np.empty_like(gt_cids)
    for q in range(Q):
        inv = np.empty(nlist, dtype=np.int32)
        inv[c_order_np[q]] = np.arange(nlist, dtype=np.int32)
        ranks[q] = inv[gt_cids[q]]

    grid = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64,
            80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, nlist]
    grid = sorted(list(set([p for p in grid if p <= nlist])))

    dummy_sq8 = torch.zeros((1, 1), device=dev, dtype=torch.uint8)
    dummy_f32 = torch.zeros((D,), device=dev, dtype=torch.float32)
    dummy_bounds = torch.tensor([0, D], dtype=torch.int32, device=dev)

    def eval_exact_ivf(p):
        co = c_order_i32[:, :p].contiguous()
        cd = cdists_sorted_f32[:, :p].contiguous()
        top_ids, top_dists, _, _, _ = p2a_ext.run_phase2a_sim(
            xb_orig_gpu, xq_orig_gpu, order_gpu, r_sorted_gpu,
            dummy_sq8, xq_orig_gpu, dummy_f32, dummy_f32, dummy_f32,
            offsets_gpu, co, cd, dummy_bounds, 256, 0, 1
        )
        torch.cuda.synchronize()
        ids_np = top_ids.cpu().numpy()
        dists_np = top_dists.cpu().numpy()
        # Recall@10 vs brute-force gt10
        hits = 0
        for q in range(Q):
            hits += len(set(ids_np[q]).intersection(set(gt10_np[q])))
        rec = hits / (Q * 10.0)
        return rec, ids_np, dists_np, co, cd

    selected = {}
    cache = {}
    for tr in target_recalls:
        # Start from rank-estimated candidate to minimize exact evaluations
        start_idx = 0
        for idx_p, p in enumerate(grid):
            est_rec = float((ranks < p).sum() / (Q * 10.0))
            if est_rec >= tr - 0.01:
                start_idx = max(0, idx_p - 1)
                break
        chosen = None
        for idx_p in range(start_idx, len(grid)):
            p = grid[idx_p]
            if p not in cache:
                cache[p] = eval_exact_ivf(p)
            rec, ids_np, dists_np, co, cd = cache[p]
            if rec >= tr or p == nlist:
                chosen = (p, rec, ids_np, dists_np, co, cd)
                break
        selected[tr] = chosen
    return selected


def check_consistency(tri_ids: np.ndarray, tri_dists: np.ndarray, ref_ids: np.ndarray, ref_dists: np.ndarray):
    """
    Per-query top-10 consistency vs Exact FP32 IVF-Flat.
    Returns:
      - strict_set_overlap_pct: mean |S_tri ∩ S_ref| / 10 * 100
      - query_100_match_pct: % of queries where top-10 matches 10/10 (allowing exact float32 ties at 10th neighbor)
    """
    Q = tri_ids.shape[0]
    total_overlap = 0
    perfect_queries = 0
    for q in range(Q):
        s_tri = set(tri_ids[q].tolist())
        s_ref = set(ref_ids[q].tolist())
        ov = len(s_tri.intersection(s_ref))
        total_overlap += ov
        if ov == 10:
            perfect_queries += 1
        else:
            # Check if any difference at boundary is a true float32 distance tie (within 2e-5 rel)
            d10_tri = float(tri_dists[q, 9])
            d10_ref = float(ref_dists[q, 9])
            if abs(d10_tri - d10_ref) <= 2e-5 * max(1.0, abs(d10_ref)):
                perfect_queries += 1
    return (total_overlap / (Q * 10.0)) * 100.0, (perfect_queries / float(Q)) * 100.0


def run_dataset(ds_name: str, seeds=(0, 1, 2), nlist=1024, max_queries=1000, device="cuda:0"):
    ds_dir = os.path.join(DATA_ROOT, ds_name)
    xb = np.load(os.path.join(ds_dir, "base.npy")).astype(np.float32)
    xq_all = np.load(os.path.join(ds_dir, "query.npy")).astype(np.float32)
    gt_all = np.load(os.path.join(ds_dir, "gt.npy"))

    N, D = xb.shape
    Q = min(len(xq_all), max_queries)
    xq = np.ascontiguousarray(xq_all[:Q])
    gt10 = gt_all[:Q, :10]

    dev = torch.device(device)
    xb_orig_gpu = torch.from_numpy(xb).to(dev, dtype=torch.float32).contiguous()
    xq_orig_gpu = torch.from_numpy(xq).to(dev, dtype=torch.float32).contiguous()

    rows = []
    curve_rows = []

    for seed in seeds:
        t_seed = time.time()
        # Variance order on deterministic sample
        rng = np.random.RandomState(seed)
        sub_idx = rng.choice(N, size=min(N, 100000), replace=False) if N > 100000 else np.arange(N)
        perm_var = np.argsort(-np.var(xb[sub_idx].astype(np.float64), axis=0)).astype(np.int64)
        perm_var_gpu = torch.from_numpy(perm_var).to(dev, dtype=torch.long)

        # 1) Unrotated IVF clustering (shared by orig_b32 and var_b32)
        c_orig, ord_orig, off_orig, r_orig = train_ivf_and_sort(xb, nlist=nlist, seed=seed)
        ord_orig_gpu = torch.from_numpy(ord_orig).to(dev, dtype=torch.int64)
        off_orig_gpu = torch.from_numpy(off_orig).to(dev, dtype=torch.int64)
        r_orig_gpu = torch.from_numpy(r_orig).to(dev, dtype=torch.float32)

        sel_orig = calibrate_nprobes_exact(
            xb_orig_gpu, xq_orig_gpu, gt10, c_orig, xq, ord_orig_gpu, off_orig_gpu, r_orig_gpu,
            target_recalls=(0.95, 0.99)
        )

        # 2) Orthogonal PCA rotation + RE-CLUSTERING in PCA space (shared by pca_b32 and pca_varblk)
        mean64, R64 = compute_pca_rotation(xb, seed=seed, sample_n=100000)
        mean64_gpu = torch.from_numpy(mean64).to(dev, dtype=torch.float64)
        R64_gpu = torch.from_numpy(R64).to(dev, dtype=torch.float64)

        # Materialize xb_pca (float32) on CPU for FAISS GPU k-means re-clustering
        xb_pca = np.empty_like(xb)
        chunk = 200000
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            blk_pca = torch.matmul(xb_orig_gpu[s:e].to(torch.float64) - mean64_gpu, R64_gpu).to(torch.float32)
            xb_pca[s:e] = blk_pca.cpu().numpy()
        xq_pca = torch.matmul(xq_orig_gpu.to(torch.float64) - mean64_gpu, R64_gpu).to(torch.float32).cpu().numpy()

        c_pca, ord_pca, off_pca, r_pca = train_ivf_and_sort(xb_pca, nlist=nlist, seed=seed)
        del xb_pca
        ord_pca_gpu = torch.from_numpy(ord_pca).to(dev, dtype=torch.int64)
        off_pca_gpu = torch.from_numpy(off_pca).to(dev, dtype=torch.int64)
        r_pca_gpu = torch.from_numpy(r_pca).to(dev, dtype=torch.float32)

        sel_pca = calibrate_nprobes_exact(
            xb_orig_gpu, xq_orig_gpu, gt10, c_pca, xq_pca, ord_pca_gpu, off_pca_gpu, r_pca_gpu,
            target_recalls=(0.95, 0.99)
        )

        # Precompute SQ8 representations for the 3 coordinate bases: orig, var, pca
        bases_data = {}
        bases_data["orig"] = (
            ord_orig_gpu, off_orig_gpu, r_orig_gpu, sel_orig,
            *transform_and_quantize(xb_orig_gpu, xq_orig_gpu, ord_orig_gpu, "orig")
        )
        bases_data["var"] = (
            ord_orig_gpu, off_orig_gpu, r_orig_gpu, sel_orig,
            *transform_and_quantize(xb_orig_gpu, xq_orig_gpu, ord_orig_gpu, "var", perm_var_gpu=perm_var_gpu)
        )
        bases_data["pca"] = (
            ord_pca_gpu, off_pca_gpu, r_pca_gpu, sel_pca,
            *transform_and_quantize(xb_orig_gpu, xq_orig_gpu, ord_pca_gpu, "pca", mean64_gpu=mean64_gpu, R64_gpu=R64_gpu)
        )

        configs = [
            ("orig_b32", "orig", False),
            ("var_b32", "var", False),
            ("pca_b32", "pca", False),
            ("pca_varblk", "pca", True),
        ]

        for cfg_name, basis_key, is_varblk in configs:
            ord_g, off_g, r_g, sel_map, sq8_g, xq_rot_g, vmin_g, delta_g, eps_g, var64_pct = bases_data[basis_key]
            blk_bounds_g = make_block_bounds(D, is_varblk, dev)
            num_blocks = int(blk_bounds_g.numel() - 1)

            for target_rec in (0.95, 0.99):
                nprobe, actual_rec, ref_ids, ref_dists, co_g, cd_g = sel_map[target_rec]
                for tile_size in (256, 64):
                    for l1_on in (0, 1):
                        t0 = time.time()
                        tri_ids_t, tri_dists_t, cnt_t, blk_tiles_t, blk_warps_t = p2a_ext.run_phase2a_sim(
                            xb_orig_gpu, xq_orig_gpu, ord_g, r_g,
                            sq8_g, xq_rot_g, vmin_g, delta_g, eps_g,
                            off_g, co_g, cd_g, blk_bounds_g,
                            int(tile_size), int(l1_on), 0
                        )
                        torch.cuda.synchronize()
                        dt_ms = (time.time() - t0) * 1000.0

                        tri_ids = tri_ids_t.cpu().numpy()
                        tri_dists = tri_dists_t.cpu().numpy()
                        strict_ov_pct, match100_pct = check_consistency(tri_ids, tri_dists, ref_ids, ref_dists)
                        if match100_pct < 100.0:
                            print(f"[WARN] Consistency < 100%: {ds_name} {cfg_name} seed={seed} rec={target_rec} "
                                  f"tile={tile_size} l1={l1_on} -> match={match100_pct:.2f}% ov={strict_ov_pct:.4f}%", flush=True)

                        cnt_sum = cnt_t.sum(dim=0).cpu().numpy()
                        total_vecs = int(cnt_sum[0])
                        l1_vec_kept = int(cnt_sum[1])
                        l1_tile_kept = int(cnt_sum[2])
                        vec_dims_comp = int(cnt_sum[3])
                        warp_dims_comp = int(cnt_sum[4])
                        tile_dims_comp = int(cnt_sum[5])
                        surv_fp32 = int(cnt_sum[6])
                        total_tiles_all = int(cnt_sum[7])
                        total_tiles_l1_kept = int(cnt_sum[8])
                        total_warps_l1_kept = int(cnt_sum[9])

                        total_dims_all = max(total_vecs * D, 1)
                        l1_vec_skip_pct = (1.0 - l1_vec_kept / max(total_vecs, 1)) * 100.0
                        l1_tile_skip_pct = (1.0 - l1_tile_kept / max(total_vecs, 1)) * 100.0

                        vec_saved_pct = (1.0 - vec_dims_comp / total_dims_all) * 100.0
                        warp_saved_pct = (1.0 - warp_dims_comp / total_dims_all) * 100.0
                        tile_saved_pct = (1.0 - tile_dims_comp / total_dims_all) * 100.0
                        surv_pct = (surv_fp32 / max(total_vecs, 1)) * 100.0

                        full_fp32_bytes = float(total_vecs * D * 4)
                        l1_meta_bytes = float(total_tiles_all * 8) if l1_on else 0.0
                        sq8_only_bytes_ratio = (total_vecs * D * 1.0 + surv_fp32 * D * 4.0) / full_fp32_bytes
                        tri_tile_bytes_ratio = (tile_dims_comp * 1.0 + surv_fp32 * D * 4.0 + l1_meta_bytes) / full_fp32_bytes
                        tri_warp_bytes_ratio = (warp_dims_comp * 1.0 + surv_fp32 * D * 4.0 + l1_meta_bytes) / full_fp32_bytes

                        tri_tile_vs_sq8 = tri_tile_bytes_ratio / max(sq8_only_bytes_ratio, 1e-9)
                        tri_warp_vs_sq8 = tri_warp_bytes_ratio / max(sq8_only_bytes_ratio, 1e-9)

                        row = {
                            "dataset": ds_name,
                            "dim": D,
                            "nlist": nlist,
                            "seed": seed,
                            "num_queries": Q,
                            "config": cfg_name,
                            "l1_on": l1_on,
                            "tile_size": tile_size,
                            "target_recall": target_rec,
                            "nprobe": nprobe,
                            "actual_recall": round(actual_rec, 4),
                            "ivf_match100_pct": round(match100_pct, 4),
                            "ivf_overlap_pct": round(strict_ov_pct, 4),
                            "first64_var_pct": round(var64_pct, 2),
                            "l1_vec_skip_pct": round(l1_vec_skip_pct, 2),
                            "l1_tile_skip_pct": round(l1_tile_skip_pct, 2),
                            "vec_saved_pct": round(vec_saved_pct, 2),
                            "warp_saved_pct": round(warp_saved_pct, 2),
                            "tile_saved_pct": round(tile_saved_pct, 2),
                            "surv_fp32_pct": round(surv_pct, 3),
                            "sq8_only_bytes": round(sq8_only_bytes_ratio, 4),
                            "tri_tile_bytes": round(tri_tile_bytes_ratio, 4),
                            "tri_warp_bytes": round(tri_warp_bytes_ratio, 4),
                            "tri_tile_vs_sq8": round(tri_tile_vs_sq8, 4),
                            "tri_warp_vs_sq8": round(tri_warp_vs_sq8, 4),
                            "sim_ms": round(dt_ms, 2),
                        }
                        rows.append(row)

                        # Record per-block active tile & warp curves for seed=0 (or all seeds)
                        bt_np = blk_tiles_t.cpu().numpy()
                        bw_np = blk_warps_t.cpu().numpy()
                        denom_tiles = max(total_tiles_all, 1)
                        denom_warps = max(total_warps_l1_kept if not l1_on else int(total_tiles_all * (tile_size // 32)), 1)
                        # Better exact warp denominator: when l1_on=0, total_warps_l1_kept == total warps across all tiles
                        for b_idx in range(num_blocks):
                            curve_rows.append({
                                "dataset": ds_name,
                                "seed": seed,
                                "config": cfg_name,
                                "l1_on": l1_on,
                                "tile_size": tile_size,
                                "target_recall": target_rec,
                                "block_idx": b_idx,
                                "dim_start": int(blk_bounds_g[b_idx].item()),
                                "dim_end": int(blk_bounds_g[b_idx + 1].item()),
                                "active_tile_ratio_all": round(float(bt_np[b_idx]) / max(total_tiles_all, 1), 4),
                                "active_tile_ratio_post_l1": round(float(bt_np[b_idx]) / max(total_tiles_l1_kept, 1), 4),
                                "active_warp_ratio_post_l1": round(float(bw_np[b_idx]) / max(total_warps_l1_kept, 1), 4),
                            })

                        if tile_size == 256 and l1_on == 1 and target_rec == 0.95:
                            print(
                                f"[{ds_name} s={seed} {cfg_name:<10} rec=0.95->({actual_rec:.4f}) p={nprobe:<3} T=256 L1=1] "
                                f"Match={match100_pct:.1f}% Var64={var64_pct:.1f}% L1_T={l1_tile_skip_pct:.1f}% "
                                f"Saved(Vec/Warp/Tile)={vec_saved_pct:.1f}%/{warp_saved_pct:.1f}%/{tile_saved_pct:.1f}% "
                                f"Surv={surv_pct:.2f}% TRI_T/SQ8={tri_tile_vs_sq8:.3f} ({dt_ms:.1f}ms)",
                                flush=True,
                            )
        print(f"--- Finished {ds_name} seed={seed} in {time.time() - t_seed:.1f}s ---", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(RES_ROOT, f"{ds_name}_phase2a.csv"), index=False)
    return rows, curve_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["sift1m", "nuswide", "glove100", "msong", "gist1m", "bge1024"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--max_queries", type=int, default=1000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    all_rows = []
    all_curves = []
    for ds in args.datasets:
        r, c = run_dataset(ds, seeds=args.seeds, nlist=args.nlist, max_queries=args.max_queries, device=args.device)
        all_rows.extend(r)
        all_curves.extend(c)

    pd.DataFrame(all_rows).to_csv(os.path.join(RES_ROOT, "all_phase2a_raw.csv"), index=False)
    pd.DataFrame(all_curves).to_csv(os.path.join(RES_ROOT, "active_tile_curves.csv"), index=False)
    print("PHASE2A_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
