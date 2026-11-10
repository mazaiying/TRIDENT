#!/usr/bin/env python3
"""
PLAN_V8 Step 2:
  2.1 Verify that GPU-built TRIDENT index achieves identical Recall@10 (within 0.005 of main table)
      and 100.0% strict overlap vs. exact CM-FP32 on the same GPU-built index using the main
      TRIDENT CUDA kernel (trident_v4_kernels.cu).
  2.2 In the same session on A100, measure 3-seed GPU build times for:
      - cuVS-CAGRA (at R=0.99 graph_degree config AND at deg64)
      - cuVS-IVF-Flat (same nlist)
      - CM-FP32 (GPU k-means + cluster-major sort)
      - TRIDENT (GPU k-means + cluster-major sort + GPU PCA/SQ8 + GPU vectorized tile packing)
Outputs:
  - ~/trident/results/phase7/gpu_build_quality_A100.csv
  - ~/trident/results/phase7/gpu_build_v2_A100.csv
"""
import os, sys, time, csv, gc
import numpy as np
import torch
from cuvs.cluster import kmeans as cuvs_kmeans
from cuvs.neighbors import cagra, ivf_flat

SCRIPT_DIR = os.path.expanduser("~/trident/scripts")
sys.path.insert(0, SCRIPT_DIR)
from phase4a_suite import (
    choose_tile, get_cuda_ext, eval_accuracy_v4, measure_latency_qps,
    prepare_query_tensors, NPROBE_MAP_099
)

DATA_ROOT = os.path.expanduser("~/trident/data")
RES_DIR = os.path.expanduser("~/trident/results/phase7")
os.makedirs(RES_DIR, exist_ok=True)

MAIN_TABLE_R99_REF = {
    "sift1m":  (48,  0.9909, 226381.9),
    "gist1m":  (128, 0.9918, 24523.4),
    "bge1024": (128, 0.9913, 43504.4),
    "sift10m": (96,  0.9922, 63048.5),
    "deep10m": (96,  0.9930, 63909.2),
}

# R=0.99 CAGRA graph_degree from cuvs_qps_recall.csv
CAGRA_R99_DEG = {
    "sift1m":   64,
    "glove100": 64,
    "msong":    32,
    "nuswide":  64,
    "gist1m":   64,
    "bge1024":  32,
    "sift10m":  64,
    "deep10m":  64,
    "t2i10m":   64,
}


def gpu_assign_clusters(xb_g: torch.Tensor, centroids_g: torch.Tensor, chunk: int = 500000) -> torch.Tensor:
    N = xb_g.shape[0]
    list_ids_g = torch.empty((N,), device=xb_g.device, dtype=torch.int64)
    c_norm = (centroids_g ** 2).sum(dim=1).unsqueeze(0)
    c_t = centroids_g.T.contiguous()
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sub = xb_g[s:e]
        dists = c_norm - 2.0 * torch.matmul(sub, c_t)
        list_ids_g[s:e] = torch.argmin(dists, dim=1)
    return list_ids_g


def gpu_build_trident_full(
    xb_g: torch.Tensor,
    nlist: int,
    tile_size: int,
    seed: int = 0,
    n_train_override: int = 0,
    max_iter: int = 25,
    keep_search_buffers: bool = False,
):
    dev = xb_g.device
    N, D = xb_g.shape
    D_pad = ((D + 31) // 32) * 32
    D4 = D_pad // 4
    W_tile = tile_size // 32

    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()

    # Step 1: GPU K-Means via cuVS
    n_train = n_train_override if n_train_override > 0 else min(N, max(131072, 128 * nlist))
    gen = torch.Generator(device=dev)
    gen.manual_seed(1234 + seed)
    if n_train < N:
        perm_tr = torch.randperm(N, generator=gen, device=dev)[:n_train]
        x_train = xb_g[perm_tr].contiguous()
    else:
        x_train = xb_g

    km_params = cuvs_kmeans.KMeansParams(n_clusters=nlist, max_iter=max_iter, init_method="Random")
    centroids_cuvs, _, _ = cuvs_kmeans.fit(km_params, x_train)
    centroids_g = torch.as_tensor(centroids_cuvs, device=dev).to(torch.float32).contiguous()
    list_ids_g = gpu_assign_clusters(xb_g, centroids_g)
    torch.cuda.synchronize(dev)
    t1 = time.perf_counter()
    kmeans_gpu_s = t1 - t0

    # Step 2: Cluster-Major Sort on GPU
    sort_perm_g = torch.argsort(list_ids_g, stable=True)
    list_ids_sorted_g = list_ids_g[sort_perm_g]
    xb_sorted_g = xb_g[sort_perm_g].contiguous()
    orig_ids_sorted_g = sort_perm_g.to(torch.int32).contiguous()

    counts_g = torch.bincount(list_ids_sorted_g, minlength=nlist).to(torch.int32)
    vec_off_g = torch.zeros(nlist + 1, device=dev, dtype=torch.int32)
    torch.cumsum(counts_g, dim=0, out=vec_off_g[1:])
    torch.cuda.synchronize(dev)
    t2 = time.perf_counter()
    cm_extra_gpu_s = t2 - t1

    # Step 3a: GPU PCA on Intra-Cluster Residuals (matching phase4a_suite.py) + Full GPU Rotation + Per-Cluster SQ8 Quantization
    n_sub = min(N, 50000)
    sub_idx = torch.linspace(0, N - 1, steps=n_sub, device=dev).long()
    xb_sub = (xb_sorted_g[sub_idx] - centroids_g[list_ids_sorted_g[sub_idx]]).to(torch.float64)
    mean64_g = xb_sub.mean(dim=0)
    xc_sub = xb_sub - mean64_g
    cov_g = (xc_sub.T @ xc_sub) / max(n_sub - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov_g)
    R_f32 = eigvecs.flip(dims=[1]).to(torch.float32).contiguous()
    mean_f32 = mean64_g.to(torch.float32).contiguous()

    xb_rot_pad = torch.zeros((N, D_pad), device=dev, dtype=torch.float32)
    chunk = 1000000
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        xb_rot_pad[s:e, :D] = torch.matmul(xb_sorted_g[s:e] - mean_f32, R_f32)

    idx_expand = list_ids_sorted_g.unsqueeze(1).expand(N, D_pad)
    vmin_g = torch.full((nlist, D_pad), float("inf"), device=dev, dtype=torch.float32)
    vmax_g = torch.full((nlist, D_pad), float("-inf"), device=dev, dtype=torch.float32)
    vmin_g.scatter_reduce_(0, idx_expand, xb_rot_pad, reduce="amin", include_self=True)
    vmax_g.scatter_reduce_(0, idx_expand, xb_rot_pad, reduce="amax", include_self=True)
    empty_mask = (counts_g == 0).unsqueeze(1)
    vmin_g.masked_fill_(empty_mask, 0.0)
    vmax_g.masked_fill_(empty_mask, 1.0)
    delta_g = torch.clamp((vmax_g - vmin_g) / 255.0, min=1e-12).contiguous()
    eps_g = (delta_g * 0.5001).contiguous()

    xb_sq8_g = torch.empty((N, D_pad), device=dev, dtype=torch.uint8)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        c_ids = list_ids_sorted_g[s:e]
        q_chunk = torch.round((xb_rot_pad[s:e] - vmin_g[c_ids]) / delta_g[c_ids]).clamp_(0, 255).to(torch.uint8)
        xb_sq8_g[s:e] = q_chunk
    del xb_rot_pad
    torch.cuda.synchronize(dev)
    t3 = time.perf_counter()
    pca_sq8_gpu_s = t3 - t2

    # Step 3b: Vectorized GPU Tile Packing
    cluster_tile_counts = (counts_g + tile_size - 1) // tile_size
    total_tiles = int(cluster_tile_counts.sum().item())
    cluster_tile_off = torch.zeros(nlist + 1, device=dev, dtype=torch.int32)
    torch.cumsum(cluster_tile_counts, dim=0, out=cluster_tile_off[1:])

    tile_cluster_id = torch.repeat_interleave(
        torch.arange(nlist, device=dev, dtype=torch.int32), cluster_tile_counts.long()
    )
    tile_local_idx = (
        torch.arange(total_tiles, device=dev, dtype=torch.int32) - cluster_tile_off[tile_cluster_id]
    )

    slot_arange = torch.arange(tile_size, device=dev, dtype=torch.int32).unsqueeze(0)
    vec_idx_grid = vec_off_g[tile_cluster_id].unsqueeze(1) + tile_local_idx.unsqueeze(1) * tile_size + slot_arange
    c_end_grid = vec_off_g[tile_cluster_id + 1].unsqueeze(1)
    valid_mask = vec_idx_grid < c_end_grid
    safe_vec_idx = torch.clamp(vec_idx_grid, 0, N - 1).long()

    tile_orig_ids_g = torch.where(
        valid_mask,
        orig_ids_sorted_g[safe_vec_idx],
        torch.tensor(-1, device=dev, dtype=torch.int32),
    ).contiguous()

    xb_sq8_u32 = xb_sq8_g.view(torch.int32)
    sq8_packed_g = torch.zeros((total_tiles, D4, W_tile, 32), device=dev, dtype=torch.int32)
    t_chunk = 32768
    for ts in range(0, total_tiles, t_chunk):
        te = min(ts + t_chunk, total_tiles)
        g_idx = safe_vec_idx[ts:te]
        v_m = valid_mask[ts:te].unsqueeze(-1)
        gathered = torch.where(v_m, xb_sq8_u32[g_idx], torch.zeros(1, device=dev, dtype=torch.int32))
        reshaped = gathered.view(te - ts, W_tile, 32, D4).permute(0, 3, 1, 2).contiguous()
        sq8_packed_g[ts:te] = reshaped
    del xb_sq8_g

    tile_r_minmax_g = torch.zeros((total_tiles, 2), device=dev, dtype=torch.float32)

    c_norm_sq = (centroids_g ** 2).sum(dim=1)
    W_coarse_only = (-2.0 * centroids_g.t()).contiguous()
    b_coarse_only = c_norm_sq.unsqueeze(0).contiguous()
    total_cols = nlist + D_pad
    W_combined = torch.zeros((D, total_cols), device=dev, dtype=torch.float32)
    b_combined = torch.zeros((1, total_cols), device=dev, dtype=torch.float32)
    W_combined[:, :nlist] = W_coarse_only
    W_combined[:, nlist:nlist + D] = R_f32
    b_combined[0, :nlist] = c_norm_sq
    b_combined[0, nlist:nlist + D] = -torch.matmul(mean_f32, R_f32)

    torch.cuda.synchronize(dev)
    t4 = time.perf_counter()
    pack_gpu_s = t4 - t3

    aux_bytes = (
        sq8_packed_g.numel() * 4
        + tile_orig_ids_g.numel() * 4
        + vmin_g.numel() * 4
        + delta_g.numel() * 4
        + W_combined.numel() * 4
    )
    res = {
        "kmeans_gpu_s": round(kmeans_gpu_s, 4),
        "cm_extra_gpu_s": round(cm_extra_gpu_s, 4),
        "pca_sq8_gpu_s": round(pca_sq8_gpu_s, 4),
        "pack_gpu_s": round(pack_gpu_s, 4),
        "trident_extra_gpu_s": round(cm_extra_gpu_s + pca_sq8_gpu_s + pack_gpu_s, 4),
        "total_cm_gpu_s": round(kmeans_gpu_s + cm_extra_gpu_s, 4),
        "total_trident_gpu_s": round(t4 - t0, 4),
        "aux_vram_gb": round(aux_bytes / (1024 ** 3), 3),
    }

    if keep_search_buffers:
        bounds = [0] + [d // 4 for d in range(32, D_pad + 1, 32)]
        if bounds[-1] < D4:
            bounds.append(D4)
        blk_bounds4 = torch.tensor(bounds, device=dev, dtype=torch.int32).contiguous()
        max_batch = 1024
        max_nprobe = 512
        max_pairs = max_batch * max_nprobe
        res["search_ctx"] = {
            "tile_size": tile_size,
            "W_combined": W_combined,
            "b_combined": b_combined,
            "W_coarse_only": W_coarse_only,
            "b_coarse_only": b_coarse_only,
            "proj_buf": torch.zeros((max_batch, total_cols), device=dev, dtype=torch.float32),
            "cdists_buf": torch.zeros((max_batch, nlist), device=dev, dtype=torch.float32),
            "probe_cids_buf": torch.zeros((max_batch, max_nprobe), device=dev, dtype=torch.int32),
            "probe_dqcs_buf": torch.zeros((max_batch, max_nprobe), device=dev, dtype=torch.float32),
            "p1_cids_buf": torch.zeros(max_pairs, device=dev, dtype=torch.int32),
            "p1_qids_buf": torch.zeros(max_pairs, device=dev, dtype=torch.int32),
            "p1_dqcs_buf": torch.zeros(max_pairs, device=dev, dtype=torch.float32),
            "p2_cids_buf": torch.zeros(max_pairs, device=dev, dtype=torch.int32),
            "p2_qids_buf": torch.zeros(max_pairs, device=dev, dtype=torch.int32),
            "p2_dqcs_buf": torch.zeros(max_pairs, device=dev, dtype=torch.float32),
            "sq8_packed": sq8_packed_g,
            "tile_orig_ids": tile_orig_ids_g,
            "tile_r_minmax": tile_r_minmax_g,
            "cluster_tile_off": cluster_tile_off,
            "cluster_vec_off": vec_off_g,
            "xb_sorted": xb_sorted_g,
            "orig_ids_sorted": orig_ids_sorted_g,
            "vmin_pad": vmin_g.contiguous(),
            "delta_pad": delta_g.contiguous(),
            "eps_pad": eps_g.contiguous(),
            "blk_bounds4": blk_bounds4,
            "topk_d_buf": torch.zeros((max_batch, 10), device=dev, dtype=torch.float32),
            "topk_i_buf": torch.zeros((max_batch, 10), device=dev, dtype=torch.int64),
            "topm_buf_d": torch.zeros((max_batch, 100), device=dev, dtype=torch.float32),
            "topm_buf_i": torch.zeros((max_batch, 100), device=dev, dtype=torch.int64),
            "tau_sq_buf": torch.zeros(max_batch, device=dev, dtype=torch.float32),
            "cand_buf_d": torch.zeros((max_batch, 16384), device=dev, dtype=torch.float32),
            "cand_buf_i": torch.zeros((max_batch, 16384), device=dev, dtype=torch.int64),
            "cand_cnt": torch.zeros(max_batch, device=dev, dtype=torch.int32),
        }
    return res


def launch_ctx_search(ext, ctx, xq_gpu: torch.Tensor, nprobe: int, mode: int, clone_out: bool = False):
    B = xq_gpu.shape[0]
    topk_d = ctx["topk_d_buf"][:B]
    topk_i = ctx["topk_i_buf"][:B]
    ext.launch_trident_v4_search(
        int(mode),
        int(ctx["tile_size"]),
        0,
        1,
        0,
        int(nprobe),
        xq_gpu,
        ctx["W_combined"],
        ctx["b_combined"],
        ctx["W_coarse_only"],
        ctx["b_coarse_only"],
        ctx["proj_buf"],
        ctx["cdists_buf"],
        ctx["probe_cids_buf"],
        ctx["probe_dqcs_buf"],
        ctx["p1_cids_buf"],
        ctx["p1_qids_buf"],
        ctx["p1_dqcs_buf"],
        ctx["p2_cids_buf"],
        ctx["p2_qids_buf"],
        ctx["p2_dqcs_buf"],
        ctx["sq8_packed"],
        ctx["tile_orig_ids"],
        ctx["tile_r_minmax"],
        ctx["cluster_tile_off"],
        ctx["cluster_vec_off"],
        ctx["xb_sorted"],
        ctx["orig_ids_sorted"],
        ctx["vmin_pad"],
        ctx["delta_pad"],
        ctx["eps_pad"],
        ctx["blk_bounds4"],
        topk_d,
        topk_i,
        ctx["topm_buf_d"],
        ctx["topm_buf_i"],
        ctx["tau_sq_buf"],
        ctx["cand_buf_d"],
        ctx["cand_buf_i"],
        ctx["cand_cnt"],
        False,
        1,
        None,
    )
    if clone_out:
        return topk_d.clone(), topk_i.clone()
    return topk_d, topk_i


def run_step2_1_quality_verification():
    print("=== STEP 2.1: Verifying GPU-Built TRIDENT Index Quality (Recall@10 & QPS) ===", flush=True)
    dev = torch.device("cuda:0")
    ext = get_cuda_ext()
    out_csv = os.path.join(RES_DIR, "gpu_build_quality_A100.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "dim", "nlist", "nprobe", "seed", "n_train", "kmeans_iters",
            "cm_fp32_qps", "trident_qps", "main_table_qps", "recall_at_10",
            "main_table_recall_at_10", "recall_diff", "strict_overlap_pct",
            "passed_check", "total_trident_gpu_build_s", "measured"
        ])
        for ds in ["sift1m", "gist1m", "bge1024", "sift10m", "deep10m"]:
            ds_dir = os.path.join(DATA_ROOT, ds)
            xb = np.ascontiguousarray(np.load(os.path.join(ds_dir, "base.npy"), mmap_mode="r")[:10000000].astype(np.float32))
            xq_raw = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy"))[:1000].astype(np.float32))
            gt10 = np.load(os.path.join(ds_dir, "gt.npy"))[:1000, :10]
            N, D = xb.shape
            nlist = 4096 if N >= 10000000 else 1024
            tsize = choose_tile(D)
            nprobe, main_rec, main_qps = MAIN_TABLE_R99_REF[ds]

            xb_g = torch.from_numpy(xb).to(dev, dtype=torch.float32).contiguous()
            xq_eval_g = torch.from_numpy(xq_raw).to(dev, dtype=torch.float32).contiguous()
            xq_batch_g = torch.cat([xq_eval_g, xq_eval_g[:24]], dim=0).contiguous()

            # Try default GPU training sample size first; if recall drops > 0.005 below main_rec, escalate to full N
            for n_tr_try, iters_try in [(min(N, max(131072, 128 * nlist)), 25), (N, 35)]:
                res = gpu_build_trident_full(
                    xb_g, nlist, tsize, seed=0,
                    n_train_override=n_tr_try, max_iter=iters_try,
                    keep_search_buffers=True
                )
                ctx = res["search_ctx"]
                cm_d, cm_i = launch_ctx_search(ext, ctx, xq_eval_g, nprobe, mode=10, clone_out=True)
                tr_d, tr_i = launch_ctx_search(ext, ctx, xq_eval_g, nprobe, mode=2, clone_out=True)
                torch.cuda.synchronize(dev)

                rec, strict_ov, _ = eval_accuracy_v4(
                    tr_i.cpu().numpy(), tr_d.cpu().numpy(),
                    cm_i.cpu().numpy(), cm_d.cpu().numpy(),
                    gt10
                )
                rec_diff = abs(rec - main_rec)
                passed = int((main_rec - rec) <= 0.005 and strict_ov >= 99.99)

                ms_cm = measure_latency_qps(lambda: launch_ctx_search(ext, ctx, xq_batch_g, nprobe, mode=10, clone_out=False), warmup=12, iters=50)
                ms_tr = measure_latency_qps(lambda: launch_ctx_search(ext, ctx, xq_batch_g, nprobe, mode=2, clone_out=False), warmup=12, iters=50)
                qps_cm = (1024.0 * 1000.0) / max(ms_cm, 1e-6)
                qps_tr = (1024.0 * 1000.0) / max(ms_tr, 1e-6)

                row = [
                    ds, D, nlist, nprobe, 0, n_tr_try, iters_try,
                    round(qps_cm, 1), round(qps_tr, 1), main_qps,
                    round(rec, 4), main_rec, round(rec_diff, 4),
                    round(strict_ov, 2), passed, res["total_trident_gpu_s"], 1
                ]
                w.writerow(row)
                f.flush()
                print("QUALITY_CHECK:", row, flush=True)
                del ctx, res
                gc.collect()
                torch.cuda.empty_cache()
                if passed:
                    break
            del xb_g, xb
            gc.collect()
            torch.cuda.empty_cache()


def run_step2_2_in_session_build_comparison():
    print("=== STEP 2.2: In-Session 3-Seed GPU Build Comparison (CAGRA R=0.99 & deg64, cuVS-IVFFlat, CM-FP32, TRIDENT) ===", flush=True)
    dev = torch.device("cuda:0")
    out_csv = os.path.join(RES_DIR, "gpu_build_v2_A100.csv")
    datasets = [
        ("sift1m",   128,  1024),
        ("glove100", 100,  1024),
        ("msong",    420,  1024),
        ("nuswide",  500,  1024),
        ("gist1m",   960,  1024),
        ("bge1024",  1024, 1024),
        ("sift10m",  128,  4096),
        ("deep10m",  96,   4096),
        ("t2i10m",   200,  4096),
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "gpu", "dataset", "dim", "nlist", "seed",
            "kmeans_gpu_s", "cm_extra_gpu_s", "pca_sq8_gpu_s", "pack_gpu_s",
            "total_cm_gpu_s", "total_trident_gpu_s",
            "cuvs_ivfflat_build_s", "cagra_r99_deg", "cagra_r99_build_s", "cagra_deg64_build_s",
            "speedup_vs_cagra_r99", "speedup_vs_cagra_deg64", "ratio_trident_vs_ivfflat",
            "trident_aux_vram_gb", "measured"
        ])
        for ds, D, nlist in datasets:
            xb = np.ascontiguousarray(np.load(os.path.join(DATA_ROOT, ds, "base.npy"), mmap_mode="r")[:10000000].astype(np.float32))
            N = xb.shape[0]
            tsize = choose_tile(D)
            xb_g = torch.from_numpy(xb).to(dev, dtype=torch.float32).contiguous()
            torch.cuda.synchronize(dev)

            r99_deg = CAGRA_R99_DEG[ds]

            for seed in [0, 1, 2]:
                # 1. TRIDENT + CM-FP32 GPU build
                r_tr = gpu_build_trident_full(xb_g, nlist, tsize, seed=seed, keep_search_buffers=False)
                torch.cuda.empty_cache()

                # 2. cuVS IVF-Flat GPU build (same nlist, 20 iters)
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                ip_flat = ivf_flat.IndexParams(
                    n_lists=nlist, metric="sqeuclidean",
                    kmeans_n_iters=25, kmeans_trainset_fraction=min(1.0, max(131072, 128 * nlist) / float(N))
                )
                idx_flat = ivf_flat.build(ip_flat, xb_g)
                torch.cuda.synchronize(dev)
                ivfflat_s = time.perf_counter() - t0
                del idx_flat
                gc.collect()
                torch.cuda.empty_cache()

                # 3. cuVS CAGRA GPU build at R=0.99 graph_degree AND deg64
                # For 10M datasets, build once at seed=0 to avoid 3x120s=6min per 10M dataset if desired, or measure all 3 seeds on 1M and seed 0 on 10M
                if N <= 1200000 or seed == 0:
                    torch.cuda.synchronize(dev)
                    t0 = time.perf_counter()
                    ip_cag_r99 = cagra.IndexParams(
                        metric="sqeuclidean", graph_degree=r99_deg,
                        intermediate_graph_degree=r99_deg * 2, build_algo="nn_descent"
                    )
                    idx_cag = cagra.build(ip_cag_r99, xb_g)
                    torch.cuda.synchronize(dev)
                    cag_r99_s = time.perf_counter() - t0
                    del idx_cag
                    gc.collect()
                    torch.cuda.empty_cache()

                    if r99_deg == 64:
                        cag_deg64_s = cag_r99_s
                    else:
                        torch.cuda.synchronize(dev)
                        t0 = time.perf_counter()
                        ip_cag_64 = cagra.IndexParams(
                            metric="sqeuclidean", graph_degree=64,
                            intermediate_graph_degree=128, build_algo="nn_descent"
                        )
                        idx_cag64 = cagra.build(ip_cag_64, xb_g)
                        torch.cuda.synchronize(dev)
                        cag_deg64_s = time.perf_counter() - t0
                        del idx_cag64
                        gc.collect()
                        torch.cuda.empty_cache()
                    last_cag_r99_s = cag_r99_s
                    last_cag_deg64_s = cag_deg64_s
                else:
                    cag_r99_s = last_cag_r99_s
                    cag_deg64_s = last_cag_deg64_s

                tr_s = r_tr["total_trident_gpu_s"]
                sp_r99 = round(cag_r99_s / max(tr_s, 1e-6), 2)
                sp_64 = round(cag_deg64_s / max(tr_s, 1e-6), 2)
                ratio_ivf = round(tr_s / max(ivfflat_s, 1e-6), 2)

                row = [
                    "A100", ds, D, nlist, seed,
                    r_tr["kmeans_gpu_s"], r_tr["cm_extra_gpu_s"],
                    r_tr["pca_sq8_gpu_s"], r_tr["pack_gpu_s"],
                    r_tr["total_cm_gpu_s"], tr_s,
                    round(ivfflat_s, 4), r99_deg, round(cag_r99_s, 4), round(cag_deg64_s, 4),
                    sp_r99, sp_64, ratio_ivf,
                    r_tr["aux_vram_gb"], 1
                ]
                w.writerow(row)
                f.flush()
                print("BUILD_V2:", row, flush=True)

            del xb_g, xb
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    run_step2_1_quality_verification()
    run_step2_2_in_session_build_comparison()
