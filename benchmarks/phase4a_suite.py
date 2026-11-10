#!/usr/bin/env python3
"""
TRIDENT Phase 4A Benchmark Suite (PLAN_V5)
Implements Phase 4A requirements:
  - 4A-1: Fixed rule choose_tile(D)
  - 4A-2: Ablation redo (B0..B4 chain + Leave-One-Out) on A100 (5 seeds)
  - 4A-3: Complete Recall@10=0.99 benchmark (all baselines) on A100 (5 seeds)
  - 4A-4: QPS-Recall Pareto curves across nprobes with Faiss-IVFPQ+Refine (k=2,4,8) (3 seeds)
  - 4A-5: Unified Build Time & Memory Footprint breakdown
  - 4A-6: Pruning Statistics with atomic counters (stats_buf)
  - 4A-7: TRIDENT-Auto Cost Model calibration and multi-batch benchmark
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
from torch.utils.cpp_extension import load

# ----------------- CONFIG & PATHS -----------------
DATA_ROOT = os.path.expanduser("~/trident/data")
RES_ROOT = os.path.expanduser("~/trident/results/phase4a")
CACHE_ROOT = os.path.expanduser("~/trident/cache_v3")
os.makedirs(RES_ROOT, exist_ok=True)
os.makedirs(CACHE_ROOT, exist_ok=True)

NPROBE_MAP_095 = {
    "gist1m": 56,
    "bge1024": 14,
    "sift1m": 24,
    "msong": 10,
    "glove100": 128,
    "nuswide": 3,
}

NPROBE_MAP_099 = {
    "gist1m": 128,
    "bge1024": 128,
    "sift1m": 48,
    "msong": 24,
    "glove100": 384,
    "nuswide": 8,
}

M_PQ_MAP = {
    "bge1024": 32,
    "gist1m": 48,
    "sift1m": 16,
    "msong": 20,
    "glove100": 20,
    "nuswide": 20,
}

def choose_tile(dim: int) -> int:
    return 256 if dim >= 256 else 64

# ----------------- JIT LOAD CUDA EXTENSION -----------------
def get_cuda_ext():
    cuda_src = os.path.expanduser("~/trident/scripts/trident_v4_kernels.cu")
    return load(
        name="trident_v4_ext",
        sources=[cuda_src],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=False,
    )

# ----------------- CLUSTERING STORAGE & INDEX -----------------
class SharedClusterStorage:
    def __init__(self, xb_raw: np.ndarray, list_ids: np.ndarray, r_vec: np.ndarray, nlist: int, device="cuda:0"):
        self.dev = torch.device(device)
        self.N, self.D = xb_raw.shape
        self.nlist = nlist

        sort_perm = np.argsort(list_ids, kind="stable").astype(np.int32)
        xb_sorted_np = np.ascontiguousarray(xb_raw[sort_perm])
        self.xb_sorted = torch.from_numpy(xb_sorted_np).to(self.dev, dtype=torch.float32).contiguous()
        self.orig_ids_sorted = torch.from_numpy(sort_perm).to(self.dev, dtype=torch.int32).contiguous()

        counts = np.bincount(list_ids, minlength=nlist).astype(np.int32)
        vec_offsets = np.zeros(nlist + 1, dtype=np.int32)
        vec_offsets[1:] = np.cumsum(counts)
        self.cluster_vec_off = torch.from_numpy(vec_offsets).to(self.dev, dtype=torch.int32).contiguous()
        self.sort_perm = sort_perm
        self.counts = counts
        self.r_vec_sorted = r_vec[sort_perm]


class TridentV4Index:
    def __init__(
        self,
        storage: SharedClusterStorage,
        centroids: np.ndarray,
        mean64: np.ndarray,
        R64: np.ndarray,
        tile_size: int = 256,
        per_cluster_sq8: int = 1,
        use_pca: int = 1,
        var_block: int = 0, # default 0 (uniform block 32 as per 4A-2)
        max_batch: int = 1024,
        max_nprobe: int = 512,
        device="cuda:0",
    ):
        self.storage = storage
        self.dev = torch.device(device)
        self.N = storage.N
        self.D = storage.D
        self.nlist = storage.nlist
        self.tile_size = tile_size
        self.per_cluster_sq8 = int(per_cluster_sq8)
        self.use_pca = int(use_pca)
        self.var_block = int(var_block)
        self.W_tile = tile_size // 32
        self.D_pad = ((self.D + 31) // 32) * 32
        self.D4 = self.D_pad // 4

        if use_pca:
            mean64_g = torch.from_numpy(mean64).to(self.dev, dtype=torch.float64)
            R64_g = torch.from_numpy(R64).to(self.dev, dtype=torch.float64)
        else:
            mean64_g = torch.zeros((self.D,), device=self.dev, dtype=torch.float64)
            R64_g = torch.eye(self.D, device=self.dev, dtype=torch.float64)

        mean_f32 = mean64_g.to(torch.float32).contiguous()
        R_f32 = R64_g.to(torch.float32).contiguous()

        centroids_g = torch.from_numpy(centroids).to(self.dev, dtype=torch.float32).contiguous()
        c_norm_sq = (centroids_g ** 2).sum(dim=1)

        self.W_coarse_only = (-2.0 * centroids_g.t()).contiguous()
        self.b_coarse_only = c_norm_sq.unsqueeze(0).contiguous()

        total_cols = self.nlist + self.D_pad
        self.W_combined = torch.zeros((self.D, total_cols), device=self.dev, dtype=torch.float32)
        self.b_combined = torch.zeros((1, total_cols), device=self.dev, dtype=torch.float32)
        self.W_combined[:, :self.nlist] = self.W_coarse_only
        self.W_combined[:, self.nlist:self.nlist + self.D] = R_f32
        self.b_combined[0, :self.nlist] = c_norm_sq
        self.b_combined[0, self.nlist:self.nlist + self.D] = -torch.matmul(mean_f32, R_f32)

        # Quantize vectors to SQ8
        xb_sorted_cpu = storage.xb_sorted.cpu().numpy()
        if use_pca:
            xb_rot = (xb_sorted_cpu.astype(np.float64) - mean64) @ R64
        else:
            xb_rot = xb_sorted_cpu.astype(np.float64)

        xb_rot_pad = np.zeros((self.N, self.D_pad), dtype=np.float32)
        xb_rot_pad[:, :self.D] = xb_rot.astype(np.float32)

        if self.per_cluster_sq8:
            vmin_np = np.zeros((self.nlist, self.D_pad), dtype=np.float32)
            delta_np = np.zeros((self.nlist, self.D_pad), dtype=np.float32)
            xb_sq8_np = np.zeros((self.N, self.D_pad), dtype=np.uint8)
            counts = storage.counts
            vec_off = storage.cluster_vec_off.cpu().numpy()
            for c in range(self.nlist):
                s, e = vec_off[c], vec_off[c + 1]
                if s < e:
                    c_vecs = xb_rot_pad[s:e]
                    c_min = c_vecs.min(axis=0)
                    c_max = c_vecs.max(axis=0)
                    c_delta = np.maximum((c_max - c_min) / 255.0, 1e-12)
                    vmin_np[c] = c_min
                    delta_np[c] = c_delta
                    q = np.clip(np.round((c_vecs - c_min) / c_delta), 0, 255).astype(np.uint8)
                    xb_sq8_np[s:e] = q
                else:
                    vmin_np[c] = 0.0
                    delta_np[c] = 1.0
            self.vmin_pad = torch.from_numpy(vmin_np).to(self.dev, dtype=torch.float32).contiguous()
            self.delta_pad = torch.from_numpy(delta_np).to(self.dev, dtype=torch.float32).contiguous()
            self.eps_pad = (self.delta_pad * 0.5001).contiguous()
        else:
            vmin_1 = xb_rot_pad.min(axis=0)
            vmax_1 = xb_rot_pad.max(axis=0)
            delta_1 = np.maximum((vmax_1 - vmin_1) / 255.0, 1e-12).astype(np.float32)
            self.vmin_pad = torch.from_numpy(vmin_1).unsqueeze(0).to(self.dev, dtype=torch.float32).contiguous()
            self.delta_pad = torch.from_numpy(delta_1).unsqueeze(0).to(self.dev, dtype=torch.float32).contiguous()
            self.eps_pad = (self.delta_pad * 0.5001).contiguous()
            xb_sq8_np = np.clip(np.round((xb_rot_pad - vmin_1) / delta_1), 0, 255).astype(np.uint8)

        # Tile packing
        cluster_tile_counts = (storage.counts + tile_size - 1) // tile_size
        total_tiles = int(cluster_tile_counts.sum())
        cluster_tile_off = np.zeros(self.nlist + 1, dtype=np.int32)
        cluster_tile_off[1:] = np.cumsum(cluster_tile_counts)
        self.cluster_tile_off = torch.from_numpy(cluster_tile_off).to(self.dev, dtype=torch.int32).contiguous()

        sq8_packed_np = np.zeros((total_tiles, self.D4, self.W_tile, 32), dtype=np.uint32)
        tile_orig_ids_np = np.full((total_tiles, tile_size), -1, dtype=np.int32)
        tile_r_minmax_np = np.zeros((total_tiles, 2), dtype=np.float32)

        orig_ids_sorted_np = storage.sort_perm
        r_vec_sorted_np = storage.r_vec_sorted
        vec_off = storage.cluster_vec_off.cpu().numpy()

        for c in range(self.nlist):
            s_vec, e_vec = vec_off[c], vec_off[c + 1]
            n_vec = e_vec - s_vec
            if n_vec == 0:
                continue
            t_start = cluster_tile_off[c]
            n_tiles = cluster_tile_counts[c]
            for t in range(n_tiles):
                tile_idx = t_start + t
                vs = s_vec + t * tile_size
                ve = min(s_vec + (t + 1) * tile_size, e_vec)
                k_cur = ve - vs
                tile_orig_ids_np[tile_idx, :k_cur] = orig_ids_sorted_np[vs:ve]
                r_slice = r_vec_sorted_np[vs:ve]
                tile_r_minmax_np[tile_idx, 0] = r_slice.min()
                tile_r_minmax_np[tile_idx, 1] = r_slice.max()

                sub_sq8 = xb_sq8_np[vs:ve].view(np.uint32)
                for w in range(self.W_tile):
                    w_s = w * 32
                    w_e = min((w + 1) * 32, k_cur)
                    if w_s < w_e:
                        sq8_packed_np[tile_idx, :, w, :w_e - w_s] = sub_sq8[w_s:w_e].T

        self.sq8_packed = torch.from_numpy(sq8_packed_np.view(np.int32)).to(self.dev).contiguous()
        self.tile_orig_ids = torch.from_numpy(tile_orig_ids_np).to(self.dev).contiguous()
        self.tile_r_minmax = torch.from_numpy(tile_r_minmax_np).to(self.dev).contiguous()

        # Dimension block bounds
        if var_block:
            if self.D_pad <= 128:
                blk_dims = [16, 16, 32, 32, 32]
            elif self.D_pad <= 256:
                blk_dims = [16, 16, 32, 32, 64, 64, 32]
            else:
                blk_dims = [16, 16, 32, 32, 64, 64, 64, 128, 128, 256, 256]
            bounds = [0]
            cur = 0
            for bd in blk_dims:
                cur += bd
                if cur <= self.D_pad:
                    bounds.append(cur // 4)
            if bounds[-1] < self.D4:
                bounds.append(self.D4)
        else:
            bounds = [0]
            for d in range(32, self.D_pad + 1, 32):
                bounds.append(d // 4)
            if bounds[-1] < self.D4:
                bounds.append(self.D4)

        self.blk_bounds4 = torch.tensor(bounds, device=self.dev, dtype=torch.int32).contiguous()

        # Buffers
        self.max_batch = max_batch
        self.max_nprobe = max_nprobe
        self.proj_buf = torch.zeros((max_batch, total_cols), device=self.dev, dtype=torch.float32)
        self.cdists_buf = torch.zeros((max_batch, self.nlist), device=self.dev, dtype=torch.float32)
        self.probe_cids_buf = torch.zeros((max_batch, max_nprobe), device=self.dev, dtype=torch.int32)
        self.probe_dqcs_buf = torch.zeros((max_batch, max_nprobe), device=self.dev, dtype=torch.float32)
        max_pairs = max_batch * max_nprobe
        self.p1_cids_buf = torch.zeros(max_pairs, device=self.dev, dtype=torch.int32)
        self.p1_qids_buf = torch.zeros(max_pairs, device=self.dev, dtype=torch.int32)
        self.p1_dqcs_buf = torch.zeros(max_pairs, device=self.dev, dtype=torch.float32)
        self.p2_cids_buf = torch.zeros(max_pairs, device=self.dev, dtype=torch.int32)
        self.p2_qids_buf = torch.zeros(max_pairs, device=self.dev, dtype=torch.int32)
        self.p2_dqcs_buf = torch.zeros(max_pairs, device=self.dev, dtype=torch.float32)
        self.topk_d_buf = torch.zeros((max_batch, 10), device=self.dev, dtype=torch.float32)
        self.topk_i_buf = torch.zeros((max_batch, 10), device=self.dev, dtype=torch.int64)
        self.topm_buf_d = torch.zeros((max_batch, 100), device=self.dev, dtype=torch.float32)
        self.topm_buf_i = torch.zeros((max_batch, 100), device=self.dev, dtype=torch.int64)
        self.tau_sq_buf = torch.zeros(max_batch, device=self.dev, dtype=torch.float32)
        self.cand_buf_d = torch.zeros((max_batch, 16384), device=self.dev, dtype=torch.float32)
        self.cand_buf_i = torch.zeros((max_batch, 16384), device=self.dev, dtype=torch.int64)
        self.cand_cnt = torch.zeros(max_batch, device=self.dev, dtype=torch.int32)

    def search(
        self,
        xq_gpu: torch.Tensor,
        nprobe: int,
        mode: int = 2,
        use_l1: int = 0,
        rerank_m: int = 0,
        tile_override: int = 0,
        warp_refine: int = 1,
        stats_buf: Optional[torch.Tensor] = None,
        record_kernel_ms: bool = False,
    ):
        ext = get_cuda_ext()
        B = xq_gpu.shape[0]
        tsize = tile_override if tile_override > 0 else self.tile_size
        topk_d = self.topk_d_buf[:B]
        topk_i = self.topk_i_buf[:B]
        kern_ms = ext.launch_trident_v4_search(
            int(mode),
            int(tsize),
            int(use_l1),
            int(self.per_cluster_sq8),
            int(rerank_m),
            int(nprobe),
            xq_gpu,
            self.W_combined,
            self.b_combined,
            self.W_coarse_only,
            self.b_coarse_only,
            self.proj_buf,
            self.cdists_buf,
            self.probe_cids_buf,
            self.probe_dqcs_buf,
            self.p1_cids_buf,
            self.p1_qids_buf,
            self.p1_dqcs_buf,
            self.p2_cids_buf,
            self.p2_qids_buf,
            self.p2_dqcs_buf,
            self.sq8_packed,
            self.tile_orig_ids,
            self.tile_r_minmax,
            self.cluster_tile_off,
            self.storage.cluster_vec_off,
            self.storage.xb_sorted,
            self.storage.orig_ids_sorted,
            self.vmin_pad,
            self.delta_pad,
            self.eps_pad,
            self.blk_bounds4,
            topk_d,
            topk_i,
            self.topm_buf_d,
            self.topm_buf_i,
            self.tau_sq_buf,
            self.cand_buf_d,
            self.cand_buf_i,
            self.cand_cnt,
            bool(record_kernel_ms),
            int(warp_refine),
            stats_buf,
        )
        return topk_d, topk_i, kern_ms

# ----------------- ACCURACY & TIMING UTILITIES -----------------
def eval_accuracy_v4(pred_ids, pred_dists, cm_ids, cm_dists, gt10):
    k = 10
    Q = pred_ids.shape[0]
    total_slots = Q * k
    rec_matches = 0
    cm_strict_matches = 0
    cm_m100_queries = 0

    for q in range(Q):
        g_set = set(gt10[q].tolist())
        p_row = pred_ids[q].tolist()
        rec_matches += len(set(p_row) & g_set)

        c_row = cm_ids[q].tolist()
        cm_row_matches = 0
        for j in range(k):
            if p_row[j] == c_row[j]:
                cm_row_matches += 1
            else:
                d_p = pred_dists[q, j]
                d_c = cm_dists[q, j]
                if abs(d_p - d_c) <= 1e-4 * max(abs(d_c), 1.0) and d_p >= 0:
                    cm_row_matches += 1
        cm_strict_matches += cm_row_matches
        if cm_row_matches == k:
            cm_m100_queries += 1

    rec_at_10 = rec_matches / float(total_slots)
    cm_strict_ov_pct = (cm_strict_matches / float(total_slots)) * 100.0
    cm_m100_pct = (cm_m100_queries / float(Q)) * 100.0
    return rec_at_10, cm_strict_ov_pct, cm_m100_pct

def measure_latency_qps(fn, warmup=12, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000.0
    return total_ms / float(iters)

def load_or_train_clustering(dataset: str, xb: np.ndarray, nlist: int, seed: int):
    import faiss
    cache_file = os.path.join(CACHE_ROOT, f"{dataset}_nlist{nlist}_s{seed}.npz")
    if os.path.exists(cache_file):
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
    d = xb.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    index.cp.seed = seed + 1000
    index.train(xb)
    index.add(xb)
    train_time = time.perf_counter() - t0

    centroids = index.quantizer.reconstruct_n(0, nlist)
    invlists = index.invlists
    list_ids = np.empty(xb.shape[0], dtype=np.int32)
    for c in range(nlist):
        sz = invlists.list_size(c)
        if sz > 0:
            ids = faiss.rev_swig_ptr(invlists.get_ids(c), sz).copy()
            list_ids[ids] = c

    diff = xb - centroids[list_ids]
    r_vec = np.linalg.norm(diff, axis=1).astype(np.float32)

    sample_size = min(50000, xb.shape[0])
    np.random.seed(seed + 42)
    idx_sample = np.random.choice(xb.shape[0], sample_size, replace=False)
    diff_sample = diff[idx_sample].astype(np.float64)

    mean64 = np.mean(diff_sample, axis=0)
    cov = np.cov(diff_sample, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    sort_idx = np.argsort(evals)[::-1]
    R64 = evecs[:, sort_idx]

    tmp_file = cache_file + f".tmp_{os.getpid()}"
    np.savez(tmp_file, centroids=centroids, list_ids=list_ids, r_vec=r_vec, mean64=mean64, R64=R64, train_time_s=train_time)
    os.replace(tmp_file, cache_file)
    return centroids, list_ids, r_vec, mean64, R64, train_time

def prepare_query_tensors(xq_raw, dataset, dev):
    Q_orig = xq_raw.shape[0]
    distinct_q = Q_orig
    b_max = 1024
    if dataset == "nuswide":
        b_max = 200
        distinct_q = 200
        xq_batch = xq_raw[:200]
        Q_eval = 200
    elif Q_orig < 1024:
        needed = 1024 - Q_orig
        xq_batch = np.vstack([xq_raw, xq_raw[:needed]])
        Q_eval = Q_orig
    else:
        xq_batch = xq_raw[:1024]
        Q_eval = 1024

    xq_eval_gpu = torch.from_numpy(xq_raw[:Q_eval]).to(dev, dtype=torch.float32).contiguous()
    xq_batch_gpu = torch.from_numpy(xq_batch).to(dev, dtype=torch.float32).contiguous()
    return Q_eval, xq_eval_gpu, b_max, distinct_q, xq_batch_gpu

# ----------------- FAISS GPU HELPERS -----------------
def build_faiss_gpu_indexes(xb, centroids, list_ids, nlist, dev_id=0):
    import faiss
    d = xb.shape[1]
    res = faiss.StandardGpuResources()

    # 1. IVFFlat
    q_flat = faiss.IndexFlatL2(d)
    cpu_flat = faiss.IndexIVFFlat(q_flat, d, nlist, faiss.METRIC_L2)
    cpu_flat.quantizer.add(centroids)
    cpu_flat.is_trained = True
    cpu_flat.add(xb)
    gpu_flat = faiss.index_cpu_to_gpu(res, dev_id, cpu_flat)

    # 2. IVFSQ8
    q_sq = faiss.IndexFlatL2(d)
    cpu_sq = faiss.IndexIVFScalarQuantizer(q_sq, d, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2)
    cpu_sq.quantizer.add(centroids)
    cpu_sq.train(xb)
    cpu_sq.add(xb)
    gpu_sq = faiss.index_cpu_to_gpu(res, dev_id, cpu_sq)

    # 3. IVFPQ
    gpu_pq = None
    M_pq = M_PQ_MAP.get(None, 16) # placeholder
    return gpu_flat, gpu_sq, res

# ----------------- 4A-2: ABLATION REDO -----------------
def run_ablation_phase4a(args):
    dev = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(dev)
    print(f"[Phase 4A Ablation] Running on {gpu_name} ({args.gpu_tag})", flush=True)

    out_csv = os.path.join(RES_ROOT, f"phase4a_ablation_{args.gpu_tag}.csv")
    rows = []

    for ds in args.datasets:
        ds_dir = os.path.join(DATA_ROOT, ds)
        xb = np.ascontiguousarray(np.load(os.path.join(ds_dir, "base.npy")).astype(np.float32))
        xq_raw = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy")).astype(np.float32))
        gt = np.load(os.path.join(ds_dir, "gt.npy"))
        dim = xb.shape[1]
        t_rule = choose_tile(dim)

        Q_eval, xq_eval_gpu, b_max, distinct_q, xq_batch_gpu = prepare_query_tensors(xq_raw, ds, dev)
        gt10 = gt[:Q_eval, :10]
        nprobe = NPROBE_MAP_095[ds]
        xq_b = xq_batch_gpu[:b_max].contiguous()

        ablation_specs = [
            ("B0_CM_FP32",                   dict(ts=t_rule, pc=0, pca=1, vb=0, mode=10, l1=0, rrm=0,   wr=1)),
            ("B1_GlobalSQ8_Uni32_Serial",    dict(ts=t_rule, pc=0, pca=0, vb=0, mode=2,  l1=0, rrm=0,   wr=0)),
            ("B2_PerClusterSQ8_Uni32_Serial",dict(ts=t_rule, pc=1, pca=0, vb=0, mode=2,  l1=0, rrm=0,   wr=0)),
            ("B3_PCA_PerClusterSQ8_Serial",  dict(ts=t_rule, pc=1, pca=1, vb=0, mode=2,  l1=0, rrm=0,   wr=0)),
            ("B4_Full_TRIDENT",              dict(ts=t_rule, pc=1, pca=1, vb=0, mode=2,  l1=0, rrm=0,   wr=1)),
            ("Full_minus_PCA",               dict(ts=t_rule, pc=1, pca=0, vb=0, mode=2,  l1=0, rrm=0,   wr=1)),
            ("Full_minus_PerClusterSQ8",     dict(ts=t_rule, pc=0, pca=1, vb=0, mode=2,  l1=0, rrm=0,   wr=1)),
            ("Full_minus_EarlyStop",         dict(ts=t_rule, pc=1, pca=1, vb=0, mode=0,  l1=0, rrm=100, wr=1)),
            ("Full_minus_WarpRefine",        dict(ts=t_rule, pc=1, pca=1, vb=0, mode=2,  l1=0, rrm=0,   wr=0)),
            ("Full_plus_VarBlk",             dict(ts=t_rule, pc=1, pca=1, vb=1, mode=2,  l1=0, rrm=0,   wr=1)),
            ("Full_plus_L1",                 dict(ts=t_rule, pc=1, pca=1, vb=0, mode=2,  l1=1, rrm=0,   wr=1)),
        ]

        for seed in args.seeds:
            centroids, list_ids, r_vec, mean64, R64, _ = load_or_train_clustering(ds, xb, args.nlist, seed)
            storage = SharedClusterStorage(xb, list_ids, r_vec, args.nlist, device="cuda:0")

            idx_ref = TridentV4Index(storage, centroids, mean64, R64, tile_size=t_rule, per_cluster_sq8=0, use_pca=1, var_block=0, max_batch=1024, device="cuda:0")
            cm_d_t, cm_i_t, _ = idx_ref.search(xq_eval_gpu, nprobe, mode=10, use_l1=0, rerank_m=0)
            torch.cuda.synchronize()
            cm_dists, cm_ids = cm_d_t.cpu().numpy(), cm_i_t.cpu().numpy()
            del cm_d_t, cm_i_t, idx_ref
            torch.cuda.empty_cache()

            for a_name, sc in ablation_specs:
                idx = TridentV4Index(
                    storage, centroids, mean64, R64,
                    tile_size=sc["ts"], per_cluster_sq8=sc["pc"], use_pca=sc["pca"], var_block=sc["vb"],
                    max_batch=1024, device="cuda:0"
                )
                td, ti, _ = idx.search(xq_eval_gpu, nprobe, mode=sc["mode"], use_l1=sc["l1"], rerank_m=sc["rrm"], warp_refine=sc["wr"])
                torch.cuda.synchronize()
                rec_val, cm_ov, cm_m100 = eval_accuracy_v4(ti.cpu().numpy(), td.cpu().numpy(), cm_ids, cm_dists, gt10)
                del td, ti

                e2e_ms = measure_latency_qps(
                    lambda: idx.search(xq_b, nprobe, mode=sc["mode"], use_l1=sc["l1"], rerank_m=sc["rrm"], warp_refine=sc["wr"], record_kernel_ms=False),
                    warmup=12, iters=50
                )
                k_acc = 0.0
                for _ in range(10):
                    _, _, km = idx.search(xq_b, nprobe, mode=sc["mode"], use_l1=sc["l1"], rerank_m=sc["rrm"], warp_refine=sc["wr"], record_kernel_ms=True)
                    k_acc += km
                kern_ms = k_acc / 10.0
                qps = (b_max * 1000.0) / max(e2e_ms, 1e-6)

                row = {
                    "gpu": args.gpu_tag,
                    "dataset": ds,
                    "dim": dim,
                    "nprobe": nprobe,
                    "seed": seed,
                    "batch": b_max,
                    "distinct_q": distinct_q,
                    "ablation_step": a_name,
                    "e2e_ms": round(e2e_ms, 4),
                    "kernel_ms": round(kern_ms, 4),
                    "qps": round(qps, 1),
                    "recall_at_10": round(rec_val, 4),
                    "cm_fp32_strict_ov_pct": round(cm_ov, 4),
                    "cm_fp32_match100_pct": round(cm_m100, 4),
                }
                rows.append(row)
                print(f"[Ablation {ds} s={seed} {a_name:30s}] e2e={e2e_ms:7.3f}ms kern={kern_ms:7.3f}ms QPS={qps:9.1f} StrictOv={cm_ov:.2f}%", flush=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] {out_csv}", flush=True)


# ----------------- 4A-3: RECALL@10=0.99 BENCHMARK -----------------
def run_recall99_phase4a(args):
    dev = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(dev)
    print(f"[Phase 4A Recall=0.99] Running on {gpu_name} ({args.gpu_tag})", flush=True)

    out_csv = os.path.join(RES_ROOT, f"phase4a_recall99_{args.gpu_tag}.csv")
    rows = []

    for ds in args.datasets:
        ds_dir = os.path.join(DATA_ROOT, ds)
        xb = np.ascontiguousarray(np.load(os.path.join(ds_dir, "base.npy")).astype(np.float32))
        xq_raw = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy")).astype(np.float32))
        gt = np.load(os.path.join(ds_dir, "gt.npy"))
        dim = xb.shape[1]
        t_rule = choose_tile(dim)

        Q_eval, xq_eval_gpu, b_max, distinct_q, xq_batch_gpu = prepare_query_tensors(xq_raw, ds, dev)
        gt10 = gt[:Q_eval, :10]
        nprobe = NPROBE_MAP_099[ds]
        xq_b = xq_batch_gpu[:b_max].contiguous()
        xq_b_cpu = xq_b.cpu().numpy()

        for seed in args.seeds:
            centroids, list_ids, r_vec, mean64, R64, _ = load_or_train_clustering(ds, xb, args.nlist, seed)
            storage = SharedClusterStorage(xb, list_ids, r_vec, args.nlist, device="cuda:0")

            gpu_flat, gpu_sq, _ = build_faiss_gpu_indexes(xb, centroids, list_ids, args.nlist, dev_id=0)

            # References
            idx_ref = TridentV4Index(storage, centroids, mean64, R64, tile_size=256, per_cluster_sq8=0, use_pca=1, var_block=0, max_batch=1024, device="cuda:0")
            cm_d_t, cm_i_t, _ = idx_ref.search(xq_eval_gpu, nprobe, mode=10, use_l1=0, rerank_m=0)
            torch.cuda.synchronize()
            cm_dists, cm_ids = cm_d_t.cpu().numpy(), cm_i_t.cpu().numpy()
            del cm_d_t, cm_i_t, idx_ref
            torch.cuda.empty_cache()

            idx_k2 = TridentV4Index(storage, centroids, mean64, R64, tile_size=t_rule, per_cluster_sq8=1, use_pca=1, var_block=0, max_batch=1024, device="cuda:0")
            idx_global_sq8 = TridentV4Index(storage, centroids, mean64, R64, tile_size=256, per_cluster_sq8=0, use_pca=1, var_block=0, max_batch=1024, device="cuda:0")

            method_cfgs = [
                ("Faiss_IVFFlat", "faiss_flat"),
                ("Faiss_IVFSQ8",  "faiss_sq"),
                ("CM_FP32_T256",  dict(mode=10, ts=256, rrm=0)),
                ("CM_FP32_T64",   dict(mode=10, ts=64,  rrm=0)),
                ("CM_SQ8_RR20",   dict(mode=0,  ts=256, rrm=20)),
                ("CM_SQ8_RR50",   dict(mode=0,  ts=256, rrm=50)),
                ("CM_SQ8_RR100",  dict(mode=0,  ts=256, rrm=100)),
                ("TRIDENT",       dict(mode=2,  ts=t_rule, rrm=0, wr=1, use_k2=True)),
            ]

            for m_name, cfg in method_cfgs:
                if m_name == "Faiss_IVFFlat":
                    gpu_flat.nprobe = nprobe
                    fd, fi = gpu_flat.search(xq_eval_gpu.cpu().numpy(), 10)
                    rec_val, cm_ov, _ = eval_accuracy_v4(fi, fd, cm_ids, cm_dists, gt10)
                    e2e_ms = measure_latency_qps(lambda: gpu_flat.search(xq_b_cpu, 10), warmup=12, iters=50)
                    kern_ms = -1.0
                elif m_name == "Faiss_IVFSQ8":
                    gpu_sq.nprobe = nprobe
                    fd, fi = gpu_sq.search(xq_eval_gpu.cpu().numpy(), 10)
                    rec_val, cm_ov, _ = eval_accuracy_v4(fi, fd, cm_ids, cm_dists, gt10)
                    e2e_ms = measure_latency_qps(lambda: gpu_sq.search(xq_b_cpu, 10), warmup=12, iters=50)
                    kern_ms = -1.0
                else:
                    active_idx = idx_k2 if cfg.get("use_k2", False) else idx_global_sq8
                    td, ti, _ = active_idx.search(
                        xq_eval_gpu, nprobe, mode=cfg["mode"], use_l1=0,
                        rerank_m=cfg["rrm"], tile_override=cfg["ts"], warp_refine=cfg.get("wr", 1)
                    )
                    torch.cuda.synchronize()
                    rec_val, cm_ov, _ = eval_accuracy_v4(ti.cpu().numpy(), td.cpu().numpy(), cm_ids, cm_dists, gt10)
                    del td, ti
                    e2e_ms = measure_latency_qps(
                        lambda: active_idx.search(
                            xq_b, nprobe, mode=cfg["mode"], use_l1=0,
                            rerank_m=cfg["rrm"], tile_override=cfg["ts"], warp_refine=cfg.get("wr", 1), record_kernel_ms=False
                        ),
                        warmup=12, iters=50
                    )
                    k_acc = 0.0
                    for _ in range(10):
                        _, _, km = active_idx.search(
                            xq_b, nprobe, mode=cfg["mode"], use_l1=0,
                            rerank_m=cfg["rrm"], tile_override=cfg["ts"], warp_refine=cfg.get("wr", 1), record_kernel_ms=True
                        )
                        k_acc += km
                    kern_ms = k_acc / 10.0

                qps = (b_max * 1000.0) / max(e2e_ms, 1e-6)
                row = {
                    "gpu": args.gpu_tag,
                    "dataset": ds,
                    "dim": dim,
                    "nprobe_99": nprobe,
                    "seed": seed,
                    "batch": b_max,
                    "distinct_q": distinct_q,
                    "method": m_name,
                    "e2e_ms": round(e2e_ms, 4),
                    "kernel_ms": round(kern_ms, 4) if kern_ms >= 0 else "N/A",
                    "qps": round(qps, 1),
                    "recall_at_10": round(rec_val, 4),
                    "cm_fp32_strict_ov_pct": round(cm_ov, 4),
                }
                rows.append(row)
                print(f"[R99 {ds} s={seed} {m_name:16s}] QPS={qps:9.1f} Rec={rec_val:.4f} Strict={cm_ov:.2f}%", flush=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] {out_csv}", flush=True)


# ----------------- 4A-4: QPS-RECALL CURVE (PARETO FRONTIER) -----------------
def run_qps_recall_phase4a(args):
    import faiss
    dev = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(dev)
    print(f"[Phase 4A QPS-Recall Curve] Running on {gpu_name} ({args.gpu_tag})", flush=True)

    out_csv = os.path.join(RES_ROOT, f"phase4a_qps_recall_{args.gpu_tag}.csv")
    rows = []

    nprobe_list_base = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    for ds in args.datasets:
        ds_dir = os.path.join(DATA_ROOT, ds)
        xb = np.ascontiguousarray(np.load(os.path.join(ds_dir, "base.npy")).astype(np.float32))
        xq_raw = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy")).astype(np.float32))
        gt = np.load(os.path.join(ds_dir, "gt.npy"))
        dim = xb.shape[1]
        t_rule = choose_tile(dim)

        Q_eval, xq_eval_gpu, b_max, distinct_q, xq_batch_gpu = prepare_query_tensors(xq_raw, ds, dev)
        gt10 = gt[:Q_eval, :10]
        xq_b = xq_batch_gpu[:b_max].contiguous()
        xq_b_cpu = xq_b.cpu().numpy()
        xq_eval_cpu = xq_eval_gpu.cpu().numpy()

        nprobes = list(nprobe_list_base)
        if ds == "glove100":
            nprobes.extend([384, 512])

        seeds_to_run = args.seeds[:3] # 3 seeds for curve as per plan

        for seed in seeds_to_run:
            centroids, list_ids, r_vec, mean64, R64, _ = load_or_train_clustering(ds, xb, args.nlist, seed)
            storage = SharedClusterStorage(xb, list_ids, r_vec, args.nlist, device="cuda:0")

            gpu_flat, gpu_sq, res = build_faiss_gpu_indexes(xb, centroids, list_ids, args.nlist, dev_id=0)

            # Build GPU IVFPQ
            m_pq = M_PQ_MAP[ds]
            q_pq = faiss.IndexFlatL2(dim)
            cpu_pq = faiss.IndexIVFPQ(q_pq, dim, args.nlist, m_pq, 8, faiss.METRIC_L2)
            cpu_pq.quantizer.add(centroids)
            cpu_pq.train(xb)
            cpu_pq.add(xb)
            gpu_pq = faiss.index_cpu_to_gpu(res, 0, cpu_pq)

            idx_k2 = TridentV4Index(storage, centroids, mean64, R64, tile_size=t_rule, per_cluster_sq8=1, use_pca=1, var_block=0, max_batch=1024, device="cuda:0")
            idx_global_sq8 = TridentV4Index(storage, centroids, mean64, R64, tile_size=t_rule, per_cluster_sq8=0, use_pca=1, var_block=0, max_batch=1024, device="cuda:0")

            xb_gpu = torch.from_numpy(xb).to(dev, dtype=torch.float32)

            for np_val in nprobes:
                # CM-FP32 Reference for this nprobe
                cm_d_t, cm_i_t, _ = idx_global_sq8.search(xq_eval_gpu, np_val, mode=10, use_l1=0, rerank_m=0, tile_override=t_rule)
                torch.cuda.synchronize()
                cm_dists, cm_ids = cm_d_t.cpu().numpy(), cm_i_t.cpu().numpy()
                del cm_d_t, cm_i_t

                # 1. Faiss_IVFFlat
                gpu_flat.nprobe = np_val
                fd, fi = gpu_flat.search(xq_eval_cpu, 10)
                rec, st, _ = eval_accuracy_v4(fi, fd, cm_ids, cm_dists, gt10)
                e2e_ms = measure_latency_qps(lambda: gpu_flat.search(xq_b_cpu, 10), warmup=5, iters=20)
                rows.append({"dataset": ds, "method": "Faiss_IVFFlat", "nprobe": np_val, "k_factor": 0, "seed": seed, "recall_at_10": round(rec, 4), "qps": round((b_max*1000.0)/e2e_ms, 1), "strict_min": round(st, 2)})

                # 2. Faiss_IVFSQ8
                gpu_sq.nprobe = np_val
                fd, fi = gpu_sq.search(xq_eval_cpu, 10)
                rec, st, _ = eval_accuracy_v4(fi, fd, cm_ids, cm_dists, gt10)
                e2e_ms = measure_latency_qps(lambda: gpu_sq.search(xq_b_cpu, 10), warmup=5, iters=20)
                rows.append({"dataset": ds, "method": "Faiss_IVFSQ8", "nprobe": np_val, "k_factor": 0, "seed": seed, "recall_at_10": round(rec, 4), "qps": round((b_max*1000.0)/e2e_ms, 1), "strict_min": round(st, 2)})

                # 3. Faiss_IVFPQ (unrefined)
                gpu_pq.nprobe = np_val
                fd, fi = gpu_pq.search(xq_eval_cpu, 10)
                rec, st, _ = eval_accuracy_v4(fi, fd, cm_ids, cm_dists, gt10)
                e2e_ms = measure_latency_qps(lambda: gpu_pq.search(xq_b_cpu, 10), warmup=5, iters=20)
                rows.append({"dataset": ds, "method": "Faiss_IVFPQ", "nprobe": np_val, "k_factor": 0, "seed": seed, "recall_at_10": round(rec, 4), "qps": round((b_max*1000.0)/e2e_ms, 1), "strict_min": round(st, 2)})

                # 4. Faiss_IVFPQ + Refine (k_factor in 2, 4, 8)
                for kf in [2, 4, 8]:
                    top_m_cand = 10 * kf
                    def run_ivfpq_refine(q_in):
                        D_c, I_c = gpu_pq.search(q_in.cpu().numpy(), top_m_cand)
                        I_c_gpu = torch.from_numpy(I_c).to(dev, dtype=torch.int64)
                        cand_vecs = xb_gpu[I_c_gpu] # [B, top_m_cand, D]
                        diff = q_in.unsqueeze(1) - cand_vecs
                        dists = (diff ** 2).sum(dim=-1)
                        top_d, top_sub_i = torch.topk(dists, 10, largest=False)
                        top_i = torch.gather(I_c_gpu, 1, top_sub_i)
                        return top_d, top_i

                    td, ti = run_ivfpq_refine(xq_eval_gpu)
                    rec, st, _ = eval_accuracy_v4(ti.cpu().numpy(), td.cpu().numpy(), cm_ids, cm_dists, gt10)
                    e2e_ms = measure_latency_qps(lambda: run_ivfpq_refine(xq_b), warmup=5, iters=20)
                    rows.append({"dataset": ds, "method": f"Faiss_IVFPQ_Refine_k{kf}", "nprobe": np_val, "k_factor": kf, "seed": seed, "recall_at_10": round(rec, 4), "qps": round((b_max*1000.0)/e2e_ms, 1), "strict_min": round(st, 2)})

                # 5. CM-FP32
                e2e_ms = measure_latency_qps(lambda: idx_global_sq8.search(xq_b, np_val, mode=10, use_l1=0, rerank_m=0, tile_override=t_rule), warmup=5, iters=20)
                rec, st, _ = eval_accuracy_v4(cm_ids, cm_dists, cm_ids, cm_dists, gt10)
                rows.append({"dataset": ds, "method": "CM_FP32", "nprobe": np_val, "k_factor": 0, "seed": seed, "recall_at_10": round(rec, 4), "qps": round((b_max*1000.0)/e2e_ms, 1), "strict_min": 100.0})

                # 6. CM_SQ8_RR50
                td, ti, _ = idx_global_sq8.search(xq_eval_gpu, np_val, mode=0, use_l1=0, rerank_m=50, tile_override=t_rule)
                rec, st, _ = eval_accuracy_v4(ti.cpu().numpy(), td.cpu().numpy(), cm_ids, cm_dists, gt10)
                e2e_ms = measure_latency_qps(lambda: idx_global_sq8.search(xq_b, np_val, mode=0, use_l1=0, rerank_m=50, tile_override=t_rule), warmup=5, iters=20)
                rows.append({"dataset": ds, "method": "CM_SQ8_RR50", "nprobe": np_val, "k_factor": 0, "seed": seed, "recall_at_10": round(rec, 4), "qps": round((b_max*1000.0)/e2e_ms, 1), "strict_min": round(st, 2)})

                # 7. TRIDENT
                td, ti, _ = idx_k2.search(xq_eval_gpu, np_val, mode=2, use_l1=0, rerank_m=0, tile_override=t_rule, warp_refine=1)
                rec, st, _ = eval_accuracy_v4(ti.cpu().numpy(), td.cpu().numpy(), cm_ids, cm_dists, gt10)
                e2e_ms = measure_latency_qps(lambda: idx_k2.search(xq_b, np_val, mode=2, use_l1=0, rerank_m=0, tile_override=t_rule, warp_refine=1), warmup=5, iters=20)
                rows.append({"dataset": ds, "method": "TRIDENT", "nprobe": np_val, "k_factor": 0, "seed": seed, "recall_at_10": round(rec, 4), "qps": round((b_max*1000.0)/e2e_ms, 1), "strict_min": round(st, 2)})
                print(f"[Curve {ds} np={np_val:3d} TRIDENT] QPS={(b_max*1000.0)/e2e_ms:9.1f} Rec={rec:.4f} Strict={st:.2f}%", flush=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] {out_csv}", flush=True)


# ----------------- 4A-5: UNIFIED BUILD TIME & MEMORY -----------------
def run_build_mem_phase4a(args):
    import faiss
    dev = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(dev)
    print(f"[Phase 4A Build & Memory] Running on {gpu_name} ({args.gpu_tag})", flush=True)

    out_csv = os.path.join(RES_ROOT, f"phase4a_build_mem_{args.gpu_tag}.csv")
    rows = []

    for ds in args.datasets:
        ds_dir = os.path.join(DATA_ROOT, ds)
        xb = np.ascontiguousarray(np.load(os.path.join(ds_dir, "base.npy")).astype(np.float32))
        N, dim = xb.shape
        t_rule = choose_tile(dim)
        seed = 0

        # 1. K-Means clustering (measured cleanly once per dataset)
        t0 = time.perf_counter()
        q_km = faiss.IndexFlatL2(dim)
        idx_km = faiss.IndexIVFFlat(q_km, dim, args.nlist, faiss.METRIC_L2)
        idx_km.cp.seed = seed + 1000
        idx_km.train(xb)
        kmeans_s = time.perf_counter() - t0

        centroids, list_ids, r_vec, mean64, R64, _ = load_or_train_clustering(ds, xb, args.nlist, seed)

        # 2. Faiss IVFFlat Add
        res = faiss.StandardGpuResources()
        torch.cuda.reset_peak_memory_stats()
        mem0 = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        q_flat = faiss.IndexFlatL2(dim)
        cpu_flat = faiss.IndexIVFFlat(q_flat, dim, args.nlist, faiss.METRIC_L2)
        cpu_flat.quantizer.add(centroids)
        cpu_flat.is_trained = True
        cpu_flat.add(xb)
        gpu_flat = faiss.index_cpu_to_gpu(res, 0, cpu_flat)
        torch.cuda.synchronize()
        add_flat_s = time.perf_counter() - t0
        mem_flat_mib = (torch.cuda.max_memory_allocated() - mem0) / (1024.0 * 1024.0)
        del gpu_flat, cpu_flat
        torch.cuda.empty_cache()

        # 3. Faiss IVFSQ8 Add
        torch.cuda.reset_peak_memory_stats()
        mem0 = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        q_sq = faiss.IndexFlatL2(dim)
        cpu_sq = faiss.IndexIVFScalarQuantizer(q_sq, dim, args.nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2)
        cpu_sq.quantizer.add(centroids)
        cpu_sq.train(xb)
        cpu_sq.add(xb)
        gpu_sq = faiss.index_cpu_to_gpu(res, 0, cpu_sq)
        torch.cuda.synchronize()
        add_sq_s = time.perf_counter() - t0
        mem_sq_mib = (torch.cuda.max_memory_allocated() - mem0) / (1024.0 * 1024.0)
        del gpu_sq, cpu_sq
        torch.cuda.empty_cache()

        # 4. CM-FP32 (Pack only)
        torch.cuda.reset_peak_memory_stats()
        mem0 = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        storage = SharedClusterStorage(xb, list_ids, r_vec, args.nlist, device="cuda:0")
        torch.cuda.synchronize()
        pack_cm_s = time.perf_counter() - t0
        mem_cm_mib = (torch.cuda.max_memory_allocated() - mem0) / (1024.0 * 1024.0)

        # 5. TRIDENT (PCA + SQ8 + Pack)
        torch.cuda.reset_peak_memory_stats()
        mem0 = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        idx_trident = TridentV4Index(storage, centroids, mean64, R64, tile_size=t_rule, per_cluster_sq8=1, use_pca=1, var_block=0, max_batch=1024, device="cuda:0")
        torch.cuda.synchronize()
        trident_extra_s = pack_cm_s + (time.perf_counter() - t0)
        mem_trident_mib = (torch.cuda.max_memory_allocated() - mem0) / (1024.0 * 1024.0)
        del idx_trident, storage
        torch.cuda.empty_cache()

        base_bpv = dim * 4 # IVFFlat bytes per vector
        sq_bpv = dim * 1
        cm_bpv = dim * 4
        d_pad = ((dim + 31) // 32) * 32
        trident_bpv = dim * 4 + d_pad * 1

        rows.append({"gpu": args.gpu_tag, "dataset": ds, "dim": dim, "method": "Faiss_IVFFlat", "kmeans_s": round(kmeans_s, 3), "extra_s": round(add_flat_s, 3), "total_s": round(kmeans_s + add_flat_s, 3), "bytes_per_vector": base_bpv, "ratio_vs_ivfflat": 1.00, "index_vram_mib": round(mem_flat_mib, 1)})
        rows.append({"gpu": args.gpu_tag, "dataset": ds, "dim": dim, "method": "Faiss_IVFSQ8",  "kmeans_s": round(kmeans_s, 3), "extra_s": round(add_sq_s, 3), "total_s": round(kmeans_s + add_sq_s, 3), "bytes_per_vector": sq_bpv, "ratio_vs_ivfflat": round(sq_bpv/base_bpv, 2), "index_vram_mib": round(mem_sq_mib, 1)})
        rows.append({"gpu": args.gpu_tag, "dataset": ds, "dim": dim, "method": "CM_FP32",        "kmeans_s": round(kmeans_s, 3), "extra_s": round(pack_cm_s, 3), "total_s": round(kmeans_s + pack_cm_s, 3), "bytes_per_vector": cm_bpv, "ratio_vs_ivfflat": 1.00, "index_vram_mib": round(mem_cm_mib, 1)})
        rows.append({"gpu": args.gpu_tag, "dataset": ds, "dim": dim, "method": "TRIDENT",        "kmeans_s": round(kmeans_s, 3), "extra_s": round(trident_extra_s, 3), "total_s": round(kmeans_s + trident_extra_s, 3), "bytes_per_vector": trident_bpv, "ratio_vs_ivfflat": round(trident_bpv/base_bpv, 2), "index_vram_mib": round(mem_trident_mib, 1)})
        print(f"[BuildMem {ds}] TRIDENT total={kmeans_s + trident_extra_s:.2f}s (kmeans={kmeans_s:.2f}s, extra={trident_extra_s:.2f}s) VRAM={mem_trident_mib:.1f}MiB ({trident_bpv/base_bpv:.2f}x IVFFlat)", flush=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] {out_csv}", flush=True)


# ----------------- 4A-6: PRUNING STATISTICS -----------------
def run_prune_stats_phase4a(args):
    dev = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(dev)
    print(f"[Phase 4A Pruning Statistics] Running on {gpu_name} ({args.gpu_tag})", flush=True)

    out_csv = os.path.join(RES_ROOT, f"phase4a_prune_stats.csv")
    rows = []

    for ds in args.datasets:
        ds_dir = os.path.join(DATA_ROOT, ds)
        xb = np.ascontiguousarray(np.load(os.path.join(ds_dir, "base.npy")).astype(np.float32))
        xq_raw = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy")).astype(np.float32))
        dim = xb.shape[1]
        t_rule = choose_tile(dim)
        seed = 0

        centroids, list_ids, r_vec, mean64, R64, _ = load_or_train_clustering(ds, xb, args.nlist, seed)
        storage = SharedClusterStorage(xb, list_ids, r_vec, args.nlist, device="cuda:0")
        idx_k2 = TridentV4Index(storage, centroids, mean64, R64, tile_size=t_rule, per_cluster_sq8=1, use_pca=1, var_block=0, max_batch=1024, device="cuda:0")

        Q_eval, xq_eval_gpu, b_max, distinct_q, xq_batch_gpu = prepare_query_tensors(xq_raw, ds, dev)
        xq_b = xq_batch_gpu[:b_max].contiguous()

        for target_rec, np_val in [("Recall_0.95", NPROBE_MAP_095[ds]), ("Recall_0.99", NPROBE_MAP_099[ds])]:
            stats_buf = torch.zeros(4, dtype=torch.int64, device=dev)
            idx_k2.search(xq_b, np_val, mode=2, use_l1=0, rerank_m=0, tile_override=t_rule, warp_refine=1, stats_buf=stats_buf)
            torch.cuda.synchronize()

            s_np = stats_buf.cpu().numpy()
            cand_total = int(s_np[0])
            sum_dims = int(s_np[1])
            survivor_total = int(s_np[2])
            refined_valid = int(s_np[3])

            avg_dims = (sum_dims / float(max(cand_total, 1)))
            prune_dim_pct = (1.0 - avg_dims / float(dim)) * 100.0
            surv_ratio = (survivor_total / float(max(cand_total, 1))) * 100.0
            ref_ratio = (refined_valid / float(max(survivor_total, 1))) * 100.0

            row = {
                "dataset": ds,
                "dim": dim,
                "target_recall": target_rec,
                "nprobe": np_val,
                "batch": b_max,
                "cand_total": cand_total,
                "avg_dims_scanned": round(avg_dims, 1),
                "prune_dim_pct": round(prune_dim_pct, 2),
                "survivor_total": survivor_total,
                "survivor_ratio_pct": round(surv_ratio, 2),
                "refined_valid_total": refined_valid,
                "refine_ratio_pct": round(ref_ratio, 2),
            }
            rows.append(row)
            print(f"[PruneStats {ds} {target_rec}] cand={cand_total} avg_dims={avg_dims:.1f}/{dim} (pruned {prune_dim_pct:.1f}%) surv_ratio={surv_ratio:.2f}%", flush=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] {out_csv}", flush=True)


# ----------------- 4A-7: TRIDENT-AUTO SCHEDULER -----------------
def run_auto_phase4a(args):
    dev = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(dev)
    print(f"[Phase 4A TRIDENT-Auto] Running on {gpu_name} ({args.gpu_tag})", flush=True)

    calib_csv = os.path.join(RES_ROOT, f"phase4a_auto_calib_{args.gpu_tag}.csv")
    auto_csv = os.path.join(RES_ROOT, f"phase4a_auto_{args.gpu_tag}.csv")

    # Step 1: Calibrate W0 on 1000 sampled train vectors from sift1m and glove100
    print("--- Calibrating W0 threshold on sampled train queries (sift1m & glove100) ---")
    calib_rows = []
    crossover_works = []

    for ds in ["sift1m", "glove100"]:
        ds_dir = os.path.join(DATA_ROOT, ds)
        xb = np.ascontiguousarray(np.load(os.path.join(ds_dir, "base.npy")).astype(np.float32))
        N, dim = xb.shape
        t_rule = choose_tile(dim)

        np.random.seed(999)
        idx_train = np.random.choice(N, 1000, replace=False)
        xq_calib_gpu = torch.from_numpy(xb[idx_train]).to(dev, dtype=torch.float32).contiguous()

        centroids, list_ids, r_vec, mean64, R64, _ = load_or_train_clustering(ds, xb, args.nlist, seed=0)
        storage = SharedClusterStorage(xb, list_ids, r_vec, args.nlist, device="cuda:0")
        idx_k2 = TridentV4Index(storage, centroids, mean64, R64, tile_size=t_rule, per_cluster_sq8=1, use_pca=1, var_block=0, max_batch=1024, device="cuda:0")

        avg_list_len = N / float(args.nlist)

        for b_test in [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            q_slice = xq_calib_gpu[:b_test].contiguous()
            np_test = NPROBE_MAP_095[ds]
            work = float(b_test * np_test * avg_list_len * dim)

            ms_cm = measure_latency_qps(lambda: idx_k2.search(q_slice, np_test, mode=10, use_l1=0, rerank_m=0, tile_override=t_rule), warmup=5, iters=20)
            ms_k2 = measure_latency_qps(lambda: idx_k2.search(q_slice, np_test, mode=2,  use_l1=0, rerank_m=0, tile_override=t_rule, warp_refine=1), warmup=5, iters=20)
            qps_cm = (b_test * 1000.0) / ms_cm
            qps_k2 = (b_test * 1000.0) / ms_k2

            calib_rows.append({"dataset": ds, "batch": b_test, "nprobe": np_test, "work": work, "qps_cm": round(qps_cm, 1), "qps_k2": round(qps_k2, 1), "k2_faster": bool(qps_k2 >= qps_cm)})
            if qps_k2 >= qps_cm:
                crossover_works.append(work)

    with open(calib_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(calib_rows[0].keys()))
        writer.writeheader()
        writer.writerows(calib_rows)

    W0 = min(crossover_works) if crossover_works else 5.0e7
    print(f"[Calibrated W0] = {W0:.2e} (saved to {calib_csv})", flush=True)

    # Step 2: Multi-batch Benchmark (B in 1, 8, 32, 128, B_max) × 5 seeds
    print("--- Running TRIDENT-Auto Multi-Batch Benchmark ---")
    auto_rows = []

    for ds in args.datasets:
        ds_dir = os.path.join(DATA_ROOT, ds)
        xb = np.ascontiguousarray(np.load(os.path.join(ds_dir, "base.npy")).astype(np.float32))
        xq_raw = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy")).astype(np.float32))
        gt = np.load(os.path.join(ds_dir, "gt.npy"))
        N, dim = xb.shape
        t_rule = choose_tile(dim)
        nprobe = NPROBE_MAP_095[ds]
        avg_list_len = N / float(args.nlist)

        Q_eval, xq_eval_gpu, b_max, distinct_q, xq_batch_gpu = prepare_query_tensors(xq_raw, ds, dev)
        batches = [1, 8, 32, 128, b_max]

        for seed in args.seeds:
            centroids, list_ids, r_vec, mean64, R64, _ = load_or_train_clustering(ds, xb, args.nlist, seed)
            storage = SharedClusterStorage(xb, list_ids, r_vec, args.nlist, device="cuda:0")
            gpu_flat, _, _ = build_faiss_gpu_indexes(xb, centroids, list_ids, args.nlist, dev_id=0)
            idx_k2 = TridentV4Index(storage, centroids, mean64, R64, tile_size=t_rule, per_cluster_sq8=1, use_pca=1, var_block=0, max_batch=1024, device="cuda:0")

            for b_cur in batches:
                xq_cur = xq_batch_gpu[:b_cur].contiguous()
                xq_cur_cpu = xq_cur.cpu().numpy()
                work = float(b_cur * nprobe * avg_list_len * dim)

                # Faiss IVFFlat
                gpu_flat.nprobe = nprobe
                ms_faiss = measure_latency_qps(lambda: gpu_flat.search(xq_cur_cpu, 10), warmup=8, iters=30)
                qps_faiss = (b_cur * 1000.0) / ms_faiss

                # CM-FP32
                ms_cm = measure_latency_qps(lambda: idx_k2.search(xq_cur, nprobe, mode=10, use_l1=0, rerank_m=0, tile_override=t_rule), warmup=8, iters=30)
                qps_cm = (b_cur * 1000.0) / ms_cm

                # TRIDENT (pure K2)
                ms_trident = measure_latency_qps(lambda: idx_k2.search(xq_cur, nprobe, mode=2, use_l1=0, rerank_m=0, tile_override=t_rule, warp_refine=1), warmup=8, iters=30)
                qps_trident = (b_cur * 1000.0) / ms_trident

                # TRIDENT-Auto
                mode_auto = 10 if work < W0 else 2
                chosen_name = "CM_FP32" if mode_auto == 10 else "TRIDENT"
                ms_auto = measure_latency_qps(lambda: idx_k2.search(xq_cur, nprobe, mode=mode_auto, use_l1=0, rerank_m=0, tile_override=t_rule, warp_refine=1), warmup=8, iters=30)
                qps_auto = (b_cur * 1000.0) / ms_auto

                best_base = max(qps_faiss, qps_cm)
                ratio_vs_max_base = qps_auto / float(best_base)

                auto_rows.append({
                    "gpu": args.gpu_tag, "dataset": ds, "dim": dim, "batch": b_cur, "seed": seed,
                    "work": work, "chosen_path": chosen_name,
                    "qps_faiss": round(qps_faiss, 1), "qps_cm_fp32": round(qps_cm, 1),
                    "qps_trident": round(qps_trident, 1), "qps_trident_auto": round(qps_auto, 1),
                    "ratio_vs_max_base": round(ratio_vs_max_base, 3)
                })
                print(f"[Auto {ds} B={b_cur:4d}] path={chosen_name:7s} AutoQPS={qps_auto:9.1f} (vs Faiss={qps_faiss:9.1f}, CM={qps_cm:9.1f}) Ratio={ratio_vs_max_base:.2f}x", flush=True)
            del gpu_flat, idx_k2, storage
            torch.cuda.empty_cache()

        del xb, xq_raw, gt, xq_eval_gpu, xq_batch_gpu
        torch.cuda.empty_cache()

    with open(auto_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(auto_rows[0].keys()))
        writer.writeheader()
        writer.writerows(auto_rows)
    print(f"[SAVED] {auto_csv}", flush=True)


# ----------------- CLI DISPATCHER -----------------
# ----------------- NCU SINGLE PROFILING TARGET -----------------
def run_ncu_single_phase4a(args):
    import faiss
    dev = torch.device("cuda:0")
    ds = args.datasets[0]
    seed = args.seeds[0]
    ds_dir = os.path.join(DATA_ROOT, ds)
    xb = np.ascontiguousarray(np.load(os.path.join(ds_dir, "base.npy")).astype(np.float32))
    xq_raw = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy")).astype(np.float32))
    dim = xb.shape[1]
    t_rule = choose_tile(dim)
    _, _, b_max, _, xq_b = prepare_query_tensors(xq_raw, ds, dev)
    nprobe = NPROBE_MAP_095[ds]
    centroids, list_ids, r_vec, mean64, R64, _ = load_or_train_clustering(ds, xb, args.nlist, seed)

    t = args.ncu_target
    if t == "Faiss_IVFFlat":
        gpu_flat, _, _ = build_faiss_gpu_indexes(xb, centroids, list_ids, args.nlist, dev_id=0)
        gpu_flat.nprobe = nprobe
        xq_cpu = xq_b.cpu().numpy()
        gpu_flat.search(xq_cpu, 10)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        gpu_flat.search(xq_cpu, 10)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return

    storage = SharedClusterStorage(xb, list_ids, r_vec, args.nlist, device="cuda:0")
    ablation_map = {
        "B0_CM_FP32":                   dict(ts=t_rule, pc=0, pca=1, vb=0, mode=10, l1=0, rrm=0,   wr=1),
        "B1_GlobalSQ8_Uni32_Serial":    dict(ts=t_rule, pc=0, pca=0, vb=0, mode=2,  l1=0, rrm=0,   wr=0),
        "B2_PerClusterSQ8_Uni32_Serial":dict(ts=t_rule, pc=1, pca=0, vb=0, mode=2,  l1=0, rrm=0,   wr=0),
        "B3_PCA_PerClusterSQ8_Serial":  dict(ts=t_rule, pc=1, pca=1, vb=0, mode=2,  l1=0, rrm=0,   wr=0),
        "B4_Full_TRIDENT":              dict(ts=t_rule, pc=1, pca=1, vb=0, mode=2,  l1=0, rrm=0,   wr=1),
        "Full_minus_PCA":               dict(ts=t_rule, pc=1, pca=0, vb=0, mode=2,  l1=0, rrm=0,   wr=1),
        "Full_minus_PerClusterSQ8":     dict(ts=t_rule, pc=0, pca=1, vb=0, mode=2,  l1=0, rrm=0,   wr=1),
        "Full_minus_EarlyStop":         dict(ts=t_rule, pc=1, pca=1, vb=0, mode=0,  l1=0, rrm=100, wr=1),
        "Full_minus_WarpRefine":        dict(ts=t_rule, pc=1, pca=1, vb=0, mode=2,  l1=0, rrm=0,   wr=0),
        "Full_plus_VarBlk":             dict(ts=t_rule, pc=1, pca=1, vb=1, mode=2,  l1=0, rrm=0,   wr=1),
        "Full_plus_L1":                 dict(ts=t_rule, pc=1, pca=1, vb=0, mode=2,  l1=1, rrm=0,   wr=1),
    }
    c = ablation_map[t]
    idx = TridentV4Index(
        storage, centroids, mean64, R64,
        tile_size=c["ts"], per_cluster_sq8=c["pc"], use_pca=c["pca"], var_block=c["vb"],
        max_batch=1024, device="cuda:0"
    )
    idx.search(xq_b, nprobe, mode=c["mode"], use_l1=c["l1"], rerank_m=c["rrm"], warp_refine=c["wr"])
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    idx.search(xq_b, nprobe, mode=c["mode"], use_l1=c["l1"], rerank_m=c["rrm"], warp_refine=c["wr"])
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

def main():
    parser = argparse.ArgumentParser(description="TRIDENT Phase 4A Suite")
    parser.add_argument("--mode", type=str, required=True, choices=["ablation", "recall99", "qps_recall", "build_mem", "prune_stats", "auto", "ncu_single", "all"])
    parser.add_argument("--ncu_target", type=str, default="")
    parser.add_argument("--gpu_tag", type=str, default="A100")
    parser.add_argument("--datasets", nargs="+", default=["gist1m", "bge1024", "sift1m", "msong", "glove100", "nuswide"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--nlist", type=int, default=1024)
    args = parser.parse_args()

    if args.mode == "ncu_single":
        run_ncu_single_phase4a(args)
    if args.mode == "ablation" or args.mode == "all":
        run_ablation_phase4a(args)
    if args.mode == "recall99" or args.mode == "all":
        run_recall99_phase4a(args)
    if args.mode == "qps_recall" or args.mode == "all":
        run_qps_recall_phase4a(args)
    if args.mode == "build_mem" or args.mode == "all":
        run_build_mem_phase4a(args)
    if args.mode == "prune_stats" or args.mode == "all":
        run_prune_stats_phase4a(args)
    if args.mode == "auto" or args.mode == "all":
        run_auto_phase4a(args)

if __name__ == "__main__":
    main()
