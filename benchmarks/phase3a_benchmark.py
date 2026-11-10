#!/usr/bin/env python3
"""
Phase 3A Comprehensive Benchmark & Real GPU Kernel Ablation (PLAN_V4 §1).
Covers:
  1. Fair Cluster-Major Baselines on identical IVF clustering (nlist=1024) and nprobe:
     - CM_FP32_T256 / CM_FP32_T64 (exact cluster-major FP32 full scan, no early-stop)
     - CM_SQ8_RR20 / CM_SQ8_RR50 / CM_SQ8_RR100 (K0 + top-m exact FP32 reranking)
     - Faiss_IVFFlat, Faiss_IVFSQ8, Faiss_IVFPQ
  2. Real GPU Kernel Ablation (A0 -> A1 -> A2 -> A3 -> A4 -> A5 -> A6_T256 / A6_T64)
  3. Fused C++/CUDA Frontend (1 cuBLAS addmm_out + CUDA top-nprobe + shared-memory Counting Sort)
  4. Per-Cluster SQ8 + Warp-Parallel FP32 Survivor Refinement over cluster-contiguous xb_sorted
  5. 5 seeds (0..4), warmup=12 (>=10), iters=50 (>=50), distinct query accounting (nuswide B_max=200)
  6. Batch size curve B in {1, 8, 32, 128, 512, 1024}, Recall@10=0.99 recalibration, build time & VRAM footprint
  7. Nsight Compute (--ncu_target) hardware dram__bytes_read.sum profiling
"""
import os, sys, time, math, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.cpp_extension import load
import faiss
import faiss.contrib.torch_utils

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.9")
os.environ["PATH"] = "/usr/local/cuda-13/bin:" + os.environ.get("PATH", "")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.expanduser("~/trident/data")
RES_ROOT = os.path.expanduser("~/trident/results/phase3a")
CACHE_ROOT = os.path.expanduser("~/trident/cache_v3")
os.makedirs(RES_ROOT, exist_ok=True)
os.makedirs(CACHE_ROOT, exist_ok=True)

print("[JIT] Loading trident_v4_kernels.cu ...", flush=True)
tri_v4_ext = load(
    name="trident_v4_ext",
    sources=[os.path.join(SCRIPT_DIR, "trident_v4_kernels.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)
print("[JIT] trident_v4_ext loaded successfully.", flush=True)

NPROBE_MAP_095 = {
    "bge1024": 14,
    "gist1m": 56,
    "sift1m": 24,
    "msong": 10,
    "glove100": 128,
    "nuswide": 3,
}

PQ_M_MAP = {
    "bge1024": 32,
    "gist1m": 48,
    "sift1m": 16,
    "msong": 20,
    "glove100": 20,
    "nuswide": 20,
}


def compute_pca(xb: np.ndarray, seed: int, sample_n: int = 100000):
    t0 = time.perf_counter()
    N, D = xb.shape
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, size=min(N, sample_n), replace=False) if N > sample_n else np.arange(N)
    sub = xb[idx].astype(np.float64)
    mean64 = sub.mean(axis=0)
    centered = sub - mean64
    cov = (centered.T @ centered) / max(len(sub) - 1, 1)
    _, eigvecs = np.linalg.eigh(cov)
    R64 = eigvecs[:, ::-1].copy()
    pca_s = time.perf_counter() - t0
    return mean64, R64, pca_s


def train_shared_ivf_centroids(xb: np.ndarray, nlist: int, seed: int):
    N, D = xb.shape
    clus = faiss.Clustering(D, nlist)
    clus.niter = 20
    clus.seed = seed
    clus.max_points_per_centroid = 256
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.device = 0
    flat_gpu = faiss.GpuIndexFlatL2(res, D, cfg)
    clus.train(xb, flat_gpu)
    centroids = faiss.vector_float_to_array(clus.centroids).reshape(nlist, D).copy()
    return centroids


def load_or_train_clustering(ds: str, xb: np.ndarray, nlist: int, seed: int):
    cache_file = os.path.join(CACHE_ROOT, f"{ds}_nlist{nlist}_s{seed}.npz")
    if os.path.exists(cache_file):
        t0 = time.perf_counter()
        cdata = np.load(cache_file)
        centroids = cdata["centroids"]
        list_ids = cdata["list_ids"]
        r_vec = cdata["r_vec"]
        mean64 = cdata["mean64"]
        R64 = cdata["R64"]
        pca_s = float(cdata["pca_s"]) if "pca_s" in cdata else 0.25
        return centroids, list_ids, r_vec, mean64, R64, pca_s

    centroids = train_shared_ivf_centroids(xb, nlist=nlist, seed=seed)
    quant_tmp = faiss.IndexFlatL2(xb.shape[1])
    quant_tmp.add(centroids)
    dists_sq, lids = quant_tmp.search(xb, 1)
    list_ids = lids.reshape(-1)
    r_vec = np.sqrt(np.maximum(dists_sq.reshape(-1), 0.0)).astype(np.float32)
    mean64, R64, pca_s = compute_pca(xb, seed=seed)
    tmp_file = f"{cache_file}.tmp.{os.getpid()}.npz"
    np.savez(tmp_file, centroids=centroids, list_ids=list_ids, r_vec=r_vec, mean64=mean64, R64=R64, pca_s=pca_s)
    os.replace(tmp_file, cache_file)
    return centroids, list_ids, r_vec, mean64, R64, pca_s


def build_faiss_gpu_single(xb: np.ndarray, centroids: np.ndarray, nlist: int, kind: str, ds: str, seed: int):
    """
    Builds a single Faiss-GPU index ('flat', 'sq8', or 'pq') and measures build time & GPU VRAM delta.
    """
    N, D = xb.shape
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    vram_before = torch.cuda.memory_allocated()

    res = faiss.StandardGpuResources()
    res.setTempMemory(256 * 1024 * 1024)

    t0 = time.perf_counter()
    quant = faiss.IndexFlatL2(D)
    quant.add(centroids)

    if kind == "flat":
        cpu_idx = faiss.IndexIVFFlat(quant, D, nlist, faiss.METRIC_L2)
        cpu_idx.is_trained = True
        cpu_idx.add(xb)
        gpu_idx = faiss.index_cpu_to_gpu(res, 0, cpu_idx)
        theo_mib = (N * (D * 4 + 8) + nlist * D * 4) / (1024.0 ** 2)
    elif kind == "sq8":
        cpu_idx = faiss.IndexIVFScalarQuantizer(quant, D, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2)
        cpu_idx.train(xb[:min(N, 100000)])
        cpu_idx.add(xb)
        gpu_idx = faiss.index_cpu_to_gpu(res, 0, cpu_idx)
        theo_mib = (N * (D + 8) + nlist * D * 4) / (1024.0 ** 2)
    elif kind == "pq":
        m_pq = PQ_M_MAP[ds]
        pq_cache = os.path.join(CACHE_ROOT, f"{ds}_ivfpq_m{m_pq}_nlist{nlist}_s{seed}.faiss")
        if os.path.exists(pq_cache):
            cpu_idx = faiss.read_index(pq_cache)
        else:
            cpu_idx = faiss.IndexIVFPQ(quant, D, nlist, m_pq, 8, faiss.METRIC_L2)
            rng = np.random.RandomState(seed)
            sub_idx = rng.choice(N, size=min(N, 50000), replace=False)
            cpu_idx.train(xb[sub_idx])
            cpu_idx.add(xb)
            tmp_pq = f"{pq_cache}.tmp.{os.getpid()}"
            faiss.write_index(cpu_idx, tmp_pq)
            os.replace(tmp_pq, pq_cache)
        gpu_idx = faiss.index_cpu_to_gpu(res, 0, cpu_idx)
        theo_mib = (N * (m_pq + 8) + nlist * D * 4) / (1024.0 ** 2)
    else:
        raise ValueError(kind)

    torch.cuda.synchronize()
    build_s = time.perf_counter() - t0
    return res, gpu_idx, build_s, theo_mib


class SharedClusterStorage:
    """
    Holds cluster-sorted contiguous FP32 vectors xb_sorted [N, D] and orig_ids_sorted [N] on GPU.
    Shared between CM-FP32 and TRIDENT indexes so we never duplicate the 4GB FP32 base vectors in VRAM!
    """
    def __init__(self, xb: np.ndarray, list_ids: np.ndarray, r_vec: np.ndarray, nlist: int, device="cuda:0"):
        self.dev = torch.device(device)
        self.N, self.D = xb.shape
        self.nlist = nlist

        t0 = time.perf_counter()
        self.order_global = np.lexsort((r_vec, list_ids)).astype(np.int64)
        sorted_cids = list_ids[self.order_global]
        counts = np.bincount(sorted_cids, minlength=nlist)
        self.offsets = np.zeros(nlist + 1, dtype=np.int64)
        np.cumsum(counts, out=self.offsets[1:])
        self.r_sorted = r_vec[self.order_global]

        xb_sorted_np = np.ascontiguousarray(xb[self.order_global], dtype=np.float32)
        orig_ids_np = self.order_global.astype(np.int32)
        vec_off_np = self.offsets.astype(np.int32)

        self.xb_sorted = torch.from_numpy(xb_sorted_np).to(self.dev, dtype=torch.float32).contiguous()
        self.orig_ids_sorted = torch.from_numpy(orig_ids_np).to(self.dev, dtype=torch.int32).contiguous()
        self.cluster_vec_off = torch.from_numpy(vec_off_np).to(self.dev, dtype=torch.int32).contiguous()
        torch.cuda.synchronize()
        self.sort_s = time.perf_counter() - t0


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
        var_block: int = 1,
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

        t_quant0 = time.perf_counter()
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

        # Fused projection matrices:
        # W_coarse_only: [D, nlist] = -2 * centroids^T, b_coarse_only: [1, nlist] = ||centroids||^2
        # W_combined: [D, nlist + D_pad] = [-2 * centroids^T | R_f32 | 0], b_combined: [1, nlist + D_pad] = [||c||^2 | -mean@R | 0]
        self.W_coarse_only = (-2.0 * centroids_g.t()).contiguous()
        self.b_coarse_only = c_norm_sq.unsqueeze(0).contiguous()

        total_cols = self.nlist + self.D_pad
        self.W_combined = torch.zeros((self.D, total_cols), device=self.dev, dtype=torch.float32)
        self.b_combined = torch.zeros((1, total_cols), device=self.dev, dtype=torch.float32)
        self.W_combined[:, :self.nlist] = self.W_coarse_only
        self.W_combined[:, self.nlist:self.nlist + self.D] = R_f32
        self.b_combined[0, :self.nlist] = c_norm_sq
        mean_rot = torch.matmul(mean_f32.unsqueeze(0), R_f32).squeeze(0)
        self.b_combined[0, self.nlist:self.nlist + self.D] = -mean_rot
        b_pca_row = self.b_combined[:, self.nlist:self.nlist + self.D].contiguous()

        sq8_sorted = torch.empty((self.N, self.D), device=self.dev, dtype=torch.uint8)
        offsets = storage.offsets

        if self.per_cluster_sq8:
            self.vmin_pad = torch.zeros((self.nlist, self.D_pad), device=self.dev, dtype=torch.float32)
            self.delta_pad = torch.ones((self.nlist, self.D_pad), device=self.dev, dtype=torch.float32)
            self.eps_pad = torch.zeros((self.nlist, self.D_pad), device=self.dev, dtype=torch.float32)
            max_q_err_all = torch.zeros((self.nlist, self.D), device=self.dev, dtype=torch.float32)
            max_rot_f32_err = torch.zeros((self.D,), device=self.dev, dtype=torch.float32)

            c_step = 64
            for c_start in range(0, self.nlist, c_step):
                c_end = min(self.nlist, c_start + c_step)
                s = int(offsets[c_start])
                e = int(offsets[c_end])
                if e <= s:
                    continue
                raw_blk = storage.xb_sorted[s:e]
                blk_f64 = torch.matmul(raw_blk.to(torch.float64) - mean64_g, R64_g)
                blk_f32 = torch.addmm(b_pca_row.expand(e - s, self.D), raw_blk, R_f32)
                rot_err = torch.abs(blk_f64 - blk_f32.to(torch.float64)).max(dim=0).values.to(torch.float32)
                max_rot_f32_err = torch.maximum(max_rot_f32_err, rot_err)

                for c in range(c_start, c_end):
                    ls = int(offsets[c]) - s
                    le = int(offsets[c + 1]) - s
                    if le <= ls:
                        continue
                    c_f64 = blk_f64[ls:le]
                    c_f32 = c_f64.to(torch.float32)
                    vmin_c = c_f32.min(dim=0).values
                    vmax_c = c_f32.max(dim=0).values
                    delta_c = torch.where(vmax_c > vmin_c, (vmax_c - vmin_c) / 255.0, torch.ones_like(vmin_c))
                    codes_c = torch.clamp(torch.round((c_f32 - vmin_c) / delta_c), 0.0, 255.0).to(torch.uint8)
                    sq8_sorted[s + ls:s + le] = codes_c
                    x_hat_c = vmin_c + codes_c.to(torch.float32) * delta_c
                    q_err_c = torch.abs(c_f64 - x_hat_c.to(torch.float64)).max(dim=0).values.to(torch.float32)

                    self.vmin_pad[c, :self.D] = vmin_c
                    self.delta_pad[c, :self.D] = delta_c
                    max_q_err_all[c] = q_err_c
                del raw_blk, blk_f64, blk_f32

            self.eps_pad[:, :self.D] = (max_q_err_all + 2.0 * max_rot_f32_err.unsqueeze(0)) * (1.0 + 1e-5) + 2e-6
            del max_q_err_all, max_rot_f32_err
        else:
            vmin = torch.full((self.D,), float("inf"), device=self.dev, dtype=torch.float32)
            vmax = torch.full((self.D,), float("-inf"), device=self.dev, dtype=torch.float32)
            chunk = 50000
            for s in range(0, self.N, chunk):
                e = min(self.N, s + chunk)
                blk_f64 = torch.matmul(storage.xb_sorted[s:e].to(torch.float64) - mean64_g, R64_g)
                blk_f32 = blk_f64.to(torch.float32)
                vmin = torch.minimum(vmin, blk_f32.min(dim=0).values)
                vmax = torch.maximum(vmax, blk_f32.max(dim=0).values)
                del blk_f64, blk_f32

            delta = (vmax - vmin) / 255.0
            delta = torch.where(delta > 0.0, delta, torch.ones_like(delta))
            max_q_err = torch.zeros((self.D,), device=self.dev, dtype=torch.float32)
            max_rot_f32_err = torch.zeros((self.D,), device=self.dev, dtype=torch.float32)

            for s in range(0, self.N, chunk):
                e = min(self.N, s + chunk)
                raw_blk = storage.xb_sorted[s:e]
                blk_f64 = torch.matmul(raw_blk.to(torch.float64) - mean64_g, R64_g)
                blk_f32 = torch.addmm(b_pca_row.expand(e - s, self.D), raw_blk, R_f32)
                rot_err = torch.abs(blk_f64 - blk_f32.to(torch.float64)).max(dim=0).values.to(torch.float32)
                max_rot_f32_err = torch.maximum(max_rot_f32_err, rot_err)

                codes = torch.clamp(torch.round((blk_f64.to(torch.float32) - vmin) / delta), 0.0, 255.0).to(torch.uint8)
                sq8_sorted[s:e] = codes
                x_hat = vmin + codes.to(torch.float32) * delta
                err = torch.abs(blk_f64 - x_hat.to(torch.float64)).max(dim=0).values.to(torch.float32)
                max_q_err = torch.maximum(max_q_err, err)
                del raw_blk, blk_f64, blk_f32, rot_err, codes, x_hat, err

            eps_d = (max_q_err + 2.0 * max_rot_f32_err) * (1.0 + 1e-5) + 2e-6
            self.vmin_pad = torch.zeros((1, self.D_pad), device=self.dev, dtype=torch.float32)
            self.delta_pad = torch.ones((1, self.D_pad), device=self.dev, dtype=torch.float32)
            self.eps_pad = torch.zeros((1, self.D_pad), device=self.dev, dtype=torch.float32)
            self.vmin_pad[0, :self.D] = vmin
            self.delta_pad[0, :self.D] = delta
            self.eps_pad[0, :self.D] = eps_d

        del mean64_g, R64_g
        torch.cuda.synchronize()
        self.sq8_quant_s = time.perf_counter() - t_quant0

        t_pack0 = time.perf_counter()
        if self.var_block:
            bounds_dim = [0]
            for b in (16, 32):
                if b < self.D_pad:
                    bounds_dim.append(b)
            cur = 32
            while cur + 32 < self.D_pad:
                cur += 32
                bounds_dim.append(cur)
            if bounds_dim[-1] != self.D_pad:
                bounds_dim.append(self.D_pad)
        else:
            bounds_dim = list(range(0, self.D_pad + 1, 32))
            if bounds_dim[-1] != self.D_pad:
                bounds_dim.append(self.D_pad)
        bounds4 = [b // 4 for b in bounds_dim]
        self.blk_bounds4 = torch.tensor(bounds4, dtype=torch.int32, device=self.dev)

        cluster_tile_off = np.zeros(self.nlist + 1, dtype=np.int32)
        for c in range(self.nlist):
            M_c = int(offsets[c + 1] - offsets[c])
            cluster_tile_off[c + 1] = cluster_tile_off[c] + ((M_c + tile_size - 1) // tile_size)
        total_tiles = int(cluster_tile_off[-1])
        self.total_tiles = total_tiles

        tile_orig_ids_np = np.full((total_tiles, tile_size), -1, dtype=np.int32)
        tile_r_minmax_np = np.zeros((total_tiles, 2), dtype=np.float32)
        tile_pos_map = np.zeros((total_tiles, tile_size), dtype=np.int64)
        tile_valid_mask = np.zeros((total_tiles, tile_size), dtype=bool)

        order_global = storage.order_global
        r_sorted = storage.r_sorted
        for c in range(self.nlist):
            s, e = int(offsets[c]), int(offsets[c + 1])
            M_c = e - s
            t_base = int(cluster_tile_off[c])
            for t_idx in range((M_c + tile_size - 1) // tile_size):
                ts = s + t_idx * tile_size
                te = min(e, ts + tile_size)
                length = te - ts
                tid_global = t_base + t_idx
                tile_orig_ids_np[tid_global, :length] = order_global[ts:te].astype(np.int32)
                tile_r_minmax_np[tid_global, 0] = r_sorted[ts]
                tile_r_minmax_np[tid_global, 1] = r_sorted[te - 1]
                tile_pos_map[tid_global, :length] = np.arange(ts, te, dtype=np.int64)
                tile_valid_mask[tid_global, :length] = True

        self.cluster_tile_off = torch.from_numpy(cluster_tile_off).to(self.dev, dtype=torch.int32)
        self.tile_orig_ids = torch.from_numpy(tile_orig_ids_np).to(self.dev, dtype=torch.int32)
        self.tile_r_minmax = torch.from_numpy(tile_r_minmax_np).to(self.dev, dtype=torch.float32)

        pos_map_g = torch.from_numpy(tile_pos_map.reshape(-1)).to(self.dev, dtype=torch.int64)
        valid_g = torch.from_numpy(tile_valid_mask.reshape(-1, 1)).to(self.dev)
        gathered = torch.where(valid_g, sq8_sorted[pos_map_g], torch.zeros((1, 1), device=self.dev, dtype=torch.uint8))
        if self.D_pad > self.D:
            gathered = torch.nn.functional.pad(gathered, (0, self.D_pad - self.D), value=0)
        packed_u8 = gathered.view(total_tiles, self.W_tile, 32, self.D4, 4).permute(0, 3, 1, 2, 4).contiguous()
        self.sq8_packed = packed_u8.view(torch.int32)
        del sq8_sorted, gathered, packed_u8, pos_map_g, valid_g
        torch.cuda.synchronize()
        self.tile_pack_s = time.perf_counter() - t_pack0

        # Pre-allocated search workspace buffers (ZERO dynamic allocations during search!)
        self.max_batch = max_batch
        self.max_nprobe = max_nprobe
        self.proj_buf = torch.empty((max_batch, total_cols), device=self.dev, dtype=torch.float32)
        self.cdists_buf = torch.empty((max_batch, self.nlist), device=self.dev, dtype=torch.float32)
        self.probe_cids_buf = torch.empty((max_batch, max_nprobe), device=self.dev, dtype=torch.int32)
        self.probe_dqcs_buf = torch.empty((max_batch, max_nprobe), device=self.dev, dtype=torch.float32)
        max_pairs = max_batch * max_nprobe
        self.p1_cids_buf = torch.empty((max_pairs,), device=self.dev, dtype=torch.int32)
        self.p1_qids_buf = torch.empty((max_pairs,), device=self.dev, dtype=torch.int32)
        self.p1_dqcs_buf = torch.empty((max_pairs,), device=self.dev, dtype=torch.float32)
        self.p2_cids_buf = torch.empty((max_pairs,), device=self.dev, dtype=torch.int32)
        self.p2_qids_buf = torch.empty((max_pairs,), device=self.dev, dtype=torch.int32)
        self.p2_dqcs_buf = torch.empty((max_pairs,), device=self.dev, dtype=torch.float32)

        self.topk_d_buf = torch.empty((max_batch, 10), device=self.dev, dtype=torch.float32)
        self.topk_i_buf = torch.empty((max_batch, 10), device=self.dev, dtype=torch.int64)
        self.topm_buf_d = torch.empty((max_batch, 100), device=self.dev, dtype=torch.float32)
        self.topm_buf_i = torch.empty((max_batch, 100), device=self.dev, dtype=torch.int64)
        self.tau_sq_buf = torch.empty((max_batch,), device=self.dev, dtype=torch.float32)
        self.max_cands = max(1024 * 10, max_nprobe * 32)
        self.cand_buf_d = torch.empty((max_batch, self.max_cands), device=self.dev, dtype=torch.float32)
        self.cand_buf_i = torch.empty((max_batch, self.max_cands), device=self.dev, dtype=torch.int64)
        self.cand_cnt = torch.zeros((max_batch,), device=self.dev, dtype=torch.int32)

    def index_vram_mib(self, include_fp32: bool = True, include_sq8: bool = True):
        total_bytes = self.W_combined.numel() * 4 + self.b_combined.numel() * 4 + self.cluster_tile_off.numel() * 4
        if include_fp32:
            total_bytes += self.storage.xb_sorted.numel() * 4 + self.storage.orig_ids_sorted.numel() * 4 + self.storage.cluster_vec_off.numel() * 4
        if include_sq8:
            total_bytes += (
                self.sq8_packed.numel() * 4
                + self.tile_orig_ids.numel() * 4
                + self.tile_r_minmax.numel() * 4
                + self.vmin_pad.numel() * 4
                + self.delta_pad.numel() * 4
                + self.eps_pad.numel() * 4
            )
        return total_bytes / (1024.0 ** 2)

    def search(
        self,
        xq_gpu: torch.Tensor,
        nprobe: int,
        mode: int = 2,
        use_l1: int = 1,
        rerank_m: int = 0,
        tile_override: int = 0,
        record_kernel_ms: bool = False,
    ):
        B = xq_gpu.shape[0]
        tsize = tile_override if tile_override > 0 else self.tile_size
        topk_d = self.topk_d_buf[:B]
        topk_i = self.topk_i_buf[:B]
        kern_ms = tri_v4_ext.launch_trident_v4_search(
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
        )
        return topk_d, topk_i, kern_ms


def eval_accuracy_v4(
    pred_ids: np.ndarray,
    pred_dists: np.ndarray,
    cm_ids: np.ndarray,
    cm_dists: np.ndarray,
    faiss_ids: np.ndarray,
    faiss_dists: np.ndarray,
    gt10: np.ndarray,
):
    Q = len(gt10)
    rec_hits = 0
    cm_strict_ov = 0
    cm_match100 = 0
    ivf_ov = 0
    ivf_match100 = 0
    for q in range(Q):
        sp = set(pred_ids[q].tolist())
        sg = set(gt10[q].tolist())
        scm = set(cm_ids[q].tolist())
        sf = set(faiss_ids[q].tolist())
        rec_hits += len(sp.intersection(sg))

        ov_cm = len(sp.intersection(scm))
        cm_strict_ov += ov_cm
        if ov_cm == 10:
            cm_match100 += 1
        else:
            dp10 = float(pred_dists[q, 9])
            dc10 = float(cm_dists[q, 9])
            if abs(dp10 - dc10) <= 2e-5 * max(1.0, abs(dc10)):
                cm_match100 += 1

        ov_f = len(sp.intersection(sf))
        ivf_ov += ov_f
        if ov_f == 10:
            ivf_match100 += 1
        else:
            dp10 = float(pred_dists[q, 9])
            df10 = float(faiss_dists[q, 9])
            if abs(dp10 - df10) <= 2e-5 * max(1.0, abs(df10)):
                ivf_match100 += 1

    return (
        rec_hits / (Q * 10.0),
        (cm_strict_ov / (Q * 10.0)) * 100.0,
        (cm_match100 / float(Q)) * 100.0,
        (ivf_ov / (Q * 10.0)) * 100.0,
        (ivf_match100 / float(Q)) * 100.0,
    )


def measure_latency_qps(fn, warmup: int = 12, iters: int = 50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / float(iters)


def prepare_query_tensors(xq_raw: np.ndarray, ds: str, dev: torch.device):
    Q_eval = min(len(xq_raw), 1000)
    xq_eval_gpu = torch.from_numpy(xq_raw[:Q_eval]).to(dev, dtype=torch.float32).contiguous()

    if ds == "nuswide":
        b_max = min(len(xq_raw), 200)
        distinct_q = b_max
        xq_batch_np = xq_raw[:b_max].copy()
    elif len(xq_raw) >= 1024:
        b_max = 1024
        distinct_q = 1024
        xq_batch_np = xq_raw[:1024].copy()
    else:
        b_max = 1024
        distinct_q = len(xq_raw)
        reps = (1024 + len(xq_raw) - 1) // len(xq_raw)
        xq_batch_np = np.tile(xq_raw, (reps, 1))[:1024].copy()

    xq_batch_gpu = torch.from_numpy(xq_batch_np).to(dev, dtype=torch.float32).contiguous()
    return Q_eval, xq_eval_gpu, b_max, distinct_q, xq_batch_gpu


def run_ncu_profile(args):
    dev = torch.device("cuda:0")
    ds = args.datasets[0]
    seed = args.seeds[0]
    ds_dir = os.path.join(DATA_ROOT, ds)
    xb = np.ascontiguousarray(np.load(os.path.join(ds_dir, "base.npy")).astype(np.float32))
    xq_raw = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy")).astype(np.float32))
    _, _, b_max, _, xq_b = prepare_query_tensors(xq_raw, ds, dev)
    nprobe = args.nprobe_override if args.nprobe_override > 0 else NPROBE_MAP_095[ds]
    centroids, list_ids, r_vec, mean64, R64, _ = load_or_train_clustering(ds, xb, args.nlist, seed)

    t = args.ncu_target
    if t in ("Faiss_IVFFlat", "Faiss_IVFSQ8", "Faiss_IVFPQ"):
        kind = {"Faiss_IVFFlat": "flat", "Faiss_IVFSQ8": "sq8", "Faiss_IVFPQ": "pq"}[t]
        res, gidx, _, _ = build_faiss_gpu_single(xb, centroids, args.nlist, kind, ds, seed)
        gidx.nprobe = nprobe
        gidx.search(xq_b, 10)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        gidx.search(xq_b, 10)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return

    storage = SharedClusterStorage(xb, list_ids, r_vec, args.nlist, device="cuda:0")
    # Ablation & baseline targets:
    # A0_CM_FP32, CM_FP32_T64, A1_CM_SQ8_RR100, CM_SQ8_RR20, CM_SQ8_RR50,
    # A2_Orig_Uni32, A3_PCA_Uni32, A4_PCA_VarBlk, A5_PCA_VarBlk_L1, A6_PerClusterSQ8_T256, A6_PerClusterSQ8_T64, TRIDENT_K0_T256
    cfg_map = {
        "A0_CM_FP32":            dict(ts=256, pc=0, pca=1, vb=1, mode=10, l1=0, rrm=0),
        "CM_FP32_T64":           dict(ts=64,  pc=0, pca=1, vb=1, mode=10, l1=0, rrm=0),
        "TRIDENT_K0_T256":       dict(ts=256, pc=0, pca=1, vb=1, mode=0,  l1=0, rrm=0),
        "CM_SQ8_RR20":           dict(ts=256, pc=0, pca=1, vb=1, mode=0,  l1=0, rrm=20),
        "CM_SQ8_RR50":           dict(ts=256, pc=0, pca=1, vb=1, mode=0,  l1=0, rrm=50),
        "A1_CM_SQ8_RR100":       dict(ts=256, pc=0, pca=1, vb=1, mode=0,  l1=0, rrm=100),
        "A2_Orig_Uni32":         dict(ts=256, pc=0, pca=0, vb=0, mode=2,  l1=0, rrm=0),
        "A3_PCA_Uni32":          dict(ts=256, pc=0, pca=1, vb=0, mode=2,  l1=0, rrm=0),
        "A4_PCA_VarBlk":         dict(ts=256, pc=0, pca=1, vb=1, mode=2,  l1=0, rrm=0),
        "A5_PCA_VarBlk_L1":      dict(ts=256, pc=0, pca=1, vb=1, mode=2,  l1=1, rrm=0),
        "A6_PerClusterSQ8_T256": dict(ts=256, pc=1, pca=1, vb=1, mode=2,  l1=1, rrm=0),
        "A6_PerClusterSQ8_T64":  dict(ts=64,  pc=1, pca=1, vb=1, mode=2,  l1=1, rrm=0),
    }
    c = cfg_map[t]
    idx = TridentV4Index(
        storage, centroids, mean64, R64,
        tile_size=c["ts"], per_cluster_sq8=c["pc"], use_pca=c["pca"], var_block=c["vb"],
        max_batch=1024, device="cuda:0"
    )
    idx.search(xq_b, nprobe, mode=c["mode"], use_l1=c["l1"], rerank_m=c["rrm"])
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    idx.search(xq_b, nprobe, mode=c["mode"], use_l1=c["l1"], rerank_m=c["rrm"])
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def run_main_benchmark(args):
    dev = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(dev)
    print(f"[Phase 3A Main] Running on {gpu_name} ({args.gpu_tag})", flush=True)

    out_csv = os.path.join(RES_ROOT, f"phase3a_main_{args.gpu_tag}.csv")
    build_csv = os.path.join(RES_ROOT, f"phase3a_build_mem_{args.gpu_tag}.csv")
    rows = []
    build_rows = []

    for ds in args.datasets:
        ds_dir = os.path.join(DATA_ROOT, ds)
        xb = np.ascontiguousarray(np.load(os.path.join(ds_dir, "base.npy")).astype(np.float32))
        xq_raw = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy")).astype(np.float32))
        gt = np.load(os.path.join(ds_dir, "gt.npy"))

        Q_eval, xq_eval_gpu, b_max, distinct_q, xq_batch_gpu = prepare_query_tensors(xq_raw, ds, dev)
        gt10 = gt[:Q_eval, :10]
        nprobe = args.nprobe_override if args.nprobe_override > 0 else NPROBE_MAP_095[ds]
        ds_batches = [1, 32, b_max] if args.batches == [1, 32, 1024] else [min(b, b_max) for b in args.batches]

        for seed in args.seeds:
            centroids, list_ids, r_vec, mean64, R64, pca_s = load_or_train_clustering(ds, xb, args.nlist, seed)

            # 1. Benchmark Faiss-GPU baselines sequentially so RTX 4080 VRAM stays low
            faiss_timing = {}
            faiss_acc_raw = {}
            f_ids, f_dists = None, None

            for f_name, f_kind in [("Faiss_IVFFlat", "flat"), ("Faiss_IVFSQ8", "sq8"), ("Faiss_IVFPQ", "pq")]:
                res_f, gidx, f_build_s, f_vram_mib = build_faiss_gpu_single(xb, centroids, args.nlist, f_kind, ds, seed)
                gidx.nprobe = nprobe
                if seed == 0:
                    build_rows.append({
                        "gpu": args.gpu_tag, "dataset": ds, "dim": xb.shape[1], "method": f_name,
                        "pca_s": 0.0, "sq8_quant_s": 0.0, "tile_pack_s": 0.0,
                        "total_build_s": round(f_build_s, 3), "index_vram_mib": round(f_vram_mib, 1),
                    })
                td_t, ti_t = gidx.search(xq_eval_gpu, 10)
                torch.cuda.synchronize()
                td_np, ti_np = td_t.cpu().numpy(), ti_t.cpu().numpy()
                if f_kind == "flat":
                    f_ids, f_dists = ti_np, td_np
                faiss_acc_raw[f_name] = (ti_np, td_np)
                del td_t, ti_t

                for B in ds_batches:
                    xq_b = xq_batch_gpu[:B].contiguous()
                    faiss_timing[(f_name, B)] = measure_latency_qps(lambda: gidx.search(xq_b, 10), warmup=12, iters=50)
                del gidx, res_f
                torch.cuda.empty_cache()

            # 2. Build SharedClusterStorage + TRIDENT indexes sequentially
            storage = SharedClusterStorage(xb, list_ids, r_vec, args.nlist, device="cuda:0")
            tri_timing = {}
            tri_acc = {}

            # Stage A: Global SQ8 T=256 (for CM_FP32_T256, CM_FP32_T64, CM_SQ8_RR20/50/100, TRIDENT_K0_T256)
            idx_glob_256 = TridentV4Index(
                storage, centroids, mean64, R64,
                tile_size=256, per_cluster_sq8=0, use_pca=1, var_block=1, max_batch=1024, device="cuda:0"
            )
            torch.cuda.empty_cache()

            # Get CM_FP32 exact reference IDs & distances on xq_eval_gpu
            cm_d_t, cm_i_t, _ = idx_glob_256.search(xq_eval_gpu, nprobe, mode=10, use_l1=0, rerank_m=0, tile_override=256)
            torch.cuda.synchronize()
            cm_dists, cm_ids = cm_d_t.cpu().numpy(), cm_i_t.cpu().numpy()
            del cm_d_t, cm_i_t

            for f_name, (ti_np, td_np) in faiss_acc_raw.items():
                tri_acc[f_name] = eval_accuracy_v4(ti_np, td_np, cm_ids, cm_dists, f_ids, f_dists, gt10)

            stage_a_methods = [
                ("CM_FP32_T256",    dict(mode=10, l1=0, rrm=0,   to=256)),
                ("CM_FP32_T64",     dict(mode=10, l1=0, rrm=0,   to=64)),
                ("TRIDENT_K0_T256", dict(mode=0,  l1=0, rrm=0,   to=256)),
                ("CM_SQ8_RR20",     dict(mode=0,  l1=0, rrm=20,  to=256)),
                ("CM_SQ8_RR50",     dict(mode=0,  l1=0, rrm=50,  to=256)),
                ("CM_SQ8_RR100",    dict(mode=0,  l1=0, rrm=100, to=256)),
            ]
            for m_name, mc in stage_a_methods:
                td, ti, _ = idx_glob_256.search(xq_eval_gpu, nprobe, mode=mc["mode"], use_l1=mc["l1"], rerank_m=mc["rrm"], tile_override=mc["to"])
                torch.cuda.synchronize()
                tri_acc[m_name] = eval_accuracy_v4(ti.cpu().numpy(), td.cpu().numpy(), cm_ids, cm_dists, f_ids, f_dists, gt10)
                del td, ti
                for B in ds_batches:
                    xq_b = xq_batch_gpu[:B].contiguous()
                    e2e_ms = measure_latency_qps(
                        lambda: idx_glob_256.search(xq_b, nprobe, mode=mc["mode"], use_l1=mc["l1"], rerank_m=mc["rrm"], tile_override=mc["to"], record_kernel_ms=False),
                        warmup=12, iters=50
                    )
                    k_acc = 0.0
                    for _ in range(10):
                        _, _, km = idx_glob_256.search(xq_b, nprobe, mode=mc["mode"], use_l1=mc["l1"], rerank_m=mc["rrm"], tile_override=mc["to"], record_kernel_ms=True)
                        k_acc += km
                    tri_timing[(m_name, B)] = (e2e_ms, k_acc / 10.0)

            if seed == 0:
                build_rows.append({
                    "gpu": args.gpu_tag, "dataset": ds, "dim": xb.shape[1], "method": "CM_FP32",
                    "pca_s": 0.0, "sq8_quant_s": 0.0, "tile_pack_s": round(storage.sort_s, 3),
                    "total_build_s": round(storage.sort_s, 3),
                    "index_vram_mib": round(idx_glob_256.index_vram_mib(include_fp32=True, include_sq8=False), 1),
                })
                build_rows.append({
                    "gpu": args.gpu_tag, "dataset": ds, "dim": xb.shape[1], "method": "CM_SQ8_RR100",
                    "pca_s": round(pca_s, 3), "sq8_quant_s": round(idx_glob_256.sq8_quant_s, 3),
                    "tile_pack_s": round(storage.sort_s + idx_glob_256.tile_pack_s, 3),
                    "total_build_s": round(pca_s + storage.sort_s + idx_glob_256.sq8_quant_s + idx_glob_256.tile_pack_s, 3),
                    "index_vram_mib": round(idx_glob_256.index_vram_mib(include_fp32=True, include_sq8=True), 1),
                })

            del idx_glob_256
            torch.cuda.empty_cache()

            # Stage B: Per-Cluster SQ8 T=256 (TRIDENT_K1_T256, TRIDENT_K2_T256)
            idx_pc_256 = TridentV4Index(
                storage, centroids, mean64, R64,
                tile_size=256, per_cluster_sq8=1, use_pca=1, var_block=1, max_batch=1024, device="cuda:0"
            )
            torch.cuda.empty_cache()
            for m_name, m_code in [("TRIDENT_K1_T256", 1), ("TRIDENT_K2_T256", 2)]:
                td, ti, _ = idx_pc_256.search(xq_eval_gpu, nprobe, mode=m_code, use_l1=1)
                torch.cuda.synchronize()
                tri_acc[m_name] = eval_accuracy_v4(ti.cpu().numpy(), td.cpu().numpy(), cm_ids, cm_dists, f_ids, f_dists, gt10)
                del td, ti
                for B in ds_batches:
                    xq_b = xq_batch_gpu[:B].contiguous()
                    e2e_ms = measure_latency_qps(
                        lambda: idx_pc_256.search(xq_b, nprobe, mode=m_code, use_l1=1, record_kernel_ms=False),
                        warmup=12, iters=50
                    )
                    k_acc = 0.0
                    for _ in range(10):
                        _, _, km = idx_pc_256.search(xq_b, nprobe, mode=m_code, use_l1=1, record_kernel_ms=True)
                        k_acc += km
                    tri_timing[(m_name, B)] = (e2e_ms, k_acc / 10.0)

            if seed == 0:
                build_rows.append({
                    "gpu": args.gpu_tag, "dataset": ds, "dim": xb.shape[1], "method": "TRIDENT_K2_T256",
                    "pca_s": round(pca_s, 3), "sq8_quant_s": round(idx_pc_256.sq8_quant_s, 3),
                    "tile_pack_s": round(storage.sort_s + idx_pc_256.tile_pack_s, 3),
                    "total_build_s": round(pca_s + storage.sort_s + idx_pc_256.sq8_quant_s + idx_pc_256.tile_pack_s, 3),
                    "index_vram_mib": round(idx_pc_256.index_vram_mib(include_fp32=True, include_sq8=True), 1),
                })
            del idx_pc_256
            torch.cuda.empty_cache()

            # Stage C: Per-Cluster SQ8 T=64 (TRIDENT_K2_T64)
            idx_pc_64 = TridentV4Index(
                storage, centroids, mean64, R64,
                tile_size=64, per_cluster_sq8=1, use_pca=1, var_block=1, max_batch=1024, device="cuda:0"
            )
            torch.cuda.empty_cache()
            for m_name, m_code in [("TRIDENT_K2_T64", 2)]:
                td, ti, _ = idx_pc_64.search(xq_eval_gpu, nprobe, mode=m_code, use_l1=1)
                torch.cuda.synchronize()
                tri_acc[m_name] = eval_accuracy_v4(ti.cpu().numpy(), td.cpu().numpy(), cm_ids, cm_dists, f_ids, f_dists, gt10)
                del td, ti
                for B in ds_batches:
                    xq_b = xq_batch_gpu[:B].contiguous()
                    e2e_ms = measure_latency_qps(
                        lambda: idx_pc_64.search(xq_b, nprobe, mode=m_code, use_l1=1, record_kernel_ms=False),
                        warmup=12, iters=50
                    )
                    k_acc = 0.0
                    for _ in range(10):
                        _, _, km = idx_pc_64.search(xq_b, nprobe, mode=m_code, use_l1=1, record_kernel_ms=True)
                        k_acc += km
                    tri_timing[(m_name, B)] = (e2e_ms, k_acc / 10.0)

            if seed == 0:
                build_rows.append({
                    "gpu": args.gpu_tag, "dataset": ds, "dim": xb.shape[1], "method": "TRIDENT_K2_T64",
                    "pca_s": round(pca_s, 3), "sq8_quant_s": round(idx_pc_64.sq8_quant_s, 3),
                    "tile_pack_s": round(storage.sort_s + idx_pc_64.tile_pack_s, 3),
                    "total_build_s": round(pca_s + storage.sort_s + idx_pc_64.sq8_quant_s + idx_pc_64.tile_pack_s, 3),
                    "index_vram_mib": round(idx_pc_64.index_vram_mib(include_fp32=True, include_sq8=True), 1),
                })
                pd.DataFrame(build_rows).to_csv(build_csv, index=False)

            del idx_pc_64, storage
            torch.cuda.empty_cache()

            all_methods = [
                "Faiss_IVFFlat", "Faiss_IVFSQ8", "Faiss_IVFPQ",
                "CM_FP32_T256", "CM_FP32_T64",
                "CM_SQ8_RR20", "CM_SQ8_RR50", "CM_SQ8_RR100",
                "TRIDENT_K0_T256", "TRIDENT_K1_T256",
                "TRIDENT_K2_T256", "TRIDENT_K2_T64",
            ]
            for B in ds_batches:
                dq = min(B, distinct_q)
                for m_name in all_methods:
                    rec_val, cm_ov, cm_m100, ivf_ov, ivf_m100 = tri_acc[m_name]
                    if m_name.startswith("Faiss_"):
                        e2e_ms = faiss_timing[(m_name, B)]
                        kern_ms = float("nan")
                    else:
                        e2e_ms, kern_ms = tri_timing[(m_name, B)]
                    qps = (B * 1000.0) / max(e2e_ms, 1e-6)
                    row = {
                        "gpu": args.gpu_tag,
                        "gpu_name": gpu_name,
                        "dataset": ds,
                        "dim": xb.shape[1],
                        "nlist": args.nlist,
                        "nprobe": nprobe,
                        "seed": seed,
                        "batch": B,
                        "distinct_q": dq,
                        "method": m_name,
                        "e2e_ms": round(e2e_ms, 4),
                        "kernel_ms": round(kern_ms, 4) if not math.isnan(kern_ms) else "N/A",
                        "qps": round(qps, 1),
                        "recall_at_10": round(rec_val, 4),
                        "cm_fp32_strict_ov_pct": round(cm_ov, 4),
                        "cm_fp32_match100_pct": round(cm_m100, 4),
                        "ivfflat_overlap_pct": round(ivf_ov, 4),
                        "ivfflat_match100_pct": round(ivf_m100, 4),
                    }
                    rows.append(row)
                    if seed == 0 or B == b_max:
                        print(f"[{args.gpu_tag} {ds} s={seed} B={B:<4} {m_name:<16}] "
                              f"e2e={e2e_ms:>7.3f}ms kern={str(row['kernel_ms']):>7}ms QPS={qps:>9.1f} "
                              f"Rec={rec_val:.4f} CM_Strict={cm_ov:.2f}%", flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"[SAVED] {out_csv}", flush=True)


def run_ablation_benchmark(args):
    dev = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(dev)
    print(f"[Phase 3A Ablation] Running on {gpu_name} ({args.gpu_tag})", flush=True)

    out_csv = os.path.join(RES_ROOT, f"phase3a_ablation_{args.gpu_tag}.csv")
    rows = []

    ablation_specs = [
        ("A0_CM_FP32",            dict(ts=256, pc=0, pca=1, vb=1, mode=10, l1=0, rrm=0)),
        ("A1_CM_SQ8_RR100",       dict(ts=256, pc=0, pca=1, vb=1, mode=0,  l1=0, rrm=100)),
        ("A2_Orig_Uni32",         dict(ts=256, pc=0, pca=0, vb=0, mode=2,  l1=0, rrm=0)),
        ("A3_PCA_Uni32",          dict(ts=256, pc=0, pca=1, vb=0, mode=2,  l1=0, rrm=0)),
        ("A4_PCA_VarBlk",         dict(ts=256, pc=0, pca=1, vb=1, mode=2,  l1=0, rrm=0)),
        ("A5_PCA_VarBlk_L1",      dict(ts=256, pc=0, pca=1, vb=1, mode=2,  l1=1, rrm=0)),
        ("A6_PerClusterSQ8_T256", dict(ts=256, pc=1, pca=1, vb=1, mode=2,  l1=1, rrm=0)),
        ("A6_PerClusterSQ8_T64",  dict(ts=64,  pc=1, pca=1, vb=1, mode=2,  l1=1, rrm=0)),
    ]

    for ds in args.datasets:
        ds_dir = os.path.join(DATA_ROOT, ds)
        xb = np.ascontiguousarray(np.load(os.path.join(ds_dir, "base.npy")).astype(np.float32))
        xq_raw = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy")).astype(np.float32))
        gt = np.load(os.path.join(ds_dir, "gt.npy"))

        Q_eval, xq_eval_gpu, b_max, distinct_q, xq_batch_gpu = prepare_query_tensors(xq_raw, ds, dev)
        gt10 = gt[:Q_eval, :10]
        nprobe = args.nprobe_override if args.nprobe_override > 0 else NPROBE_MAP_095[ds]
        xq_b = xq_batch_gpu[:b_max].contiguous()

        for seed in args.seeds:
            centroids, list_ids, r_vec, mean64, R64, _ = load_or_train_clustering(ds, xb, args.nlist, seed)
            storage = SharedClusterStorage(xb, list_ids, r_vec, args.nlist, device="cuda:0")

            # Compute CM_FP32 reference first
            idx_ref = TridentV4Index(storage, centroids, mean64, R64, tile_size=256, per_cluster_sq8=0, use_pca=1, var_block=1, max_batch=1024, device="cuda:0")
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
                td, ti, _ = idx.search(xq_eval_gpu, nprobe, mode=sc["mode"], use_l1=sc["l1"], rerank_m=sc["rrm"])
                torch.cuda.synchronize()
                rec_val, cm_ov, cm_m100, _, _ = eval_accuracy_v4(ti.cpu().numpy(), td.cpu().numpy(), cm_ids, cm_dists, cm_ids, cm_dists, gt10)
                del td, ti

                e2e_ms = measure_latency_qps(
                    lambda: idx.search(xq_b, nprobe, mode=sc["mode"], use_l1=sc["l1"], rerank_m=sc["rrm"], record_kernel_ms=False),
                    warmup=12, iters=50
                )
                k_acc = 0.0
                for _ in range(10):
                    _, _, km = idx.search(xq_b, nprobe, mode=sc["mode"], use_l1=sc["l1"], rerank_m=sc["rrm"], record_kernel_ms=True)
                    k_acc += km
                kern_ms = k_acc / 10.0
                qps = (b_max * 1000.0) / max(e2e_ms, 1e-6)

                row = {
                    "gpu": args.gpu_tag,
                    "dataset": ds,
                    "dim": xb.shape[1],
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
                print(f"[Ablation {ds} s={seed} {a_name:<22}] e2e={e2e_ms:>7.3f}ms kern={kern_ms:>7.3f}ms QPS={qps:>9.1f} StrictOv={cm_ov:.2f}%", flush=True)
                del idx
                torch.cuda.empty_cache()

            del storage
            torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"[SAVED] {out_csv}", flush=True)


def run_supplements_benchmark(args):
    dev = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(dev)
    print(f"[Phase 3A Supplements] Running on {gpu_name} ({args.gpu_tag})", flush=True)

    # 1. Batch Curve B in {1, 8, 32, 128, 512, 1024} on gist1m, bge1024, sift1m
    bcurve_csv = os.path.join(RES_ROOT, f"phase3a_batch_curve_{args.gpu_tag}.csv")
    bcurve_rows = []
    bcurve_batches = [1, 8, 32, 128, 512, 1024]

    for ds in ["gist1m", "bge1024", "sift1m"]:
        if ds not in args.datasets:
            continue
        ds_dir = os.path.join(DATA_ROOT, ds)
        xb = np.ascontiguousarray(np.load(os.path.join(ds_dir, "base.npy")).astype(np.float32))
        xq_raw = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy")).astype(np.float32))
        _, _, b_max, distinct_q, xq_batch_gpu = prepare_query_tensors(xq_raw, ds, dev)
        nprobe = NPROBE_MAP_095[ds]

        for seed in args.seeds:
            centroids, list_ids, r_vec, mean64, R64, _ = load_or_train_clustering(ds, xb, args.nlist, seed)

            for f_name, f_kind in [("Faiss_IVFFlat", "flat"), ("Faiss_IVFSQ8", "sq8"), ("Faiss_IVFPQ", "pq")]:
                res_f, gidx, _, _ = build_faiss_gpu_single(xb, centroids, args.nlist, f_kind, ds, seed)
                gidx.nprobe = nprobe
                for B in bcurve_batches:
                    xq_b = xq_batch_gpu[:B].contiguous()
                    e2e_ms = measure_latency_qps(lambda: gidx.search(xq_b, 10), warmup=12, iters=50)
                    bcurve_rows.append({
                        "gpu": args.gpu_tag, "dataset": ds, "dim": xb.shape[1], "nprobe": nprobe,
                        "seed": seed, "batch": B, "distinct_q": min(B, distinct_q), "method": f_name,
                        "e2e_ms": round(e2e_ms, 4), "kernel_ms": "N/A", "qps": round((B * 1000.0) / max(e2e_ms, 1e-6), 1),
                    })
                del gidx, res_f
                torch.cuda.empty_cache()

            storage = SharedClusterStorage(xb, list_ids, r_vec, args.nlist, device="cuda:0")
            idx_glob = TridentV4Index(storage, centroids, mean64, R64, tile_size=256, per_cluster_sq8=0, use_pca=1, var_block=1, max_batch=1024, device="cuda:0")
            for m_name, mc in [("CM_FP32_T256", dict(mode=10, l1=0, rrm=0)), ("CM_SQ8_RR100", dict(mode=0, l1=0, rrm=100))]:
                for B in bcurve_batches:
                    xq_b = xq_batch_gpu[:B].contiguous()
                    e2e_ms = measure_latency_qps(lambda: idx_glob.search(xq_b, nprobe, mode=mc["mode"], use_l1=mc["l1"], rerank_m=mc["rrm"]), warmup=12, iters=50)
                    _, _, km = idx_glob.search(xq_b, nprobe, mode=mc["mode"], use_l1=mc["l1"], rerank_m=mc["rrm"], record_kernel_ms=True)
                    bcurve_rows.append({
                        "gpu": args.gpu_tag, "dataset": ds, "dim": xb.shape[1], "nprobe": nprobe,
                        "seed": seed, "batch": B, "distinct_q": min(B, distinct_q), "method": m_name,
                        "e2e_ms": round(e2e_ms, 4), "kernel_ms": round(km, 4), "qps": round((B * 1000.0) / max(e2e_ms, 1e-6), 1),
                    })
            del idx_glob
            torch.cuda.empty_cache()

            for tsize, m_name in [(256, "TRIDENT_K2_T256"), (64, "TRIDENT_K2_T64")]:
                idx_pc = TridentV4Index(storage, centroids, mean64, R64, tile_size=tsize, per_cluster_sq8=1, use_pca=1, var_block=1, max_batch=1024, device="cuda:0")
                for B in bcurve_batches:
                    xq_b = xq_batch_gpu[:B].contiguous()
                    e2e_ms = measure_latency_qps(lambda: idx_pc.search(xq_b, nprobe, mode=2, use_l1=1), warmup=12, iters=50)
                    _, _, km = idx_pc.search(xq_b, nprobe, mode=2, use_l1=1, record_kernel_ms=True)
                    bcurve_rows.append({
                        "gpu": args.gpu_tag, "dataset": ds, "dim": xb.shape[1], "nprobe": nprobe,
                        "seed": seed, "batch": B, "distinct_q": min(B, distinct_q), "method": m_name,
                        "e2e_ms": round(e2e_ms, 4), "kernel_ms": round(km, 4), "qps": round((B * 1000.0) / max(e2e_ms, 1e-6), 1),
                    })
                del idx_pc
                torch.cuda.empty_cache()
            del storage
            torch.cuda.empty_cache()
            pd.DataFrame(bcurve_rows).to_csv(bcurve_csv, index=False)

    # 2. Recall@10 = 0.99 nprobe recalibration & B = B_max comparison across all 6 datasets
    rec99_csv = os.path.join(RES_ROOT, f"phase3a_recall99_{args.gpu_tag}.csv")
    rec99_rows = []
    for ds in args.datasets:
        ds_dir = os.path.join(DATA_ROOT, ds)
        xb = np.ascontiguousarray(np.load(os.path.join(ds_dir, "base.npy")).astype(np.float32))
        xq_raw = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy")).astype(np.float32))
        gt = np.load(os.path.join(ds_dir, "gt.npy"))
        Q_eval, xq_eval_gpu, b_max, distinct_q, xq_batch_gpu = prepare_query_tensors(xq_raw, ds, dev)
        gt10 = gt[:Q_eval, :10]
        xq_b = xq_batch_gpu[:b_max].contiguous()

        # Recalibrate nprobe for Recall@10 >= 0.99 on seed 0
        c0, l0, r0, m0, R0, _ = load_or_train_clustering(ds, xb, args.nlist, 0)
        res0, gflat0, _, _ = build_faiss_gpu_single(xb, c0, args.nlist, "flat", ds, 0)
        nprobe_99 = args.nlist
        for cand_p in [8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512]:
            gflat0.nprobe = cand_p
            _, ti_t = gflat0.search(xq_eval_gpu, 10)
            torch.cuda.synchronize()
            ti_np = ti_t.cpu().numpy()
            hits = sum(len(set(ti_np[q].tolist()).intersection(set(gt10[q].tolist()))) for q in range(Q_eval))
            rec = hits / (Q_eval * 10.0)
            if rec >= 0.9900:
                nprobe_99 = cand_p
                break
        del gflat0, res0
        torch.cuda.empty_cache()
        print(f"[Recall@0.99 Calib] {ds}: nprobe_99 = {nprobe_99}", flush=True)

        for seed in args.seeds:
            centroids, list_ids, r_vec, mean64, R64, _ = load_or_train_clustering(ds, xb, args.nlist, seed)
            res_f, gflat, _, _ = build_faiss_gpu_single(xb, centroids, args.nlist, "flat", ds, seed)
            gflat.nprobe = nprobe_99
            fd_t, fi_t = gflat.search(xq_eval_gpu, 10)
            torch.cuda.synchronize()
            f_dists, f_ids = fd_t.cpu().numpy(), fi_t.cpu().numpy()
            f_e2e = measure_latency_qps(lambda: gflat.search(xq_b, 10), warmup=12, iters=50)
            del gflat, res_f, fd_t, fi_t
            torch.cuda.empty_cache()

            storage = SharedClusterStorage(xb, list_ids, r_vec, args.nlist, device="cuda:0")
            idx_glob = TridentV4Index(storage, centroids, mean64, R64, tile_size=256, per_cluster_sq8=0, use_pca=1, var_block=1, max_batch=1024, max_nprobe=512, device="cuda:0")
            cm_d_t, cm_i_t, _ = idx_glob.search(xq_eval_gpu, nprobe_99, mode=10, use_l1=0)
            torch.cuda.synchronize()
            cm_dists, cm_ids = cm_d_t.cpu().numpy(), cm_i_t.cpu().numpy()
            f_rec, f_cm_ov, f_cm_m100, _, _ = eval_accuracy_v4(f_ids, f_dists, cm_ids, cm_dists, f_ids, f_dists, gt10)
            rec99_rows.append({
                "gpu": args.gpu_tag, "dataset": ds, "dim": xb.shape[1], "nprobe_99": nprobe_99,
                "seed": seed, "batch": b_max, "distinct_q": distinct_q, "method": "Faiss_IVFFlat",
                "e2e_ms": round(f_e2e, 4), "kernel_ms": "N/A", "qps": round((b_max * 1000.0) / max(f_e2e, 1e-6), 1),
                "recall_at_10": round(f_rec, 4), "cm_fp32_strict_ov_pct": round(f_cm_ov, 4),
            })

            for m_name, mc in [("CM_FP32_T256", dict(mode=10, l1=0, rrm=0)), ("CM_SQ8_RR100", dict(mode=0, l1=0, rrm=100))]:
                td, ti, _ = idx_glob.search(xq_eval_gpu, nprobe_99, mode=mc["mode"], use_l1=mc["l1"], rerank_m=mc["rrm"])
                torch.cuda.synchronize()
                rec_val, cm_ov, _, _, _ = eval_accuracy_v4(ti.cpu().numpy(), td.cpu().numpy(), cm_ids, cm_dists, f_ids, f_dists, gt10)
                e2e_ms = measure_latency_qps(lambda: idx_glob.search(xq_b, nprobe_99, mode=mc["mode"], use_l1=mc["l1"], rerank_m=mc["rrm"]), warmup=12, iters=50)
                _, _, km = idx_glob.search(xq_b, nprobe_99, mode=mc["mode"], use_l1=mc["l1"], rerank_m=mc["rrm"], record_kernel_ms=True)
                rec99_rows.append({
                    "gpu": args.gpu_tag, "dataset": ds, "dim": xb.shape[1], "nprobe_99": nprobe_99,
                    "seed": seed, "batch": b_max, "distinct_q": distinct_q, "method": m_name,
                    "e2e_ms": round(e2e_ms, 4), "kernel_ms": round(km, 4), "qps": round((b_max * 1000.0) / max(e2e_ms, 1e-6), 1),
                    "recall_at_10": round(rec_val, 4), "cm_fp32_strict_ov_pct": round(cm_ov, 4),
                })
            del idx_glob
            torch.cuda.empty_cache()

            for tsize, m_name in [(256, "TRIDENT_K2_T256"), (64, "TRIDENT_K2_T64")]:
                idx_pc = TridentV4Index(storage, centroids, mean64, R64, tile_size=tsize, per_cluster_sq8=1, use_pca=1, var_block=1, max_batch=1024, max_nprobe=512, device="cuda:0")
                td, ti, _ = idx_pc.search(xq_eval_gpu, nprobe_99, mode=2, use_l1=1)
                torch.cuda.synchronize()
                rec_val, cm_ov, _, _, _ = eval_accuracy_v4(ti.cpu().numpy(), td.cpu().numpy(), cm_ids, cm_dists, f_ids, f_dists, gt10)
                e2e_ms = measure_latency_qps(lambda: idx_pc.search(xq_b, nprobe_99, mode=2, use_l1=1), warmup=12, iters=50)
                _, _, km = idx_pc.search(xq_b, nprobe_99, mode=2, use_l1=1, record_kernel_ms=True)
                rec99_rows.append({
                    "gpu": args.gpu_tag, "dataset": ds, "dim": xb.shape[1], "nprobe_99": nprobe_99,
                    "seed": seed, "batch": b_max, "distinct_q": distinct_q, "method": m_name,
                    "e2e_ms": round(e2e_ms, 4), "kernel_ms": round(km, 4), "qps": round((b_max * 1000.0) / max(e2e_ms, 1e-6), 1),
                    "recall_at_10": round(rec_val, 4), "cm_fp32_strict_ov_pct": round(cm_ov, 4),
                })
                del idx_pc
                torch.cuda.empty_cache()

            del storage
            torch.cuda.empty_cache()
            pd.DataFrame(rec99_rows).to_csv(rec99_csv, index=False)

    print(f"[SAVED] {bcurve_csv} and {rec99_csv}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_mode", choices=["main", "ablation", "supplements", "ncu"], default="main")
    ap.add_argument("--datasets", nargs="+", default=["gist1m", "bge1024", "sift1m", "msong", "glove100", "nuswide"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 32, 1024])
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--nprobe_override", type=int, default=0)
    ap.add_argument("--gpu_tag", default="A100")
    ap.add_argument("--ncu_target", default="")
    args = ap.parse_args()

    if args.ncu_target or args.run_mode == "ncu":
        run_ncu_profile(args)
    elif args.run_mode == "main":
        run_main_benchmark(args)
    elif args.run_mode == "ablation":
        run_ablation_benchmark(args)
    elif args.run_mode == "supplements":
        run_supplements_benchmark(args)


if __name__ == "__main__":
    main()
