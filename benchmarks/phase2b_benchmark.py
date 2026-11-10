#!/usr/bin/env python3
"""
Phase 2B: Real GPU Kernel Kill-Test & Baseline Benchmark (PLAN_V3 §4).
Compares on identical IVF clustering (same nlist=1024, same centroids, same cluster assignments, same nprobe):
  - Faiss-GPU IVFFlat (exact FP32 baseline, GPU-resident tensors via faiss.contrib.torch_utils)
  - Faiss-GPU IVFSQ8  (lossy SQ8 baseline, GPU-resident tensors)
  - TRIDENT-K0        (PCA + SQ8 full-dimension cluster-major scan, no early-stop, no FP32)
  - TRIDENT-K1        (PCA + variable-block (16,16,32,...) SQ8 tile early-stop + L1, no FP32)
  - TRIDENT-K2        (K1 + exact FP32 survivor refinement -> 100% lossless vs Faiss-GPU IVFFlat)
Evaluated across:
  - tile_size in {256, 64}
  - batch in {1, 32, 1024}
  - GPUs: NVIDIA A100 80GB PCIe and NVIDIA GeForce RTX 4080 16GB
  - --ncu_target <method> for exact hardware HBM dram__bytes_read.sum profiling via Nsight Compute!
"""
import os, sys, time, math, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.cpp_extension import load
import faiss
import faiss.contrib.torch_utils  # enables zero-copy GPU tensor search on Faiss-GPU!

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13")
os.environ["PATH"] = "/usr/local/cuda-13/bin:" + os.environ.get("PATH", "")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.expanduser("~/trident/data")
RES_ROOT = os.path.expanduser("~/trident/results/phase2b")
os.makedirs(RES_ROOT, exist_ok=True)

print("[JIT] Compiling trident_kernels.cu ...", flush=True)
tri_ext = load(
    name="trident_kernels_v3_ext",
    sources=[os.path.join(SCRIPT_DIR, "trident_kernels.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)
print("[JIT] trident_kernels_v2_ext compiled successfully.", flush=True)


def compute_pca(xb: np.ndarray, seed: int, sample_n: int = 100000):
    N, D = xb.shape
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, size=min(N, sample_n), replace=False) if N > sample_n else np.arange(N)
    sub = xb[idx].astype(np.float64)
    mean64 = sub.mean(axis=0)
    centered = sub - mean64
    cov = (centered.T @ centered) / max(len(sub) - 1, 1)
    _, eigvecs = np.linalg.eigh(cov)
    R64 = eigvecs[:, ::-1].copy()
    return mean64, R64


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


def build_faiss_gpu_indexes(xb: np.ndarray, centroids: np.ndarray, nlist: int):
    N, D = xb.shape
    res = faiss.StandardGpuResources()
    res.setTempMemory(512 * 1024 * 1024)

    quant_flat = faiss.IndexFlatL2(D)
    quant_flat.add(centroids)
    cpu_ivfflat = faiss.IndexIVFFlat(quant_flat, D, nlist, faiss.METRIC_L2)
    cpu_ivfflat.is_trained = True
    cpu_ivfflat.add(xb)
    gpu_ivfflat = faiss.index_cpu_to_gpu(res, 0, cpu_ivfflat)

    quant_sq = faiss.IndexFlatL2(D)
    quant_sq.add(centroids)
    cpu_ivfsq8 = faiss.IndexIVFScalarQuantizer(quant_sq, D, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2)
    cpu_ivfsq8.train(xb[:min(N, 100000)])
    cpu_ivfsq8.add(xb)
    gpu_ivfsq8 = faiss.index_cpu_to_gpu(res, 0, cpu_ivfsq8)

    dists_sq, list_ids = quant_flat.search(xb, 1)
    list_ids = list_ids.reshape(-1)
    r_vec = np.sqrt(np.maximum(dists_sq.reshape(-1), 0.0)).astype(np.float32)
    return res, gpu_ivfflat, gpu_ivfsq8, list_ids, r_vec


class TridentGPUIndex:
    def __init__(self, xb: np.ndarray, centroids: np.ndarray, list_ids: np.ndarray, r_vec: np.ndarray,
                 mean64: np.ndarray, R64: np.ndarray, tile_size: int = 256, max_batch: int = 1024,
                 device="cuda:0", shared_xb_gpu: torch.Tensor = None):
        self.dev = torch.device(device)
        self.N, self.D = xb.shape
        self.nlist = centroids.shape[0]
        self.tile_size = tile_size
        self.W_tile = tile_size // 32
        self.D_pad = ((self.D + 31) // 32) * 32
        self.D4 = self.D_pad // 4

        order_global = np.lexsort((r_vec, list_ids)).astype(np.int64)
        sorted_cids = list_ids[order_global]
        counts = np.bincount(sorted_cids, minlength=self.nlist)
        offsets = np.zeros(self.nlist + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        r_sorted = r_vec[order_global]

        if shared_xb_gpu is not None:
            self.xb_orig = shared_xb_gpu
        else:
            self.xb_orig = torch.from_numpy(xb).to(self.dev, dtype=torch.float32).contiguous()
        self.centroids_gpu = torch.from_numpy(centroids).to(self.dev, dtype=torch.float32).contiguous()
        self.c_norm_sq = (self.centroids_gpu ** 2).sum(dim=1)

        mean64_g = torch.from_numpy(mean64).to(self.dev, dtype=torch.float64)
        R64_g = torch.from_numpy(R64).to(self.dev, dtype=torch.float64)
        self.mean_f32 = mean64_g.to(torch.float32).contiguous()
        self.R_f32 = R64_g.to(torch.float32).contiguous()

        order_g = torch.from_numpy(order_global).to(self.dev, dtype=torch.int64)
        vmin = torch.full((self.D,), float("inf"), device=self.dev, dtype=torch.float32)
        vmax = torch.full((self.D,), float("-inf"), device=self.dev, dtype=torch.float32)
        chunk = 50000
        for s in range(0, self.N, chunk):
            e = min(self.N, s + chunk)
            blk_f64 = torch.matmul(self.xb_orig[order_g[s:e]].to(torch.float64) - mean64_g, R64_g)
            blk_f32 = blk_f64.to(torch.float32)
            vmin = torch.minimum(vmin, blk_f32.min(dim=0).values)
            vmax = torch.maximum(vmax, blk_f32.max(dim=0).values)
            del blk_f64, blk_f32

        delta = (vmax - vmin) / 255.0
        delta = torch.where(delta > 0.0, delta, torch.ones_like(delta))

        sq8_sorted = torch.empty((self.N, self.D), device=self.dev, dtype=torch.uint8)
        max_q_err = torch.zeros((self.D,), device=self.dev, dtype=torch.float32)
        max_rot_f32_err = torch.zeros((self.D,), device=self.dev, dtype=torch.float32)
        for s in range(0, self.N, chunk):
            e = min(self.N, s + chunk)
            raw_blk = self.xb_orig[order_g[s:e]]
            blk_f64 = torch.matmul(raw_blk.to(torch.float64) - mean64_g, R64_g)
            blk_f32 = torch.matmul(raw_blk - self.mean_f32, self.R_f32)
            rot_err = torch.abs(blk_f64 - blk_f32.to(torch.float64)).max(dim=0).values.to(torch.float32)
            max_rot_f32_err = torch.maximum(max_rot_f32_err, rot_err)

            codes = torch.clamp(torch.round((blk_f64.to(torch.float32) - vmin) / delta), 0.0, 255.0).to(torch.uint8)
            sq8_sorted[s:e] = codes
            x_hat = vmin + codes.to(torch.float32) * delta
            err = torch.abs(blk_f64 - x_hat.to(torch.float64)).max(dim=0).values.to(torch.float32)
            max_q_err = torch.maximum(max_q_err, err)
            del raw_blk, blk_f64, blk_f32, rot_err, codes, x_hat, err
        del order_g, mean64_g, R64_g

        eps_d = (max_q_err + 2.0 * max_rot_f32_err) * (1.0 + 1e-5) + 2e-6

        self.vmin_pad = torch.zeros((self.D_pad,), device=self.dev, dtype=torch.float32)
        self.delta_pad = torch.zeros((self.D_pad,), device=self.dev, dtype=torch.float32)
        self.eps_pad = torch.zeros((self.D_pad,), device=self.dev, dtype=torch.float32)
        self.vmin_pad[:self.D] = vmin
        self.delta_pad[:self.D] = delta
        self.eps_pad[:self.D] = eps_d

        # Variable block bounds in units of d4 (dim / 4): (16, 16, 32, 32, ...)
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
        del sq8_sorted, gathered, packed_u8

        self.max_batch = max_batch
        self.xq_pca_buf = torch.zeros((max_batch, self.D_pad), device=self.dev, dtype=torch.float32)
        self.topk_d_buf = torch.empty((max_batch, 10), device=self.dev, dtype=torch.float32)
        self.topk_i_buf = torch.empty((max_batch, 10), device=self.dev, dtype=torch.int64)
        self.tau_sq_buf = torch.empty((max_batch,), device=self.dev, dtype=torch.float32)
        self.max_cands = 1024 * 10
        self.cand_buf_d = torch.empty((max_batch, self.max_cands), device=self.dev, dtype=torch.float32)
        self.cand_buf_i = torch.empty((max_batch, self.max_cands), device=self.dev, dtype=torch.int64)
        self.cand_cnt = torch.zeros((max_batch,), device=self.dev, dtype=torch.int32)
        self.empty_i32 = torch.empty((0,), device=self.dev, dtype=torch.int32)
        self.empty_f32 = torch.empty((0,), device=self.dev, dtype=torch.float32)
        self.qid_arange = torch.arange(max_batch, device=self.dev, dtype=torch.int32).unsqueeze(1)

    def _sort_cluster_pairs(self, cids_2d: torch.Tensor, dqcs_2d: torch.Tensor):
        B, P = cids_2d.shape
        if P == 0:
            return self.empty_i32, self.empty_i32, self.empty_f32
        flat_cids = cids_2d.reshape(-1)
        flat_qids = self.qid_arange[:B].expand(B, P).reshape(-1)
        flat_dqcs = dqcs_2d.reshape(-1)
        if B == 1:
            return flat_cids.contiguous(), flat_qids.contiguous(), flat_dqcs.contiguous()
        perm = torch.argsort(flat_cids)
        return flat_cids[perm].contiguous(), flat_qids[perm].contiguous(), flat_dqcs[perm].contiguous()

    def search(self, xq_gpu: torch.Tensor, nprobe: int, mode: int = 2, use_l1: int = 1, record_kernel_ms: bool = False):
        B = xq_gpu.shape[0]
        # 1. Coarse quantizer L2 distances (identical centroids to Faiss-GPU)
        q_norm_sq = (xq_gpu ** 2).sum(dim=1, keepdim=True)
        cdists_sq = torch.addmm(self.c_norm_sq.unsqueeze(0), xq_gpu, self.centroids_gpu.t(), beta=1.0, alpha=-2.0)
        cdists_sq.add_(q_norm_sq).clamp_(min=0.0)
        top_c_sq, top_c_ids = torch.topk(cdists_sq, k=nprobe, dim=1, largest=False, sorted=True)
        probe_cids = top_c_ids.to(torch.int32)
        probe_dqcs = torch.sqrt(top_c_sq)

        # 2. PCA query rotation
        xq_pca = self.xq_pca_buf[:B]
        torch.matmul(xq_gpu - self.mean_f32, self.R_f32, out=xq_pca[:, :self.D])

        # 3. Two-stage Cluster-Major Grouping
        if nprobe <= 1 or (mode == 0 and B == 1):
            p_init = 0
        elif mode == 0:
            p_init = 1
        elif B == 1:
            p_init = 1
        elif nprobe <= 16:
            p_init = 2
        else:
            p_init = 3

        p1_cids, p1_qids, p1_dqcs = self._sort_cluster_pairs(probe_cids[:, :p_init], probe_dqcs[:, :p_init])
        p2_cids, p2_qids, p2_dqcs = self._sort_cluster_pairs(probe_cids[:, p_init:], probe_dqcs[:, p_init:])

        max_cands_q = min(self.max_cands, max(nprobe * 10, 64))
        topk_d = self.topk_d_buf[:B]
        topk_i = self.topk_i_buf[:B]
        tau_sq = self.tau_sq_buf[:B]
        cand_d = self.cand_buf_d[:B, :max_cands_q]
        cand_i = self.cand_buf_i[:B, :max_cands_q]
        cand_c = self.cand_cnt[:B]

        kern_ms = tri_ext.launch_trident_scan(
            int(mode), int(self.tile_size), int(use_l1),
            self.sq8_packed, self.tile_orig_ids, self.tile_r_minmax, self.cluster_tile_off,
            self.xb_orig, xq_gpu, xq_pca,
            self.vmin_pad, self.delta_pad, self.eps_pad, self.blk_bounds4,
            p1_cids, p1_qids, p1_dqcs, p2_cids, p2_qids, p2_dqcs,
            topk_d, topk_i, tau_sq, cand_d, cand_i, cand_c,
            bool(record_kernel_ms)
        )
        return topk_d, topk_i, kern_ms


def eval_accuracy(pred_ids: np.ndarray, pred_dists: np.ndarray,
                  faiss_ids: np.ndarray, faiss_dists: np.ndarray, gt10: np.ndarray):
    Q = len(gt10)
    rec_hits = 0
    ivf_overlap = 0
    ivf_match100 = 0
    for q in range(Q):
        sp = set(pred_ids[q].tolist())
        sg = set(gt10[q].tolist())
        sf = set(faiss_ids[q].tolist())
        rec_hits += len(sp.intersection(sg))
        ov = len(sp.intersection(sf))
        ivf_overlap += ov
        if ov == 10:
            ivf_match100 += 1
        else:
            dp10 = float(pred_dists[q, 9])
            df10 = float(faiss_dists[q, 9])
            if abs(dp10 - df10) <= 2e-5 * max(1.0, abs(df10)):
                ivf_match100 += 1
    return (rec_hits / (Q * 10.0)), (ivf_overlap / (Q * 10.0)) * 100.0, (ivf_match100 / float(Q)) * 100.0


def measure_latency_qps(fn, warmup: int, iters: int):
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


NPROBE_MAP_095 = {
    "bge1024": 14,
    "gist1m": 56,
    "sift1m": 24,
    "msong": 10,
    "glove100": 128,
    "nuswide": 3,
}


def run_benchmark(args):
    dev = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(dev)
    print(f"[GPU] Running Phase 2B on {gpu_name}", flush=True)

    rows = []
    for ds in args.datasets:
        ds_dir = os.path.join(DATA_ROOT, ds)
        xb = np.ascontiguousarray(np.load(os.path.join(ds_dir, "base.npy")).astype(np.float32))
        xq_raw = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy")).astype(np.float32))
        gt = np.load(os.path.join(ds_dir, "gt.npy"))
        Q_eval = min(len(xq_raw), 1000)
        gt10 = gt[:Q_eval, :10]

        if len(xq_raw) >= 1024:
            xq_1024 = xq_raw[:1024].copy()
        else:
            reps = (1024 + len(xq_raw) - 1) // len(xq_raw)
            xq_1024 = np.tile(xq_raw, (reps, 1))[:1024].copy()
        xq_1024_gpu = torch.from_numpy(xq_1024).to(dev, dtype=torch.float32).contiguous()
        xq_eval_gpu = torch.from_numpy(xq_raw[:Q_eval]).to(dev, dtype=torch.float32).contiguous()

        nprobe = args.nprobe_override if args.nprobe_override > 0 else NPROBE_MAP_095[ds]

        for seed in args.seeds:
            cache_file = os.path.join(args.cache_dir, f"{ds}_nlist{args.nlist}_s{seed}.npz") if args.cache_dir else ""
            if cache_file and os.path.exists(cache_file):
                cdata = np.load(cache_file)
                centroids = cdata["centroids"]
                list_ids = cdata["list_ids"]
                r_vec = cdata["r_vec"]
                mean64 = cdata["mean64"]
                R64 = cdata["R64"]
            else:
                centroids = train_shared_ivf_centroids(xb, nlist=args.nlist, seed=seed)
                quant_tmp = faiss.IndexFlatL2(xb.shape[1])
                quant_tmp.add(centroids)
                dists_sq, lids = quant_tmp.search(xb, 1)
                list_ids = lids.reshape(-1)
                r_vec = np.sqrt(np.maximum(dists_sq.reshape(-1), 0.0)).astype(np.float32)
                mean64, R64 = compute_pca(xb, seed=seed)
                if cache_file:
                    os.makedirs(args.cache_dir, exist_ok=True)
                    np.savez(cache_file, centroids=centroids, list_ids=list_ids, r_vec=r_vec, mean64=mean64, R64=R64)

            # Exact hardware HBM dram__bytes_read.sum profiling mode under Nsight Compute
            if args.ncu_target:
                xq_b = xq_1024_gpu
                if args.ncu_target in ("faiss_flat", "faiss_sq8"):
                    res_f, gpu_ivfflat, gpu_ivfsq8, _, _ = build_faiss_gpu_indexes(xb, centroids, args.nlist)
                    gpu_ivfflat.nprobe = nprobe
                    gpu_ivfsq8.nprobe = nprobe
                    target_idx = gpu_ivfflat if args.ncu_target == "faiss_flat" else gpu_ivfsq8
                    target_idx.search(xq_b, 10)
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStart()
                    target_idx.search(xq_b, 10)
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()
                elif args.ncu_target in ("K0_T256", "K1_T256", "K2_T256", "K0_T64", "K1_T64", "K2_T64"):
                    m_map = {"K0": 0, "K1": 1, "K2": 2}
                    m_code = m_map[args.ncu_target.split("_")[0]]
                    tsize = 64 if args.ncu_target.endswith("T64") else 256
                    idx_obj = TridentGPUIndex(xb, centroids, list_ids, r_vec, mean64, R64, tile_size=tsize, max_batch=1024, device="cuda:0")
                    idx_obj.search(xq_b, nprobe, mode=m_code, use_l1=1)
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStart()
                    idx_obj.search(xq_b, nprobe, mode=m_code, use_l1=1)
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()
                return

            # Stage 1: Build & benchmark Faiss-GPU baselines, then free Faiss VRAM (keeps peak VRAM <= 6.3 GB for RTX 4080)
            res_f, gpu_ivfflat, gpu_ivfsq8, list_ids, r_vec = build_faiss_gpu_indexes(xb, centroids, args.nlist)
            gpu_ivfflat.nprobe = nprobe
            gpu_ivfsq8.nprobe = nprobe

            f_dists_t, f_ids_t = gpu_ivfflat.search(xq_eval_gpu, 10)
            torch.cuda.synchronize()
            f_dists = f_dists_t.cpu().numpy()
            f_ids = f_ids_t.cpu().numpy()
            f_rec, _, _ = eval_accuracy(f_ids, f_dists, f_ids, f_dists, gt10)

            sq_dists_t, sq_ids_t = gpu_ivfsq8.search(xq_eval_gpu, 10)
            torch.cuda.synchronize()
            sq_rec, sq_ov, sq_m100 = eval_accuracy(
                sq_ids_t.cpu().numpy(), sq_dists_t.cpu().numpy(), f_ids, f_dists, gt10
            )
            del f_dists_t, f_ids_t, sq_dists_t, sq_ids_t

            faiss_timing = {}
            for B in args.batches:
                xq_b = xq_1024_gpu[:B].contiguous()
                warmup = 10 if B < 1024 else 5
                iters = 30 if B < 1024 else 15
                faiss_timing[("Faiss_IVFFlat", B)] = measure_latency_qps(lambda: gpu_ivfflat.search(xq_b, 10), warmup, iters)
                faiss_timing[("Faiss_IVFSQ8", B)] = measure_latency_qps(lambda: gpu_ivfsq8.search(xq_b, 10), warmup, iters)

            del gpu_ivfflat, gpu_ivfsq8, res_f
            torch.cuda.empty_cache()

            # Stage 2: Build & benchmark TRIDENT T=256 then T=64 sequentially (peak VRAM <= 5.5 GB for RTX 4080)
            shared_xb_gpu = torch.from_numpy(xb).to(dev, dtype=torch.float32).contiguous()
            acc_cache = {
                "Faiss_IVFFlat": (f_rec, 100.0, 100.0),
                "Faiss_IVFSQ8": (sq_rec, sq_ov, sq_m100),
            }
            tri_timing = {}
            for tsize, t_methods in [
                (256, [("TRIDENT_K0_T256", 0), ("TRIDENT_K1_T256", 1), ("TRIDENT_K2_T256", 2)]),
                (64,  [("TRIDENT_K0_T64", 0),  ("TRIDENT_K1_T64", 1),  ("TRIDENT_K2_T64", 2)]),
            ]:
                idx_obj = TridentGPUIndex(xb, centroids, list_ids, r_vec, mean64, R64, tile_size=tsize, max_batch=1024, device="cuda:0", shared_xb_gpu=shared_xb_gpu)
                torch.cuda.empty_cache()
                for m_name, m_code in t_methods:
                    td, ti, _ = idx_obj.search(xq_eval_gpu, nprobe, mode=m_code, use_l1=1)
                    torch.cuda.synchronize()
                    acc_cache[m_name] = eval_accuracy(ti.cpu().numpy(), td.cpu().numpy(), f_ids, f_dists, gt10)
                    del td, ti
                for B in args.batches:
                    xq_b = xq_1024_gpu[:B].contiguous()
                    warmup = 10 if B < 1024 else 5
                    iters = 30 if B < 1024 else 15
                    for m_name, m_code in t_methods:
                        e2e_ms = measure_latency_qps(
                            lambda: idx_obj.search(xq_b, nprobe, mode=m_code, use_l1=1, record_kernel_ms=False),
                            warmup, iters
                        )
                        k_acc = 0.0
                        for _ in range(5):
                            _, _, km = idx_obj.search(xq_b, nprobe, mode=m_code, use_l1=1, record_kernel_ms=True)
                            k_acc += km
                        tri_timing[(m_name, B)] = (e2e_ms, k_acc / 5.0)
                del idx_obj
                torch.cuda.empty_cache()

            del shared_xb_gpu
            torch.cuda.empty_cache()

            method_order = [
                ("Faiss_IVFFlat", "faiss_flat"),
                ("Faiss_IVFSQ8", "faiss_sq8"),
                ("TRIDENT_K0_T256", "trident"),
                ("TRIDENT_K1_T256", "trident"),
                ("TRIDENT_K2_T256", "trident"),
                ("TRIDENT_K0_T64", "trident"),
                ("TRIDENT_K1_T64", "trident"),
                ("TRIDENT_K2_T64", "trident"),
            ]

            for B in args.batches:
                for m_name, kind in method_order:
                    rec_val, ov_pct, m100_pct = acc_cache[m_name]
                    if kind in ("faiss_flat", "faiss_sq8"):
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
                        "method": m_name,
                        "e2e_ms": round(e2e_ms, 4),
                        "kernel_ms": round(kern_ms, 4) if not math.isnan(kern_ms) else "N/A",
                        "qps": round(qps, 1),
                        "recall_at_10": round(rec_val, 4),
                        "ivfflat_overlap_pct": round(ov_pct, 4),
                        "ivfflat_match100_pct": round(m100_pct, 4),
                    }
                    rows.append(row)
                    if seed == 0 or B == 1024:
                        print(f"[{args.gpu_tag} {ds} s={seed} B={B:<4} {m_name:<16}] "
                              f"e2e={e2e_ms:>7.3f}ms kern={str(row['kernel_ms']):>7}ms QPS={qps:>9.1f} "
                              f"Rec={rec_val:.4f} Match={m100_pct:.2f}%", flush=True)

            if not args.ncu_target:
                out_csv = os.path.join(RES_ROOT, f"phase2b_{args.gpu_tag}.csv")
                pd.DataFrame(rows).to_csv(out_csv, index=False)

    if not args.ncu_target:
        out_csv = os.path.join(RES_ROOT, f"phase2b_{args.gpu_tag}.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"SAVED {out_csv}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["gist1m", "bge1024", "sift1m", "msong", "glove100", "nuswide"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 32, 1024])
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--nprobe_override", type=int, default=0)
    ap.add_argument("--gpu_tag", default="A100")
    ap.add_argument("--ncu_target", default="")
    ap.add_argument("--cache_dir", default="")
    args = ap.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
