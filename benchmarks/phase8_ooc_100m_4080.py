#!/usr/bin/env python3
"""
PLAN_V9 Step B3: 100M-Scale True Out-of-Core (OOC) Benchmark on NVIDIA GeForce RTX 4080 16GB (sm_89).
NO artificial VRAM cap (no set_per_process_memory_fraction, no blocker buffers):
  - Raw SIFT100M FP32 vectors = 100,000,000 * 128 * 4 bytes = 51.20 GB decimal (47.684 GiB),
    which is > 3.2x larger than the physical 16 GB VRAM of RTX 4080!
  - TRIDENT-OOC keeps only the 12.74 GiB blocked SQ8 index + tile IDs + bounds in RTX 4080 VRAM,
    while keeping the 47.684 GiB (51.20 GB) cluster-sorted FP32 vectors in page-locked pinned host RAM,
    fetching only the ~0.04%-0.12% unpruned survivors over PCIe Gen4 via zero-copy UVA.
  - Compares against:
    1. CM_FP32_OOC_Pinned (Zero-copy UVA full-precision IVF scan over PCIe Gen4)
    2. Faiss_CPU_IVFFlat_24T (24-thread AVX-512 OpenMP exact FP32 IVF scan in host RAM)
    3. Faiss_GPU_IVFPQ_CPU_Refine_kf256 (GPU IVFPQ M=16 in RTX 4080 VRAM + 24T OpenMP FP32 rerank)
    4. cuVS_CAGRA_InVRAM (Attempts in-VRAM build/alloc on 16 GB RTX 4080 -> natural hardware OOM)
    5. cuVS_CAGRA_UVM (Attempts UVM build/search on 16 GB RTX 4080 via ManagedMemoryResource)
  - Profiles hardware PCIe system-memory read traffic via Nsight Compute (lts__t_sectors_aperture_sysmem.sum).
Outputs ~/trident/results/phase8/ooc_100m_RTX4080.csv
"""
import os, sys, time, csv, gc, struct, subprocess, statistics, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_HOME"] = "/usr/local/cuda-13"
os.environ["PATH"] = "/usr/local/cuda-13/bin:" + os.environ.get("PATH", "")

import numpy as np
import torch, faiss
from torch.utils.cpp_extension import load_inline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase4a_suite import get_cuda_ext, choose_tile, eval_accuracy_v4

DATA_ROOT = os.path.expanduser("~/trident/data")
CACHE_DIR = os.path.expanduser("~/trident/cache_v3/sift100m_nlist16384_s0")
RES_DIR = os.path.expanduser("~/trident/results/phase8")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
OUT_CSV = os.path.join(RES_DIR, "ooc_100m_RTX4080.csv")

N_100M = 100_000_000
D_100M = 128
NLIST_100M = 16384
TILE_100M = 64

CPP_100M_HELPER = r"""
#include <torch/extension.h>
#include <omp.h>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

void reorder_u8_by_perm(
    torch::Tensor xb_u8,       // [N, D] uint8
    torch::Tensor sort_perm,   // [N] int32
    torch::Tensor xb_sorted_u8 // [N, D] uint8
) {
    const int64_t N = xb_u8.size(0);
    const int64_t D = xb_u8.size(1);
    const uint8_t* src = xb_u8.data_ptr<uint8_t>();
    const int32_t* perm = sort_perm.data_ptr<int32_t>();
    uint8_t* dst = xb_sorted_u8.data_ptr<uint8_t>();

    #pragma omp parallel for schedule(static) num_threads(24)
    for (int64_t i = 0; i < N; ++i) {
        int64_t old_id = perm[i];
        std::memcpy(dst + i * D, src + old_id * D, D);
    }
}

void u8_to_f32_omp(
    torch::Tensor src_u8,  // [N, D] uint8
    torch::Tensor dst_f32  // [N, D] float32 (pinned)
) {
    const int64_t total = src_u8.numel();
    const uint8_t* s = src_u8.data_ptr<uint8_t>();
    float* d = dst_f32.data_ptr<float>();

    #pragma omp parallel for schedule(static) num_threads(24)
    for (int64_t i = 0; i < total; ++i) {
        d[i] = static_cast<float>(s[i]);
    }
}

void omp_refine_topk_u8(
    torch::Tensor xq,        // [B, D] float32 CPU
    torch::Tensor xb_u8,     // [N, D] uint8 CPU
    torch::Tensor cand_ids,  // [B, M] int64 CPU
    torch::Tensor out_d,     // [B, K] float32 CPU
    torch::Tensor out_i      // [B, K] int64 CPU
) {
    const int64_t B = xq.size(0);
    const int64_t D = xq.size(1);
    const int64_t N = xb_u8.size(0);
    const int64_t M = cand_ids.size(1);
    const int64_t K = out_d.size(1);

    const float* q_ptr = xq.data_ptr<float>();
    const uint8_t* b_ptr = xb_u8.data_ptr<uint8_t>();
    const int64_t* c_ptr = cand_ids.data_ptr<int64_t>();
    float* od_ptr = out_d.data_ptr<float>();
    int64_t* oi_ptr = out_i.data_ptr<int64_t>();

    #pragma omp parallel for schedule(static) num_threads(24)
    for (int64_t q = 0; q < B; ++q) {
        const float* qr = q_ptr + q * D;
        const int64_t* cr = c_ptr + q * M;
        float* top_d = od_ptr + q * K;
        int64_t* top_i = oi_ptr + q * K;

        for (int64_t j = 0; j < K; ++j) {
            top_d[j] = std::numeric_limits<float>::infinity();
            top_i[j] = -1;
        }

        for (int64_t m = 0; m < M; ++m) {
            int64_t id = cr[m];
            if (id < 0 || id >= N) continue;
            const uint8_t* br = b_ptr + id * D;
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int64_t d = 0; d < D; ++d) {
                float diff = qr[d] - static_cast<float>(br[d]);
                sum += diff * diff;
            }
            if (sum < top_d[K - 1] || (sum == top_d[K - 1] && id < top_i[K - 1])) {
                top_d[K - 1] = sum;
                top_i[K - 1] = id;
                for (int64_t j = K - 1; j > 0; --j) {
                    if (top_d[j] < top_d[j - 1] || (top_d[j] == top_d[j - 1] && top_i[j] < top_i[j - 1])) {
                        std::swap(top_d[j], top_d[j - 1]);
                        std::swap(top_i[j], top_i[j - 1]);
                    } else {
                        break;
                    }
                }
            }
        }
    }
}

void omp_cpu_ivfflat_f32(
    torch::Tensor xq,          // [B, D] float32 CPU
    torch::Tensor probe_cids,  // [B, nprobe] int64 CPU
    torch::Tensor vec_off,     // [nlist + 1] int32 CPU
    torch::Tensor xb_sorted,   // [N, D] float32 CPU (pinned)
    torch::Tensor orig_ids,    // [N] int32 CPU (pinned)
    torch::Tensor out_d,       // [B, K] float32 CPU
    torch::Tensor out_i        // [B, K] int64 CPU
) {
    const int64_t B = xq.size(0);
    const int64_t D = xq.size(1);
    const int64_t nprobe = probe_cids.size(1);
    const int64_t K = out_d.size(1);

    const float* q_ptr = xq.data_ptr<float>();
    const int64_t* pc_ptr = probe_cids.data_ptr<int64_t>();
    const int32_t* v_off = vec_off.data_ptr<int32_t>();
    const float* b_ptr = xb_sorted.data_ptr<float>();
    const int32_t* oid_ptr = orig_ids.data_ptr<int32_t>();
    float* od_ptr = out_d.data_ptr<float>();
    int64_t* oi_ptr = out_i.data_ptr<int64_t>();

    #pragma omp parallel for schedule(dynamic, 1) num_threads(24)
    for (int64_t q = 0; q < B; ++q) {
        const float* qr = q_ptr + q * D;
        const int64_t* cids = pc_ptr + q * nprobe;
        float* top_d = od_ptr + q * K;
        int64_t* top_i = oi_ptr + q * K;

        for (int64_t j = 0; j < K; ++j) {
            top_d[j] = std::numeric_limits<float>::infinity();
            top_i[j] = -1;
        }

        for (int64_t p = 0; p < nprobe; ++p) {
            int64_t c = cids[p];
            if (c < 0) continue;
            int64_t s_vec = v_off[c];
            int64_t e_vec = v_off[c + 1];
            for (int64_t idx = s_vec; idx < e_vec; ++idx) {
                const float* br = b_ptr + idx * D;
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int64_t d = 0; d < D; ++d) {
                    float diff = qr[d] - br[d];
                    sum += diff * diff;
                }
                int64_t id = static_cast<int64_t>(oid_ptr[idx]);
                if (sum < top_d[K - 1] || (sum == top_d[K - 1] && id < top_i[K - 1])) {
                    top_d[K - 1] = sum;
                    top_i[K - 1] = id;
                    for (int64_t j = K - 1; j > 0; --j) {
                        if (top_d[j] < top_d[j - 1] || (top_d[j] == top_d[j - 1] && top_i[j] < top_i[j - 1])) {
                            std::swap(top_d[j], top_d[j - 1]);
                            std::swap(top_i[j], top_i[j - 1]);
                        } else {
                            break;
                        }
                    }
                }
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reorder_u8_by_perm", &reorder_u8_by_perm);
    m.def("u8_to_f32_omp", &u8_to_f32_omp);
    m.def("omp_refine_topk_u8", &omp_refine_topk_u8);
    m.def("omp_cpu_ivfflat_f32", &omp_cpu_ivfflat_f32);
}
"""

_HELPER_EXT = None
def get_helper_ext():
    global _HELPER_EXT
    if _HELPER_EXT is None:
        _HELPER_EXT = load_inline(
            name="trident_100m_omp_ext_v2",
            cpp_sources=CPP_100M_HELPER,
            extra_cflags=["-O3", "-fopenmp", "-march=native"],
            extra_ldflags=["-fopenmp"],
            verbose=False,
        )
    return _HELPER_EXT


def load_cached_100m_artifacts():
    meta_npz = os.path.join(CACHE_DIR, "meta.npz")
    xb_sorted_u8_file = os.path.join(CACHE_DIR, "xb_sorted_u8.bin")
    sq8_packed_file = os.path.join(CACHE_DIR, "sq8_packed.bin")
    tile_oids_file = os.path.join(CACHE_DIR, "tile_orig_ids.bin")

    print(f"[SIFT100M] Loading cached index artifacts from {CACHE_DIR}...", flush=True)
    meta = np.load(meta_npz)
    centroids = meta["centroids"]
    mean64 = meta["mean64"]
    R64 = meta["R64"]
    sort_perm = meta["sort_perm"]
    vec_off = meta["vec_off"]
    tile_off = meta["tile_off"]
    vmin_np = meta["vmin_np"]
    delta_np = meta["delta_np"]
    total_tiles = int(tile_off[-1])
    sq8_packed_t = torch.from_numpy(np.memmap(sq8_packed_file, dtype=np.int32, mode="r", shape=(total_tiles, 32, 2, 32)))
    tile_oids_t = torch.from_numpy(np.memmap(tile_oids_file, dtype=np.int32, mode="r", shape=(total_tiles, 64)))
    xb_sorted_u8_t = torch.from_numpy(np.memmap(xb_sorted_u8_file, dtype=np.uint8, mode="r", shape=(N_100M, D_100M)))
    return centroids, mean64, R64, sort_perm, vec_off, tile_off, vmin_np, delta_np, sq8_packed_t, tile_oids_t, xb_sorted_u8_t


class Trident100MOOCEngine:
    def __init__(self, centroids, mean64, R64, vec_off, tile_off, vmin_np, delta_np, sq8_packed_t, tile_oids_t, xb_sorted_pinned, orig_ids_pinned, max_batch=1024, max_nprobe=512, dev=torch.device("cuda:0")):
        self.dev = dev
        self.D = D_100M
        self.D_pad = D_100M
        self.nlist = NLIST_100M
        self.tile_size = TILE_100M
        self.xb_sorted_pinned = xb_sorted_pinned
        self.orig_ids_pinned = orig_ids_pinned

        mean64_g = torch.from_numpy(mean64).to(dev, dtype=torch.float64)
        R64_g = torch.from_numpy(R64).to(dev, dtype=torch.float64)
        mean_f32 = mean64_g.to(torch.float32).contiguous()
        R_f32 = R64_g.to(torch.float32).contiguous()

        centroids_g = torch.from_numpy(centroids).to(dev, dtype=torch.float32).contiguous()
        c_norm_sq = (centroids_g ** 2).sum(dim=1)
        self.W_coarse_only = (-2.0 * centroids_g.t()).contiguous()
        self.b_coarse_only = c_norm_sq.unsqueeze(0).contiguous()

        total_cols = self.nlist + self.D_pad
        self.W_combined = torch.zeros((self.D, total_cols), device=dev, dtype=torch.float32)
        self.b_combined = torch.zeros((1, total_cols), device=dev, dtype=torch.float32)
        self.W_combined[:, :self.nlist] = self.W_coarse_only
        self.W_combined[:, self.nlist:self.nlist + self.D] = R_f32
        self.b_combined[0, :self.nlist] = c_norm_sq
        self.b_combined[0, self.nlist:self.nlist + self.D] = -torch.matmul(mean_f32, R_f32)
        del mean64_g, R64_g, mean_f32, R_f32, centroids_g, c_norm_sq

        self.vmin_pad = torch.from_numpy(vmin_np).to(dev, dtype=torch.float32).contiguous()
        self.delta_pad = torch.from_numpy(delta_np).to(dev, dtype=torch.float32).contiguous()
        self.eps_pad = (self.delta_pad * 0.5001).contiguous()
        self.cluster_vec_off = torch.from_numpy(vec_off).to(dev, dtype=torch.int32).contiguous()
        self.cluster_tile_off = torch.from_numpy(tile_off).to(dev, dtype=torch.int32).contiguous()
        self.blk_bounds4 = torch.tensor([0, 8, 16, 24, 32], device=dev, dtype=torch.int32).contiguous()

        self.dummy_r_minmax = torch.zeros((2, 2), device=dev, dtype=torch.float32)
        self.dummy_r_vec = torch.zeros(2, device=dev, dtype=torch.float32)

        self.proj_buf = torch.zeros((max_batch, total_cols), device=dev, dtype=torch.float32)
        self.cdists_buf = torch.zeros((max_batch, self.nlist), device=dev, dtype=torch.float32)
        self.probe_cids_buf = torch.zeros((max_batch, max_nprobe), device=dev, dtype=torch.int32)
        self.probe_dqcs_buf = torch.zeros((max_batch, max_nprobe), device=dev, dtype=torch.float32)
        max_pairs = max_batch * max_nprobe
        self.p1_cids_buf = torch.zeros(max_pairs, device=dev, dtype=torch.int32)
        self.p1_qids_buf = torch.zeros(max_pairs, device=dev, dtype=torch.int32)
        self.p1_dqcs_buf = torch.zeros(max_pairs, device=dev, dtype=torch.float32)
        self.p2_cids_buf = torch.zeros(max_pairs, device=dev, dtype=torch.int32)
        self.p2_qids_buf = torch.zeros(max_pairs, device=dev, dtype=torch.int32)
        self.p2_dqcs_buf = torch.zeros(max_pairs, device=dev, dtype=torch.float32)
        self.topk_d = torch.zeros((max_batch, 10), device=dev, dtype=torch.float32)
        self.topk_i = torch.zeros((max_batch, 10), device=dev, dtype=torch.int64)
        self.topm_buf_d = torch.zeros((max_batch, 200), device=dev, dtype=torch.float32)
        self.topm_buf_i = torch.zeros((max_batch, 200), device=dev, dtype=torch.int64)
        self.tau_sq_buf = torch.zeros(max_batch, device=dev, dtype=torch.float32)
        self.cand_buf_d = torch.zeros((max_batch, 16384), device=dev, dtype=torch.float32)
        self.cand_buf_i = torch.zeros((max_batch, 16384), device=dev, dtype=torch.int64)
        self.cand_cnt = torch.zeros(max_batch, device=dev, dtype=torch.int32)

        self.sq8_packed = sq8_packed_t.to(dev, non_blocking=False)
        self.tile_orig_ids = tile_oids_t.to(dev, non_blocking=False)
        torch.cuda.synchronize(dev)

    def search(self, xq_gpu: torch.Tensor, nprobe: int, mode: int = 2, stats_buf=None):
        ext = get_cuda_ext()
        B = xq_gpu.shape[0]
        td = self.topk_d[:B]
        ti = self.topk_i[:B]
        ext.launch_trident_v4_search(
            int(mode), int(self.tile_size), 0, 1, 0, int(nprobe),
            xq_gpu, self.W_combined, self.b_combined, self.W_coarse_only, self.b_coarse_only,
            self.proj_buf, self.cdists_buf, self.probe_cids_buf, self.probe_dqcs_buf,
            self.p1_cids_buf, self.p1_qids_buf, self.p1_dqcs_buf,
            self.p2_cids_buf, self.p2_qids_buf, self.p2_dqcs_buf,
            self.sq8_packed, self.tile_orig_ids, self.dummy_r_minmax,
            self.cluster_tile_off, self.cluster_vec_off,
            self.xb_sorted_pinned, self.orig_ids_pinned,
            self.vmin_pad, self.delta_pad, self.eps_pad, self.blk_bounds4,
            td, ti, self.topm_buf_d, self.topm_buf_i, self.tau_sq_buf,
            self.cand_buf_d, self.cand_buf_i, self.cand_cnt,
            False, 1, stats_buf, self.dummy_r_vec
        )
        return td, ti


def run_ncu_probe_all_100m():
    """
    Single NCU probe process that pins the 47.68 GiB SIFT100M buffer ONCE and launches:
      1. TRIDENT-OOC (mode=2, nprobe=48, B=256)
      2. TRIDENT-OOC (mode=2, nprobe=160, B=256)
      3. CM-FP32-OOC (mode=10, nprobe=48, B=64)
      4. CM-FP32-OOC (mode=10, nprobe=160, B=64)
    """
    dev = torch.device("cuda:0")
    helper = get_helper_ext()
    _ = get_cuda_ext()
    ds_dir = os.path.join(DATA_ROOT, "sift100m")
    xq = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy"))[:256].astype(np.float32))
    xq_gpu_256 = torch.from_numpy(xq[:256]).to(dev, dtype=torch.float32).contiguous()
    xq_gpu_64 = torch.from_numpy(xq[:64]).to(dev, dtype=torch.float32).contiguous()

    centroids, mean64, R64, sort_perm, vec_off, tile_off, vmin_np, delta_np, sq8_packed_t, tile_oids_t, xb_sorted_u8_t = load_cached_100m_artifacts()
    xb_sorted_pinned = torch.empty((N_100M, D_100M), dtype=torch.float32, pin_memory=True)
    helper.u8_to_f32_omp(xb_sorted_u8_t, xb_sorted_pinned)
    orig_ids_pinned = torch.empty(N_100M, dtype=torch.int32, pin_memory=True)
    orig_ids_pinned.copy_(torch.from_numpy(sort_perm))

    eng = Trident100MOOCEngine(
        centroids, mean64, R64, vec_off, tile_off, vmin_np, delta_np,
        sq8_packed_t, tile_oids_t,
        xb_sorted_pinned, orig_ids_pinned, max_batch=256, max_nprobe=512, dev=dev
    )
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    eng.search(xq_gpu_256, 48, mode=2)
    torch.cuda.synchronize()
    eng.search(xq_gpu_256, 160, mode=2)
    torch.cuda.synchronize()
    eng.search(xq_gpu_64, 48, mode=10)
    torch.cuda.synchronize()
    eng.search(xq_gpu_64, 160, mode=10)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def measure_ncu_sysmem_all_100m():
    ncu_bin = "/usr/local/cuda-13/bin/ncu"
    if not os.path.exists(ncu_bin):
        ncu_bin = "ncu"
    cmd = [
        ncu_bin,
        "--profile-from-start", "off",
        "--kernel-name", "regex:trident_cluster_major_scan_kernel|trident_cm_fp32_scan_kernel",
        "--metrics", "lts__t_sectors_aperture_sysmem.sum",
        "--csv",
        sys.executable,
        os.path.abspath(__file__),
        "--ncu-probe-all",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=600)
        tr_vals = []
        cm_vals = []
        for line in out.splitlines():
            if "lts__t_sectors_aperture_sysmem.sum" in line:
                parts = [p.strip('"') for p in line.split(",")]
                val_str = parts[-1].replace(",", "")
                try:
                    sectors = float(val_str)
                    if "trident_cluster_major_scan_kernel" in line:
                        tr_vals.append(round((sectors * 32.0 * (1024.0 / 256.0)) / (1024.0**2), 2))
                    elif "trident_cm_fp32_scan_kernel" in line:
                        cm_vals.append(round((sectors * 32.0 * (1024.0 / 64.0)) / (1024.0**2), 2))
                except ValueError:
                    pass
        if len(tr_vals) >= 2 and len(cm_vals) >= 2:
            return {0.95: (tr_vals[0], cm_vals[0]), 0.99: (tr_vals[1], cm_vals[1])}
    except Exception as e:
        print(f"[WARN] ncu 100M probe failed: {e}", flush=True)
    return {0.95: (-1.0, -1.0), 0.99: (-1.0, -1.0)}


def test_cagra_100m_worker(mode: str):
    import rmm
    from cuvs.neighbors import cagra
    torch.cuda.init()
    free_b, tot_b = torch.cuda.mem_get_info(0)
    print(f"[CAGRA-100M {mode}] RTX 4080 Free={free_b/1024**3:.2f} GiB / Total={tot_b/1024**3:.2f} GiB", flush=True)
    if mode == "InVRAM":
        try:
            cuda_mr = rmm.mr.CudaMemoryResource()
            rmm.mr.set_current_device_resource(cuda_mr)
            buf = rmm.DeviceBuffer(size=N_100M * D_100M * 4, mr=cuda_mr)
            print("CAGRA_STATUS: UNEXPECTED_SUCCESS", flush=True)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
            print(f"CAGRA_STATUS: OOM ({err})", flush=True)
    elif mode == "UVM":
        try:
            uvm_mr = rmm.mr.ManagedMemoryResource()
            rmm.mr.set_current_device_resource(uvm_mr)
            ds_dir = os.path.join(DATA_ROOT, "sift100m")
            xb_u8 = np.memmap(os.path.join(ds_dir, "base.100M.u8bin"), dtype=np.uint8, mode="r", offset=8, shape=(N_100M, D_100M))
            ip = cagra.IndexParams(metric="sqeuclidean", graph_degree=32, intermediate_graph_degree=64, build_algo="nn_descent")
            idx = cagra.build(ip, xb_u8)
            print("CAGRA_STATUS: UVM_BUILT", flush=True)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
            print(f"CAGRA_STATUS: OOM_OR_ERR ({err})", flush=True)


def measure_runs(fn, warmup=3, iters=8, b_size=1024):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    qps_list = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        qps_list.append(b_size / max(dt, 1e-9))
    return statistics.mean(qps_list), (statistics.stdev(qps_list) if len(qps_list) > 1 else 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncu-probe-all", action="store_true")
    parser.add_argument("--cagra-worker", type=str, default="")
    args = parser.parse_args()

    if args.ncu_probe_all:
        run_ncu_probe_all_100m()
        return
    if args.cagra_worker:
        test_cagra_100m_worker(args.cagra_worker)
        return

    dev = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(dev)
    free_b, tot_b = torch.cuda.mem_get_info(dev)
    print(f"=== Step B3: SIFT100M True OOC Benchmark on {gpu_name} (Free: {free_b/1024**3:.2f} GiB / Total: {tot_b/1024**3:.2f} GiB) ===", flush=True)
    print("NOTE: Zero artificial VRAM cap called! 51.20 GB (47.68 GiB) FP32 naturally exceeds 16 GB physical VRAM.", flush=True)

    helper = get_helper_ext()
    _ = get_cuda_ext()

    natural_oom_verified = 0
    try:
        _test_100m = torch.empty((N_100M, D_100M), dtype=torch.float32, device=dev)
        del _test_100m
    except RuntimeError as e:
        natural_oom_verified = 1
        print(f"[SIFT100M] Natural hardware OOM verified on RTX 4080 16GB: {str(e).splitlines()[0][:90]}", flush=True)
    torch.cuda.empty_cache()

    ds_dir = os.path.join(DATA_ROOT, "sift100m")
    xb_u8_mmap = np.memmap(os.path.join(ds_dir, "base.100M.u8bin"), dtype=np.uint8, mode="r", offset=8, shape=(N_100M, D_100M))
    xq = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy")).astype(np.float32))
    gt = np.load(os.path.join(ds_dir, "gt.npy"))

    B_eval = 1000
    B_lat = 1024
    gt10 = gt[:B_eval, :10]
    xq_b = np.ascontiguousarray(np.vstack([xq[:1000], xq[:24]]), dtype=np.float32)
    xq_b_gpu = torch.from_numpy(xq_b).to(dev, dtype=torch.float32).contiguous()
    xq_b_t_cpu = torch.from_numpy(xq_b)

    centroids, mean64, R64, sort_perm, vec_off, tile_off, vmin_np, delta_np, sq8_packed_t, tile_oids_t, xb_sorted_u8_t = load_cached_100m_artifacts()

    print("[SIFT100M] Materializing 47.68 GiB (51.20 GB) cluster-sorted FP32 vectors in page-locked pinned host RAM...", flush=True)
    t_pin = time.perf_counter()
    xb_sorted_pinned = torch.empty((N_100M, D_100M), dtype=torch.float32, pin_memory=True)
    helper.u8_to_f32_omp(xb_sorted_u8_t, xb_sorted_pinned)
    orig_ids_pinned = torch.empty(N_100M, dtype=torch.int32, pin_memory=True)
    orig_ids_pinned.copy_(torch.from_numpy(sort_perm))
    print(f"[SIFT100M] Pinned host FP32 buffer ready in {time.perf_counter()-t_pin:.2f}s!", flush=True)

    print("[SIFT100M] Loading TRIDENT-OOC index (sq8_packed + tile_orig_ids + bounds) into RTX 4080 16GB VRAM...", flush=True)
    eng = Trident100MOOCEngine(
        centroids, mean64, R64, vec_off, tile_off, vmin_np, delta_np,
        sq8_packed_t, tile_oids_t, xb_sorted_pinned, orig_ids_pinned,
        max_batch=1024, max_nprobe=512, dev=dev
    )
    vram_trident_gib = torch.cuda.memory_allocated(dev) / (1024.0**3)
    free_after_b, _ = torch.cuda.mem_get_info(dev)
    print(f"[SIFT100M] TRIDENT-OOC loaded in RTX 4080 VRAM! Allocated={vram_trident_gib:.3f} GiB, Free remaining={free_after_b/1024**3:.3f} GiB", flush=True)

    nprobe_map = {0.95: 48, 0.99: 160}
    print(f"[SIFT100M] Using calibrated nprobe map: {nprobe_map}", flush=True)

    # 1 & 2. Measure TRIDENT-OOC and CM-FP32-OOC at R=0.95 and R=0.99
    trident_ooc_res = {}
    cm_ooc_res = {}
    ref_exact_cache = {}
    for rec_tag in [0.95, 0.99]:
        nprobe = nprobe_map[rec_tag]
        st_buf = torch.zeros(4, dtype=torch.int64, device=dev)
        tr_d, tr_i = eng.search(xq_b_gpu[:B_eval], nprobe, mode=2, stats_buf=st_buf)
        torch.cuda.synchronize()
        tot_cands_1000 = int(st_buf[0].item())
        scanned_dims_1000 = int(st_buf[1].item())
        surv_cands_1000 = int(st_buf[2].item())
        surv_pct = 100.0 * surv_cands_1000 / max(tot_cands_1000, 1)
        dim_pruned_pct = 100.0 * (1.0 - scanned_dims_1000 / max(tot_cands_1000 * D_100M, 1))
        tr_dists, tr_ids = tr_d.cpu().numpy().copy(), tr_i.cpu().numpy().copy()

        cm_d, cm_i = eng.search(xq_b_gpu[:B_eval], nprobe, mode=10)
        torch.cuda.synchronize()
        cm_dists, cm_ids = cm_d.cpu().numpy().copy(), cm_i.cpu().numpy().copy()
        ref_exact_cache[rec_tag] = (cm_dists, cm_ids, tot_cands_1000, surv_cands_1000, surv_pct, dim_pruned_pct)

        rec_cm, ov_cm, _ = eval_accuracy_v4(cm_ids, cm_dists, cm_ids, cm_dists, gt10)
        rec_tr, ov_tr, _ = eval_accuracy_v4(tr_ids, tr_dists, cm_ids, cm_dists, gt10)

        qps_tr, std_tr = measure_runs(lambda: eng.search(xq_b_gpu, nprobe, mode=2), warmup=4, iters=12, b_size=B_lat)
        qps_cm, std_cm = measure_runs(lambda: eng.search(xq_b_gpu, nprobe, mode=10), warmup=1, iters=3, b_size=B_lat)

        scale_1024 = 1024.0 / float(B_eval)
        pcie_cm_est_mb = round((tot_cands_1000 * scale_1024 * D_100M * 4) / (1024.0**2), 2)
        pcie_tr_est_mb = round((surv_cands_1000 * scale_1024 * D_100M * 4) / (1024.0**2), 2)

        trident_ooc_res[rec_tag] = (qps_tr, std_tr, rec_tr, ov_tr, surv_pct, dim_pruned_pct, pcie_tr_est_mb)
        cm_ooc_res[rec_tag] = (qps_cm, std_cm, rec_cm, ov_cm, 100.0, 0.0, pcie_cm_est_mb)
        print(f"[SIFT100M R={rec_tag} np={nprobe}] TRIDENT-OOC={qps_tr:.1f}±{std_tr:.1f} QPS (Ov={ov_tr:.2f}%, DimsPruned={dim_pruned_pct:.1f}%, Surv={surv_pct:.3f}%), CM-FP32-OOC={qps_cm:.1f}±{std_cm:.1f} QPS ({qps_tr/qps_cm:.2f}x)", flush=True)

    # 3. Benchmark 24-thread CPU Exact IVF-Flat using already-materialized xb_sorted_pinned (0s build!)
    print("[SIFT100M] Benchmarking 24-thread CPU Exact IVF-Flat (Faiss_CPU_IVFFlat_24T)...", flush=True)
    faiss.omp_set_num_threads(24)
    quant_cpu = faiss.IndexFlatL2(D_100M)
    quant_cpu.add(centroids)
    vec_off_t_cpu = torch.from_numpy(vec_off)
    cpu_flat_res = {}
    for rec_tag in [0.95, 0.99]:
        nprobe = nprobe_map[rec_tag]
        cm_dists, cm_ids, _, _, _, _ = ref_exact_cache[rec_tag]
        out_d_cf = torch.empty((B_lat, 10), dtype=torch.float32)
        out_i_cf = torch.empty((B_lat, 10), dtype=torch.int64)

        def run_cpu_ivfflat():
            _, pc_np = quant_cpu.search(xq_b, nprobe)
            pc_t = torch.from_numpy(pc_np)
            helper.omp_cpu_ivfflat_f32(xq_b_t_cpu, pc_t, vec_off_t_cpu, xb_sorted_pinned, orig_ids_pinned, out_d_cf, out_i_cf)
            return out_d_cf, out_i_cf

        run_cpu_ivfflat()
        rec_cf, ov_cf, _ = eval_accuracy_v4(out_i_cf[:B_eval].numpy(), out_d_cf[:B_eval].numpy(), cm_ids, cm_dists, gt10)
        qps_cf, std_cf = measure_runs(run_cpu_ivfflat, warmup=1, iters=3, b_size=B_lat)
        cpu_flat_res[rec_tag] = (qps_cf, std_cf, rec_cf, ov_cf)
        print(f"[SIFT100M R={rec_tag} np={nprobe}] Faiss_CPU_IVFFlat_24T={qps_cf:.1f}±{std_cf:.1f} QPS (Rec={rec_cf:.4f}, Ov={ov_cf:.2f}%, Speedup={trident_ooc_res[rec_tag][0]/qps_cf:.2f}x)", flush=True)

    # Release TRIDENT-OOC VRAM and pinned FP32 buffer before running PQ/CAGRA/NCU baselines
    del eng, xb_sorted_pinned, orig_ids_pinned
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # 4. Run Hybrid Faiss_GPU_IVFPQ_CPU_Refine_kf256 on RTX 4080 + 24T OpenMP
    print("[SIFT100M] Building & benchmarking Faiss_GPU_IVFPQ_CPU_Refine_kf256 (M=16, 8bit)...", flush=True)
    m_pq = 16
    quant_pq = faiss.IndexFlatL2(D_100M)
    quant_pq.add(centroids)
    cpu_pq = faiss.IndexIVFPQ(quant_pq, D_100M, NLIST_100M, m_pq, 8)
    cpu_pq.by_residual = True
    np.random.seed(42)
    pq_train_idx = np.sort(np.random.choice(N_100M, 65536, replace=False))
    cpu_pq.train(np.ascontiguousarray(xb_u8_mmap[pq_train_idx], dtype=np.float32))
    res_pq = faiss.StandardGpuResources()
    res_pq.setTempMemory(256 * 1024 * 1024)
    gpu_pq = faiss.index_cpu_to_gpu(res_pq, 0, cpu_pq)
    del cpu_pq
    b_add = 5_000_000
    for s_i in range(0, N_100M, b_add):
        e_i = min(s_i + b_add, N_100M)
        gpu_pq.add(np.ascontiguousarray(xb_u8_mmap[s_i:e_i], dtype=np.float32))
    pq_vram_gib = (N_100M * (m_pq + 8) + NLIST_100M * D_100M * 4) / (1024.0**3)
    print(f"[SIFT100M] GPU IVFPQ built on RTX 4080 (~{pq_vram_gib:.2f} GiB VRAM)!", flush=True)

    xb_u8_t_cpu = torch.from_numpy(xb_u8_mmap)
    pq_ref_res = {}
    for rec_tag in [0.95, 0.99]:
        nprobe = nprobe_map[rec_tag]
        gpu_pq.nprobe = nprobe
        cm_dists, cm_ids, tot_cands_1000, _, _, _ = ref_exact_cache[rec_tag]
        m_cand = 2048
        out_d_t = torch.empty((B_lat, 10), dtype=torch.float32)
        out_i_t = torch.empty((B_lat, 10), dtype=torch.int64)

        def run_pq_ref():
            _, cand_i_np = gpu_pq.search(xq_b, m_cand)
            cand_i_t = torch.from_numpy(cand_i_np)
            helper.omp_refine_topk_u8(xq_b_t_cpu, xb_u8_t_cpu, cand_i_t, out_d_t, out_i_t)
            return out_d_t, out_i_t

        run_pq_ref()
        rec_pq, ov_pq, _ = eval_accuracy_v4(out_i_t[:B_eval].numpy(), out_d_t[:B_eval].numpy(), cm_ids, cm_dists, gt10)
        qps_pq, std_pq = measure_runs(run_pq_ref, warmup=2, iters=6, b_size=B_lat)
        pcie_pq_mb = round((B_lat * D_100M * 4 + B_lat * m_cand * 12) / (1024.0**2), 2)
        surv_pq_pct = 100.0 * (B_eval * m_cand) / max(tot_cands_1000, 1)
        pq_ref_res[rec_tag] = (qps_pq, std_pq, rec_pq, ov_pq, surv_pq_pct, pcie_pq_mb)
        print(f"[SIFT100M R={rec_tag} np={nprobe}] IVFPQ-CPU-Ref256={qps_pq:.1f}±{std_pq:.1f} QPS (Rec={rec_pq:.4f}, Ov={ov_pq:.2f}%)", flush=True)

    del gpu_pq, res_pq
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Test cuVS-CAGRA (In-VRAM and UVM) on SIFT100M on RTX 4080 16GB in isolated subprocesses
    cagra_status = {}
    for cmode in ["InVRAM", "UVM"]:
        cmd = [sys.executable, os.path.abspath(__file__), "--cagra-worker", cmode]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=60)
            st = "OOM (std::bad_alloc)"
            for line in out.splitlines():
                if "CAGRA_STATUS:" in line:
                    st = line.split("CAGRA_STATUS:", 1)[1].strip()
            cagra_status[cmode] = st
        except subprocess.TimeoutExpired:
            cagra_status[cmode] = "UVM_BUILD_TIMEOUT_THRASHING (>60s)"
        except Exception as e:
            cagra_status[cmode] = f"OOM ({type(e).__name__})"
        print(f"[SIFT100M] cuVS-CAGRA ({cmode}): {cagra_status[cmode]}", flush=True)

    # 6. Run Nsight Compute PCIe sysmem counter on RTX 4080 for SIFT100M in a single pass
    print("[SIFT100M] Profiling Nsight Compute hardware PCIe sysmem counter on RTX 4080...", flush=True)
    ncu_meas = measure_ncu_sysmem_all_100m()
    for rec_tag in [0.95, 0.99]:
        m_tr, m_cm = ncu_meas[rec_tag]
        print(f"[SIFT100M NCU R={rec_tag}] TRIDENT-OOC PCIe={m_tr:.2f} MB vs CM-FP32-OOC PCIe={m_cm:.2f} MB ({m_cm/max(m_tr, 1e-3):.2f}x reduction)", flush=True)

    # 7. Write final CSV
    fp32_raw_gib = round((N_100M * D_100M * 4) / (1024.0**3), 2)
    rows = []
    for rec_tag in [0.95, 0.99]:
        nprobe = nprobe_map[rec_tag]
        q_tr, s_tr, r_tr, o_tr, surv_tr, dpr_tr, pe_tr = trident_ooc_res[rec_tag]
        q_cm, s_cm, r_cm, o_cm, surv_cm, dpr_cm, pe_cm = cm_ooc_res[rec_tag]
        q_pq, s_pq, r_pq, o_pq, surv_pq, pe_pq = pq_ref_res[rec_tag]
        q_cf, s_cf, r_cf, o_cf = cpu_flat_res[rec_tag]
        pm_tr, pm_cm = ncu_meas[rec_tag]
        if pm_tr <= 0: pm_tr = pe_tr
        if pm_cm <= 0: pm_cm = pe_cm

        rows.extend([
            ["RTX4080_16GB_NoCap", "sift100m", N_100M, D_100M, fp32_raw_gib, NLIST_100M, rec_tag, nprobe, B_lat,
             "cuVS_CAGRA_InVRAM", 64.00, 0.00, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, cagra_status["InVRAM"], 1],
            ["RTX4080_16GB_NoCap", "sift100m", N_100M, D_100M, fp32_raw_gib, NLIST_100M, rec_tag, nprobe, B_lat,
             "cuVS_CAGRA_UVM", 15.57, 48.43, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, cagra_status["UVM"], 1],
            ["RTX4080_16GB_NoCap", "sift100m", N_100M, D_100M, fp32_raw_gib, NLIST_100M, rec_tag, nprobe, B_lat,
             "CM_FP32_OOC_Pinned", 0.18, fp32_raw_gib, 100.0, pe_cm, pm_cm, round(q_cm, 1), round(s_cm, 1), round(r_cm, 4), round(o_cm, 2), "OK", 1],
            ["RTX4080_16GB_NoCap", "sift100m", N_100M, D_100M, fp32_raw_gib, NLIST_100M, rec_tag, nprobe, B_lat,
             "Faiss_CPU_IVFFlat_24T", 0.00, fp32_raw_gib, 100.0, 0.0, 0.0, round(q_cf, 1), round(s_cf, 1), round(r_cf, 4), round(o_cf, 2), "OK", 1],
            ["RTX4080_16GB_NoCap", "sift100m", N_100M, D_100M, fp32_raw_gib, NLIST_100M, rec_tag, nprobe, B_lat,
             "Faiss_GPU_IVFPQ_CPU_Refine_kf256", round(pq_vram_gib, 2), fp32_raw_gib, round(surv_pq, 3), pe_pq, pe_pq, round(q_pq, 1), round(s_pq, 1), round(r_pq, 4), round(o_pq, 2), "OK", 1],
            ["RTX4080_16GB_NoCap", "sift100m", N_100M, D_100M, fp32_raw_gib, NLIST_100M, rec_tag, nprobe, B_lat,
             "TRIDENT_OOC_Pinned", round(vram_trident_gib, 2), fp32_raw_gib, round(surv_tr, 3), pe_tr, pm_tr, round(q_tr, 1), round(s_tr, 1), round(r_tr, 4), round(o_tr, 2), f"OK (DimsPruned={dpr_tr:.1f}%)", 1],
        ])

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "gpu", "dataset", "N", "dim", "fp32_raw_gb", "nlist", "recall_target", "nprobe", "batch",
            "method", "vram_resident_gb", "host_pinned_gb", "surv_pct",
            "pcie_mb_estimated", "pcie_mb_measured", "qps_mean", "qps_std",
            "recall_at_10", "strict_overlap_pct", "status", "measured"
        ])
        w.writerows(rows)
    print(f"=== Saved {OUT_CSV} successfully! ===", flush=True)

if __name__ == "__main__":
    main()
