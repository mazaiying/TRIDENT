#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <cstdlib>  // PLAN_V10.1: std::getenv for TRIDENT_UB_SEED

namespace {

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int K>
__device__ __forceinline__ void insert_topk(
    float* sh_d,
    int64_t* sh_i,
    float d_val,
    int64_t id_val
) {
    if (d_val < sh_d[K - 1] || (d_val == sh_d[K - 1] && id_val < sh_i[K - 1])) {
        sh_d[K - 1] = d_val;
        sh_i[K - 1] = id_val;
        #pragma unroll
        for (int j = K - 1; j > 0; --j) {
            if (sh_d[j] < sh_d[j - 1] || (sh_d[j] == sh_d[j - 1] && sh_i[j] < sh_i[j - 1])) {
                float td = sh_d[j - 1]; sh_d[j - 1] = sh_d[j]; sh_d[j] = td;
                int64_t ti = sh_i[j - 1]; sh_i[j - 1] = sh_i[j]; sh_i[j] = ti;
            } else {
                break;
            }
        }
    }
}

__device__ __forceinline__ void insert_topk_dyn(
    float* sh_d,
    int64_t* sh_i,
    int K,
    float d_val,
    int64_t id_val
) {
    if (d_val < sh_d[K - 1] || (d_val == sh_d[K - 1] && id_val < sh_i[K - 1])) {
        sh_d[K - 1] = d_val;
        sh_i[K - 1] = id_val;
        for (int j = K - 1; j > 0; --j) {
            if (sh_d[j] < sh_d[j - 1] || (sh_d[j] == sh_d[j - 1] && sh_i[j] < sh_i[j - 1])) {
                float td = sh_d[j - 1]; sh_d[j - 1] = sh_d[j]; sh_d[j] = td;
                int64_t ti = sh_i[j - 1]; sh_i[j - 1] = sh_i[j]; sh_i[j] = ti;
            } else {
                break;
            }
        }
    }
}

/*
 * Fused Frontend Kernel 1:
 *   Given cdists_part [B, stride] where columns [0, nlist) contain ||c||^2 - 2<q, c>,
 *   computes ||q||^2 from xq_orig [B, D], adds ||q||^2, and selects the top-nprobe smallest
 *   centroid distances in exact sorted order (ties broken by smaller cluster ID).
 */
__global__ void trident_topk_centroids_kernel(
    const float* __restrict__ cdists_part, // [B, stride]
    const float* __restrict__ xq_orig,     // [B, D]
    int32_t* __restrict__ probe_cids,      // [B, nprobe]
    float* __restrict__ probe_dqcs,        // [B, nprobe]
    const int B,
    const int D,
    const int nlist,
    const int stride,
    const int nprobe
) {
    const int q = blockIdx.x;
    if (q >= B) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    extern __shared__ float sh_cdist[]; // [nlist]
    __shared__ float sh_warp_sum[8];
    __shared__ float sh_q_norm_sq;
    __shared__ float sh_w_min_d[8];
    __shared__ int32_t sh_w_min_c[8];

    const float* q_ptr = xq_orig + (int64_t)q * D;
    float q_part = 0.0f;
    for (int d = tid; d < D; d += 256) {
        float v = q_ptr[d];
        q_part = fmaf(v, v, q_part);
    }
    q_part = warp_reduce_sum(q_part);
    if (lane == 0) sh_warp_sum[warp_id] = q_part;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) total += sh_warp_sum[w];
        sh_q_norm_sq = total;
    }
    __syncthreads();

    const float q_nsq = sh_q_norm_sq;
    const float* c_row = cdists_part + (int64_t)q * stride;
    for (int c = tid; c < nlist; c += 256) {
        sh_cdist[c] = fmaxf(c_row[c] + q_nsq, 0.0f);
    }
    __syncthreads();

    for (int p = 0; p < nprobe; ++p) {
        float best_d = INFINITY;
        int32_t best_c = 2147483647;
        for (int c = tid; c < nlist; c += 256) {
            float d = sh_cdist[c];
            if (d < best_d || (d == best_d && c < best_c)) {
                best_d = d;
                best_c = c;
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float od = __shfl_down_sync(0xffffffff, best_d, offset);
            int32_t oc = __shfl_down_sync(0xffffffff, best_c, offset);
            if (od < best_d || (od == best_d && oc < best_c)) {
                best_d = od;
                best_c = oc;
            }
        }
        if (lane == 0) {
            sh_w_min_d[warp_id] = best_d;
            sh_w_min_c[warp_id] = best_c;
        }
        __syncthreads();

        if (tid < 8) {
            float wd = sh_w_min_d[tid];
            int32_t wc = sh_w_min_c[tid];
            #pragma unroll
            for (int offset = 4; offset > 0; offset >>= 1) {
                float od = __shfl_down_sync(0xff, wd, offset);
                int32_t oc = __shfl_down_sync(0xff, wc, offset);
                if (od < wd || (od == wd && oc < wc)) {
                    wd = od;
                    wc = oc;
                }
            }
            if (tid == 0) {
                probe_cids[q * nprobe + p] = wc;
                probe_dqcs[q * nprobe + p] = sqrtf(wd);
                sh_cdist[wc] = INFINITY;
            }
        }
        __syncthreads();
    }
}

/*
 * Fused Frontend Kernel 2:
 *   Shared-memory Counting Sort of (cid, qid, dqc) pairs by cluster_id in [0, nlist).
 */
__global__ void trident_counting_sort_pairs_kernel(
    const int32_t* __restrict__ probe_cids, // [B, nprobe]
    const float* __restrict__ probe_dqcs,   // [B, nprobe]
    const int B,
    const int nprobe,
    const int p_init,
    const int nlist,
    int32_t* __restrict__ p1_cids,
    int32_t* __restrict__ p1_qids,
    float* __restrict__ p1_dqcs,
    int32_t* __restrict__ p2_cids,
    int32_t* __restrict__ p2_qids,
    float* __restrict__ p2_dqcs
) {
    const int pass_id = blockIdx.x;
    const int p_start = (pass_id == 0) ? 0 : p_init;
    const int p_end   = (pass_id == 0) ? p_init : nprobe;
    const int P       = p_end - p_start;
    if (P <= 0) return;

    int32_t* out_cids = (pass_id == 0) ? p1_cids : p2_cids;
    int32_t* out_qids = (pass_id == 0) ? p1_qids : p2_qids;
    float*   out_dqcs = (pass_id == 0) ? p1_dqcs : p2_dqcs;

    const int tid = threadIdx.x;
    const int total = B * P;

    if (B == 1) {
        for (int idx = tid; idx < total; idx += 256) {
            int p = p_start + idx;
            out_cids[idx] = probe_cids[p];
            out_qids[idx] = 0;
            out_dqcs[idx] = probe_dqcs[p];
        }
        return;
    }

    extern __shared__ int32_t sh_hist[]; // [nlist + 1]
    __shared__ int32_t sh_warp_tot[8];

    for (int c = tid; c <= nlist; c += 256) {
        sh_hist[c] = 0;
    }
    __syncthreads();

    for (int idx = tid; idx < total; idx += 256) {
        int q = idx / P;
        int p = p_start + (idx - q * P);
        int32_t cid = probe_cids[q * nprobe + p];
        atomicAdd(&sh_hist[cid], 1);
    }
    __syncthreads();

    int32_t running_base = 0;
    for (int base = 0; base < nlist; base += 256) {
        int c = base + tid;
        int32_t val = (c < nlist) ? sh_hist[c] : 0;
        int lane = tid & 31;
        int warp_id = tid >> 5;
        int32_t scan = val;
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            int32_t n = __shfl_up_sync(0xffffffff, scan, offset);
            if (lane >= offset) scan += n;
        }
        if (lane == 31) sh_warp_tot[warp_id] = scan;
        __syncthreads();

        if (tid < 8) {
            int32_t wscan = sh_warp_tot[tid];
            #pragma unroll
            for (int offset = 1; offset < 8; offset <<= 1) {
                int32_t n = __shfl_up_sync(0xff, wscan, offset);
                if (tid >= offset) wscan += n;
            }
            sh_warp_tot[tid] = wscan;
        }
        __syncthreads();

        int32_t warp_prefix = (warp_id > 0) ? sh_warp_tot[warp_id - 1] : 0;
        int32_t exc = running_base + warp_prefix + scan - val;
        if (c < nlist) sh_hist[c] = exc;
        running_base += sh_warp_tot[7];
        __syncthreads();
    }

    for (int idx = tid; idx < total; idx += 256) {
        int q = idx / P;
        int p = p_start + (idx - q * P);
        int32_t cid = probe_cids[q * nprobe + p];
        int32_t dst = atomicAdd(&sh_hist[cid], 1);
        out_cids[dst] = cid;
        out_qids[dst] = q;
        out_dqcs[dst] = probe_dqcs[q * nprobe + p];
    }
}

/*
 * MODE 10: CM-FP32 (Cluster-Major Exact FP32 Full-Dimension Scan, No Early-Stop)
 */
template <int TILE_SIZE>
__global__ void trident_cm_fp32_scan_kernel(
    const float* __restrict__ xb_sorted,          // [N, D] contiguous in cluster order
    const int32_t* __restrict__ orig_ids_sorted,  // [N] original vector IDs
    const int32_t* __restrict__ cluster_vec_off,  // [nlist + 1]
    const float* __restrict__ xq_orig,            // [B, D]
    const int32_t* __restrict__ sorted_cids,      // [M_pairs]
    const int32_t* __restrict__ sorted_qids,      // [M_pairs]
    const int M_pairs,
    const int D,
    const int K,
    const int max_cands_per_q,
    float* __restrict__ cand_buf_d,               // [B, max_cands_per_q]
    int64_t* __restrict__ cand_buf_i,             // [B, max_cands_per_q]
    int32_t* __restrict__ cand_cnt,               // [B]
    int64_t* __restrict__ stats_out = nullptr     // [4] optional counters
) {
    const int pair_idx = blockIdx.x;
    if (pair_idx >= M_pairs) return;

    const int32_t cid = sorted_cids[pair_idx];
    const int32_t q   = sorted_qids[pair_idx];
    const int32_t v_start = cluster_vec_off[cid];
    const int32_t v_end   = cluster_vec_off[cid + 1];
    if (v_start >= v_end) return;

    constexpr int W_TILE = TILE_SIZE >> 5;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int K_CAP = (K <= 100) ? K : 100;

    extern __shared__ float smem_fp32[];
    float* sh_q_orig = smem_fp32; // [D]

    __shared__ float sh_warp_d[W_TILE][10];
    __shared__ int64_t sh_warp_i[W_TILE][10];

    const float* q_ptr = xq_orig + (int64_t)q * D;
    for (int d = tid; d < D; d += TILE_SIZE) {
        sh_q_orig[d] = q_ptr[d];
    }
    for (int j = lane; j < K_CAP; j += 32) {
        sh_warp_d[warp_id][j] = INFINITY;
        sh_warp_i[warp_id][j] = -1;
    }
    __syncthreads();

    float* my_wd = sh_warp_d[warp_id];
    int64_t* my_wi = sh_warp_i[warp_id];
    float warp_tau_sq = INFINITY;

    for (int32_t v = v_start + warp_id; v < v_end; v += W_TILE) {
        const float* x_ptr = xb_sorted + (int64_t)v * D;
        float part = 0.0f;
        int d = lane * 4;
        for (; d + 3 < D; d += 128) {
            float4 xv = *reinterpret_cast<const float4*>(x_ptr + d);
            float4 qv = *reinterpret_cast<const float4*>(sh_q_orig + d);
            float df0 = xv.x - qv.x; part = fmaf(df0, df0, part);
            float df1 = xv.y - qv.y; part = fmaf(df1, df1, part);
            float df2 = xv.z - qv.z; part = fmaf(df2, df2, part);
            float df3 = xv.w - qv.w; part = fmaf(df3, df3, part);
        }
        for (int dr = d; dr < d + 4 && dr < D; ++dr) {
            float df = x_ptr[dr] - sh_q_orig[dr];
            part = fmaf(df, df, part);
        }
        part = warp_reduce_sum(part);
        if (lane == 0 && part <= warp_tau_sq) {
            int64_t oid = (int64_t)orig_ids_sorted[v];
            if (K_CAP == 10) insert_topk<10>(my_wd, my_wi, part, oid);
            else if (K_CAP == 1) insert_topk<1>(my_wd, my_wi, part, oid);
            else insert_topk_dyn(my_wd, my_wi, K_CAP, part, oid);
            warp_tau_sq = my_wd[K_CAP - 1];
        }
    }
    __syncthreads();

    if (tid == 0) {
        float* loc_d = sh_warp_d[0];
        int64_t* loc_i = sh_warp_i[0];
        for (int w = 1; w < W_TILE; ++w) {
            for (int j = 0; j < K_CAP; ++j) {
                if (sh_warp_i[w][j] >= 0) {
                    if (K_CAP == 10) insert_topk<10>(loc_d, loc_i, sh_warp_d[w][j], sh_warp_i[w][j]);
                    else if (K_CAP == 1) insert_topk<1>(loc_d, loc_i, sh_warp_d[w][j], sh_warp_i[w][j]);
                    else insert_topk_dyn(loc_d, loc_i, K_CAP, sh_warp_d[w][j], sh_warp_i[w][j]);
                }
            }
        }
        int k_emit = 0;
        for (int j = 0; j < K_CAP; ++j) {
            if (loc_i[j] >= 0) k_emit++;
            else break;
        }
        if (k_emit > 0) {
            int base_slot = atomicAdd(&cand_cnt[q], k_emit);
            for (int j = 0; j < k_emit; ++j) {
                if (base_slot + j < max_cands_per_q) {
                    cand_buf_d[(int64_t)q * max_cands_per_q + base_slot + j] = loc_d[j];
                    cand_buf_i[(int64_t)q * max_cands_per_q + base_slot + j] = loc_i[j];
                }
            }
        }
    }
}

/*
 * MODE 11: Tribase-GPU (Cluster-Major Exact FP32 with Triangle Inequality Lower-Bound Pruning | ||q-c|| - ||x-c|| | > tau)
 */
template <int TILE_SIZE>
__global__ void trident_tribase_fp32_scan_kernel(
    const float* __restrict__ xb_sorted,          // [N, D] contiguous in cluster order
    const int32_t* __restrict__ orig_ids_sorted,  // [N] original vector IDs
    const float* __restrict__ r_vec_sorted,       // [N] precomputed ||x - c|| in cluster order
    const int32_t* __restrict__ cluster_vec_off,  // [nlist + 1]
    const float* __restrict__ xq_orig,            // [B, D]
    const int32_t* __restrict__ sorted_cids,      // [M_pairs]
    const int32_t* __restrict__ sorted_qids,      // [M_pairs]
    const float* __restrict__ sorted_dqcs,        // [M_pairs] ||q - c||
    const float* __restrict__ init_tau_sq,        // [B] from Pass 1 (nullptr in Pass 1)
    const int M_pairs,
    const int D,
    const int K,
    const int max_cands_per_q,
    float* __restrict__ cand_buf_d,               // [B, max_cands_per_q]
    int64_t* __restrict__ cand_buf_i,             // [B, max_cands_per_q]
    int32_t* __restrict__ cand_cnt,               // [B]
    int64_t* __restrict__ stats_out = nullptr     // [4] optional counters
) {
    const int pair_idx = blockIdx.x;
    if (pair_idx >= M_pairs) return;

    const int32_t cid = sorted_cids[pair_idx];
    const int32_t q   = sorted_qids[pair_idx];
    const float d_qc  = sorted_dqcs[pair_idx];
    const int32_t v_start = cluster_vec_off[cid];
    const int32_t v_end   = cluster_vec_off[cid + 1];
    if (v_start >= v_end) return;

    constexpr int W_TILE = TILE_SIZE >> 5;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int K_CAP = (K <= 100) ? K : 100;

    extern __shared__ float smem_fp32[];
    float* sh_q_orig = smem_fp32; // [D]

    __shared__ float sh_warp_d[W_TILE][100];
    __shared__ int64_t sh_warp_i[W_TILE][100];
    __shared__ float sh_tau_sq;
    __shared__ float sh_tau;

    const float* q_ptr = xq_orig + (int64_t)q * D;
    for (int d = tid; d < D; d += TILE_SIZE) {
        sh_q_orig[d] = q_ptr[d];
    }
    for (int j = lane; j < K_CAP; j += 32) {
        sh_warp_d[warp_id][j] = INFINITY;
        sh_warp_i[warp_id][j] = -1;
    }
    if (tid == 0) {
        float tsq = (init_tau_sq != nullptr) ? init_tau_sq[q] : INFINITY;
        sh_tau_sq = tsq;
        sh_tau = isinf(tsq) ? INFINITY : sqrtf(fmaxf(tsq, 0.0f));
    }
    __syncthreads();

    float* my_wd = sh_warp_d[warp_id];
    int64_t* my_wi = sh_warp_i[warp_id];
    float warp_tau_sq = sh_tau_sq;
    float warp_tau = sh_tau;
    const float q_init_tau_sq = sh_tau_sq;

    for (int32_t v = v_start + warp_id; v < v_end; v += W_TILE) {
        int pruned = 0;
        if (lane == 0) {
            if (__builtin_expect(stats_out != nullptr, 0)) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&stats_out[0]), 1ULL);
            }
            float r_v = r_vec_sorted[v];
            float lb = fabsf(d_qc - r_v);
            if (!isinf(warp_tau) && lb > warp_tau * (1.0f + 1e-5f) + 1e-5f) {
                pruned = 1;
            } else if (__builtin_expect(stats_out != nullptr, 0)) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&stats_out[1]), (unsigned long long)D);
                atomicAdd(reinterpret_cast<unsigned long long*>(&stats_out[2]), 1ULL);
            }
        }
        pruned = __shfl_sync(0xffffffff, pruned, 0);
        if (pruned) continue;

        const float* x_ptr = xb_sorted + (int64_t)v * D;
        float part = 0.0f;
        int d = lane * 4;
        for (; d + 3 < D; d += 128) {
            float4 xv = *reinterpret_cast<const float4*>(x_ptr + d);
            float4 qv = *reinterpret_cast<const float4*>(sh_q_orig + d);
            float df0 = xv.x - qv.x; part = fmaf(df0, df0, part);
            float df1 = xv.y - qv.y; part = fmaf(df1, df1, part);
            float df2 = xv.z - qv.z; part = fmaf(df2, df2, part);
            float df3 = xv.w - qv.w; part = fmaf(df3, df3, part);
        }
        for (int dr = d; dr < d + 4 && dr < D; ++dr) {
            float df = x_ptr[dr] - sh_q_orig[dr];
            part = fmaf(df, df, part);
        }
        part = warp_reduce_sum(part);
        if (lane == 0 && part <= warp_tau_sq) {
            int64_t oid = (int64_t)orig_ids_sorted[v];
            if (K_CAP == 10) insert_topk<10>(my_wd, my_wi, part, oid);
            else if (K_CAP == 1) insert_topk<1>(my_wd, my_wi, part, oid);
            else insert_topk_dyn(my_wd, my_wi, K_CAP, part, oid);
            if (my_wd[K_CAP - 1] < warp_tau_sq) {
                warp_tau_sq = my_wd[K_CAP - 1];
                warp_tau = sqrtf(fmaxf(warp_tau_sq, 0.0f));
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        float* loc_d = sh_warp_d[0];
        int64_t* loc_i = sh_warp_i[0];
        for (int w = 1; w < W_TILE; ++w) {
            for (int j = 0; j < K_CAP; ++j) {
                if (sh_warp_i[w][j] >= 0) {
                    if (K_CAP == 10) insert_topk<10>(loc_d, loc_i, sh_warp_d[w][j], sh_warp_i[w][j]);
                    else if (K_CAP == 1) insert_topk<1>(loc_d, loc_i, sh_warp_d[w][j], sh_warp_i[w][j]);
                    else insert_topk_dyn(loc_d, loc_i, K_CAP, sh_warp_d[w][j], sh_warp_i[w][j]);
                }
            }
        }
        int k_emit = 0;
        for (int j = 0; j < K_CAP; ++j) {
            if (loc_i[j] >= 0 && (isinf(q_init_tau_sq) || loc_d[j] <= q_init_tau_sq)) {
                k_emit++;
            } else {
                break;
            }
        }
        if (k_emit > 0) {
            int base_slot = atomicAdd(&cand_cnt[q], k_emit);
            for (int j = 0; j < k_emit; ++j) {
                if (base_slot + j < max_cands_per_q) {
                    cand_buf_d[(int64_t)q * max_cands_per_q + base_slot + j] = loc_d[j];
                    cand_buf_i[(int64_t)q * max_cands_per_q + base_slot + j] = loc_i[j];
                }
            }
        }
    }
}

/*
 * MODE:
 *   0 = K0 / CM-SQ8+RR (SQ8 full-dimension cluster-major scan, no early-stop; emits top-k_local SQ8 candidates per cluster)
 *   1 = K1 (SQ8 tile/warp dimension-block early-stop + L1 band, no FP32 refinement)
 *   2 = K2 (K1 + warp-parallel exact FP32 survivor refinement over cluster-contiguous xb_sorted -> 100% lossless)
 */
template <int MODE, int TILE_SIZE, int WARP_REFINE = 1, int UB_ENABLE = 0>
__global__ void trident_cluster_major_scan_kernel(
    const uint32_t* __restrict__ sq8_packed,      // [total_tiles, D4, W_tile, 32]
    const int32_t* __restrict__ tile_orig_ids,    // [total_tiles, TILE_SIZE]
    const float* __restrict__ tile_r_minmax,      // [total_tiles, 2]
    const int32_t* __restrict__ cluster_tile_off, // [nlist + 1]
    const int32_t* __restrict__ cluster_vec_off,  // [nlist + 1]
    const float* __restrict__ xb_sorted,          // [N, D] contiguous in cluster order!
    const float* __restrict__ xq_orig,            // [B, D]
    const float* __restrict__ xq_pca,             // [B, xq_pca_stride] (offset to PCA cols passed by caller)
    const int xq_pca_stride,
    const float* __restrict__ vmin,               // [1 or nlist, D_pad]
    const float* __restrict__ delta,              // [1 or nlist, D_pad]
    const float* __restrict__ eps_dim,            // [1 or nlist, D_pad]
    const int per_cluster_sq8,                    // 0 = global [1, D_pad], 1 = per-cluster [nlist, D_pad]
    const int32_t* __restrict__ blk_bounds4,      // [num_blocks + 1] (in units of d4 = dim/4)
    const int32_t* __restrict__ sorted_cids,      // [M_pairs] sorted by cluster_id!
    const int32_t* __restrict__ sorted_qids,      // [M_pairs]
    const float* __restrict__ sorted_dqcs,        // [M_pairs]
    const float* __restrict__ init_tau_sq,        // [B] from Pass 1 (nullptr in Pass 1)
    const int M_pairs,
    const int D,
    const int D_pad,
    const int num_blocks,
    const int use_l1,
    const int k_local,                            // 10 for normal top-10, up to 100 for CM-SQ8+RR
    const int max_cands_per_q,
    float* __restrict__ cand_buf_d,               // [B, max_cands_per_q]
    int64_t* __restrict__ cand_buf_i,             // [B, max_cands_per_q]
    int32_t* __restrict__ cand_cnt,               // [B]
    int64_t* __restrict__ stats_out = nullptr     // [4] optional counters
) {
    const int pair_idx = blockIdx.x;
    if (pair_idx >= M_pairs) return;

    const int32_t cid = sorted_cids[pair_idx];
    const int32_t q   = sorted_qids[pair_idx];
    const float d_qc  = sorted_dqcs[pair_idx];

    const int32_t t_start = cluster_tile_off[cid];
    const int32_t t_end   = cluster_tile_off[cid + 1];
    if (t_start >= t_end) return;

    const int32_t c_vec_start = cluster_vec_off[cid];

    constexpr int W_TILE = TILE_SIZE >> 5;
    constexpr int MAX_K_LOCAL = (MODE == 0) ? 200 : 10;
    const int K_LOC = (MODE == 0) ? ((k_local <= MAX_K_LOCAL) ? ((k_local >= 1) ? k_local : 1) : MAX_K_LOCAL) : 10;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int D4 = D_pad >> 2;

    extern __shared__ float smem[];
    float* sh_base   = smem;                   // [D_pad]
    float* sh_delta  = smem + D_pad;           // [D_pad]
    float* sh_eps    = smem + 2 * D_pad;       // [D_pad]
    float* sh_q_orig = smem + 3 * D_pad;       // [D]

    __shared__ float sh_local_d[MAX_K_LOCAL];
    __shared__ int64_t sh_local_i[MAX_K_LOCAL];
    __shared__ float sh_tau_sq;
    __shared__ float sh_tau;
    __shared__ int sh_warp_alive[2][8];
    __shared__ int sh_warp_cand[2][8];
    __shared__ int32_t sh_surv_oids[TILE_SIZE];
    __shared__ int32_t sh_surv_spos[TILE_SIZE];
    __shared__ float sh_cand_hat[TILE_SIZE];
    __shared__ int sh_n_surv;
    __shared__ int sh_q_orig_ready;

    const float* q_pca_ptr = xq_pca + (int64_t)q * xq_pca_stride;
    const int64_t sq_off = per_cluster_sq8 ? ((int64_t)cid * D_pad) : 0;

    for (int d = tid; d < D_pad; d += TILE_SIZE) {
        sh_base[d]  = vmin[sq_off + d] - q_pca_ptr[d];
        sh_delta[d] = delta[sq_off + d];
        if constexpr (MODE != 0) {
            sh_eps[d] = eps_dim[sq_off + d];
        }
    }
    for (int j = tid; j < MAX_K_LOCAL; j += TILE_SIZE) {
        sh_local_d[j] = INFINITY;
        sh_local_i[j] = -1;
    }
    if (tid == 0) {
        float tsq = (init_tau_sq != nullptr) ? init_tau_sq[q] : INFINITY;
        sh_tau_sq = tsq;
        sh_tau = isinf(tsq) ? INFINITY : sqrtf(fmaxf(tsq, 0.0f));
        sh_n_surv = 0;
        sh_q_orig_ready = 0;
    }
    __syncthreads();

    const float q_init_tau_sq = sh_tau_sq;

    // ===== PLAN_V10.1: lossless SQ8 upper-bound threshold seeding (compile-time isolated via UB_ENABLE) =====
    // For every candidate x of cluster c: ||q - x||^2 <= UB(x) = sum_d (|r_cd(x_d) - q_d| + eps_cd)^2,
    // since |x_d - r_cd(x_d)| <= 0.5*delta_cd <= eps_cd (eps = 0.5001*delta). The 10th-smallest UB in
    // this block therefore upper-bounds the true 10th-NN distance of q, so LB > tau pruning stays lossless.
    if constexpr (MODE == 2 && UB_ENABLE != 0) {
        for (int32_t ub_tile = t_start; ub_tile < t_end; ++ub_tile) {
            const int32_t ub_oid = tile_orig_ids[(int64_t)ub_tile * TILE_SIZE + tid];
            float ub = INFINITY;
            if (ub_oid >= 0) {
                const uint32_t* ub_ptr = sq8_packed + (int64_t)ub_tile * D4 * TILE_SIZE;
                float acc = 0.0f;
                for (int d4 = 0; d4 < D4; ++d4) {
                    const uint32_t pack = __ldg(&ub_ptr[(d4 * W_TILE + warp_id) * 32 + lane]);
                    const int d = d4 << 2;
                    const float a0 = fabsf(fmaf((float)(pack & 0xFFu),         sh_delta[d],     sh_base[d]))     + sh_eps[d];
                    const float a1 = fabsf(fmaf((float)((pack >> 8) & 0xFFu),  sh_delta[d + 1], sh_base[d + 1])) + sh_eps[d + 1];
                    const float a2 = fabsf(fmaf((float)((pack >> 16) & 0xFFu), sh_delta[d + 2], sh_base[d + 2])) + sh_eps[d + 2];
                    const float a3 = fabsf(fmaf((float)(pack >> 24),           sh_delta[d + 3], sh_base[d + 3])) + sh_eps[d + 3];
                    acc = fmaf(a0, a0, acc);
                    acc = fmaf(a1, a1, acc);
                    acc = fmaf(a2, a2, acc);
                    acc = fmaf(a3, a3, acc);
                }
                ub = acc * (1.0f + 1e-4f) + 1e-4f;   // safety margin for FP32 rotation / accumulation error
            }
            sh_cand_hat[tid]  = ub;
            sh_surv_oids[tid] = ub_oid;
            __syncthreads();
            if (tid == 0) {
                for (int u = 0; u < TILE_SIZE; ++u) {
                    if (sh_surv_oids[u] >= 0 && sh_cand_hat[u] < sh_local_d[9]) {
                        insert_topk<10>(sh_local_d, sh_local_i, sh_cand_hat[u], (int64_t)sh_surv_oids[u]);
                    }
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            if (sh_local_d[9] < sh_tau_sq) {
                sh_tau_sq = sh_local_d[9];
                sh_tau = sqrtf(fmaxf(sh_tau_sq, 0.0f));
            }
            #pragma unroll
            for (int j = 0; j < 10; ++j) {
                sh_local_d[j] = INFINITY;
                sh_local_i[j] = -1;
            }
        }
        __syncthreads();
    }
    // ===== end PLAN_V10.1 UB seeding =====


    for (int32_t tile_id = t_start; tile_id < t_end; ++tile_id) {
        if (tile_id > t_start) {
            __syncthreads();
        }
        float cur_tau_sq = sh_tau_sq;
        float cur_tau    = sh_tau;

        if constexpr (MODE != 0) {
            if (use_l1 && !isinf(cur_tau)) {
                float r_min = tile_r_minmax[tile_id * 2 + 0];
                float r_max = tile_r_minmax[tile_id * 2 + 1];
                float tol = cur_tau * 1.0001f + 1e-4f;
                if (r_max < d_qc - tol || r_min > d_qc + tol) {
                    continue;
                }
            }
        }

        int32_t my_oid = tile_orig_ids[(int64_t)tile_id * TILE_SIZE + tid];
        int32_t my_spos = c_vec_start + (tile_id - t_start) * TILE_SIZE + tid;
        bool alive = (my_oid >= 0);
        if (__builtin_expect(stats_out != nullptr, 0)) {
            if (alive) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&stats_out[0]), 1ULL);
            }
        }
        float acc_hat = 0.0f;
        float tau_sq_safe = isinf(cur_tau_sq) ? INFINITY : (cur_tau_sq * (1.0f + 2e-5f) + 1e-6f);

        if constexpr (MODE == 0) {
            if (alive) {
                const uint32_t* t_ptr = sq8_packed + (int64_t)tile_id * D4 * TILE_SIZE;
                for (int d4 = 0; d4 < D4; ++d4) {
                    uint32_t pack = __ldg(&t_ptr[(d4 * W_TILE + warp_id) * 32 + lane]);
                    int d = d4 << 2;
                    float e0 = fmaf((float)(pack & 0xFFu),         sh_delta[d],     sh_base[d]);
                    float e1 = fmaf((float)((pack >> 8) & 0xFFu),  sh_delta[d + 1], sh_base[d + 1]);
                    float e2 = fmaf((float)((pack >> 16) & 0xFFu), sh_delta[d + 2], sh_base[d + 2]);
                    float e3 = fmaf((float)(pack >> 24),           sh_delta[d + 3], sh_base[d + 3]);
                    acc_hat = fmaf(e0, e0, acc_hat);
                    acc_hat = fmaf(e1, e1, acc_hat);
                    acc_hat = fmaf(e2, e2, acc_hat);
                    acc_hat = fmaf(e3, e3, acc_hat);
                }
            }
        } else {
            // MODE 1 (K1) or MODE 2 (K2): Tile & Warp dimension-block early-stop
            float acc_lb = 0.0f;
            const uint32_t* t_ptr = sq8_packed + (int64_t)tile_id * D4 * TILE_SIZE;
            bool tile_dead = false;
            int my_scanned_d = D_pad;

            for (int b = 0; b < num_blocks; ++b) {
                unsigned w_mask = __ballot_sync(0xffffffff, alive);
                if (b > 0) {
                    if (lane == 0) sh_warp_alive[b & 1][warp_id] = (w_mask != 0u) ? 1 : 0;
                    __syncthreads();
                    int tile_any = 0;
                    #pragma unroll
                    for (int w = 0; w < W_TILE; ++w) {
                        tile_any |= sh_warp_alive[b & 1][w];
                    }
                    if (!tile_any) {
                        tile_dead = true;
                        break;
                    }
                }

                if (w_mask != 0u && alive) {
                    int d4_s = blk_bounds4[b];
                    int d4_e = blk_bounds4[b + 1];
                    for (int d4 = d4_s; d4 < d4_e; ++d4) {
                        uint32_t pack = __ldg(&t_ptr[(d4 * W_TILE + warp_id) * 32 + lane]);
                        int d = d4 << 2;
                        float u0 = fabsf(fmaf((float)(pack & 0xFFu),         sh_delta[d],     sh_base[d]));
                        float u1 = fabsf(fmaf((float)((pack >> 8) & 0xFFu),  sh_delta[d + 1], sh_base[d + 1]));
                        float u2 = fabsf(fmaf((float)((pack >> 16) & 0xFFu), sh_delta[d + 2], sh_base[d + 2]));
                        float u3 = fabsf(fmaf((float)(pack >> 24),           sh_delta[d + 3], sh_base[d + 3]));
                        if constexpr (MODE == 1) {
                            acc_hat = fmaf(u0, u0, acc_hat);
                            acc_hat = fmaf(u1, u1, acc_hat);
                            acc_hat = fmaf(u2, u2, acc_hat);
                            acc_hat = fmaf(u3, u3, acc_hat);
                        }
                        float l0 = fmaxf(u0 - sh_eps[d],     0.0f); acc_lb = fmaf(l0, l0, acc_lb);
                        float l1 = fmaxf(u1 - sh_eps[d + 1], 0.0f); acc_lb = fmaf(l1, l1, acc_lb);
                        float l2 = fmaxf(u2 - sh_eps[d + 2], 0.0f); acc_lb = fmaf(l2, l2, acc_lb);
                        float l3 = fmaxf(u3 - sh_eps[d + 3], 0.0f); acc_lb = fmaf(l3, l3, acc_lb);
                    }
                    if (acc_lb > tau_sq_safe) {
                        alive = false;
                        my_scanned_d = blk_bounds4[b + 1] * 4;
                    }
                }
            }
            if (__builtin_expect(stats_out != nullptr, 0)) {
                if (my_oid >= 0) {
                    atomicAdd(reinterpret_cast<unsigned long long*>(&stats_out[1]), (unsigned long long)my_scanned_d);
                }
            }
            if (tile_dead) continue;
        }

        bool has_cand;
        if constexpr (MODE == 0) {
            has_cand = alive && (isinf(cur_tau_sq) || acc_hat <= cur_tau_sq);
        } else if constexpr (MODE == 1) {
            has_cand = alive && (acc_hat <= tau_sq_safe);
        } else {
            has_cand = alive;
        }

        if (__builtin_expect(stats_out != nullptr, 0)) {
            if (has_cand) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&stats_out[2]), 1ULL);
            }
        }
        const int t_parity = tile_id & 1;
        unsigned c_mask = __ballot_sync(0xffffffff, has_cand);
        if (lane == 0) sh_warp_cand[t_parity][warp_id] = (c_mask != 0u) ? 1 : 0;
        __syncthreads();

        int any_cand = 0;
        #pragma unroll
        for (int w = 0; w < W_TILE; ++w) {
            any_cand |= sh_warp_cand[t_parity][w];
        }
        if (!any_cand) continue;

        if (c_mask != 0u) {
            int leader = __ffs(c_mask) - 1;
            int warp_base = 0;
            if (lane == leader) {
                warp_base = atomicAdd(&sh_n_surv, __popc(c_mask));
            }
            warp_base = __shfl_sync(c_mask, warp_base, leader);
            if (has_cand) {
                int offset = __popc(c_mask & ((1u << lane) - 1u));
                int slot = warp_base + offset;
                sh_surv_oids[slot] = my_oid;
                sh_surv_spos[slot] = my_spos;
                if constexpr (MODE != 2) {
                    sh_cand_hat[slot] = acc_hat;
                }
            }
        }
        __syncthreads();

        const int ns = sh_n_surv;
        if constexpr (MODE != 2) {
            if (tid == 0) {
                const bool pack_spos = (MODE == 0 && use_l1 != 0);
                for (int u = 0; u < ns; ++u) {
                    int64_t id_val = pack_spos
                        ? ((((int64_t)sh_surv_spos[u]) << 32) | ((uint32_t)sh_surv_oids[u]))
                        : ((int64_t)sh_surv_oids[u]);
                    if (K_LOC == 10) {
                        insert_topk<10>(sh_local_d, sh_local_i, sh_cand_hat[u], id_val);
                    } else if (K_LOC == 1) {
                        insert_topk<1>(sh_local_d, sh_local_i, sh_cand_hat[u], id_val);
                    } else {
                        insert_topk_dyn(sh_local_d, sh_local_i, K_LOC, sh_cand_hat[u], id_val);
                    }
                }
                if (sh_local_d[K_LOC - 1] < sh_tau_sq) {
                    sh_tau_sq = sh_local_d[K_LOC - 1];
                    if constexpr (MODE == 1) {
                        sh_tau = sqrtf(fmaxf(sh_tau_sq, 0.0f));
                    }
                }
                sh_n_surv = 0;
            }
            __syncthreads();
        } else {
            // MODE 2 (K2): Lazy-load xq_orig & WARP-PARALLEL exact FP32 L2 refinement over xb_sorted!
            if (!sh_q_orig_ready) {
                const float* q_orig_ptr = xq_orig + (int64_t)q * D;
                for (int d = tid; d < D; d += TILE_SIZE) {
                    sh_q_orig[d] = q_orig_ptr[d];
                }
                __syncthreads();
                if (tid == 0) sh_q_orig_ready = 1;
                __syncthreads();
            }

            if constexpr (WARP_REFINE == 1) {
                // Each warp w in [0, W_TILE) refines survivors u = w, w + W_TILE, ... in parallel!
                for (int u = warp_id; u < ns; u += W_TILE) {
                    int32_t spos = sh_surv_spos[u];
                    const float* x_ptr = xb_sorted + (int64_t)spos * D;
                    float part = 0.0f;
                    int d = lane * 4;
                    for (; d + 3 < D; d += 128) {
                        float4 xv = *reinterpret_cast<const float4*>(x_ptr + d);
                        float4 qv = *reinterpret_cast<const float4*>(sh_q_orig + d);
                        float df0 = xv.x - qv.x; part = fmaf(df0, df0, part);
                        float df1 = xv.y - qv.y; part = fmaf(df1, df1, part);
                        float df2 = xv.z - qv.z; part = fmaf(df2, df2, part);
                        float df3 = xv.w - qv.w; part = fmaf(df3, df3, part);
                    }
                    for (int dr = d; dr < d + 4 && dr < D; ++dr) {
                        float df = x_ptr[dr] - sh_q_orig[dr];
                        part = fmaf(df, df, part);
                    }
                    part = warp_reduce_sum(part);
                    if (lane == 0) {
                        sh_cand_hat[u] = part;
                    }
                }
            } else {
                // Serial refinement across warps (only warp 0 refines all survivors sequentially)
                if (warp_id == 0) {
                    for (int u = 0; u < ns; ++u) {
                        int32_t spos = sh_surv_spos[u];
                        const float* x_ptr = xb_sorted + (int64_t)spos * D;
                        float part = 0.0f;
                        int d = lane * 4;
                        for (; d + 3 < D; d += 128) {
                            float4 xv = *reinterpret_cast<const float4*>(x_ptr + d);
                            float4 qv = *reinterpret_cast<const float4*>(sh_q_orig + d);
                            float df0 = xv.x - qv.x; part = fmaf(df0, df0, part);
                            float df1 = xv.y - qv.y; part = fmaf(df1, df1, part);
                            float df2 = xv.z - qv.z; part = fmaf(df2, df2, part);
                            float df3 = xv.w - qv.w; part = fmaf(df3, df3, part);
                        }
                        for (int dr = d; dr < d + 4 && dr < D; ++dr) {
                            float df = x_ptr[dr] - sh_q_orig[dr];
                            part = fmaf(df, df, part);
                        }
                        part = warp_reduce_sum(part);
                        if (lane == 0) {
                            sh_cand_hat[u] = part;
                        }
                    }
                }
            }
            __syncthreads();

            if (tid == 0) {
                for (int u = 0; u < ns; ++u) {
                    float d_exact = sh_cand_hat[u];
                    if (isinf(q_init_tau_sq) || d_exact <= q_init_tau_sq) {
                        if (__builtin_expect(stats_out != nullptr, 0)) {
                            atomicAdd(reinterpret_cast<unsigned long long*>(&stats_out[3]), 1ULL);
                        }
                        if (K_LOC == 10) {
                            insert_topk<10>(sh_local_d, sh_local_i, d_exact, (int64_t)sh_surv_oids[u]);
                        } else if (K_LOC == 1) {
                            insert_topk<1>(sh_local_d, sh_local_i, d_exact, (int64_t)sh_surv_oids[u]);
                        } else {
                            insert_topk_dyn(sh_local_d, sh_local_i, K_LOC, d_exact, (int64_t)sh_surv_oids[u]);
                        }
                        if (sh_local_d[K_LOC - 1] < sh_tau_sq) {
                            sh_tau_sq = sh_local_d[K_LOC - 1];
                            sh_tau = sqrtf(fmaxf(sh_tau_sq, 0.0f));
                        }
                    }
                }
                sh_n_surv = 0;
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        int k_emit = 0;
        for (int j = 0; j < K_LOC; ++j) {
            if (sh_local_i[j] >= 0 && (isinf(q_init_tau_sq) || sh_local_d[j] <= q_init_tau_sq)) {
                k_emit++;
            } else {
                break;
            }
        }
        if (k_emit > 0) {
            int base_slot = atomicAdd(&cand_cnt[q], k_emit);
            for (int j = 0; j < k_emit; ++j) {
                if (base_slot + j < max_cands_per_q) {
                    cand_buf_d[(int64_t)q * max_cands_per_q + base_slot + j] = sh_local_d[j];
                    cand_buf_i[(int64_t)q * max_cands_per_q + base_slot + j] = sh_local_i[j];
                }
            }
        }
    }
}

// Warp-parallel Merge kernel (1 CTA = 256 threads = 8 warps per query, merges cand_buf into topk_d/topk_i [B, K])
template <int MAX_K = 10>
__global__ void trident_merge_topk_kernel(
    float* __restrict__ topk_d,              // [B, K] in/out
    int64_t* __restrict__ topk_i,            // [B, K] in/out
    float* __restrict__ tau_sq_out,          // [B] out (may be nullptr)
    const float* __restrict__ cand_buf_d,    // [B, max_cands_per_q]
    const int64_t* __restrict__ cand_buf_i,  // [B, max_cands_per_q]
    const int32_t* __restrict__ cand_cnt,    // [B]
    const int B,
    const int K,
    const int max_cands_per_q,
    const int reset_top10
) {
    const int q = blockIdx.x;
    if (q >= B) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int K_CAP = (MAX_K == 10) ? 10 : ((K <= 100) ? ((K >= 1) ? K : 1) : 100);

    __shared__ float sh_wd[8][MAX_K];
    __shared__ int64_t sh_wi[8][MAX_K];

    for (int j = lane; j < K_CAP; j += 32) {
        if (warp_id == 0 && !reset_top10) {
            sh_wd[0][j] = topk_d[(int64_t)q * K_CAP + j];
            sh_wi[0][j] = topk_i[(int64_t)q * K_CAP + j];
        } else {
            sh_wd[warp_id][j] = INFINITY;
            sh_wi[warp_id][j] = -1;
        }
    }
    __syncthreads();

    int cnt = cand_cnt[q];
    if (cnt > max_cands_per_q) cnt = max_cands_per_q;
    const float* cd = cand_buf_d + (int64_t)q * max_cands_per_q;
    const int64_t* ci = cand_buf_i + (int64_t)q * max_cands_per_q;

    if (lane == 0) {
        float* my_d = sh_wd[warp_id];
        int64_t* my_i = sh_wi[warp_id];
        for (int u = warp_id; u < cnt; u += 8) {
            if (K_CAP == 10) insert_topk<10>(my_d, my_i, cd[u], ci[u]);
            else if (K_CAP == 1) insert_topk<1>(my_d, my_i, cd[u], ci[u]);
            else insert_topk_dyn(my_d, my_i, K_CAP, cd[u], ci[u]);
        }
    }
    __syncthreads();

    if (tid == 0) {
        float* loc_d = sh_wd[0];
        int64_t* loc_i = sh_wi[0];
        for (int w = 1; w < 8; ++w) {
            for (int j = 0; j < K_CAP; ++j) {
                if (sh_wi[w][j] >= 0) {
                    if (K_CAP == 10) insert_topk<10>(loc_d, loc_i, sh_wd[w][j], sh_wi[w][j]);
                    else if (K_CAP == 1) insert_topk<1>(loc_d, loc_i, sh_wd[w][j], sh_wi[w][j]);
                    else insert_topk_dyn(loc_d, loc_i, K_CAP, sh_wd[w][j], sh_wi[w][j]);
                }
            }
        }
        for (int j = 0; j < K_CAP; ++j) {
            topk_d[(int64_t)q * K_CAP + j] = loc_d[j];
            topk_i[(int64_t)q * K_CAP + j] = loc_i[j];
        }
        if (tau_sq_out != nullptr) {
            tau_sq_out[q] = loc_d[K_CAP - 1];
        }
    }
}

/*
 * Pass 1 Top-M Merge or Pass 2 Top-M Merge + Warp-Parallel Exact FP32 Reranking Kernel (for CM-SQ8+RR):
 *   - When do_rerank == 0 (Pass 1): merges cand_buf into topm_buf_d/i [B, 200] and sets tau_sq_out[q] = sh_topm_d[M - 1].
 *   - When do_rerank == 1 (Pass 2): merges cand_buf into topm_buf_d/i (initialized from Pass 1 if reset_topm == 0),
 *     then the 8 warps compute exact FP32 L2 distances over xb_sorted[spos * D] in parallel for all M candidates,
 *     and output the exact top-K to topk_d/topk_i!
 */
__global__ void trident_merge_and_rerank_topm_kernel(
    float* __restrict__ topk_d,              // [B, K] out
    int64_t* __restrict__ topk_i,            // [B, K] out
    float* __restrict__ topm_buf_d,          // [B, 200] in/out
    int64_t* __restrict__ topm_buf_i,        // [B, 200] in/out
    float* __restrict__ tau_sq_out,          // [B] out (may be nullptr)
    const float* __restrict__ xb_sorted,     // [N, D]
    const float* __restrict__ xq_orig,       // [B, D]
    const float* __restrict__ cand_buf_d,    // [B, max_cands_per_q]
    const int64_t* __restrict__ cand_buf_i,  // [B, max_cands_per_q]
    const int32_t* __restrict__ cand_cnt,    // [B]
    const int B,
    const int D,
    const int K,
    const int rerank_m,
    const int max_cands_per_q,
    const int reset_topm,
    const int do_rerank
) {
    const int q = blockIdx.x;
    if (q >= B) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int K_CAP = (K <= 100) ? ((K >= 1) ? K : 1) : 100;
    const int M_REQ = (rerank_m >= K_CAP) ? rerank_m : K_CAP;
    const int M = (M_REQ <= 200) ? M_REQ : 200;

    extern __shared__ float sh_q_rr[]; // [D]
    __shared__ float sh_topm_d[200];
    __shared__ int64_t sh_topm_i[200];
    __shared__ float sh_exact_d[200];
    __shared__ float sh_out_d[100];
    __shared__ int64_t sh_out_i[100];

    if (do_rerank) {
        const float* q_ptr = xq_orig + (int64_t)q * D;
        for (int d = tid; d < D; d += 256) {
            sh_q_rr[d] = q_ptr[d];
        }
    }
    for (int j = tid; j < 200; j += 256) {
        if (reset_topm || j >= M) {
            sh_topm_d[j] = INFINITY;
            sh_topm_i[j] = -1;
        } else {
            sh_topm_d[j] = topm_buf_d[(int64_t)q * 200 + j];
            sh_topm_i[j] = topm_buf_i[(int64_t)q * 200 + j];
        }
    }
    __syncthreads();

    if (tid == 0) {
        int cnt = cand_cnt[q];
        if (cnt > max_cands_per_q) cnt = max_cands_per_q;
        const float* cd = cand_buf_d + (int64_t)q * max_cands_per_q;
        const int64_t* ci = cand_buf_i + (int64_t)q * max_cands_per_q;
        for (int u = 0; u < cnt; ++u) {
            insert_topk_dyn(sh_topm_d, sh_topm_i, M, cd[u], ci[u]);
        }
        if (tau_sq_out != nullptr) {
            tau_sq_out[q] = sh_topm_d[M - 1];
        }
    }
    __syncthreads();

    if (!do_rerank) {
        for (int j = tid; j < M; j += 256) {
            topm_buf_d[(int64_t)q * 200 + j] = sh_topm_d[j];
            topm_buf_i[(int64_t)q * 200 + j] = sh_topm_i[j];
        }
        return;
    }

    for (int u = warp_id; u < M; u += 8) {
        int64_t packed = sh_topm_i[u];
        if (packed < 0) {
            if (lane == 0) sh_exact_d[u] = INFINITY;
            continue;
        }
        int32_t spos = (int32_t)(packed >> 32);
        const float* x_ptr = xb_sorted + (int64_t)spos * D;
        float part = 0.0f;
        int d = lane * 4;
        for (; d + 3 < D; d += 128) {
            float4 xv = *reinterpret_cast<const float4*>(x_ptr + d);
            float4 qv = *reinterpret_cast<const float4*>(sh_q_rr + d);
            float df0 = xv.x - qv.x; part = fmaf(df0, df0, part);
            float df1 = xv.y - qv.y; part = fmaf(df1, df1, part);
            float df2 = xv.z - qv.z; part = fmaf(df2, df2, part);
            float df3 = xv.w - qv.w; part = fmaf(df3, df3, part);
        }
        for (int dr = d; dr < d + 4 && dr < D; ++dr) {
            float df = x_ptr[dr] - sh_q_rr[dr];
            part = fmaf(df, df, part);
        }
        part = warp_reduce_sum(part);
        if (lane == 0) {
            sh_exact_d[u] = part;
        }
    }
    __syncthreads();

    if (tid == 0) {
        for (int j = 0; j < K_CAP; ++j) {
            sh_out_d[j] = INFINITY;
            sh_out_i[j] = -1;
        }
        for (int u = 0; u < M; ++u) {
            int64_t packed = sh_topm_i[u];
            if (packed >= 0) {
                int64_t oid = (int64_t)((uint32_t)(packed & 0xffffffffu));
                if (K_CAP == 10) insert_topk<10>(sh_out_d, sh_out_i, sh_exact_d[u], oid);
                else if (K_CAP == 1) insert_topk<1>(sh_out_d, sh_out_i, sh_exact_d[u], oid);
                else insert_topk_dyn(sh_out_d, sh_out_i, K_CAP, sh_exact_d[u], oid);
            }
        }
        for (int j = 0; j < K_CAP; ++j) {
            topk_d[(int64_t)q * K_CAP + j] = sh_out_d[j];
            topk_i[(int64_t)q * K_CAP + j] = sh_out_i[j];
        }
    }
}

} // namespace

float launch_trident_v4_search(
    int mode,               // 0=K0/CM-SQ8+RR, 1=K1, 2=K2, 10=CM-FP32, 11=Tribase-GPU
    int tile_size,          // 64 or 256
    int use_l1,             // 0 or 1
    int per_cluster_sq8,    // 0 or 1
    int rerank_m,           // 0 for normal top-K; 20/50/100/200 for CM-SQ8+RR
    int nprobe,
    torch::Tensor xq_orig,          // [B, D]
    torch::Tensor W_combined,       // [D, nlist + D_pad] (for modes 0,1,2)
    torch::Tensor b_combined,       // [1, nlist + D_pad]
    torch::Tensor W_coarse_only,    // [D, nlist] (for mode 10 CM-FP32 and mode 11 Tribase-GPU)
    torch::Tensor b_coarse_only,    // [1, nlist]
    torch::Tensor proj_buf,         // [max_B, nlist + D_pad]
    torch::Tensor cdists_buf,       // [max_B, nlist] contiguous for mode 10/11
    torch::Tensor probe_cids_buf,   // [max_B, max_nprobe]
    torch::Tensor probe_dqcs_buf,   // [max_B, max_nprobe]
    torch::Tensor p1_cids_buf,      // [max_B * max_nprobe]
    torch::Tensor p1_qids_buf,
    torch::Tensor p1_dqcs_buf,
    torch::Tensor p2_cids_buf,
    torch::Tensor p2_qids_buf,
    torch::Tensor p2_dqcs_buf,
    torch::Tensor sq8_packed,
    torch::Tensor tile_orig_ids,
    torch::Tensor tile_r_minmax,
    torch::Tensor cluster_tile_off,
    torch::Tensor cluster_vec_off,
    torch::Tensor xb_sorted,
    torch::Tensor orig_ids_sorted,
    torch::Tensor vmin,
    torch::Tensor delta,
    torch::Tensor eps_dim,
    torch::Tensor blk_bounds4,
    torch::Tensor topk_d,           // [B, K]
    torch::Tensor topk_i,           // [B, K]
    torch::Tensor topm_buf_d,       // [max_B, 200]
    torch::Tensor topm_buf_i,       // [max_B, 200]
    torch::Tensor tau_sq,
    torch::Tensor cand_buf_d,
    torch::Tensor cand_buf_i,
    torch::Tensor cand_cnt,
    bool record_events,
    int warp_refine = 1,
    torch::optional<torch::Tensor> stats_buf = c10::nullopt,
    torch::optional<torch::Tensor> r_vec_sorted = c10::nullopt
) {
    const int B = xq_orig.size(0);
    const int D = xq_orig.size(1);
    const int K = topk_d.size(1);
    const int nlist = W_coarse_only.size(1);
    const int total_cols = W_combined.size(1);
    const int D_pad = total_cols - nlist;
    const int num_blocks = blk_bounds4.size(0) - 1;
    const int max_cands_per_q = cand_buf_d.size(1);
    int64_t* stats_ptr = nullptr;
    if (stats_buf.has_value() && stats_buf.value().defined() && stats_buf.value().numel() >= 4) {
        stats_ptr = stats_buf.value().data_ptr<int64_t>();
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 1. Coarse Quantizer (+ Fused PCA Rotation when mode not in {10, 11}) in ONE single cuBLAS addmm_out!
    const float* cdists_ptr = nullptr;
    int proj_stride = 0;
    if (mode == 10 || mode == 11) {
        auto c_slice = cdists_buf.slice(0, 0, B);
        at::addmm_out(c_slice, b_coarse_only.expand({B, nlist}), xq_orig, W_coarse_only);
        cdists_ptr = cdists_buf.data_ptr<float>();
        proj_stride = nlist;
    } else {
        auto proj_slice = proj_buf.slice(0, 0, B);
        at::addmm_out(proj_slice, b_combined.expand({B, total_cols}), xq_orig, W_combined);
        cdists_ptr = proj_buf.data_ptr<float>();
        proj_stride = total_cols;
    }

    // 2. Top-nprobe centroid selection in CUDA
    if ((size_t)(nlist + 1) * sizeof(float) > 48000) {
        cudaFuncSetAttribute(trident_topk_centroids_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)((nlist + 64) * sizeof(float)));
        cudaFuncSetAttribute(trident_counting_sort_pairs_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)((nlist + 64) * sizeof(int32_t)));
    }
    trident_topk_centroids_kernel<<<B, 256, nlist * sizeof(float), stream>>>(
        cdists_ptr,
        xq_orig.data_ptr<float>(),
        probe_cids_buf.data_ptr<int32_t>(),
        probe_dqcs_buf.data_ptr<float>(),
        B, D, nlist, proj_stride, nprobe
    );

    // 3. Determine two-pass split
    const char* ub_env = std::getenv("TRIDENT_UB_SEED");
    const int ub_v = ub_env ? std::atoi(ub_env) : 0;
    int p_init = 0;
    if (mode == 0) {
        if (nprobe <= 1 || (B == 1 && rerank_m == 0)) p_init = 0;
        else if ((rerank_m > 20 || K > 10) && nprobe >= 4) p_init = 2;
        else p_init = 1;
    } else if (mode == 1 || mode == 2 || mode == 11) {
        if (nprobe <= 1) p_init = 0;
        else if (B == 1 && K <= 10) p_init = 1;
        else if (nprobe <= 16) p_init = 2;
        else p_init = 3;
    }
    const int M_p1 = B * p_init;
    const int M_p2 = B * (nprobe - p_init);

    // 4. Shared-memory Counting Sort of cluster pairs for Pass 1 & Pass 2 simultaneously
    trident_counting_sort_pairs_kernel<<<2, 256, (nlist + 1) * sizeof(int32_t), stream>>>(
        probe_cids_buf.data_ptr<int32_t>(),
        probe_dqcs_buf.data_ptr<float>(),
        B, nprobe, p_init, nlist,
        p1_cids_buf.data_ptr<int32_t>(),
        p1_qids_buf.data_ptr<int32_t>(),
        p1_dqcs_buf.data_ptr<float>(),
        p2_cids_buf.data_ptr<int32_t>(),
        p2_qids_buf.data_ptr<int32_t>(),
        p2_dqcs_buf.data_ptr<float>()
    );

    cudaEvent_t ev_start, ev_stop;
    if (record_events) {
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);
        cudaEventRecord(ev_start, stream);
    }

    if (mode == 10) {
        size_t smem_fp32 = (size_t)D * sizeof(float);
        cudaMemsetAsync(cand_cnt.data_ptr<int32_t>(), 0, B * sizeof(int32_t), stream);
        if (tile_size == 64) {
            trident_cm_fp32_scan_kernel<64><<<M_p2, 64, smem_fp32, stream>>>(
                xb_sorted.data_ptr<float>(),
                orig_ids_sorted.data_ptr<int32_t>(),
                cluster_vec_off.data_ptr<int32_t>(),
                xq_orig.data_ptr<float>(),
                p2_cids_buf.data_ptr<int32_t>(),
                p2_qids_buf.data_ptr<int32_t>(),
                M_p2, D, K, max_cands_per_q,
                cand_buf_d.data_ptr<float>(),
                cand_buf_i.data_ptr<int64_t>(),
                cand_cnt.data_ptr<int32_t>(),
                stats_ptr
            );
        } else {
            trident_cm_fp32_scan_kernel<256><<<M_p2, 256, smem_fp32, stream>>>(
                xb_sorted.data_ptr<float>(),
                orig_ids_sorted.data_ptr<int32_t>(),
                cluster_vec_off.data_ptr<int32_t>(),
                xq_orig.data_ptr<float>(),
                p2_cids_buf.data_ptr<int32_t>(),
                p2_qids_buf.data_ptr<int32_t>(),
                M_p2, D, K, max_cands_per_q,
                cand_buf_d.data_ptr<float>(),
                cand_buf_i.data_ptr<int64_t>(),
                cand_cnt.data_ptr<int32_t>(),
                stats_ptr
            );
        }
        trident_merge_topk_kernel<<<B, 256, 0, stream>>>(
            topk_d.data_ptr<float>(),
            topk_i.data_ptr<int64_t>(),
            nullptr,
            cand_buf_d.data_ptr<float>(),
            cand_buf_i.data_ptr<int64_t>(),
            cand_cnt.data_ptr<int32_t>(),
            B, K, max_cands_per_q, 1
        );
    } else if (mode == 11) {
        size_t smem_fp32 = (size_t)D * sizeof(float);
        const float* r_vec_ptr = r_vec_sorted.value().data_ptr<float>();
        if (M_p1 > 0) {
            cudaMemsetAsync(cand_cnt.data_ptr<int32_t>(), 0, B * sizeof(int32_t), stream);
            if (tile_size == 64) {
                trident_tribase_fp32_scan_kernel<64><<<M_p1, 64, smem_fp32, stream>>>(
                    xb_sorted.data_ptr<float>(),
                    orig_ids_sorted.data_ptr<int32_t>(),
                    r_vec_ptr,
                    cluster_vec_off.data_ptr<int32_t>(),
                    xq_orig.data_ptr<float>(),
                    p1_cids_buf.data_ptr<int32_t>(),
                    p1_qids_buf.data_ptr<int32_t>(),
                    p1_dqcs_buf.data_ptr<float>(),
                    nullptr,
                    M_p1, D, K, max_cands_per_q,
                    cand_buf_d.data_ptr<float>(),
                    cand_buf_i.data_ptr<int64_t>(),
                    cand_cnt.data_ptr<int32_t>(),
                    stats_ptr
                );
            } else {
                trident_tribase_fp32_scan_kernel<256><<<M_p1, 256, smem_fp32, stream>>>(
                    xb_sorted.data_ptr<float>(),
                    orig_ids_sorted.data_ptr<int32_t>(),
                    r_vec_ptr,
                    cluster_vec_off.data_ptr<int32_t>(),
                    xq_orig.data_ptr<float>(),
                    p1_cids_buf.data_ptr<int32_t>(),
                    p1_qids_buf.data_ptr<int32_t>(),
                    p1_dqcs_buf.data_ptr<float>(),
                    nullptr,
                    M_p1, D, K, max_cands_per_q,
                    cand_buf_d.data_ptr<float>(),
                    cand_buf_i.data_ptr<int64_t>(),
                    cand_cnt.data_ptr<int32_t>(),
                    stats_ptr
                );
            }
            trident_merge_topk_kernel<<<B, 256, 0, stream>>>(
                topk_d.data_ptr<float>(),
                topk_i.data_ptr<int64_t>(),
                tau_sq.data_ptr<float>(),
                cand_buf_d.data_ptr<float>(),
                cand_buf_i.data_ptr<int64_t>(),
                cand_cnt.data_ptr<int32_t>(),
                B, K, max_cands_per_q, 1
            );
        }
        if (M_p2 > 0) {
            cudaMemsetAsync(cand_cnt.data_ptr<int32_t>(), 0, B * sizeof(int32_t), stream);
            if (tile_size == 64) {
                trident_tribase_fp32_scan_kernel<64><<<M_p2, 64, smem_fp32, stream>>>(
                    xb_sorted.data_ptr<float>(),
                    orig_ids_sorted.data_ptr<int32_t>(),
                    r_vec_ptr,
                    cluster_vec_off.data_ptr<int32_t>(),
                    xq_orig.data_ptr<float>(),
                    p2_cids_buf.data_ptr<int32_t>(),
                    p2_qids_buf.data_ptr<int32_t>(),
                    p2_dqcs_buf.data_ptr<float>(),
                    (M_p1 > 0) ? tau_sq.data_ptr<float>() : nullptr,
                    M_p2, D, K, max_cands_per_q,
                    cand_buf_d.data_ptr<float>(),
                    cand_buf_i.data_ptr<int64_t>(),
                    cand_cnt.data_ptr<int32_t>(),
                    stats_ptr
                );
            } else {
                trident_tribase_fp32_scan_kernel<256><<<M_p2, 256, smem_fp32, stream>>>(
                    xb_sorted.data_ptr<float>(),
                    orig_ids_sorted.data_ptr<int32_t>(),
                    r_vec_ptr,
                    cluster_vec_off.data_ptr<int32_t>(),
                    xq_orig.data_ptr<float>(),
                    p2_cids_buf.data_ptr<int32_t>(),
                    p2_qids_buf.data_ptr<int32_t>(),
                    p2_dqcs_buf.data_ptr<float>(),
                    (M_p1 > 0) ? tau_sq.data_ptr<float>() : nullptr,
                    M_p2, D, K, max_cands_per_q,
                    cand_buf_d.data_ptr<float>(),
                    cand_buf_i.data_ptr<int64_t>(),
                    cand_cnt.data_ptr<int32_t>(),
                    stats_ptr
                );
            }
            trident_merge_topk_kernel<<<B, 256, 0, stream>>>(
                topk_d.data_ptr<float>(),
                topk_i.data_ptr<int64_t>(),
                nullptr,
                cand_buf_d.data_ptr<float>(),
                cand_buf_i.data_ptr<int64_t>(),
                cand_cnt.data_ptr<int32_t>(),
                B, K, max_cands_per_q, (M_p1 == 0) ? 1 : 0
            );
        }
    } else {
        size_t smem_bytes = (size_t)(3 * D_pad + D) * sizeof(float);
        const float* xq_pca_ptr = proj_buf.data_ptr<float>() + nlist;
        const int eff_rr_m = (rerank_m > 0) ? std::max(rerank_m, K) : 0;
        const int k_local_p1 = (mode == 0 && eff_rr_m > 0) ? ((eff_rr_m <= 128) ? eff_rr_m : 128) : K;
        const int k_local_p2 = (mode == 0 && eff_rr_m > 0)
            ? std::min(eff_rr_m, std::max(K, std::max(16, (eff_rr_m + nprobe - 1) / std::max(nprobe, 1) + 8)))
            : K;
        const int mode_use_l1 = (mode == 0) ? ((eff_rr_m > 0) ? 1 : 0) : use_l1;

        auto dispatch = [&](auto mode_tag, auto tile_tag, auto warp_tag) {
            constexpr int M = decltype(mode_tag)::value;
            constexpr int T = decltype(tile_tag)::value;
            constexpr int WR = decltype(warp_tag)::value;

            if (M_p1 > 0) {
                cudaMemsetAsync(cand_cnt.data_ptr<int32_t>(), 0, B * sizeof(int32_t), stream);
                auto launch_p1 = [&](auto ub_tag) {
                    constexpr int UB = decltype(ub_tag)::value;
                    trident_cluster_major_scan_kernel<M, T, WR, UB><<<M_p1, T, smem_bytes, stream>>>(
                        reinterpret_cast<const uint32_t*>(sq8_packed.data_ptr<int32_t>()),
                        tile_orig_ids.data_ptr<int32_t>(),
                        tile_r_minmax.data_ptr<float>(),
                        cluster_tile_off.data_ptr<int32_t>(),
                        cluster_vec_off.data_ptr<int32_t>(),
                        xb_sorted.data_ptr<float>(),
                        xq_orig.data_ptr<float>(),
                        xq_pca_ptr,
                        total_cols,
                        vmin.data_ptr<float>(),
                        delta.data_ptr<float>(),
                        eps_dim.data_ptr<float>(),
                        per_cluster_sq8,
                        blk_bounds4.data_ptr<int32_t>(),
                        p1_cids_buf.data_ptr<int32_t>(),
                        p1_qids_buf.data_ptr<int32_t>(),
                        p1_dqcs_buf.data_ptr<float>(),
                        nullptr,
                        M_p1, D, D_pad, num_blocks, mode_use_l1, k_local_p1, max_cands_per_q,
                        cand_buf_d.data_ptr<float>(),
                        cand_buf_i.data_ptr<int64_t>(),
                        cand_cnt.data_ptr<int32_t>(),
                        stats_ptr
                    );
                };
                if constexpr (M == 2) {
                    if (ub_v != 0) launch_p1(std::integral_constant<int, 1>{});
                    else           launch_p1(std::integral_constant<int, 0>{});
                } else {
                    launch_p1(std::integral_constant<int, 0>{});
                }
                if (M == 0 && eff_rr_m > 0) {
                    trident_merge_and_rerank_topm_kernel<<<B, 256, 0, stream>>>(
                        topk_d.data_ptr<float>(),
                        topk_i.data_ptr<int64_t>(),
                        topm_buf_d.data_ptr<float>(),
                        topm_buf_i.data_ptr<int64_t>(),
                        tau_sq.data_ptr<float>(),
                        xb_sorted.data_ptr<float>(),
                        xq_orig.data_ptr<float>(),
                        cand_buf_d.data_ptr<float>(),
                        cand_buf_i.data_ptr<int64_t>(),
                        cand_cnt.data_ptr<int32_t>(),
                        B, D, K, eff_rr_m, max_cands_per_q, 1, 0
                    );
                } else {
                    trident_merge_topk_kernel<<<B, 256, 0, stream>>>(
                        topk_d.data_ptr<float>(),
                        topk_i.data_ptr<int64_t>(),
                        tau_sq.data_ptr<float>(),
                        cand_buf_d.data_ptr<float>(),
                        cand_buf_i.data_ptr<int64_t>(),
                        cand_cnt.data_ptr<int32_t>(),
                        B, K, max_cands_per_q, 1
                    );
                }
            }

            if (M_p2 > 0) {
                cudaMemsetAsync(cand_cnt.data_ptr<int32_t>(), 0, B * sizeof(int32_t), stream);
                auto launch_p2 = [&](auto ub_tag) {
                    constexpr int UB = decltype(ub_tag)::value;
                    trident_cluster_major_scan_kernel<M, T, WR, UB><<<M_p2, T, smem_bytes, stream>>>(
                        reinterpret_cast<const uint32_t*>(sq8_packed.data_ptr<int32_t>()),
                        tile_orig_ids.data_ptr<int32_t>(),
                        tile_r_minmax.data_ptr<float>(),
                        cluster_tile_off.data_ptr<int32_t>(),
                        cluster_vec_off.data_ptr<int32_t>(),
                        xb_sorted.data_ptr<float>(),
                        xq_orig.data_ptr<float>(),
                        xq_pca_ptr,
                        total_cols,
                        vmin.data_ptr<float>(),
                        delta.data_ptr<float>(),
                        eps_dim.data_ptr<float>(),
                        per_cluster_sq8,
                        blk_bounds4.data_ptr<int32_t>(),
                        p2_cids_buf.data_ptr<int32_t>(),
                        p2_qids_buf.data_ptr<int32_t>(),
                        p2_dqcs_buf.data_ptr<float>(),
                        (M_p1 > 0) ? tau_sq.data_ptr<float>() : nullptr,
                        M_p2, D, D_pad, num_blocks, mode_use_l1, k_local_p2, max_cands_per_q,
                        cand_buf_d.data_ptr<float>(),
                        cand_buf_i.data_ptr<int64_t>(),
                        cand_cnt.data_ptr<int32_t>(),
                        stats_ptr
                    );
                };
                if constexpr (M == 2) {
                    if (ub_v == 2 || (ub_v == 1 && M_p1 == 0)) launch_p2(std::integral_constant<int, 1>{});
                    else                                       launch_p2(std::integral_constant<int, 0>{});
                } else {
                    launch_p2(std::integral_constant<int, 0>{});
                }
                if (M == 0 && eff_rr_m > 0) {
                    trident_merge_and_rerank_topm_kernel<<<B, 256, (size_t)D * sizeof(float), stream>>>(
                        topk_d.data_ptr<float>(),
                        topk_i.data_ptr<int64_t>(),
                        topm_buf_d.data_ptr<float>(),
                        topm_buf_i.data_ptr<int64_t>(),
                        nullptr,
                        xb_sorted.data_ptr<float>(),
                        xq_orig.data_ptr<float>(),
                        cand_buf_d.data_ptr<float>(),
                        cand_buf_i.data_ptr<int64_t>(),
                        cand_cnt.data_ptr<int32_t>(),
                        B, D, K, eff_rr_m, max_cands_per_q, (M_p1 == 0) ? 1 : 0, 1
                    );
                } else {
                    trident_merge_topk_kernel<<<B, 256, 0, stream>>>(
                        topk_d.data_ptr<float>(),
                        topk_i.data_ptr<int64_t>(),
                        nullptr,
                        cand_buf_d.data_ptr<float>(),
                        cand_buf_i.data_ptr<int64_t>(),
                        cand_cnt.data_ptr<int32_t>(),
                        B, K, max_cands_per_q, (M_p1 == 0) ? 1 : 0
                    );
                }
            }
        };

        if (tile_size == 64) {
            if (mode == 0) dispatch(std::integral_constant<int, 0>{}, std::integral_constant<int, 64>{}, std::integral_constant<int, 1>{});
            else if (mode == 1) dispatch(std::integral_constant<int, 1>{}, std::integral_constant<int, 64>{}, std::integral_constant<int, 1>{});
            else {
                if (warp_refine == 0) dispatch(std::integral_constant<int, 2>{}, std::integral_constant<int, 64>{}, std::integral_constant<int, 0>{});
                else dispatch(std::integral_constant<int, 2>{}, std::integral_constant<int, 64>{}, std::integral_constant<int, 1>{});
            }
        } else {
            if (mode == 0) dispatch(std::integral_constant<int, 0>{}, std::integral_constant<int, 256>{}, std::integral_constant<int, 1>{});
            else if (mode == 1) dispatch(std::integral_constant<int, 1>{}, std::integral_constant<int, 256>{}, std::integral_constant<int, 1>{});
            else {
                if (warp_refine == 0) dispatch(std::integral_constant<int, 2>{}, std::integral_constant<int, 256>{}, std::integral_constant<int, 0>{});
                else dispatch(std::integral_constant<int, 2>{}, std::integral_constant<int, 256>{}, std::integral_constant<int, 1>{});
            }
        }
    }

    float ms = 0.0f;
    if (record_events) {
        cudaEventRecord(ev_stop, stream);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
    }
    return ms;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_trident_v4_search", &launch_trident_v4_search,
          "TRIDENT V4/V5 Fused GPU Search",
          py::arg("mode"), py::arg("tile_size"), py::arg("use_l1"),
          py::arg("per_cluster_sq8"), py::arg("rerank_m"), py::arg("nprobe"),
          py::arg("xq_orig"), py::arg("W_combined"), py::arg("b_combined"),
          py::arg("W_coarse_only"), py::arg("b_coarse_only"),
          py::arg("proj_buf"), py::arg("cdists_buf"), py::arg("probe_cids_buf"),
          py::arg("probe_dqcs_buf"), py::arg("p1_cids_buf"), py::arg("p1_qids_buf"),
          py::arg("p1_dqcs_buf"), py::arg("p2_cids_buf"), py::arg("p2_qids_buf"),
          py::arg("p2_dqcs_buf"), py::arg("sq8_packed"), py::arg("tile_orig_ids"),
          py::arg("tile_r_minmax"), py::arg("cluster_tile_off"),
          py::arg("cluster_vec_off"), py::arg("xb_sorted"), py::arg("orig_ids_sorted"),
          py::arg("vmin"), py::arg("delta"), py::arg("eps_dim"),
          py::arg("blk_bounds4"), py::arg("topk_d"), py::arg("topk_i"),
          py::arg("topm_buf_d"), py::arg("topm_buf_i"), py::arg("tau_sq"),
          py::arg("cand_buf_d"), py::arg("cand_buf_i"), py::arg("cand_cnt"),
          py::arg("record_events"),
          py::arg("warp_refine") = 1,
          py::arg("stats_buf") = py::none(),
          py::arg("r_vec_sorted") = py::none());
}
