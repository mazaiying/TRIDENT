#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ void insert_topk10(
    float* sh_d,
    int64_t* sh_i,
    float d_val,
    int64_t id_val
) {
    if (d_val < sh_d[9] || (d_val == sh_d[9] && id_val < sh_i[9])) {
        sh_d[9] = d_val;
        sh_i[9] = id_val;
        #pragma unroll
        for (int j = 9; j > 0; --j) {
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
 * MODE:
 *   0 = K0 (SQ8 full-dimension cluster-major scan, no early-stop, no FP32 refinement)
 *   1 = K1 (SQ8 tile/warp dimension-block early-stop + L1 band, no FP32 refinement)
 *   2 = K2 (K1 + exact FP32 survivor refinement -> 100% lossless IVF-Flat)
 *
 * Cluster-Major Batched Scan Kernel (used for BOTH Pass 1 and Pass 2!):
 *   - Consecutive CTAs share the same cluster_id (sorted_cids) for L2 cache reuse across queries.
 *   - Pass 1 passes init_tau_sq == nullptr (starts with tau_sq = INFINITY, tightens within cluster).
 *   - Pass 2 passes init_tau_sq != nullptr (starts with query's Pass 1 tau_sq[q]).
 */
template <int MODE, int TILE_SIZE>
__global__ void trident_cluster_major_scan_kernel(
    const uint32_t* __restrict__ sq8_packed,     // [total_tiles, D4, W_tile, 32]
    const int32_t* __restrict__ tile_orig_ids,   // [total_tiles, TILE_SIZE]
    const float* __restrict__ tile_r_minmax,     // [total_tiles, 2]
    const int32_t* __restrict__ cluster_tile_off,// [nlist + 1]
    const float* __restrict__ xb_orig,           // [N, D]
    const float* __restrict__ xq_orig,           // [B, D]
    const float* __restrict__ xq_pca,            // [B, D_pad]
    const float* __restrict__ vmin,              // [D_pad]
    const float* __restrict__ delta,             // [D_pad]
    const float* __restrict__ eps_dim,           // [D_pad]
    const int32_t* __restrict__ blk_bounds4,     // [num_blocks + 1] (in units of d4 = dim/4)
    const int32_t* __restrict__ sorted_cids,     // [M_pairs] sorted by cluster_id!
    const int32_t* __restrict__ sorted_qids,     // [M_pairs]
    const float* __restrict__ sorted_dqcs,       // [M_pairs]
    const float* __restrict__ init_tau_sq,       // [B] from Pass 1 (nullptr in Pass 1)
    const int M_pairs,
    const int D,
    const int D_pad,
    const int num_blocks,
    const int use_l1,
    const int max_cands_per_q,
    float* __restrict__ cand_buf_d,              // [B, max_cands_per_q]
    int64_t* __restrict__ cand_buf_i,            // [B, max_cands_per_q]
    int32_t* __restrict__ cand_cnt               // [B]
) {
    const int pair_idx = blockIdx.x;
    if (pair_idx >= M_pairs) return;

    const int32_t cid = sorted_cids[pair_idx];
    const int32_t q   = sorted_qids[pair_idx];
    const float d_qc  = sorted_dqcs[pair_idx];

    const int32_t t_start = cluster_tile_off[cid];
    const int32_t t_end   = cluster_tile_off[cid + 1];
    if (t_start >= t_end) return;

    constexpr int W_TILE = TILE_SIZE >> 5;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int D4 = D_pad >> 2;

    extern __shared__ float smem[];
    float* sh_base   = smem;                   // [D_pad]
    float* sh_delta  = smem + D_pad;           // [D_pad]
    float* sh_eps    = smem + 2 * D_pad;       // [D_pad]
    float* sh_q_orig = smem + 3 * D_pad;       // [D]

    __shared__ float sh_local_d[10];
    __shared__ int64_t sh_local_i[10];
    __shared__ float sh_tau_sq;
    __shared__ float sh_tau;
    __shared__ int sh_warp_alive[2][8];
    __shared__ int sh_warp_cand[2][8];
    __shared__ int32_t sh_surv_oids[TILE_SIZE];
    __shared__ float sh_cand_hat[TILE_SIZE];
    __shared__ int sh_n_surv;
    __shared__ int sh_q_orig_ready;
    __shared__ float sh_red[8];

    const float* q_pca_ptr = xq_pca + (int64_t)q * D_pad;

    for (int d = tid; d < D_pad; d += TILE_SIZE) {
        sh_base[d]  = vmin[d] - q_pca_ptr[d];
        sh_delta[d] = delta[d];
        if constexpr (MODE != 0) {
            sh_eps[d] = eps_dim[d];
        }
    }
    if (tid < 10) {
        sh_local_d[tid] = INFINITY;
        sh_local_i[tid] = -1;
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
        bool alive = (my_oid >= 0);
        float acc_hat = 0.0f;
        float tau_sq_safe = isinf(cur_tau_sq) ? INFINITY : (cur_tau_sq * (1.0f + 2e-5f) + 1e-6f);

        if constexpr (MODE == 0) {
            // Full-dimension SQ8 scan without early-stop
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
                    if (acc_lb > tau_sq_safe) alive = false;
                }
            }
            if (tile_dead) continue;
        }

        // Fast warp-ballot survivor check (skips compaction & serial loops when 0 threads survive!)
        bool has_cand;
        if constexpr (MODE == 0) {
            has_cand = alive && (isinf(cur_tau_sq) || acc_hat <= cur_tau_sq);
        } else if constexpr (MODE == 1) {
            has_cand = alive && (acc_hat <= tau_sq_safe);
        } else {
            has_cand = alive;
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

        // Warp-aggregated shared-memory compaction of surviving candidates
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
                if constexpr (MODE != 2) {
                    sh_cand_hat[slot] = acc_hat;
                }
            }
        }
        __syncthreads();

        const int ns = sh_n_surv;
        if constexpr (MODE != 2) {
            if (tid == 0) {
                for (int u = 0; u < ns; ++u) {
                    insert_topk10(sh_local_d, sh_local_i, sh_cand_hat[u], (int64_t)sh_surv_oids[u]);
                }
                if (sh_local_d[9] < sh_tau_sq) {
                    sh_tau_sq = sh_local_d[9];
                    if constexpr (MODE == 1) {
                        sh_tau = sqrtf(fmaxf(sh_tau_sq, 0.0f));
                    }
                }
                sh_n_surv = 0;
            }
            __syncthreads();
        } else {
            // MODE 2 (K2): Lazy-load xq_orig & cooperatively compute exact FP32 L2 on survivors
            if (!sh_q_orig_ready) {
                const float* q_orig_ptr = xq_orig + (int64_t)q * D;
                for (int d = tid; d < D; d += TILE_SIZE) {
                    sh_q_orig[d] = q_orig_ptr[d];
                }
                __syncthreads();
                if (tid == 0) sh_q_orig_ready = 1;
                __syncthreads();
            }

            for (int u = 0; u < ns; ++u) {
                int64_t oid = (int64_t)sh_surv_oids[u];
                const float* x_ptr = xb_orig + oid * D;
                float part = 0.0f;
                int d = tid * 4;
                for (; d + 3 < D; d += TILE_SIZE * 4) {
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
                if (lane == 0) sh_red[warp_id] = part;
                __syncthreads();
                if (tid == 0) {
                    float d_exact = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < W_TILE; ++w) d_exact += sh_red[w];
                    if (isinf(q_init_tau_sq) || d_exact <= q_init_tau_sq) {
                        insert_topk10(sh_local_d, sh_local_i, d_exact, oid);
                        if (sh_local_d[9] < sh_tau_sq) {
                            sh_tau_sq = sh_local_d[9];
                            sh_tau = sqrtf(fmaxf(sh_tau_sq, 0.0f));
                        }
                    }
                    if (u == ns - 1) sh_n_surv = 0;
                }
                __syncthreads();
            }
        }
    }

    // Emit cluster-local winners (<= q_init_tau_sq, or all <=10 if q_init_tau_sq is inf) lock-free
    if (tid == 0) {
        int k_emit = 0;
        for (int j = 0; j < 10; ++j) {
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

// Merge kernel (1 CTA = 1 query, merges cand_buf into topk_d/topk_i and writes tau_sq_out[q])
__global__ void trident_merge_topk_kernel(
    float* __restrict__ topk_d,              // [B, 10] in/out
    int64_t* __restrict__ topk_i,            // [B, 10] in/out
    float* __restrict__ tau_sq_out,          // [B] out (may be nullptr)
    const float* __restrict__ cand_buf_d,    // [B, max_cands_per_q]
    const int64_t* __restrict__ cand_buf_i,  // [B, max_cands_per_q]
    const int32_t* __restrict__ cand_cnt,    // [B]
    const int B,
    const int max_cands_per_q,
    const int reset_top10
) {
    const int q = blockIdx.x;
    if (q >= B || threadIdx.x != 0) return;

    float loc_d[10];
    int64_t loc_i[10];
    for (int j = 0; j < 10; ++j) {
        if (reset_top10) {
            loc_d[j] = INFINITY;
            loc_i[j] = -1;
        } else {
            loc_d[j] = topk_d[q * 10 + j];
            loc_i[j] = topk_i[q * 10 + j];
        }
    }

    int cnt = cand_cnt[q];
    if (cnt > max_cands_per_q) cnt = max_cands_per_q;
    const float* cd = cand_buf_d + (int64_t)q * max_cands_per_q;
    const int64_t* ci = cand_buf_i + (int64_t)q * max_cands_per_q;

    for (int u = 0; u < cnt; ++u) {
        insert_topk10(loc_d, loc_i, cd[u], ci[u]);
    }

    for (int j = 0; j < 10; ++j) {
        topk_d[q * 10 + j] = loc_d[j];
        topk_i[q * 10 + j] = loc_i[j];
    }
    if (tau_sq_out != nullptr) {
        tau_sq_out[q] = loc_d[9];
    }
}

} // namespace

float launch_trident_scan(
    int mode,               // 0=K0, 1=K1, 2=K2
    int tile_size,          // 64 or 256
    int use_l1,             // 0 or 1
    torch::Tensor sq8_packed,
    torch::Tensor tile_orig_ids,
    torch::Tensor tile_r_minmax,
    torch::Tensor cluster_tile_off,
    torch::Tensor xb_orig,
    torch::Tensor xq_orig,
    torch::Tensor xq_pca,
    torch::Tensor vmin,
    torch::Tensor delta,
    torch::Tensor eps_dim,
    torch::Tensor blk_bounds4,
    torch::Tensor p1_cids,
    torch::Tensor p1_qids,
    torch::Tensor p1_dqcs,
    torch::Tensor p2_cids,
    torch::Tensor p2_qids,
    torch::Tensor p2_dqcs,
    torch::Tensor topk_d,
    torch::Tensor topk_i,
    torch::Tensor tau_sq,
    torch::Tensor cand_buf_d,
    torch::Tensor cand_buf_i,
    torch::Tensor cand_cnt,
    bool record_events
) {
    const int B = xq_orig.size(0);
    const int D = xq_orig.size(1);
    const int D_pad = xq_pca.size(1);
    const int num_blocks = blk_bounds4.size(0) - 1;
    const int M_p1 = p1_cids.size(0);
    const int M_p2 = p2_cids.size(0);
    const int max_cands_per_q = cand_buf_d.size(1);

    size_t smem_bytes = (size_t)(3 * D_pad + D) * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cudaEvent_t ev_start, ev_stop;
    if (record_events) {
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);
        cudaEventRecord(ev_start, stream);
    }

    auto dispatch = [&](auto mode_tag, auto tile_tag) {
        constexpr int M = decltype(mode_tag)::value;
        constexpr int T = decltype(tile_tag)::value;

        if (M_p1 > 0) {
            cudaMemsetAsync(cand_cnt.data_ptr<int32_t>(), 0, B * sizeof(int32_t), stream);
            trident_cluster_major_scan_kernel<M, T><<<M_p1, T, smem_bytes, stream>>>(
                reinterpret_cast<const uint32_t*>(sq8_packed.data_ptr<int32_t>()),
                tile_orig_ids.data_ptr<int32_t>(),
                tile_r_minmax.data_ptr<float>(),
                cluster_tile_off.data_ptr<int32_t>(),
                xb_orig.data_ptr<float>(),
                xq_orig.data_ptr<float>(),
                xq_pca.data_ptr<float>(),
                vmin.data_ptr<float>(),
                delta.data_ptr<float>(),
                eps_dim.data_ptr<float>(),
                blk_bounds4.data_ptr<int32_t>(),
                p1_cids.data_ptr<int32_t>(),
                p1_qids.data_ptr<int32_t>(),
                p1_dqcs.data_ptr<float>(),
                nullptr,
                M_p1, D, D_pad, num_blocks, use_l1, max_cands_per_q,
                cand_buf_d.data_ptr<float>(),
                cand_buf_i.data_ptr<int64_t>(),
                cand_cnt.data_ptr<int32_t>()
            );
            trident_merge_topk_kernel<<<B, 32, 0, stream>>>(
                topk_d.data_ptr<float>(),
                topk_i.data_ptr<int64_t>(),
                tau_sq.data_ptr<float>(),
                cand_buf_d.data_ptr<float>(),
                cand_buf_i.data_ptr<int64_t>(),
                cand_cnt.data_ptr<int32_t>(),
                B, max_cands_per_q, 1
            );
        }

        if (M_p2 > 0) {
            cudaMemsetAsync(cand_cnt.data_ptr<int32_t>(), 0, B * sizeof(int32_t), stream);
            trident_cluster_major_scan_kernel<M, T><<<M_p2, T, smem_bytes, stream>>>(
                reinterpret_cast<const uint32_t*>(sq8_packed.data_ptr<int32_t>()),
                tile_orig_ids.data_ptr<int32_t>(),
                tile_r_minmax.data_ptr<float>(),
                cluster_tile_off.data_ptr<int32_t>(),
                xb_orig.data_ptr<float>(),
                xq_orig.data_ptr<float>(),
                xq_pca.data_ptr<float>(),
                vmin.data_ptr<float>(),
                delta.data_ptr<float>(),
                eps_dim.data_ptr<float>(),
                blk_bounds4.data_ptr<int32_t>(),
                p2_cids.data_ptr<int32_t>(),
                p2_qids.data_ptr<int32_t>(),
                p2_dqcs.data_ptr<float>(),
                (M_p1 > 0) ? tau_sq.data_ptr<float>() : nullptr,
                M_p2, D, D_pad, num_blocks, use_l1, max_cands_per_q,
                cand_buf_d.data_ptr<float>(),
                cand_buf_i.data_ptr<int64_t>(),
                cand_cnt.data_ptr<int32_t>()
            );
            trident_merge_topk_kernel<<<B, 32, 0, stream>>>(
                topk_d.data_ptr<float>(),
                topk_i.data_ptr<int64_t>(),
                nullptr,
                cand_buf_d.data_ptr<float>(),
                cand_buf_i.data_ptr<int64_t>(),
                cand_cnt.data_ptr<int32_t>(),
                B, max_cands_per_q, (M_p1 == 0) ? 1 : 0
            );
        }
    };

    if (tile_size == 64) {
        if (mode == 0) dispatch(std::integral_constant<int, 0>{}, std::integral_constant<int, 64>{});
        else if (mode == 1) dispatch(std::integral_constant<int, 1>{}, std::integral_constant<int, 64>{});
        else dispatch(std::integral_constant<int, 2>{}, std::integral_constant<int, 64>{});
    } else {
        if (mode == 0) dispatch(std::integral_constant<int, 0>{}, std::integral_constant<int, 256>{});
        else if (mode == 1) dispatch(std::integral_constant<int, 1>{}, std::integral_constant<int, 256>{});
        else dispatch(std::integral_constant<int, 2>{}, std::integral_constant<int, 256>{});
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
    m.def("launch_trident_scan", &launch_trident_scan, "TRIDENT Real GPU IVF Scan (K0, K1, K2)");
}
