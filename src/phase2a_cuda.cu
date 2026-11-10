#include <torch/extension.h>
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

__global__ void phase2a_sim_kernel(
    const float* __restrict__ xb_orig,          // [N, D]
    const float* __restrict__ xq_orig,          // [Q, D]
    const int64_t* __restrict__ orig_ids,       // [N]
    const float* __restrict__ r_sorted,         // [N]
    const uint8_t* __restrict__ sq8_sorted,     // [N, D]
    const float* __restrict__ xq_rot,           // [Q, D]
    const float* __restrict__ vmin,             // [D]
    const float* __restrict__ delta,            // [D]
    const float* __restrict__ eps_dim,          // [D]
    const int64_t* __restrict__ offsets,        // [nlist + 1]
    const int32_t* __restrict__ c_order,        // [Q, nprobe]
    const float* __restrict__ cdists,           // [Q, nprobe] (Euclidean distance, not sq)
    const int32_t* __restrict__ blk_bounds,     // [num_blocks + 1]
    const int N,
    const int Q,
    const int D,
    const int nprobe,
    const int num_blocks,
    const int tile_size,                        // 64 or 256
    const int l1_on,                            // 0 or 1
    const int exact_only,                       // 1 => bypass pruning, compute exact IVF-Flat top-10
    int64_t* __restrict__ out_top10_ids,        // [Q, 10]
    float* __restrict__ out_top10_dists,        // [Q, 10]
    int64_t* __restrict__ out_counters,         // [Q, 10]
    int64_t* __restrict__ out_blk_active_tiles, // [num_blocks] atomic/global sum
    int64_t* __restrict__ out_blk_active_warps  // [num_blocks] atomic/global sum
) {
    const int q = blockIdx.x;
    if (q >= Q) return;

    const int tid = threadIdx.x; // 0..255
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int W_tile = tile_size >> 5; // 2 for 64, 8 for 256

    __shared__ float sh_topk_d[10];
    __shared__ int64_t sh_topk_i[10];
    __shared__ float sh_tau_sq;
    __shared__ float sh_tau;
    __shared__ int sh_warp_alive[8];
    __shared__ int64_t sh_cand_oid[256];
    __shared__ int64_t sh_surv_oids[256];
    __shared__ int sh_n_surv;
    __shared__ float sh_red[8];

    if (tid < 10) {
        sh_topk_d[tid] = INFINITY;
        sh_topk_i[tid] = -1;
    }
    if (tid == 0) {
        sh_tau_sq = INFINITY;
        sh_tau = INFINITY;
    }
    __syncthreads();

    const float* q_orig_ptr = xq_orig + (int64_t)q * D;
    const float* q_rot_ptr  = xq_rot  + (int64_t)q * D;

    int64_t total_vecs = 0;
    int64_t l1_vec_kept = 0;
    int64_t l1_tile_kept = 0;
    int64_t vec_dims_comp = 0;
    int64_t warp_dims_comp = 0;
    int64_t tile_dims_comp = 0;
    int64_t surv_fp32 = 0;
    int64_t total_tiles_all = 0;
    int64_t total_tiles_l1_kept = 0;
    int64_t total_warps_l1_kept = 0;

    for (int p = 0; p < nprobe; ++p) {
        int32_t cid = c_order[q * nprobe + p];
        float d_qc  = cdists[q * nprobe + p];
        int64_t s = offsets[cid];
        int64_t e = offsets[cid + 1];
        int64_t M = e - s;
        if (M <= 0) continue;

        for (int64_t t_off = 0; t_off < M; t_off += tile_size) {
            int T_cur = (int)((t_off + tile_size <= M) ? tile_size : (M - t_off));
            if (tid == 0) {
                total_vecs += T_cur;
                total_tiles_all += 1;
            }

            if (exact_only) {
                // Exact unpruned IVF-Flat reference
                if (tid < T_cur) {
                    sh_surv_oids[tid] = orig_ids[s + t_off + tid];
                }
                if (tid == 0) {
                    sh_n_surv = T_cur;
                }
                __syncthreads();

                int n_surv = sh_n_surv;
                for (int u = 0; u < n_surv; ++u) {
                    int64_t oid = sh_surv_oids[u];
                    const float* x_ptr = xb_orig + oid * D;
                    float part = 0.0f;
                    for (int d = tid; d < D; d += 256) {
                        float diff = x_ptr[d] - q_orig_ptr[d];
                        part += diff * diff;
                    }
                    part = warp_reduce_sum(part);
                    if (lane == 0) sh_red[warp_id] = part;
                    __syncthreads();
                    if (tid == 0) {
                        float d_exact = sh_red[0] + sh_red[1] + sh_red[2] + sh_red[3]
                                      + sh_red[4] + sh_red[5] + sh_red[6] + sh_red[7];
                        if (d_exact < sh_topk_d[9] || (d_exact == sh_topk_d[9] && oid < sh_topk_i[9])) {
                            sh_topk_d[9] = d_exact;
                            sh_topk_i[9] = oid;
                            for (int j = 9; j > 0; --j) {
                                if (sh_topk_d[j] < sh_topk_d[j - 1] ||
                                    (sh_topk_d[j] == sh_topk_d[j - 1] && sh_topk_i[j] < sh_topk_i[j - 1])) {
                                    float td = sh_topk_d[j - 1]; sh_topk_d[j - 1] = sh_topk_d[j]; sh_topk_d[j] = td;
                                    int64_t ti = sh_topk_i[j - 1]; sh_topk_i[j - 1] = sh_topk_i[j]; sh_topk_i[j] = ti;
                                } else {
                                    break;
                                }
                            }
                            sh_tau_sq = sh_topk_d[9];
                            sh_tau = sqrtf(fmaxf(sh_tau_sq, 0.0f));
                        }
                    }
                    __syncthreads();
                }
                continue;
            }

            // Pruning mode (L1 + L2 + L3)
            float cur_tau = sh_tau;
            float cur_tau_sq = sh_tau_sq;
            float tau_sq_safe = isinf(cur_tau_sq) ? INFINITY : (cur_tau_sq * (1.0f + 2e-5f) + 1e-6f);

            bool tile_l1_pruned = false;
            if (!isinf(cur_tau)) {
                float tau_l1_safe = cur_tau * (1.0f + 1e-4f) + 1e-4f;
                float lo = d_qc - tau_l1_safe;
                float hi = d_qc + tau_l1_safe;

                // Vector-level L1 count
                int v_in_band = 0;
                if (tid < T_cur) {
                    float rv = r_sorted[s + t_off + tid];
                    if (rv >= lo && rv <= hi) v_in_band = 1;
                }
                int w_cnt = __popc(__ballot_sync(0xffffffff, v_in_band != 0));
                if (lane == 0) sh_warp_alive[warp_id] = (warp_id < W_tile) ? w_cnt : 0;
                __syncthreads();
                if (tid == 0) {
                    int sum_v = 0;
                    for (int w = 0; w < W_tile; ++w) sum_v += sh_warp_alive[w];
                    l1_vec_kept += sum_v;
                }

                if (l1_on) {
                    float r_min = r_sorted[s + t_off];
                    float r_max = r_sorted[s + t_off + T_cur - 1];
                    if (r_max < lo || r_min > hi) {
                        tile_l1_pruned = true;
                    }
                }
            } else {
                if (tid == 0) {
                    l1_vec_kept += T_cur;
                }
            }
            __syncthreads();
            if (tile_l1_pruned) {
                continue;
            }

            int valid_warps_in_tile = (T_cur + 31) >> 5;
            if (tid == 0) {
                l1_tile_kept += T_cur;
                total_tiles_l1_kept += 1;
                total_warps_l1_kept += valid_warps_in_tile;
            }

            bool alive = (tid < T_cur);
            int64_t pos = s + t_off + tid;
            float acc_lb = 0.0f;

            for (int b = 0; b < num_blocks; ++b) {
                int b_start = blk_bounds[b];
                int b_end   = blk_bounds[b + 1];
                int w_b     = b_end - b_start;

                if (warp_id < W_tile) {
                    unsigned mask_alive = __ballot_sync(0xffffffff, alive);
                    int w_alive_cnt = __popc(mask_alive);
                    int w_valid_cnt = min(32, max(0, T_cur - (warp_id << 5)));
                    if (lane == 0) {
                        sh_warp_alive[warp_id] = w_alive_cnt;
                        if (w_alive_cnt > 0 && w_valid_cnt > 0) {
                            atomicAdd((unsigned long long*)&out_blk_active_warps[b], 1ULL);
                        }
                    }
                }
                __syncthreads();

                int tile_alive_cnt = 0;
                int tile_warp_dims = 0;
                for (int w = 0; w < W_tile; ++w) {
                    int w_cnt = sh_warp_alive[w];
                    tile_alive_cnt += w_cnt;
                    if (w_cnt > 0) {
                        int w_valid_cnt = min(32, max(0, T_cur - (w << 5)));
                        tile_warp_dims += w_valid_cnt * w_b;
                    }
                }

                if (tile_alive_cnt == 0) {
                    // Entire tile is dead! Early-stop remaining blocks!
                    __syncthreads();
                    break;
                }

                if (tid == 0) {
                    vec_dims_comp  += (int64_t)tile_alive_cnt * w_b;
                    warp_dims_comp += (int64_t)tile_warp_dims;
                    tile_dims_comp += (int64_t)T_cur * w_b;
                    atomicAdd((unsigned long long*)&out_blk_active_tiles[b], 1ULL);
                }

                if (alive) {
                    const uint8_t* code_ptr = sq8_sorted + pos * D;
                    for (int d = b_start; d < b_end; ++d) {
                        float x_hat = vmin[d] + (float)code_ptr[d] * delta[d];
                        float diff = fabsf(x_hat - q_rot_ptr[d]) - eps_dim[d];
                        if (diff > 0.0f) {
                            acc_lb += diff * diff;
                        }
                    }
                    if (acc_lb > tau_sq_safe) {
                        alive = false;
                    }
                }
                __syncthreads();
            }

            // Collect survivors in deterministic order
            if (tid < T_cur) {
                sh_cand_oid[tid] = alive ? orig_ids[pos] : -1;
            }
            __syncthreads();

            if (tid == 0) {
                int ns = 0;
                for (int i = 0; i < T_cur; ++i) {
                    if (sh_cand_oid[i] >= 0) {
                        sh_surv_oids[ns++] = sh_cand_oid[i];
                    }
                }
                sh_n_surv = ns;
                surv_fp32 += ns;
            }
            __syncthreads();

            int n_surv = sh_n_surv;
            for (int u = 0; u < n_surv; ++u) {
                int64_t oid = sh_surv_oids[u];
                const float* x_ptr = xb_orig + oid * D;
                float part = 0.0f;
                for (int d = tid; d < D; d += 256) {
                    float diff = x_ptr[d] - q_orig_ptr[d];
                    part += diff * diff;
                }
                part = warp_reduce_sum(part);
                if (lane == 0) sh_red[warp_id] = part;
                __syncthreads();
                if (tid == 0) {
                    float d_exact = sh_red[0] + sh_red[1] + sh_red[2] + sh_red[3]
                                  + sh_red[4] + sh_red[5] + sh_red[6] + sh_red[7];
                    if (d_exact < sh_topk_d[9] || (d_exact == sh_topk_d[9] && oid < sh_topk_i[9])) {
                        sh_topk_d[9] = d_exact;
                        sh_topk_i[9] = oid;
                        for (int j = 9; j > 0; --j) {
                            if (sh_topk_d[j] < sh_topk_d[j - 1] ||
                                (sh_topk_d[j] == sh_topk_d[j - 1] && sh_topk_i[j] < sh_topk_i[j - 1])) {
                                float td = sh_topk_d[j - 1]; sh_topk_d[j - 1] = sh_topk_d[j]; sh_topk_d[j] = td;
                                int64_t ti = sh_topk_i[j - 1]; sh_topk_i[j - 1] = sh_topk_i[j]; sh_topk_i[j] = ti;
                            } else {
                                break;
                            }
                        }
                        sh_tau_sq = sh_topk_d[9];
                        sh_tau = sqrtf(fmaxf(sh_tau_sq, 0.0f));
                    }
                }
                __syncthreads();
            }
        }
    }

    if (tid < 10) {
        out_top10_ids[q * 10 + tid]   = sh_topk_i[tid];
        out_top10_dists[q * 10 + tid] = sh_topk_d[tid];
    }
    if (tid == 0) {
        int64_t* c_out = out_counters + q * 10;
        c_out[0] = total_vecs;
        c_out[1] = l1_vec_kept;
        c_out[2] = l1_tile_kept;
        c_out[3] = vec_dims_comp;
        c_out[4] = warp_dims_comp;
        c_out[5] = tile_dims_comp;
        c_out[6] = surv_fp32;
        c_out[7] = total_tiles_all;
        c_out[8] = total_tiles_l1_kept;
        c_out[9] = total_warps_l1_kept;
    }
}

} // namespace

std::vector<torch::Tensor> run_phase2a_sim(
    torch::Tensor xb_orig,
    torch::Tensor xq_orig,
    torch::Tensor orig_ids,
    torch::Tensor r_sorted,
    torch::Tensor sq8_sorted,
    torch::Tensor xq_rot,
    torch::Tensor vmin,
    torch::Tensor delta,
    torch::Tensor eps_dim,
    torch::Tensor offsets,
    torch::Tensor c_order,
    torch::Tensor cdists,
    torch::Tensor blk_bounds,
    int tile_size,
    int l1_on,
    int exact_only
) {
    const int N = xb_orig.size(0);
    const int D = xb_orig.size(1);
    const int Q = xq_orig.size(0);
    const int nprobe = c_order.size(1);
    const int num_blocks = blk_bounds.size(0) - 1;

    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(xb_orig.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(xb_orig.device());

    auto out_top10_ids   = torch::empty({Q, 10}, opts_i64);
    auto out_top10_dists = torch::empty({Q, 10}, opts_f32);
    auto out_counters    = torch::zeros({Q, 10}, opts_i64);
    auto out_blk_tiles   = torch::zeros({num_blocks}, opts_i64);
    auto out_blk_warps   = torch::zeros({num_blocks}, opts_i64);

    phase2a_sim_kernel<<<Q, 256>>>(
        xb_orig.data_ptr<float>(),
        xq_orig.data_ptr<float>(),
        orig_ids.data_ptr<int64_t>(),
        r_sorted.data_ptr<float>(),
        sq8_sorted.data_ptr<uint8_t>(),
        xq_rot.data_ptr<float>(),
        vmin.data_ptr<float>(),
        delta.data_ptr<float>(),
        eps_dim.data_ptr<float>(),
        offsets.data_ptr<int64_t>(),
        c_order.data_ptr<int32_t>(),
        cdists.data_ptr<float>(),
        blk_bounds.data_ptr<int32_t>(),
        N, Q, D, nprobe, num_blocks, tile_size, l1_on, exact_only,
        out_top10_ids.data_ptr<int64_t>(),
        out_top10_dists.data_ptr<float>(),
        out_counters.data_ptr<int64_t>(),
        out_blk_tiles.data_ptr<int64_t>(),
        out_blk_warps.data_ptr<int64_t>()
    );

    return {out_top10_ids, out_top10_dists, out_counters, out_blk_tiles, out_blk_warps};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_phase2a_sim", &run_phase2a_sim, "Phase 2A Exact GPU Simulator");
}
