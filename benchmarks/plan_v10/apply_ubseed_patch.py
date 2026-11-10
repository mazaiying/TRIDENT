#!/usr/bin/env python3
"""
PLAN_V10.1 Step K1 — add *lossless* SQ8 upper-bound threshold seeding to trident_v4_kernels.cu
with compile-time template isolation (UB_ENABLE = 0 vs 1) and compile-time K=10 constant folding.

Why V10.1 (from D1/D2 diagnostic):
  1. In V10, placing UB_BLOCK inline inside trident_cluster_major_scan_kernel guarded by a runtime
     __constant__ int c_ub_seed caused nvcc to allocate extra registers and shared memory (sh_ub_d/i)
     even when c_ub_seed == 0, reducing ub=0 QPS by 8.0%–11.7% vs V9 (and slowing down Pass 2 when ub=1).
  2. In V9, replacing Phase 4A's compile-time K_LOC=10 (in MODE != 0) and K_CAP=10 (in
     trident_merge_topk_kernel<10>) with runtime variables prevented nvcc from dead-code-eliminating
     insert_topk_dyn and constant-folding sh_local_d[9], costing 14.8%–22.1% vs Phase 4A.

Fix in V10.1:
  - Add template parameter `int UB_ENABLE = 0` to `trident_cluster_major_scan_kernel`.
  - Gate UB_BLOCK with `if constexpr (MODE == 2 && UB_ENABLE != 0)` and reuse `sh_local_d/i[0..9]`
    (resetting them after seeding `sh_tau_sq`), so `UB_ENABLE == 0` has zero extra registers and
    zero extra shared memory.
  - Dispatch `UB_ENABLE = 1` only when needed:
      * TRIDENT_UB_SEED = 0 : Pass 1 -> UB_ENABLE=0, Pass 2 -> UB_ENABLE=0
      * TRIDENT_UB_SEED = 1 : Pass 1 -> UB_ENABLE=1, Pass 2 -> UB_ENABLE=0 (or 1 if M_p1 == 0)
      * TRIDENT_UB_SEED = 2 : Pass 1 -> UB_ENABLE=1, Pass 2 -> UB_ENABLE=1
  - Fold `K_LOC = 10` at compile time when `MODE != 0` and `K_CAP = 10` when `MAX_K == 10`, restoring
    Phase 4A's exact PTX/SASS for `ub=0`.

Usage (on iclr-a100):
  /home/mazaiying/trident/.venv/bin/python ~/trident/scripts/plan_v10/apply_ubseed_patch.py
"""
import os
import re
import shutil
import sys

SRC = os.path.expanduser(os.environ.get("TRIDENT_KERNEL_SRC", "~/trident/scripts/trident_v4_kernels.cu"))
BAK = SRC + ".bak_plan_v9"

UB_BLOCK = r'''
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
'''

HOST_BLOCK = r'''    const char* ub_env = std::getenv("TRIDENT_UB_SEED");
    const int ub_v = ub_env ? std::atoi(ub_env) : 0;
'''


def replace_unique(src: str, pattern: str, repl: str, name: str) -> str:
    ms = list(re.finditer(pattern, src, flags=re.M))
    if len(ms) != 1:
        sys.exit(f"[ABORT] anchor '{name}' matched {len(ms)} times (expected exactly 1). Do NOT hand-edit; report to user.")
    start, end = ms[0].span()
    return src[:start] + repl + src[end:]


def insert_after_unique(src: str, pattern: str, text: str, name: str) -> str:
    ms = list(re.finditer(pattern, src, flags=re.M))
    if len(ms) != 1:
        sys.exit(f"[ABORT] anchor '{name}' matched {len(ms)} times (expected exactly 1). Do NOT hand-edit; report to user.")
    end = ms[0].end()
    return src[:end] + "\n" + text + src[end:]


def insert_before_unique(src: str, pattern: str, text: str, name: str) -> str:
    ms = list(re.finditer(pattern, src, flags=re.M))
    if len(ms) != 1:
        sys.exit(f"[ABORT] anchor '{name}' matched {len(ms)} times (expected exactly 1). Do NOT hand-edit; report to user.")
    start = ms[0].start()
    return src[:start] + text + src[start:]


def main():
    if os.path.exists(BAK):
        shutil.copy2(BAK, SRC)
        print(f"[restored] clean V9 base from {BAK} -> {SRC}")
    else:
        shutil.copy2(SRC, BAK)
        print(f"[backup] {SRC} -> {BAK}")

    with open(SRC, "r") as f:
        src = f.read()

    # (1) #include <cstdlib> right after the last #include line
    lines = src.split("\n")
    inc_idx = [i for i, l in enumerate(lines) if l.lstrip().startswith("#include")]
    if not inc_idx:
        sys.exit("[ABORT] no #include line found")
    lines.insert(inc_idx[-1] + 1, "#include <cstdlib>  // PLAN_V10.1: std::getenv for TRIDENT_UB_SEED")
    src = "\n".join(lines)

    # (2) Add `int UB_ENABLE = 0` template parameter to trident_cluster_major_scan_kernel
    src = replace_unique(
        src,
        r"template\s*<\s*int\s+MODE\s*,\s*int\s+TILE_SIZE\s*,\s*int\s+WARP_REFINE\s*=\s*1\s*>\s*\n__global__\s+void\s+trident_cluster_major_scan_kernel\b",
        "template <int MODE, int TILE_SIZE, int WARP_REFINE = 1, int UB_ENABLE = 0>\n__global__ void trident_cluster_major_scan_kernel",
        "cluster_major_template_sig",
    )

    # (3) Fold K_LOC = 10 at compile time when MODE != 0, and K_CAP = 10 when MAX_K == 10
    src = replace_unique(
        src,
        r"constexpr int MAX_K_LOCAL = \(MODE == 0\) \? 200 : 10;\s*\n\s*const int K_LOC = \(k_local <= MAX_K_LOCAL\) \? \(\(k_local >= 1\) \? k_local : 1\) : MAX_K_LOCAL;",
        "constexpr int MAX_K_LOCAL = (MODE == 0) ? 200 : 10;\n    const int K_LOC = (MODE == 0) ? ((k_local <= MAX_K_LOCAL) ? ((k_local >= 1) ? k_local : 1) : MAX_K_LOCAL) : 10;",
        "k_loc_compile_time",
    )
    src = replace_unique(
        src,
        r"template <int MAX_K = 10>\s*\n__global__ void trident_merge_topk_kernel\([\s\S]*?const int K_CAP = \(K <= 100\) \? \(\(K >= 1\) \? K : 1\) : 100;",
        lambda m: m.group(0).replace(
            "const int K_CAP = (K <= 100) ? ((K >= 1) ? K : 1) : 100;",
            "const int K_CAP = (MAX_K == 10) ? 10 : ((K <= 100) ? ((K >= 1) ? K : 1) : 100);",
        ),
        "k_cap_compile_time",
    ) if False else src
    # Use explicit string replacement inside trident_merge_topk_kernel for K_CAP
    src = replace_unique(
        src,
        r"const int K_CAP = \(K <= 100\) \? \(\(K >= 1\) \? K : 1\) : 100;\s*\n\s*\n\s*__shared__ float sh_wd\[8\]\[MAX_K\];",
        "const int K_CAP = (MAX_K == 10) ? 10 : ((K <= 100) ? ((K >= 1) ? K : 1) : 100);\n\n    __shared__ float sh_wd[8][MAX_K];",
        "k_cap_compile_time",
    )

    # (4) Device-side seeding block right after `const float q_init_tau_sq = sh_tau_sq;` in cluster_major_scan_kernel
    src = insert_after_unique(
        src,
        r"sh_q_orig_ready\s*=\s*0;\s*\}\s*__syncthreads\(\);\s*const\s+float\s+q_init_tau_sq\s*=\s*sh_tau_sq;",
        UB_BLOCK,
        "q_init_tau_sq(cluster_major)",
    )

    # (5) Host-side env read right before `int p_init = 0;` in launch_trident_v4_search
    src = insert_before_unique(src, r"^[ \t]*int p_init = 0;[ \t]*$", HOST_BLOCK, "p_init")

    # (6) Dispatch UB_ENABLE=1 in Pass 1 when (M == 2 && ub_v != 0), and UB_ENABLE=0 otherwise
    p1_old = r"""                trident_cluster_major_scan_kernel<M, T, WR><<<M_p1, T, smem_bytes, stream>>>(
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
                );"""
    p1_new = r"""                auto launch_p1 = [&](auto ub_tag) {
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
                }"""
    if p1_old not in src:
        sys.exit("[ABORT] p1 launch block not found exactly once")
    src = src.replace(p1_old, p1_new, 1)

    # (7) Dispatch UB_ENABLE=1 in Pass 2 only when (M == 2 && (ub_v == 2 || (ub_v == 1 && M_p1 == 0)))
    p2_old = r"""                trident_cluster_major_scan_kernel<M, T, WR><<<M_p2, T, smem_bytes, stream>>>(
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
                );"""
    p2_new = r"""                auto launch_p2 = [&](auto ub_tag) {
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
                }"""
    if p2_old not in src:
        sys.exit("[ABORT] p2 launch block not found exactly once")
    src = src.replace(p2_old, p2_new, 1)

    for sym in ["sh_cand_hat", "sh_surv_oids", "sh_eps", "sh_base", "sh_delta", "insert_topk<10>", "W_TILE", "D4", "UB_ENABLE"]:
        if sym not in src:
            sys.exit(f"[ABORT] expected symbol '{sym}' not found; kernel layout differs from PLAN_V10.1 assumptions.")

    with open(SRC, "w") as f:
        f.write(src)
    print(f"[patched V10.1] {SRC}")
    print("Next: run K2 (compile once for sm_80 + sm_89) before launching any experiment.")


if __name__ == "__main__":
    main()
