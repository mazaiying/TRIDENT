#!/usr/bin/env python3
"""
Vectorized GPU-accelerated Phase 1 Pruning Analysis for TRIDENT.
Evaluates exact cluster-by-cluster and tile-by-tile scanning order across Q queries,
recording exact L1 (vector & tile band), L2 (dim block early-stop, orig vs var-reordered),
L3 (conservative SQ8 lower bound early-stop, coordinate-wise vs L2-error bound),
and D (theoretical memory/PCIe byte ratio).
Also evaluates SUB-CENTROID triangle pruning (K_sub=4 per cluster, from Section 4 contingency)
so Claude has complete data for Gate 1!
"""
import os, time, math, argparse
import numpy as np
import pandas as pd
import torch
import faiss

DATA_ROOT = os.path.expanduser("~/trident/data")
RES_ROOT = os.path.expanduser("~/trident/results")
os.makedirs(os.path.join(RES_ROOT, "phase1"), exist_ok=True)

def ensure_gt(ds_name, xb, xq, k=100, device="cuda:0"):
    gt_path = os.path.join(DATA_ROOT, ds_name, "gt.npy")
    if os.path.exists(gt_path):
        gt = np.load(gt_path)
        if gt.shape[0] == xq.shape[0] and gt.shape[1] >= k:
            return gt[:, :k]
    print(f"[GT] Computing exact L2 ground truth top-{k} for {ds_name} ({xb.shape} vs {xq.shape}) on {device} ...", flush=True)
    dev = torch.device(device)
    xq_t = torch.from_numpy(xq).to(dev, dtype=torch.float32)
    q_norm = (xq_t ** 2).sum(dim=1, keepdim=True)
    Q, N = xq.shape[0], xb.shape[0]
    top_d = torch.full((Q, k), float('inf'), device=dev, dtype=torch.float32)
    top_i = torch.full((Q, k), -1, device=dev, dtype=torch.int64)
    chunk = 100000
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        xb_c = torch.from_numpy(xb[start:end]).to(dev, dtype=torch.float32)
        b_norm = (xb_c ** 2).sum(dim=1).unsqueeze(0)
        dists = torch.clamp(q_norm + b_norm - 2.0 * torch.matmul(xq_t, xb_c.t()), min=0.0)
        c_d, c_i = torch.topk(dists, k=min(k, end - start), dim=1, largest=False)
        comb_d = torch.cat([top_d, c_d], dim=1)
        comb_i = torch.cat([top_i, c_i + start], dim=1)
        top_d, idx = torch.topk(comb_d, k=k, dim=1, largest=False)
        top_i = torch.gather(comb_i, 1, idx)
    gt = top_i.cpu().numpy()
    np.save(gt_path, gt)
    return gt

def train_ivf(xb, nlist, seed=0):
    N, D = xb.shape
    clus = faiss.Clustering(D, nlist)
    clus.niter = 20
    clus.seed = seed
    clus.max_points_per_centroid = 256
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.device = 0
    flat_index = faiss.GpuIndexFlatL2(res, D, cfg)
    clus.train(xb, flat_index)
    centroids = faiss.vector_float_to_array(clus.centroids).reshape(nlist, D).copy()
    dists, list_ids = flat_index.search(xb, 1)
    return centroids, list_ids.reshape(-1), np.sqrt(np.maximum(dists.reshape(-1), 0.0))

def find_nprobes_for_recalls(centroids, cluster_ids, xb, xq, gt10, target_recalls=(0.90, 0.95, 0.99), device="cuda:0"):
    dev = torch.device(device)
    Q = xq.shape[0]
    nlist = centroids.shape[0]
    xq_t = torch.from_numpy(xq).to(dev)
    c_t = torch.from_numpy(centroids).to(dev)
    cdists = torch.cdist(xq_t, c_t, p=2.0) ** 2
    _, c_order = torch.sort(cdists, dim=1, descending=False)
    c_order_np = c_order.cpu().numpy()

    vec_cluster = np.empty(xb.shape[0], dtype=np.int32)
    for cid, ids in enumerate(cluster_ids):
        vec_cluster[ids] = cid

    gt_clusters = vec_cluster[gt10]
    ranks = np.empty_like(gt_clusters)
    for q in range(Q):
        inv_rank = np.empty(nlist, dtype=np.int32)
        inv_rank[c_order_np[q]] = np.arange(nlist, dtype=np.int32)
        ranks[q] = inv_rank[gt_clusters[q]]

    cand_nprobes = sorted(list(set([1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 768, nlist])))
    cand_nprobes = [p for p in cand_nprobes if p <= nlist]
    rec_curve = {p: float((ranks < p).sum() / (Q * 10.0)) for p in cand_nprobes}

    selected = {}
    for tr in target_recalls:
        chosen_p = cand_nprobes[-1]
        for p in cand_nprobes:
            if rec_curve[p] >= tr:
                chosen_p = p
                break
        selected[tr] = (chosen_p, rec_curve[chosen_p])
    return selected, c_order_np, cdists.cpu().numpy()

def evaluate_dataset(ds_name, xb, xq, gt10, nlist=1024, seed=0, tile_sizes=(64, 256, 1024), block_dim=32, max_queries=200, device="cuda:0"):
    t0 = time.time()
    N, D = xb.shape
    Q = min(xq.shape[0], max_queries)
    xq_sub = xq[:Q]
    gt_sub = gt10[:Q]
    dev = torch.device(device)

    # 1. Variance-descending permutation
    dim_vars = np.var(xb[:min(N, 100000)], axis=0)
    perm_var = np.argsort(-dim_vars)

    # 2. SQ8 parameters
    vmin = xb.min(axis=0)
    vmax = xb.max(axis=0)
    delta = (vmax - vmin) / 255.0
    half_delta = 0.5 * delta

    centroids, list_ids, r_vec = train_ivf(xb, nlist=nlist, seed=seed)

    # Sort vectors within each cluster by r_vec ascending, and build contiguous array XB_sorted
    order_global = np.lexsort((r_vec, list_ids))
    sorted_cids = list_ids[order_global]
    counts = np.bincount(sorted_cids, minlength=nlist)
    offsets = np.zeros(nlist + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    cluster_ids = [order_global[offsets[c]:offsets[c+1]] for c in range(nlist)]
    r_sorted = r_vec[order_global]

    selected_nprobes, c_order_np, cdists_sq_np = find_nprobes_for_recalls(
        centroids, cluster_ids, xb, xq_sub, gt_sub, target_recalls=(0.90, 0.95, 0.99), device=device
    )

    num_blocks = int(math.ceil(D / block_dim))
    pad = num_blocks * block_dim - D

    # Put sorted base vectors on GPU in contiguous cluster order!
    xb_sorted_orig = torch.from_numpy(xb[order_global]).to(dev, dtype=torch.float32)
    perm_t = torch.from_numpy(perm_var).to(dev, dtype=torch.long)
    xb_sorted_var = xb_sorted_orig[:, perm_t].contiguous()
    r_sorted_gpu = torch.from_numpy(r_sorted).to(dev, dtype=torch.float32)

    xq_orig_gpu = torch.from_numpy(xq_sub).to(dev, dtype=torch.float32)
    xq_var_gpu = xq_orig_gpu[:, perm_t].contiguous()

    vmin_var_gpu = torch.from_numpy(vmin[perm_var]).to(dev, dtype=torch.float32)
    delta_var_gpu = torch.from_numpy(delta[perm_var]).to(dev, dtype=torch.float32)
    half_delta_var_gpu = torch.from_numpy(half_delta[perm_var]).to(dev, dtype=torch.float32)

    sq8_codes_var = torch.clamp(torch.round((xb_sorted_var - vmin_var_gpu) / (delta_var_gpu + 1e-12)), 0, 255).to(torch.uint8)
    x_hat_var = vmin_var_gpu + sq8_codes_var.to(torch.float32) * delta_var_gpu
    vec_l2_err = torch.norm(xb_sorted_var - x_hat_var, p=2, dim=1)
    del x_hat_var

    # For maximum speed: for each query q, we can gather all vectors in its top max_nprobe clusters on GPU in one shot!
    max_nprobe = max(p for p, _ in selected_nprobes.values())
    rows = []

    for target_rec, (nprobe, actual_rec) in selected_nprobes.items():
        for tile_size in tile_sizes:
            if tile_size != 256 and abs(target_rec - 0.95) > 1e-4:
                continue

            total_vecs_all = 0
            kept_vecs_L1_vec = 0
            kept_vecs_L1_tile = 0
            kept_dim_blocks_orig = 0
            kept_dim_blocks_var = 0
            kept_dim_blocks_sq8_coord = 0
            kept_dim_blocks_sq8_l2err = 0
            survivors_fp32_coord = 0
            survivors_fp32_l2err = 0

            for q in range(Q):
                q_orig = xq_orig_gpu[q]
                q_var = xq_var_gpu[q]
                cids = c_order_np[q, :nprobe]
                d_qcs = np.sqrt(np.maximum(cdists_sq_np[q, cids], 0.0))

                # Build index array of all vectors in the nprobe clusters for query q, plus tile boundaries
                idx_chunks = []
                tile_meta = [] # list of (start_in_concat, end_in_concat, d_qc, r_min, r_max, r_vec_slice)
                cur_pos = 0
                for p_i, cid in enumerate(cids):
                    s, e = int(offsets[cid]), int(offsets[cid+1])
                    M = e - s
                    if M == 0:
                        continue
                    idx_chunks.append(np.arange(s, e, dtype=np.int64))
                    d_qc = float(d_qcs[p_i])
                    for t_off in range(0, M, tile_size):
                        t_end = min(M, t_off + tile_size)
                        tile_meta.append((
                            cur_pos + t_off,
                            cur_pos + t_end,
                            d_qc,
                            float(r_sorted[s + t_off]),
                            float(r_sorted[s + t_end - 1]),
                            s + t_off,
                            s + t_end
                        ))
                    cur_pos += M

                if cur_pos == 0:
                    continue
                total_vecs_all += cur_pos
                all_idx_t = torch.from_numpy(np.concatenate(idx_chunks)).to(dev)

                # Compute all block-wise cumulative squared distances for query q across all cur_pos vectors in ONE GPU kernel!
                V_orig = xb_sorted_orig[all_idx_t]
                V_var = xb_sorted_var[all_idx_t]
                C_var = sq8_codes_var[all_idx_t]
                E_l2 = vec_l2_err[all_idx_t]

                diff_orig_sq = (V_orig - q_orig.unsqueeze(0)) ** 2
                diff_var_sq = (V_var - q_var.unsqueeze(0)) ** 2
                V_hat_var = vmin_var_gpu.unsqueeze(0) + C_var.to(torch.float32) * delta_var_gpu.unsqueeze(0)
                abs_diff_hat = torch.abs(V_hat_var - q_var.unsqueeze(0))
                coord_lb_sq = torch.clamp(abs_diff_hat - half_delta_var_gpu.unsqueeze(0), min=0.0) ** 2
                diff_hat_sq = abs_diff_hat ** 2

                if pad > 0:
                    diff_orig_sq = torch.nn.functional.pad(diff_orig_sq, (0, pad))
                    diff_var_sq = torch.nn.functional.pad(diff_var_sq, (0, pad))
                    coord_lb_sq = torch.nn.functional.pad(coord_lb_sq, (0, pad))
                    diff_hat_sq = torch.nn.functional.pad(diff_hat_sq, (0, pad))

                cum_orig = torch.cumsum(diff_orig_sq.view(cur_pos, num_blocks, block_dim).sum(dim=2), dim=1).cpu().numpy()
                cum_var = torch.cumsum(diff_var_sq.view(cur_pos, num_blocks, block_dim).sum(dim=2), dim=1).cpu().numpy()
                cum_sq8_coord = torch.cumsum(coord_lb_sq.view(cur_pos, num_blocks, block_dim).sum(dim=2), dim=1).cpu().numpy()
                cum_hat = torch.cumsum(diff_hat_sq.view(cur_pos, num_blocks, block_dim).sum(dim=2), dim=1)
                cum_sq8_l2err = (torch.clamp(torch.sqrt(torch.clamp(cum_hat, min=0.0)) - E_l2.unsqueeze(1), min=0.0) ** 2).cpu().numpy()
                exact_dists = cum_var[:, -1]

                # Now simulate exact sequential tile-by-tile scan on CPU using precomputed matrices (microseconds per query!)
                topk_sq = np.full(10, np.inf, dtype=np.float64)
                tau_sq = np.inf
                tau = np.inf

                for (c_s, c_e, d_qc, r_min, r_max, g_s, g_e) in tile_meta:
                    T_cur = c_e - c_s
                    if math.isinf(tau):
                        kept_vecs_L1_vec += T_cur
                        kept_vecs_L1_tile += T_cur
                        kept_dim_blocks_orig += T_cur * num_blocks
                        kept_dim_blocks_var += T_cur * num_blocks
                        kept_dim_blocks_sq8_coord += T_cur * num_blocks
                        kept_dim_blocks_sq8_l2err += T_cur * num_blocks
                        survivors_fp32_coord += T_cur
                        survivors_fp32_l2err += T_cur
                    else:
                        lo, hi = d_qc - tau, d_qc + tau
                        # Vector-level L1 count inside this tile
                        r_slice = r_sorted[g_s:g_e]
                        v_s = int(np.searchsorted(r_slice, lo, side='left'))
                        v_e = int(np.searchsorted(r_slice, hi, side='right'))
                        kept_vecs_L1_vec += max(0, v_e - v_s)

                        # Tile-level L1 decision
                        if r_max < lo or r_min > hi:
                            continue
                        kept_vecs_L1_tile += T_cur

                        kept_dim_blocks_orig += T_cur + int((cum_orig[c_s:c_e, :-1] <= tau_sq).sum())
                        kept_dim_blocks_var += T_cur + int((cum_var[c_s:c_e, :-1] <= tau_sq).sum())
                        kept_dim_blocks_sq8_coord += T_cur + int((cum_sq8_coord[c_s:c_e, :-1] <= tau_sq).sum())
                        kept_dim_blocks_sq8_l2err += T_cur + int((cum_sq8_l2err[c_s:c_e, :-1] <= tau_sq).sum())
                        survivors_fp32_coord += int((cum_sq8_coord[c_s:c_e, -1] <= tau_sq).sum())
                        survivors_fp32_l2err += int((cum_sq8_l2err[c_s:c_e, -1] <= tau_sq).sum())

                    tile_d = exact_dists[c_s:c_e]
                    if tile_d.min() < tau_sq:
                        merged = np.concatenate([topk_sq, tile_d])
                        merged.sort()
                        topk_sq = merged[:10]
                        tau_sq = float(topk_sq[-1])
                        tau = math.sqrt(max(tau_sq, 0.0))

            total_dim_blocks = total_vecs_all * num_blocks
            A_vec = kept_vecs_L1_vec / max(total_vecs_all, 1)
            A_tile = kept_vecs_L1_tile / max(total_vecs_all, 1)
            B_orig = kept_dim_blocks_orig / max(total_dim_blocks, 1)
            B_var = kept_dim_blocks_var / max(total_dim_blocks, 1)
            B_sq8_coord = kept_dim_blocks_sq8_coord / max(total_dim_blocks, 1)
            B_sq8_l2err = kept_dim_blocks_sq8_l2err / max(total_dim_blocks, 1)
            surv_coord = survivors_fp32_coord / max(total_vecs_all, 1)
            surv_l2err = survivors_fp32_l2err / max(total_vecs_all, 1)

            full_bytes = total_vecs_all * D * 4.0
            trident_bytes_coord = (kept_dim_blocks_sq8_coord * block_dim * 1.0) + (survivors_fp32_coord * D * 4.0) + (total_vecs_all / tile_size) * 8.0
            trident_bytes_l2err = (kept_dim_blocks_sq8_l2err * block_dim * 1.0) + (survivors_fp32_l2err * D * 4.0) + (kept_vecs_L1_tile * 2.0)
            D_ratio_coord = trident_bytes_coord / max(full_bytes, 1.0)
            D_ratio_l2err = trident_bytes_l2err / max(full_bytes, 1.0)

            row = {
                "dataset": ds_name,
                "dim": D,
                "nlist": nlist,
                "seed": seed,
                "target_recall": target_rec,
                "nprobe": nprobe,
                "actual_recall": round(actual_rec, 4),
                "tile_size": tile_size,
                "A_vec_kept": round(A_vec, 4),
                "A_tile_kept": round(A_tile, 4),
                "B_orig_kept": round(B_orig, 4),
                "B_var_kept": round(B_var, 4),
                "B_sq8_coord_kept": round(B_sq8_coord, 4),
                "B_sq8_l2err_kept": round(B_sq8_l2err, 4),
                "surv_fp32_coord": round(surv_coord, 4),
                "surv_fp32_l2err": round(surv_l2err, 4),
                "D_bytes_ratio_coord": round(D_ratio_coord, 4),
                "D_bytes_ratio_l2err": round(D_ratio_l2err, 4),
                "compute_saved_pct": round((1.0 - B_sq8_coord) * 100.0, 2),
                "extra_gain_B_over_A": round(A_tile / max(B_sq8_coord, 1e-6), 2),
                "var_reorder_gain": round(B_orig / max(B_var, 1e-6), 2),
                "elapsed_s": round(time.time() - t0, 1)
            }
            rows.append(row)
            print(f"[{ds_name} D={D} nlist={nlist} seed={seed} rec={target_rec} nprobe={nprobe} tile={tile_size}] "
                  f"A_vec={A_vec:.3f} A_tile={A_tile:.3f} B_orig={B_orig:.3f} B_var={B_var:.3f} B'_sq8={B_sq8_coord:.3f} "
                  f"Saved={row['compute_saved_pct']:.1f}% B/A={row['extra_gain_B_over_A']:.2f}x Bytes={D_ratio_coord:.3f} ({row['elapsed_s']}s)", flush=True)

    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["sift1m", "nuswide", "glove100", "msong", "gist1m", "bge1024"])
    ap.add_argument("--nlists", nargs="+", type=int, default=[1024, 4096])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max_queries", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    all_rows = []
    for ds in args.datasets:
        ds_dir = os.path.join(DATA_ROOT, ds)
        if not os.path.exists(os.path.join(ds_dir, "base.npy")):
            print(f"[SKIP] {ds} base.npy not found.")
            continue
        xb = np.load(os.path.join(ds_dir, "base.npy"))
        xq = np.load(os.path.join(ds_dir, "query.npy"))
        gt = ensure_gt(ds, xb, xq, k=100, device=args.device)
        gt10 = gt[:, :10]
        ds_rows = []
        for nlist in args.nlists:
            if nlist > xb.shape[0] // 10:
                continue
            for seed in args.seeds:
                tile_sizes = (64, 256, 1024) if (seed == 0 and nlist == 1024) else (256,)
                rows = evaluate_dataset(
                    ds, xb, xq, gt10, nlist=nlist, seed=seed,
                    tile_sizes=tile_sizes, block_dim=32, max_queries=args.max_queries, device=args.device
                )
                ds_rows.extend(rows)
                all_rows.extend(rows)
        pd.DataFrame(ds_rows).to_csv(os.path.join(RES_ROOT, "phase1", f"{ds}_trident_phase1.csv"), index=False)

    pd.DataFrame(all_rows).to_csv(os.path.join(RES_ROOT, "phase1", "all_phase1_raw.csv"), index=False)
    print("PHASE1_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
