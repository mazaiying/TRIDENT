#!/usr/bin/env python3
"""
PLAN_V9 Step B2: Sanity Check of SIFT10M and Deep10M on RTX 4080 16GB (CUDA_VISIBLE_DEVICES=0).
Verifies Recall@10, strict overlap vs CM-FP32 (100.00%), In-VRAM QPS, and OOC QPS on RTX 4080.
"""
import os, sys, time, csv, gc, statistics
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_HOME"] = "/usr/local/cuda-13"
os.environ["PATH"] = "/usr/local/cuda-13/bin:" + os.environ.get("PATH", "")

import numpy as np
import torch, faiss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase4a_suite import SharedClusterStorage, TridentV4Index, choose_tile, eval_accuracy_v4
from phase4b_suite import load_or_train_clustering_10m

DATA_ROOT = os.path.expanduser("~/trident/data")
RES_DIR = os.path.expanduser("~/trident/results/phase8")
os.makedirs(RES_DIR, exist_ok=True)
OUT_B2_CSV = os.path.join(RES_DIR, "b2_10m_sanity_RTX4080.csv")

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
    dev = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(dev)
    free_b, tot_b = torch.cuda.mem_get_info(dev)
    print(f"=== Step B2: 10M Sanity Check on {gpu_name} (Free: {free_b/1024**3:.2f} GiB / Total: {tot_b/1024**3:.2f} GiB) ===", flush=True)

    rows = []
    for ds, dim, nlist, np_map in [
        ("sift10m", 128, 4096, {0.95: 32, 0.99: 96}),
        ("deep10m", 96,  4096, {0.95: 32, 0.99: 96}),
    ]:
        ds_dir = os.path.join(DATA_ROOT, ds)
        xb = np.ascontiguousarray(np.load(os.path.join(ds_dir, "base.npy")).astype(np.float32))
        xq = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy")).astype(np.float32))
        gt = np.load(os.path.join(ds_dir, "gt.npy"))
        N = xb.shape[0]
        t_rule = choose_tile(dim)
        B_eval = min(1000, len(xq))
        gt10 = gt[:B_eval, :10]
        xq_b = np.vstack([xq[:1000], xq[:24]]) if len(xq) >= 1000 else np.vstack([xq, xq[:1024-len(xq)]])
        xq_b = np.ascontiguousarray(xq_b[:1024], dtype=np.float32)
        xq_b_gpu = torch.from_numpy(xq_b).to(dev, dtype=torch.float32).contiguous()

        centroids, list_ids, r_vec, mean64, R64, _ = load_or_train_clustering_10m(ds, xb, nlist, 0)
        storage = SharedClusterStorage(xb, list_ids, r_vec, nlist, device="cuda:0")
        idx_k2 = TridentV4Index(storage, centroids, mean64, R64, tile_size=t_rule, per_cluster_sq8=1, use_pca=1, var_block=0, max_batch=1024, device="cuda:0")

        # Save a pinned host copy of xb_sorted for OOC test
        xb_sorted_pinned = storage.xb_sorted.cpu().pin_memory()

        for rec_tag, nprobe in np_map.items():
            st_buf = torch.zeros(4, dtype=torch.int64, device=dev)
            cm_d, cm_i, _ = idx_k2.search(xq_b_gpu[:B_eval], nprobe, mode=10, use_l1=0, rerank_m=0, tile_override=t_rule)
            cm_dists, cm_ids = cm_d.cpu().numpy(), cm_i.cpu().numpy()
            tr_d, tr_i, _ = idx_k2.search(xq_b_gpu[:B_eval], nprobe, mode=2, use_l1=0, rerank_m=0, tile_override=t_rule, warp_refine=1, stats_buf=st_buf)
            torch.cuda.synchronize()
            tot_cands = int(st_buf[0].item())
            surv_cands = int(st_buf[2].item())
            surv_pct = 100.0 * surv_cands / max(tot_cands, 1)

            rec_cm, ov_cm, _ = eval_accuracy_v4(cm_ids, cm_dists, cm_ids, cm_dists, gt10)
            rec_tr, ov_tr, _ = eval_accuracy_v4(tr_i.cpu().numpy(), tr_d.cpu().numpy(), cm_ids, cm_dists, gt10)

            qps_cm_vram, std_cm_vram = measure_runs(lambda: idx_k2.search(xq_b_gpu, nprobe, mode=10, use_l1=0, rerank_m=0, tile_override=t_rule))
            qps_tr_vram, std_tr_vram = measure_runs(lambda: idx_k2.search(xq_b_gpu, nprobe, mode=2, use_l1=0, rerank_m=0, tile_override=t_rule, warp_refine=1))

            print(f"[B2 In-VRAM {ds} R={rec_tag} np={nprobe}] CM-FP32={qps_cm_vram:.1f} QPS (Rec={rec_cm:.4f}), TRIDENT={qps_tr_vram:.1f} QPS ({qps_tr_vram/qps_cm_vram:.2f}x, Rec={rec_tr:.4f}, Ov={ov_tr:.2f}%, Surv={surv_pct:.3f}%)", flush=True)
            rows.append([ds, N, dim, rec_tag, nprobe, "CM_FP32_InVRAM", round(qps_cm_vram, 1), round(std_cm_vram, 1), round(rec_cm, 4), round(ov_cm, 2), 100.0])
            rows.append([ds, N, dim, rec_tag, nprobe, "TRIDENT_InVRAM", round(qps_tr_vram, 1), round(std_tr_vram, 1), round(rec_tr, 4), round(ov_tr, 2), round(surv_pct, 3)])

        # Now test OOC on RTX 4080 (move xb_sorted to pinned host RAM)
        storage.xb_sorted = None
        torch.cuda.empty_cache()
        storage.xb_sorted = xb_sorted_pinned
        torch.cuda.synchronize()

        for rec_tag, nprobe in np_map.items():
            ooc_tr_d, ooc_tr_i, _ = idx_k2.search(xq_b_gpu[:B_eval], nprobe, mode=2, use_l1=0, rerank_m=0, tile_override=t_rule, warp_refine=1)
            ooc_cm_d, ooc_cm_i, _ = idx_k2.search(xq_b_gpu[:B_eval], nprobe, mode=10, use_l1=0, rerank_m=0, tile_override=t_rule)
            torch.cuda.synchronize()
            cm_dists, cm_ids = ooc_cm_d.cpu().numpy(), ooc_cm_i.cpu().numpy()
            rec_tr_ooc, ov_tr_ooc, _ = eval_accuracy_v4(ooc_tr_i.cpu().numpy(), ooc_tr_d.cpu().numpy(), cm_ids, cm_dists, gt10)
            rec_cm_ooc, ov_cm_ooc, _ = eval_accuracy_v4(cm_ids, cm_dists, cm_ids, cm_dists, gt10)

            qps_tr_ooc, std_tr_ooc = measure_runs(lambda: idx_k2.search(xq_b_gpu, nprobe, mode=2, use_l1=0, rerank_m=0, tile_override=t_rule, warp_refine=1), warmup=2, iters=6)
            qps_cm_ooc, std_cm_ooc = measure_runs(lambda: idx_k2.search(xq_b_gpu, nprobe, mode=10, use_l1=0, rerank_m=0, tile_override=t_rule), warmup=1, iters=3)

            print(f"[B2 OOC     {ds} R={rec_tag} np={nprobe}] CM-OOC={qps_cm_ooc:.1f} QPS, TRIDENT-OOC={qps_tr_ooc:.1f} QPS ({qps_tr_ooc/qps_cm_ooc:.2f}x, Rec={rec_tr_ooc:.4f}, Ov={ov_tr_ooc:.2f}%)", flush=True)
            rows.append([ds, N, dim, rec_tag, nprobe, "CM_FP32_OOC_Pinned", round(qps_cm_ooc, 1), round(std_cm_ooc, 1), round(rec_cm_ooc, 4), round(ov_cm_ooc, 2), 100.0])
            rows.append([ds, N, dim, rec_tag, nprobe, "TRIDENT_OOC_Pinned", round(qps_tr_ooc, 1), round(std_tr_ooc, 1), round(rec_tr_ooc, 4), round(ov_tr_ooc, 2), 0.0])

        del idx_k2, storage, xb_sorted_pinned, xb
        gc.collect()
        torch.cuda.empty_cache()

    with open(OUT_B2_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "N", "dim", "recall_target", "nprobe", "method", "qps_mean", "qps_std", "recall_at_10", "strict_overlap_pct", "surv_pct"])
        w.writerows(rows)
    print(f"Saved {OUT_B2_CSV} successfully!", flush=True)

if __name__ == "__main__":
    main()
