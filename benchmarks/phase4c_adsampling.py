#!/usr/bin/env python3
"""
Phase 4C-1: ADSampling (SIGMOD'24) Benchmark Runner
Evaluates ADSampling across sift1m (128d), glove100 (100d), and gist1m (960d).
Measures: Recall@10, QPS, Latency (us), and Pruned Dimension Ratio for Recall 0.95 and Recall 0.99 targets.
"""

import os
import sys
import subprocess
import csv
import time
import numpy as np

DATA_ROOT = os.path.expanduser("~/trident/data")
BASE_DIR = os.path.expanduser("~/trident/baselines/ADSampling")
INDEX_DIR = os.path.join(BASE_DIR, "index")
RES_DIR = os.path.join(BASE_DIR, "results")
OUT_CSV = os.path.expanduser("~/trident/results/phase4c/adsampling_results.csv")

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

NPROBE_TARGETS = {
    "sift1m": [("R0.95", 24), ("R0.99", 48)],
    "glove100": [("R0.95", 128), ("R0.99", 384)],
    "gist1m": [("R0.95", 56), ("R0.99", 128)],
}

def write_fvecs(filename, array):
    n, d = array.shape
    d_col = np.full((n, 1), d, dtype=np.int32)
    merged = np.hstack([d_col, array.astype(np.float32).view(np.int32)])
    merged.tofile(filename)
    print(f"Saved {filename} ({os.path.getsize(filename)} bytes)")

def prepare_centroids(ds_name):
    cent_fvecs = os.path.join(DATA_ROOT, ds_name, "fvecs", f"{ds_name}_centroids.fvecs")
    if os.path.exists(cent_fvecs):
        return cent_fvecs

    cache_npz = os.path.expanduser(f"~/trident/cache_v3/{ds_name}_nlist1024_s0.npz")
    if os.path.exists(cache_npz):
        z = np.load(cache_npz)
        centroids = z["centroids"].astype(np.float32)
        write_fvecs(cent_fvecs, centroids)
        return cent_fvecs
    else:
        raise FileNotFoundError(f"Centroids cache not found for {ds_name} at {cache_npz}")

def run_adsampling():
    datasets = ["sift1m", "glove100", "gist1m"]
    epsilons = [0.0, 1.0, 2.0, 4.0, 8.0]
    
    rows = []
    
    for ds in datasets:
        base_fvecs = os.path.join(DATA_ROOT, ds, "fvecs", f"{ds}_base.fvecs")
        query_fvecs = os.path.join(DATA_ROOT, ds, "fvecs", f"{ds}_query.fvecs")
        gt_ivecs = os.path.join(DATA_ROOT, ds, "fvecs", f"{ds}_groundtruth.ivecs")
        
        if not (os.path.exists(base_fvecs) and os.path.exists(query_fvecs) and os.path.exists(gt_ivecs)):
            print(f"Skipping {ds}, fvecs data missing.")
            continue
            
        cent_fvecs = prepare_centroids(ds)
        idx_path = os.path.join(INDEX_DIR, f"{ds}_ivf_adapt1.bin")
        
        # Build index if not present
        if not os.path.exists(idx_path):
            print(f"Building ADSampling index for {ds}...")
            t0 = time.time()
            cmd_build = [
                os.path.join(BASE_DIR, "src", "index_ivf"),
                "-d", base_fvecs,
                "-c", cent_fvecs,
                "-i", idx_path,
                "-a", "1"
            ]
            res = subprocess.run(cmd_build, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"Index build failed: {res.stderr}")
                continue
            print(f"Built {idx_path} in {time.time()-t0:.2f}s")
            
        for target_label, nprobe in NPROBE_TARGETS[ds]:
            for eps in epsilons:
                res_txt = os.path.join(RES_DIR, f"res_{ds}_{target_label}_eps{eps}.txt")
                if os.path.exists(res_txt):
                    os.remove(res_txt)
                    
                cmd_search = [
                    os.path.join(BASE_DIR, "src", "search_ivf"),
                    "-i", idx_path,
                    "-q", query_fvecs,
                    "-g", gt_ivecs,
                    "-r", res_txt,
                    "-k", "10",
                    "-b", str(nprobe),
                    "-e", str(eps)
                ]
                res = subprocess.run(cmd_search, capture_output=True, text=True)
                if os.path.exists(res_txt) and os.path.getsize(res_txt) > 0:
                    with open(res_txt, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 4:
                                try:
                                    np_val = int(parts[0])
                                    rec = float(parts[1]) / 100.0
                                    lat_us = float(parts[2])
                                    tot_dim = float(parts[3])
                                    qps = 1e6 / max(lat_us, 1e-3)
                                    
                                    row = {
                                        "dataset": ds,
                                        "target": target_label,
                                        "nprobe": np_val,
                                        "epsilon0": eps,
                                        "recall_at_10": round(rec, 4),
                                        "latency_us": round(lat_us, 2),
                                        "qps": round(qps, 1),
                                        "tot_dimension": int(tot_dim)
                                    }
                                    rows.append(row)
                                    print(f"[ADSampling {ds:<8} {target_label} eps={eps:<3}] Rec@10={rec:.4f} QPS={qps:>8.1f} Lat={lat_us:>7.1f}us TotDim={int(tot_dim):>12}", flush=True)
                                except ValueError:
                                    pass

    if rows:
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[SAVED] {OUT_CSV}", flush=True)

if __name__ == "__main__":
    run_adsampling()
