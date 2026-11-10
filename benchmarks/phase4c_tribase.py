#!/usr/bin/env python3
"""
Phase 4C-2: Tribase (导师工作, VLDB'24) Benchmark Runner
Evaluates Tribase across sift1m, glove100, and gist1m on NUMA node 0.
Tests:
- Full cores (OMP_NUM_THREADS=24 on NUMA node 0)
- Single core (OMP_NUM_THREADS=1)
- nprobes corresponding to Recall 0.95 and Recall 0.99
- opt_levels: OPT_NONE (baseline IVF-Flat), OPT_TRIANGLE, OPT_ALL
Reports: QPS, Recall, Pruning ratio, Latency, and Strict consistency with IVF-Flat.
"""

import os
import sys
import subprocess
import csv
import time

TRIBASE_DIR = os.path.expanduser("~/trident/baselines/Tribase")
BENCHMARKS_DIR = os.path.join(TRIBASE_DIR, "benchmarks")
LOGS_DIR = os.path.join(TRIBASE_DIR, "logs")
OUT_CSV = os.path.expanduser("~/trident/results/phase4c/tribase_results.csv")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

CONFIGS = [
    {"dataset": "sift1m", "nprobes": [24, 48]},
    {"dataset": "glove100", "nprobes": [128, 384]},
    {"dataset": "gist1m", "nprobes": [56, 128]},
]

THREAD_CONFIGS = [24, 1]  # Full NUMA 0 cores vs Single core

def run_tribase_benchmark():
    all_rows = []
    
    for cfg in CONFIGS:
        ds = cfg["dataset"]
        nprobes = cfg["nprobes"]
        nprobes_str = " ".join(str(p) for p in nprobes)
        
        for nthreads in THREAD_CONFIGS:
            print(f"\n==========================================", flush=True)
            print(f"Running Tribase on {ds} (Threads: {nthreads}, nprobes: {nprobes_str})", flush=True)
            print(f"==========================================", flush=True)
            
            run_csv = os.path.join(LOGS_DIR, f"{ds}_t{nthreads}_run.csv")
            if os.path.exists(run_csv):
                os.remove(run_csv)
                
            cmd = [
                "numactl", "--cpunodebind=0", "--membind=0",
                "./release/bin/query",
                "--benchmarks_path", "./benchmarks",
                "--dataset", ds,
                "--nprobes", *[str(p) for p in nprobes],
                "--sub_nprobe_ratio", "1",
                "--opt_levels", "OPT_NONE", "OPT_TRIANGLE", "OPT_ALL",
                "--loop", "1",
                "--early_stop",
                "--csv", run_csv,
                "--verbose"
            ]
            
            # Check if index exists; if so, pass --cache
            idx_pattern = os.path.join(BENCHMARKS_DIR, ds, "index")
            if os.path.exists(idx_pattern) and len(os.listdir(idx_pattern)) > 0:
                cmd.append("--cache")
                
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(nthreads)
            
            t0 = time.time()
            res = subprocess.run(cmd, cwd=TRIBASE_DIR, env=env, capture_output=True, text=True)
            elapsed = time.time() - t0
            
            print(f"Command finished in {elapsed:.1f}s, returncode={res.returncode}", flush=True)
            if res.returncode != 0:
                print(f"[ERROR STDERR]:\n{res.stderr[-500:]}", flush=True)
                continue
                
            if os.path.exists(run_csv):
                with open(run_csv, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row["num_threads"] = nthreads
                        all_rows.append(row)
                        opt = row.get("opt_level", "")
                        qps = row.get("qps", "N/A")
                        rec = row.get("recall", "N/A")
                        np_val = row.get("nprobe", "N/A")
                        print(f"  [Tribase {ds} T={nthreads} np={np_val} opt={opt}] QPS={float(qps):.1f} Rec={float(rec):.4f}", flush=True)

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n[DONE] Saved Tribase benchmark results to {OUT_CSV}", flush=True)

if __name__ == "__main__":
    run_tribase_benchmark()
