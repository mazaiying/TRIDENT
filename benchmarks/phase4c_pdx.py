#!/usr/bin/env python3
"""
Phase 4C-3: PDX (SIGMOD'24) Benchmark Runner
Evaluates PDX IVF ADSampling (pdx_f32) across sift and gist on NUMA node 0.
Tests:
- sift: nprobes [24, 48, 64, 96, 128, 160]
- gist: nprobes [56, 128, 192, 256, 384]
Measures: Latency (avg, min, max ms), QPS, Recall@20, and writes ~/trident/results/phase4c/pdx_results.csv.
"""

import os
import sys
import subprocess
import csv
import time

PDX_BIN = os.path.expanduser("~/trident/baselines/PDX/build/benchmarks/BenchmarkEndToEnd")
RAW_RESULTS_CSV = os.path.expanduser("~/trident/baselines/PDX/benchmarks/results/DEFAULT/END_TO_END_PDX_ADSAMPLING.csv")
OUT_CSV = os.path.expanduser("~/trident/results/phase4c/pdx_results.csv")

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

PROBES_BY_DATASET = {
    "sift": [24, 48, 64, 96, 128, 160],
    "gist": [56, 128, 192, 256, 384],
}

def run_pdx():
    # Remove existing raw results CSV to ensure fresh measurements
    if os.path.exists(RAW_RESULTS_CSV):
        os.remove(RAW_RESULTS_CSV)
        
    for ds, probes in PROBES_BY_DATASET.items():
        for np_val in probes:
            print(f"Running PDX {ds} nprobe={np_val} on NUMA 0...", flush=True)
            cmd = [
                "numactl", "--cpunodebind=0", "--membind=0",
                PDX_BIN, ds, "pdx_f32", str(np_val)
            ]
            t0 = time.time()
            res = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = time.time() - t0
            print(f"  Finished in {elapsed:.2f}s, code={res.returncode}", flush=True)
            
    # Process RAW_RESULTS_CSV into clean summary OUT_CSV
    if os.path.exists(RAW_RESULTS_CSV):
        rows = []
        with open(RAW_RESULTS_CSV, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                avg_ms = float(r["avg"])
                qps = 1000.0 / avg_ms if avg_ms > 0 else 0
                rows.append({
                    "dataset": r["dataset"],
                    "algorithm": r["algorithm"],
                    "nprobe": int(r["ivf_nprobe"]),
                    "recall": round(float(r["recall"]), 4),
                    "latency_avg_ms": round(avg_ms, 3),
                    "latency_min_ms": round(float(r["min"]), 3),
                    "latency_max_ms": round(float(r["max"]), 3),
                    "qps": round(qps, 1),
                    "knn": int(r["knn"]),
                    "num_queries": int(r["n_queries"])
                })
                
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            
        print(f"\n[DONE] Saved PDX results to {OUT_CSV}", flush=True)
        for r in rows:
            print(f"  [{r['dataset']:<4} np={r['nprobe']:>3}] Rec={r['recall']:.4f} Lat={r['latency_avg_ms']:>6.2f}ms QPS={r['qps']:>7.1f}", flush=True)

if __name__ == "__main__":
    run_pdx()
