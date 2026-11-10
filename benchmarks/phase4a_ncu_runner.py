#!/usr/bin/env python3
"""
Phase 4A Nsight Compute (ncu) Hardware HBM Bytes Profiler (PLAN_V5 §1.2).
Profiles exact hardware dram__bytes_read.sum on A100 across all 6 datasets for:
  - Faiss_IVFFlat
  - New Ablation Chain:
    B0_CM_FP32 -> B1_GlobalSQ8_Uni32_Serial -> B2_PerClusterSQ8_Uni32_Serial ->
    B3_PCA_PerClusterSQ8_Serial -> B4_Full_TRIDENT
  - Leave-One-Out Ablations:
    Full_minus_PCA, Full_minus_PerClusterSQ8, Full_minus_EarlyStop,
    Full_minus_WarpRefine, Full_plus_VarBlk, Full_plus_L1
"""
import os, sys, csv, io, subprocess
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RES_ROOT = os.path.expanduser("~/trident/results/phase4a")
LOG_ROOT = os.path.expanduser("~/trident/logs/ncu_phase4a")
os.makedirs(RES_ROOT, exist_ok=True)
os.makedirs(LOG_ROOT, exist_ok=True)

NCU_BIN = "/usr/local/cuda-13/bin/ncu"
PY_BIN = "/home/mazaiying/trident/.venv/bin/python"
BENCH_SCRIPT = os.path.join(SCRIPT_DIR, "phase4a_suite.py")

DATASETS = ["gist1m", "bge1024", "sift1m", "msong", "glove100", "nuswide"]
TARGETS = [
    "Faiss_IVFFlat",
    "B0_CM_FP32",
    "B1_GlobalSQ8_Uni32_Serial",
    "B2_PerClusterSQ8_Uni32_Serial",
    "B3_PCA_PerClusterSQ8_Serial",
    "B4_Full_TRIDENT",
    "Full_minus_PCA",
    "Full_minus_PerClusterSQ8",
    "Full_minus_EarlyStop",
    "Full_minus_WarpRefine",
    "Full_plus_VarBlk",
    "Full_plus_L1",
]


def parse_ncu_csv(raw_text: str):
    lines = raw_text.splitlines()
    csv_lines = []
    started = False
    for line in lines:
        if line.startswith('"ID","Process ID"'):
            started = True
        if started and line.strip().startswith('"'):
            csv_lines.append(line)
    if not csv_lines:
        return 0, 0, 0
    reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
    num_kernels = 0
    total_bytes = 0
    scan_bytes = 0
    for row in reader:
        kname = row.get("Kernel Name", "")
        val_str = row.get("dram__bytes_read.sum", "").replace(",", "").strip()
        if not val_str or val_str == "byte":
            continue
        try:
            v = int(float(val_str))
        except ValueError:
            continue
        num_kernels += 1
        total_bytes += v
        if "trident_cluster_major_scan_kernel" in kname or "trident_cm_fp32_scan_kernel" in kname or "trident_merge_and_rerank_topm_kernel" in kname:
            scan_bytes += v
    return num_kernels, total_bytes, scan_bytes


def main():
    out_csv = os.path.join(RES_ROOT, "ncu_phase4a.csv")
    rows = []
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["CUDA_HOME"] = "/usr/local/cuda-13"
    env["PATH"] = "/usr/local/cuda-13/bin:" + env.get("PATH", "")

    for ds in DATASETS:
        b_val = 200 if ds == "nuswide" else 1024
        ref_flat_bytes = None
        ref_b0_bytes = None
        ds_rows = []
        for t in TARGETS:
            cmd = [
                NCU_BIN,
                "--csv",
                "--page", "raw",
                "--metrics", "dram__bytes_read.sum",
                "--profile-from-start", "off",
                PY_BIN,
                BENCH_SCRIPT,
                "--mode", "ncu_single",
                "--datasets", ds,
                "--seeds", "0",
                "--ncu_target", t,
            ]
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            raw_out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            with open(os.path.join(LOG_ROOT, f"ncu_{ds}_{t}.log"), "w") as f:
                f.write(raw_out)

            nk, tot_b, scan_b = parse_ncu_csv(raw_out)
            if t == "Faiss_IVFFlat":
                ref_flat_bytes = tot_b
            if t == "B0_CM_FP32":
                ref_b0_bytes = tot_b

            r = {
                "dataset": ds,
                "batch": b_val,
                "seed": 0,
                "target": t,
                "num_kernels": nk,
                "total_hbm_read_bytes": tot_b,
                "total_hbm_read_mb": round(tot_b / (1024.0 ** 2), 3),
                "scan_only_hbm_read_bytes": scan_b if not t.startswith("Faiss_") else "N/A",
                "scan_only_hbm_read_mb": round(scan_b / (1024.0 ** 2), 3) if not t.startswith("Faiss_") else "N/A",
                "ratio_vs_faiss_flat": round(tot_b / ref_flat_bytes, 4) if ref_flat_bytes else "N/A",
                "ratio_vs_B0_CM_FP32": "PENDING",
            }
            ds_rows.append(r)
            print(f"[NCU {ds:<8} B={b_val:<4} {t:<28}] kernels={nk:<2} total={r['total_hbm_read_mb']:>9.3f} MB scan={str(r['scan_only_hbm_read_mb']):>8} MB", flush=True)

        for r in ds_rows:
            r["ratio_vs_B0_CM_FP32"] = round(r["total_hbm_read_bytes"] / ref_b0_bytes, 4) if ref_b0_bytes else "N/A"
            rows.append(r)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"[SAVED] {out_csv}", flush=True)


if __name__ == "__main__":
    main()
