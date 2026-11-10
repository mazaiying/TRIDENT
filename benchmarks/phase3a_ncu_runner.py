#!/usr/bin/env python3
"""
Phase 3A Nsight Compute (ncu) Hardware HBM Bytes Profiler (PLAN_V4 §1.2).
Profiles exact hardware dram__bytes_read.sum on A100 across all 6 datasets for:
  - Faiss_IVFFlat, Faiss_IVFSQ8, Faiss_IVFPQ
  - Full Ablation Chain:
    A0_CM_FP32 -> A1_CM_SQ8_RR100 -> A2_Orig_Uni32 -> A3_PCA_Uni32 ->
    A4_PCA_VarBlk -> A5_PCA_VarBlk_L1 -> A6_PerClusterSQ8_T256 / A6_PerClusterSQ8_T64
  - Additional Baselines: TRIDENT_K0_T256, CM_SQ8_RR50
"""
import os, sys, csv, io, subprocess
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RES_ROOT = os.path.expanduser("~/trident/results/phase3a")
LOG_ROOT = os.path.expanduser("~/trident/logs/ncu_phase3a")
os.makedirs(RES_ROOT, exist_ok=True)
os.makedirs(LOG_ROOT, exist_ok=True)

NCU_BIN = "/usr/local/cuda-13/bin/ncu"
PY_BIN = "/home/mazaiying/trident/.venv/bin/python"
BENCH_SCRIPT = os.path.join(SCRIPT_DIR, "phase3a_benchmark.py")

DATASETS = ["gist1m", "bge1024", "sift1m", "msong", "glove100", "nuswide"]
TARGETS = [
    "Faiss_IVFFlat",
    "Faiss_IVFSQ8",
    "Faiss_IVFPQ",
    "TRIDENT_K0_T256",
    "CM_SQ8_RR50",
    "A0_CM_FP32",
    "A1_CM_SQ8_RR100",
    "A2_Orig_Uni32",
    "A3_PCA_Uni32",
    "A4_PCA_VarBlk",
    "A5_PCA_VarBlk_L1",
    "A6_PerClusterSQ8_T256",
    "A6_PerClusterSQ8_T64",
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
    out_csv = os.path.join(RES_ROOT, "ncu_phase3a.csv")
    rows = []
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["CUDA_HOME"] = "/usr/local/cuda-13"
    env["PATH"] = "/usr/local/cuda-13/bin:" + env.get("PATH", "")

    for ds in DATASETS:
        b_val = 200 if ds == "nuswide" else 1024
        ref_flat_bytes = None
        ref_a0_bytes = None
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
                "--run_mode", "ncu",
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
            if t == "A0_CM_FP32":
                ref_a0_bytes = tot_b

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
                "ratio_vs_A0_CM_FP32": "PENDING",
            }
            ds_rows.append(r)
            print(f"[NCU {ds:<8} B={b_val:<4} {t:<22}] kernels={nk:<2} total={r['total_hbm_read_mb']:>9.3f} MB scan={str(r['scan_only_hbm_read_mb']):>8} MB", flush=True)

        for r in ds_rows:
            r["ratio_vs_A0_CM_FP32"] = round(r["total_hbm_read_bytes"] / ref_a0_bytes, 4) if ref_a0_bytes else "N/A"
            rows.append(r)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"[SAVED] {out_csv}", flush=True)


if __name__ == "__main__":
    main()
