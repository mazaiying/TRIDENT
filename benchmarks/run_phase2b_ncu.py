#!/usr/bin/env python3
"""
Runs NVIDIA Nsight Compute (ncu) hardware HBM counter profiling (dram__bytes_read.sum)
for Phase 2B (batch=1024, seed=0) across Faiss_IVFFlat, Faiss_IVFSQ8, K0, K1, K2 (T=256 and T=64).
Saves per-kernel raw logs and aggregated CSV in ~/trident/results/phase2b/ncu_hbm_bytes.csv.
"""
import os, sys, csv, io, subprocess
import pandas as pd

NCU_BIN = "/usr/local/cuda-13/bin/ncu"
PY_BIN = "/home/mazaiying/trident/.venv/bin/python"
BENCH_SCRIPT = os.path.expanduser("~/trident/scripts/phase2b_benchmark.py")
RES_DIR = os.path.expanduser("~/trident/results/phase2b")
LOG_DIR = os.path.expanduser("~/trident/logs/ncu_phase2b")
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DATASETS = ["gist1m", "bge1024", "sift1m", "msong", "glove100", "nuswide"]
TARGETS = ["faiss_flat", "faiss_sq8", "K0_T256", "K1_T256", "K2_T256", "K0_T64", "K1_T64", "K2_T64"]


def parse_ncu_csv(raw_text: str):
    lines = [ln for ln in raw_text.splitlines() if ln.startswith('"')]
    if len(lines) < 3:
        raise RuntimeError(f"Unexpected ncu output (fewer than 3 CSV lines):\n{raw_text[-1000:]}")
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    total_bytes = 0
    scan_bytes = 0
    n_kernels = 0
    for row in reader:
        rid = row.get("ID", "").strip()
        if not rid.isdigit():
            continue  # skip unit header row
        val_str = row.get("dram__bytes_read.sum", "0").replace(",", "").strip()
        b_val = int(float(val_str))
        kname = row.get("Kernel Name", "")
        total_bytes += b_val
        n_kernels += 1
        if "trident_cluster_major_scan_kernel" in kname or "trident_merge_topk_kernel" in kname:
            scan_bytes += b_val
    return total_bytes, scan_bytes, n_kernels


def main():
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
    rows = []
    for ds in DATASETS:
        sq8_k0_t256_total = None
        sq8_k0_t64_total = None
        flat_total = None
        ds_rows = []
        for tgt in TARGETS:
            cmd = [
                NCU_BIN,
                "--profile-from-start", "off",
                "--csv",
                "--page", "raw",
                "--metrics", "dram__bytes_read.sum",
                PY_BIN, BENCH_SCRIPT,
                "--datasets", ds,
                "--seeds", "0",
                "--batches", "1024",
                "--gpu_tag", "A100_ncu",
                "--ncu_target", tgt,
                "--cache_dir", os.path.join(RES_DIR, ".ivf_cache"),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            print(f"[NCU] Profiling {ds} {tgt} on GPU {gpu_id} ...", flush=True)
            proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
            raw_log_path = os.path.join(LOG_DIR, f"ncu_{ds}_{tgt}.csv")
            with open(raw_log_path, "w") as f:
                f.write(proc.stdout)

            tot_b, scan_b, nk = parse_ncu_csv(proc.stdout)
            if tgt == "faiss_flat":
                flat_total = tot_b
            elif tgt == "K0_T256":
                sq8_k0_t256_total = tot_b
            elif tgt == "K0_T64":
                sq8_k0_t64_total = tot_b

            ds_rows.append({
                "dataset": ds,
                "batch": 1024,
                "seed": 0,
                "target": tgt,
                "num_kernels": nk,
                "total_hbm_read_bytes": tot_b,
                "total_hbm_read_mb": round(tot_b / (1024.0 * 1024.0), 3),
                "scan_only_hbm_read_bytes": scan_b if scan_b > 0 else "N/A",
            })
            print(f"  -> {ds} {tgt}: total={tot_b/(1024*1024):.2f} MiB (kernels={nk})", flush=True)

        for r in ds_rows:
            tot_b = r["total_hbm_read_bytes"]
            r["ratio_vs_faiss_flat"] = round(tot_b / flat_total, 4) if flat_total else "N/A"
            if r["target"].endswith("T64") and sq8_k0_t64_total:
                r["ratio_vs_K0_same_tile"] = round(tot_b / sq8_k0_t64_total, 4)
            elif r["target"].endswith("T256") and sq8_k0_t256_total:
                r["ratio_vs_K0_same_tile"] = round(tot_b / sq8_k0_t256_total, 4)
            else:
                r["ratio_vs_K0_same_tile"] = "N/A"
            rows.append(r)

    out_csv = os.path.join(RES_DIR, "ncu_hbm_bytes.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"SAVED {out_csv}", flush=True)


if __name__ == "__main__":
    main()
