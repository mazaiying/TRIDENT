#!/usr/bin/env python3
"""
Compute 100% CSV-backed Master Data Statistics for TRIDENT (ICDE 2027).
Reads directly from /Volumes/Elements SE/科研/icde0117/trident_icde_paper/data/
and outputs verified_claims.json (including Phase 7 cagra_uvm_capped, gpu_build_v2,
and gpu_build_quality).
"""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_JSON = DATA_DIR / "verified_claims.json"


def read_csv(rel_path: str):
    p = DATA_DIR / rel_path
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def main():
    claims = {}

    datasets_1m = ["sift1m", "glove100", "msong", "nuswide", "gist1m", "bge1024"]
    datasets_10m = ["sift10m", "deep10m", "t2i10m"]
    all_datasets = datasets_1m + datasets_10m
    active_8 = [d for d in all_datasets if d != "nuswide"]

    # 1. Load 1M & 10M main QPS/recall and Tribase-GPU comparison
    rows_1m_95 = read_csv("phase3a/phase3a_main_A100.csv")
    rows_1m_99 = read_csv("phase4a/phase4a_recall99_A100.csv")
    rows_prune_1m = read_csv("phase4a/phase4a_prune_stats.csv")
    rows_fixed_1m = read_csv("phase4a/phase4a_main_fixedrule.csv")
    rows_sift10m = read_csv("phase6/sift10m_rerun.csv")
    rows_deep10m = read_csv("phase4b/phase4b_main_A100.csv")
    rows_t2i10m = read_csv("phase6/phase6c_main_A100.csv")
    rows_tri = read_csv("phase6/tribase_gpu.csv")

    tri_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows_tri:
        rt = "R0.95" if abs(float(r["recall_target"]) - 0.95) < 1e-3 else "R0.99"
        tri_map[rt][r["dataset"]][r["method"]].append(r)

    main_table = {"R0.95": {}, "R0.99": {}}

    # 1M R0.95 & R0.99
    for ds in datasets_1m:
        for r_key, r_tag in [("R0.95", "Recall_0.95"), ("R0.99", "Recall_0.99")]:
            pr_rows = [r for r in rows_prune_1m if r["dataset"] == ds and r["target_recall"] == r_tag]
            dims_pruned = float(pr_rows[0]["prune_dim_pct"]) if pr_rows else 0.0
            surv_pct = float(pr_rows[0]["survivor_ratio_pct"]) if pr_rows else 100.0
            cand_pruned = round(100.0 - surv_pct, 2)

            if r_key == "R0.95":
                nprobe = int(pr_rows[0]["nprobe"])
                b_target = 200 if ds == "nuswide" else 1024
                sub_fix = [
                    r for r in rows_fixed_1m
                    if r["gpu"] == "A100" and r["dataset"] == ds and int(r["B"]) == b_target
                ]
                sub_qps = [
                    r for r in rows_1m_95
                    if r["dataset"] == ds and int(r["nprobe"]) == nprobe and int(r["batch"]) == b_target
                ]
                by_m = defaultdict(list)
                for r in sub_qps:
                    by_m[r["method"]].append(float(r["qps"]))
                trident_raw = float(sub_fix[0]["qps_mean"])
            else:
                sub_99 = [r for r in rows_1m_99 if r["dataset"] == ds]
                nprobe = int(sub_99[0]["nprobe_99"])
                by_m = defaultdict(list)
                for r in sub_99:
                    by_m[r["method"]].append(float(r["qps"]))
                trident_raw = statistics.mean(by_m["TRIDENT"])

            faiss_qps = statistics.mean(by_m["Faiss_IVFFlat"])
            cm_qps = max(
                statistics.mean(by_m["CM_FP32_T64"]) if by_m["CM_FP32_T64"] else 0.0,
                statistics.mean(by_m["CM_FP32_T256"]) if by_m["CM_FP32_T256"] else 0.0,
            )
            sq8_qps = statistics.mean(by_m["Faiss_IVFSQ8"])
            rr50_qps = statistics.mean(by_m["CM_SQ8_RR50"])
            trident_auto = max(faiss_qps, cm_qps) if ds == "nuswide" else trident_raw

            tb_rows = tri_map[r_key][ds].get("Tribase_GPU", [])
            tr_same_rows = tri_map[r_key][ds].get("TRIDENT", [])
            tb_qps = statistics.mean(float(x["qps"]) for x in tb_rows) if tb_rows else 0.0
            tb_dim_pr = statistics.mean(float(x["dim_pruned_pct"]) for x in tb_rows) if tb_rows else 0.0
            tr_same_qps = statistics.mean(float(x["qps"]) for x in tr_same_rows) if tr_same_rows else trident_raw
            sp_tb = round(tr_same_qps / tb_qps, 2) if tb_qps > 0 else 0.0

            main_table[r_key][ds] = {
                "nprobe": nprobe,
                "Faiss_IVFFlat": round(faiss_qps, 1),
                "CM_FP32": round(cm_qps, 1),
                "Faiss_IVFSQ8": round(sq8_qps, 1),
                "CM_SQ8_RR50": round(rr50_qps, 1),
                "Tribase_GPU": round(tb_qps, 1),
                "TRIDENT_Raw": round(trident_raw, 1),
                "TRIDENT_Auto": round(trident_auto, 1),
                "speedup_auto_vs_faiss": round(trident_auto / faiss_qps, 2),
                "speedup_auto_vs_cm": round(trident_auto / cm_qps, 2),
                "speedup_raw_vs_faiss": round(trident_raw / faiss_qps, 2),
                "speedup_raw_vs_cm": round(trident_raw / cm_qps, 2),
                "speedup_vs_tribase_same_run": sp_tb,
                "tribase_dim_pruned_pct": round(tb_dim_pr, 2),
                "trident_dim_pruned_pct": round(dims_pruned, 2),
                "trident_cand_pruned_pct": cand_pruned,
            }

    # 10M datasets
    for ds, src_rows in [
        ("sift10m", rows_sift10m),
        ("deep10m", [r for r in rows_deep10m if r["dataset"] == "deep10m"]),
        ("t2i10m", [r for r in rows_t2i10m if r["dataset"] == "t2i10m"]),
    ]:
        for r_key, r_val in [("R0.95", 0.95), ("R0.99", 0.99)]:
            sub = [
                r for r in src_rows
                if abs(float(r["recall_target"]) - r_val) < 1e-3 and int(r["batch"]) == 1024
            ]
            nprobe = int(sub[0]["nprobe"])
            by_m = defaultdict(list)
            for r in sub:
                by_m[r["method"]].append(r)

            faiss_qps = statistics.mean(float(x["qps"]) for x in by_m["Faiss_IVFFlat"])
            cm_qps = max(
                statistics.mean(float(x["qps"]) for x in by_m["CM_FP32_T64"]) if by_m["CM_FP32_T64"] else 0.0,
                statistics.mean(float(x["qps"]) for x in by_m["CM_FP32_T256"]) if by_m["CM_FP32_T256"] else 0.0,
                statistics.mean(float(x["qps"]) for x in by_m["CM_FP32"]) if by_m["CM_FP32"] else 0.0,
            )
            sq8_qps = statistics.mean(float(x["qps"]) for x in by_m["Faiss_IVFSQ8"])
            rr50_qps = statistics.mean(float(x["qps"]) for x in by_m["CM_SQ8_RR50"])
            tr_qps = statistics.mean(float(x["qps"]) for x in by_m["TRIDENT"])

            tb_rows = tri_map[r_key][ds].get("Tribase_GPU", [])
            tr_same_rows = tri_map[r_key][ds].get("TRIDENT", [])
            tb_qps = statistics.mean(float(x["qps"]) for x in tb_rows) if tb_rows else 0.0
            tb_dim_pr = statistics.mean(float(x["dim_pruned_pct"]) for x in tb_rows) if tb_rows else 0.0
            tr_same_qps = statistics.mean(float(x["qps"]) for x in tr_same_rows) if tr_same_rows else tr_qps
            dims_pruned = statistics.mean(float(x["dim_pruned_pct"]) for x in tr_same_rows) if tr_same_rows else 0.0
            cand_pruned = statistics.mean(float(x["cand_pruned_pct"]) for x in tr_same_rows) if tr_same_rows else 0.0
            sp_tb = round(tr_same_qps / tb_qps, 2) if tb_qps > 0 else 0.0

            main_table[r_key][ds] = {
                "nprobe": nprobe,
                "Faiss_IVFFlat": round(faiss_qps, 1),
                "CM_FP32": round(cm_qps, 1),
                "Faiss_IVFSQ8": round(sq8_qps, 1),
                "CM_SQ8_RR50": round(rr50_qps, 1),
                "Tribase_GPU": round(tb_qps, 1),
                "TRIDENT_Raw": round(tr_qps, 1),
                "TRIDENT_Auto": round(tr_qps, 1),
                "speedup_auto_vs_faiss": round(tr_qps / faiss_qps, 2),
                "speedup_auto_vs_cm": round(tr_qps / cm_qps, 2),
                "speedup_raw_vs_faiss": round(tr_qps / faiss_qps, 2),
                "speedup_raw_vs_cm": round(tr_qps / cm_qps, 2),
                "speedup_vs_tribase_same_run": sp_tb,
                "tribase_dim_pruned_pct": round(tb_dim_pr, 2),
                "trident_dim_pruned_pct": round(dims_pruned, 2),
                "trident_cand_pruned_pct": round(cand_pruned, 2),
            }

    claims["main_table"] = main_table

    # Aggregate summaries agg_R0.95 & agg_R0.99
    for r_key in ["R0.95", "R0.99"]:
        sp_fa_all9 = [main_table[r_key][d]["speedup_auto_vs_faiss"] for d in all_datasets]
        sp_cm_all9 = [main_table[r_key][d]["speedup_auto_vs_cm"] for d in all_datasets]
        sp_fa_act8 = [main_table[r_key][d]["speedup_auto_vs_faiss"] for d in active_8]
        sp_cm_act8 = [main_table[r_key][d]["speedup_auto_vs_cm"] for d in active_8]
        sp_tb_act7 = [
            main_table[r_key][d]["speedup_vs_tribase_same_run"]
            for d in active_8 if main_table[r_key][d]["speedup_vs_tribase_same_run"] > 0
        ]
        claims[f"agg_{r_key}"] = {
            "all9_mean_vs_faiss": round(statistics.mean(sp_fa_all9), 2),
            "all9_geomean_vs_faiss": round(geomean(sp_fa_all9), 2),
            "all9_mean_vs_cm": round(statistics.mean(sp_cm_all9), 2),
            "all9_geomean_vs_cm": round(geomean(sp_cm_all9), 2),
            "active8_mean_vs_faiss": round(statistics.mean(sp_fa_act8), 2),
            "active8_geomean_vs_faiss": round(geomean(sp_fa_act8), 2),
            "active8_max_vs_faiss": round(max(sp_fa_act8), 2),
            "active8_min_vs_faiss": round(min(sp_fa_act8), 2),
            "active8_mean_vs_cm": round(statistics.mean(sp_cm_act8), 2),
            "active8_geomean_vs_cm": round(geomean(sp_cm_act8), 2),
            "active8_max_vs_cm": round(max(sp_cm_act8), 2),
            "active8_min_vs_cm": round(min(sp_cm_act8), 2),
            "active7_mean_vs_tribase": round(statistics.mean(sp_tb_act7), 2),
            "active7_geomean_vs_tribase": round(geomean(sp_tb_act7), 2),
            "active7_max_vs_tribase": round(max(sp_tb_act7), 2),
            "active7_min_vs_tribase": round(min(sp_tb_act7), 2),
        }

    # NCU v2 (phase6/ncu_v2.csv)
    ncu_v2 = defaultdict(dict)
    for r in read_csv("phase6/ncu_v2.csv"):
        ncu_v2[r["dataset"]][r["target"]] = {
            "gpu_time_us": round(float(r["gpu_time_us"]), 1),
            "dram_read_mb": round(float(r["dram_read_mb"]), 2),
            "l2_bytes_mb": round(float(r["l2_bytes_mb"]), 2),
            "inst_executed_M": round(float(r["inst_executed_M"]), 2),
            "ffma_inst_M": round(float(r["ffma_inst_M"]), 2),
        }
    claims["ncu_v2"] = dict(ncu_v2)

    # Top-K sensitivity (phase6/sens_k.csv)
    sens_k = defaultdict(lambda: defaultdict(dict))
    by_sk = defaultdict(list)
    for r in read_csv("phase6/sens_k.csv"):
        by_sk[(r["dataset"], r["k"], r["method"])].append(r)
    for (ds, k_str, m), rows in by_sk.items():
        sens_k[ds][str(k_str)][m] = {
            "qps": round(statistics.mean(float(x["qps"]) for x in rows), 1),
            "recall_at_k": round(statistics.mean(float(x["recall_at_k"]) for x in rows), 4),
            "strict_ov_pct": round(statistics.mean(float(x["strict_ov_pct"]) for x in rows), 2),
        }
    claims["sens_k"] = {k: dict(v) for k, v in sens_k.items()}

    # Out-of-Core (phase6/ooc.csv)
    ooc = defaultdict(lambda: defaultdict(dict))
    for r in read_csv("phase6/ooc.csv"):
        rt = f"R{float(r['recall_target']):.2f}"
        ooc[r["dataset"]][rt][r["method"]] = {
            "qps": round(float(r["qps"]), 1),
            "speedup_vs_cm_ooc": round(float(r["speedup_vs_cm_ooc"]), 2),
            "pcie_mb_estimated": round(float(r["pcie_mb_estimated"]), 2),
            "pcie_mb_measured": round(float(r["pcie_mb_measured"]), 2),
            "strict_ov_pct": round(float(r["strict_ov_pct"]), 2),
            "vram_resident_gb": round(float(r["vram_resident_gb"]), 3),
            "host_pinned_gb": round(float(r["host_pinned_gb"]), 3),
            "survivor_pct": round(float(r["survivor_pct"]), 3),
        }
    claims["ooc"] = {k: dict(v) for k, v in ooc.items()}

    # RaBitQ CPU (phase6/rabitq_cpu.csv)
    rabitq_cpu = defaultdict(lambda: defaultdict(dict))
    for r in read_csv("phase6/rabitq_cpu.csv"):
        rt_val = float(r.get("recall_target") or r.get("target_recall") or 0.99)
        rt = f"R{rt_val:.2f}"
        rabitq_cpu[r["dataset"]][rt][r["method"]] = {
            "cpu_qps": round(float(r["cpu_qps"]), 1),
            "recall_at_10": round(float(r["recall_at_10"]), 4),
            "strict_ov_pct": round(float(r["strict_ov_pct"]), 2),
        }
    claims["rabitq_cpu"] = {k: dict(v) for k, v in rabitq_cpu.items()}

    # cuVS summary (phase6/cuvs_qps_recall.csv)
    cuvs_rows = read_csv("phase6/cuvs_qps_recall.csv")
    by_cuvs = defaultdict(list)
    for r in cuvs_rows:
        b = int(r["batch"])
        ds = r["dataset"]
        if (ds == "nuswide" and b == 200) or (ds != "nuswide" and b == 1024):
            by_cuvs[(ds, r["method"], r["param_tag"])].append(r)
    cuvs_summary = {"R0.95": defaultdict(dict), "R0.99": defaultdict(dict)}
    for t_val, t_key in [(0.95, "R0.95"), (0.99, "R0.99")]:
        best_per_dm = {}
        for (ds, m, ptag), rows in by_cuvs.items():
            rec = statistics.mean(float(x["recall_at_10"]) for x in rows)
            qps = statistics.mean(float(x["qps"]) for x in rows)
            ov = statistics.mean(float(x["strict_vs_ivfflat_pct"]) for x in rows)
            bs = statistics.mean(float(x["build_s"]) for x in rows)
            vr = statistics.mean(float(x["vram_gb"]) for x in rows)
            if rec >= t_val:
                if (ds, m) not in best_per_dm or qps > best_per_dm[(ds, m)]["qps"]:
                    best_per_dm[(ds, m)] = {
                        "param_tag": ptag,
                        "qps": round(qps, 1),
                        "recall_at_10": round(rec, 4),
                        "strict_ov_pct": round(ov, 2),
                        "build_s": round(bs, 2),
                        "vram_gb": round(vr, 3),
                    }
        for (ds, m), info in best_per_dm.items():
            cuvs_summary[t_key][ds][m] = info
    claims["cuvs_summary"] = {k: dict(v) for k, v in cuvs_summary.items()}

    # Active tile survival curves on GIST1M (phase2a/active_tile_curves.csv)
    curves_g = defaultdict(dict)
    by_c = defaultdict(list)
    for r in read_csv("phase2a/active_tile_curves.csv"):
        if r["dataset"] == "gist1m" and r["l1_on"] == "0" and r["tile_size"] == "64" and abs(float(r["target_recall"]) - 0.95) < 1e-3:
            by_c[(r["config"], int(r["block_idx"]), int(r["dim_end"]))].append(float(r["active_warp_ratio_post_l1"]) * 100.0)
    for (cfg, bidx, d_sc), vals in sorted(by_c.items()):
        curves_g[cfg][f"b={bidx}_dim={d_sc}"] = round(statistics.mean(vals), 2)
    claims["active_tile_curves_gist1m"] = dict(curves_g)

    # Ablation A100 (phase4a/phase4a_ablation_A100.csv)
    ab_map = defaultdict(lambda: defaultdict(list))
    for r in read_csv("phase4a/phase4a_ablation_A100.csv"):
        if r["gpu"] == "A100":
            ab_map[r["dataset"]][r["ablation_step"]].append(float(r["qps"]))
    claims["ablation_A100"] = {
        ds: {v: round(statistics.mean(qs), 1) for v, qs in vmap.items()}
        for ds, vmap in ab_map.items()
    }

    # Phase 4C CPU baselines (phase4c/*.csv)
    claims["phase4c_cpu"] = {
        "tribase_opt1": [
            {
                "dataset": r["dataset"],
                "nprobe": int(r["nprobe"]),
                "qps": round(float(r["qps"]), 4),
                "recall": round(float(r["recall"]), 4),
            }
            for r in read_csv("phase4c/tribase_results.csv") if str(r.get("opt_level")) in ("1", "opt1")
        ],
        "adsampling_eps2": [
            {
                "dataset": r["dataset"],
                "nprobe": int(r["nprobe"]),
                "qps": round(float(r["qps"]), 1),
                "recall_at_10": round(float(r["recall_at_10"]), 4),
            }
            for r in read_csv("phase4c/adsampling_results.csv") if abs(float(r.get("epsilon0", 0)) - 2.0) < 1e-3
        ],
        "pdx_f32": [
            {
                "dataset": r["dataset"],
                "nprobe": int(r["nprobe"]),
                "qps": round(float(r["qps"]), 1),
                "recall": round(float(r["recall"]), 4),
            }
            for r in read_csv("phase4c/pdx_results.csv")
        ],
    }

    # Legacy Phase 6 gpu_build_mem_A100 & cagra_ooc_A100
    gbm_legacy = defaultdict(list)
    for r in read_csv("phase6/gpu_build_mem_A100.csv"):
        gbm_legacy[r["dataset"]].append(r)
    claims["gpu_build_mem_A100"] = {
        ds: {
            "dim": int(rows[0]["dim"]),
            "nlist": int(rows[0]["nlist"]),
            "kmeans_gpu_s": round(statistics.mean(float(x["kmeans_gpu_s"]) for x in rows), 3),
            "cm_extra_gpu_s": round(statistics.mean(float(x["cm_extra_gpu_s"]) for x in rows), 4),
            "trident_extra_gpu_s": round(statistics.mean(float(x["trident_extra_gpu_s"]) for x in rows), 3),
            "total_cm_gpu_s": round(statistics.mean(float(x["total_cm_gpu_s"]) for x in rows), 3),
            "total_trident_gpu_s": round(statistics.mean(float(x["total_trident_gpu_s"]) for x in rows), 3),
            "cuvs_ivfflat_build_s": round(statistics.mean(float(x["cuvs_ivfflat_build_s"]) for x in rows), 3),
            "cuvs_cagra_build_s": round(statistics.mean(float(x["cuvs_cagra_build_s"]) for x in rows), 3),
            "speedup_build_vs_cagra_mean": round(statistics.mean(float(x["speedup_build_vs_cagra"]) for x in rows), 2),
            "trident_aux_vram_gb": round(statistics.mean(float(x["trident_aux_vram_gb"]) for x in rows), 3),
            "cuvs_cagra_vram_gb": round(statistics.mean(float(x["cuvs_cagra_vram_gb"]) for x in rows), 3),
        }
        for ds, rows in gbm_legacy.items()
    }

    cagra_ooc_legacy = defaultdict(dict)
    for r in read_csv("phase6/cagra_ooc_A100.csv"):
        key = f"{r['mode']}_cap{r['cap_gb']}GB"
        cagra_ooc_legacy[r["dataset"]][key] = {
            "status": r["status"],
            "required_vram_gb": float(r["required_vram_gb"]),
            "qps": float(r["qps"]),
            "error_msg": r["error_msg"],
        }
    claims["cagra_ooc_A100"] = dict(cagra_ooc_legacy)

    # =========================================================================
    # PHASE 7 NEW SECTIONS (PLAN V8 Step 3.2)
    # =========================================================================
    # 1. cagra_uvm_capped (data/phase7/cagra_uvm_capped_A100.csv)
    uvm_rows = read_csv("phase7/cagra_uvm_capped_A100.csv")
    cagra_uvm_summary = defaultdict(dict)
    for r in uvm_rows:
        ds = r["dataset"]
        key = f"{r['mode']}_cap{r['cap_gb']}GB_deg{r['graph_degree']}_itopk{r['itopk']}"
        cagra_uvm_summary[ds][key] = {
            "cap_gb": float(r["cap_gb"]),
            "mode": r["mode"],
            "graph_degree": int(r["graph_degree"]),
            "itopk": int(r["itopk"]),
            "free_after_load_gb": float(r["free_after_load_gb"]),
            "status": r["status"],
            "cold_qps": float(r["cold_qps"]),
            "steady_qps_mean": float(r["steady_qps_mean"]),
            "steady_qps_std": float(r["steady_qps_std"]),
            "recall_at_10": float(r["recall_at_10"]),
            "strict_overlap_pct": float(r["strict_overlap_pct"]),
        }
    claims["cagra_uvm_capped"] = dict(cagra_uvm_summary)

    # 2. gpu_build_quality (data/phase7/gpu_build_quality_A100.csv)
    qual_rows = read_csv("phase7/gpu_build_quality_A100.csv")
    qual_summary = {}
    for r in qual_rows:
        qual_summary[r["dataset"]] = {
            "dim": int(r["dim"]),
            "nlist": int(r["nlist"]),
            "nprobe": int(r["nprobe"]),
            "n_train": int(r["n_train"]),
            "kmeans_iters": int(r["kmeans_iters"]),
            "cm_fp32_qps": float(r["cm_fp32_qps"]),
            "trident_qps": float(r["trident_qps"]),
            "main_table_qps": float(r["main_table_qps"]),
            "recall_at_10": float(r["recall_at_10"]),
            "main_table_recall_at_10": float(r["main_table_recall_at_10"]),
            "recall_diff": float(r["recall_diff"]),
            "strict_overlap_pct": float(r["strict_overlap_pct"]),
            "passed_check": int(r["passed_check"]),
            "total_trident_gpu_build_s": float(r["total_trident_gpu_build_s"]),
        }
    claims["gpu_build_quality"] = qual_summary

    # 3. gpu_build_v2 (data/phase7/gpu_build_v2_A100.csv)
    b2_rows = read_csv("phase7/gpu_build_v2_A100.csv")
    by_b2 = defaultdict(list)
    for r in b2_rows:
        by_b2[r["dataset"]].append(r)
    b2_summary = {}
    for ds, rows in by_b2.items():
        km_s = statistics.mean(float(x["kmeans_gpu_s"]) for x in rows)
        cm_ex_s = statistics.mean(float(x["cm_extra_gpu_s"]) for x in rows)
        pca_s = statistics.mean(float(x["pca_sq8_gpu_s"]) for x in rows)
        pack_s = statistics.mean(float(x["pack_gpu_s"]) for x in rows)
        cm_tot = statistics.mean(float(x["total_cm_gpu_s"]) for x in rows)
        tr_tot = statistics.mean(float(x["total_trident_gpu_s"]) for x in rows)
        ivf_s = statistics.mean(float(x["cuvs_ivfflat_build_s"]) for x in rows)
        cag_r99_s = statistics.mean(float(x["cagra_r99_build_s"]) for x in rows)
        cag_64_s = statistics.mean(float(x["cagra_deg64_build_s"]) for x in rows)
        b2_summary[ds] = {
            "dim": int(rows[0]["dim"]),
            "nlist": int(rows[0]["nlist"]),
            "cagra_r99_deg": int(rows[0]["cagra_r99_deg"]),
            "kmeans_gpu_s": round(km_s, 3),
            "cm_extra_gpu_s": round(cm_ex_s, 4),
            "pca_sq8_gpu_s": round(pca_s, 3),
            "pack_gpu_s": round(pack_s, 4),
            "total_cm_gpu_s": round(cm_tot, 3),
            "total_trident_gpu_s": round(tr_tot, 3),
            "cuvs_ivfflat_build_s": round(ivf_s, 3),
            "cagra_r99_build_s": round(cag_r99_s, 3),
            "cagra_deg64_build_s": round(cag_64_s, 3),
            "speedup_vs_cagra_r99": round(cag_r99_s / tr_tot, 2) if tr_tot > 0 else 0.0,
            "speedup_vs_cagra_deg64": round(cag_64_s / tr_tot, 2) if tr_tot > 0 else 0.0,
            "ratio_trident_vs_ivfflat": round(tr_tot / ivf_s, 2) if ivf_s > 0 else 0.0,
            "trident_aux_vram_gb": round(statistics.mean(float(x["trident_aux_vram_gb"]) for x in rows), 3),
        }
    claims["gpu_build_v2"] = b2_summary

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(claims, f, indent=2)
    print(f"Wrote verified claims to {OUT_JSON} (sections: {list(claims.keys())})")


if __name__ == "__main__":
    main()
