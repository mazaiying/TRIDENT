#!/usr/bin/env python3
"""
PLAN_V11 Task B — Generate SUMMARY_V11.md strictly from CSV/JSONL files.
Zero manual copying, zero hardcoding of experimental numbers.
"""
import csv
import json
import os

ROOT = "/Volumes/Elements SE/科研/icde0117"
P10_DIR = os.path.join(ROOT, "results", "phase10")
P11_DIR = os.path.join(ROOT, "results", "phase11")
os.makedirs(P11_DIR, exist_ok=True)


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def deduplicate_rows(rows):
    seen = set()
    out = []
    for r in rows:
        key = (
            r.get("dataset"),
            r.get("recall_target"),
            r.get("method"),
            r.get("config"),
            r.get("ub_seed"),
            r.get("m_cand"),
        )
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def main():
    lossless_rows = read_csv(os.path.join(P10_DIR, "lossless_A100.csv"))
    ooc_1m10m_raw = read_csv(os.path.join(P10_DIR, "ooc_1m10m_A100.csv"))
    ooc_1m10m = deduplicate_rows(ooc_1m10m_raw)
    ncu_1m10m = read_csv(os.path.join(P10_DIR, "ncu_perpass_1m10m_A100.csv"))
    ooc_100m = deduplicate_rows(read_csv(os.path.join(P10_DIR, "ooc_100m_RTX4080_v10.csv")))
    ncu_100m = read_csv(os.path.join(P10_DIR, "ncu_perpass_100m_RTX4080.csv"))
    e3_recs = read_jsonl(os.path.join(P10_DIR, "gpu_baselines_100m_RTX4080.jsonl"))

    lines = []
    lines.append("# SUMMARY_V11 — Auto-Generated from CSV / JSONL (Zero Hand-Copied Numbers)\n")

    # 0. Lossless Audit
    lines.append("## 0. Step E0 Lossless Audit (`lossless_A100.csv`)\n")
    lines.append(f"- Total rows: {len(lossless_rows)}")
    min_ov = min(float(r["strict_overlap_pct"]) for r in lossless_rows)
    max_err = max(float(r["max_abs_dist_diff_vs_cm"]) for r in lossless_rows)
    lines.append(f"- Min `strict_overlap_pct`: `{min_ov:.4f}%`")
    lines.append(f"- Max `max_abs_dist_diff_vs_cm`: `{max_err:.3e}`\n")
    lines.append("| Dataset | Placement | R | nprobe | ub_seed | QPS (mean ± std) | Recall@10 | Strict Overlap (%) | Surv (%) | Dims Pruned (%) | Max Abs Dist Diff |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in lossless_rows:
        lines.append(
            f"| {r['dataset']} | {r['placement']} | {r['recall_target']} | {r['nprobe']} | {r['ub_seed']} | "
            f"{float(r['qps_mean']):,.1f} ± {float(r['qps_std']):.1f} | {r['recall_at_10']} | "
            f"{float(r['strict_overlap_pct']):.2f}% | {float(r['surv_pct']):.4f}% | "
            f"{float(r['dims_pruned_pct']):.2f}% | {r['max_abs_dist_diff_vs_cm']} |"
        )
    lines.append("")

    # 1. Main Table per Dataset x R
    lines.append("## 1. Full OOC Benchmark Tables (1M/10M on A100 & 100M on RTX 4080)\n")
    lines.append("| Dataset | GPU / Cap | R | nprobe | Method | Config / ub / m | QPS (mean ± std) | Recall@10 | Strict Overlap (%) | Surv (%) | Dims Pruned (%) |")
    lines.append("| :--- | :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

    all_ooc = []
    for r in ooc_1m10m:
        r_copy = dict(r)
        r_copy["_gpu_cap"] = f"A100 ({r['cap_gb']} GB cap)"
        all_ooc.append(r_copy)
    for r in ooc_100m:
        r_copy = dict(r)
        r_copy["_gpu_cap"] = "RTX 4080 (16 GB, No Cap)"
        all_ooc.append(r_copy)

    for r in all_ooc:
        ds = r["dataset"]
        gc = r["_gpu_cap"]
        rt = r["recall_target"]
        npb = r["nprobe"]
        mth = r["method"]
        cfg = r["config"]
        if r.get("m_cand"):
            cfg = f"{cfg}, m={r['m_cand']}"
        qps_m = float(r["qps_mean"])
        qps_s = float(r["qps_std"])
        rec10 = r["recall_at_10"] if r["recall_at_10"] != "" else "—"
        ov = f"{float(r['strict_overlap_pct']):.2f}%" if r["strict_overlap_pct"] != "" else "—"
        surv = f"{float(r['surv_pct']):.4f}%" if r.get("surv_pct", "") != "" else "—"
        dpr = f"{float(r['dims_pruned_pct']):.2f}%" if r.get("dims_pruned_pct", "") != "" else "—"
        lines.append(
            f"| {ds} | {gc} | {rt} | {npb} | `{mth}` | `{cfg}` | {qps_m:,.1f} ± {qps_s:.1f} | {rec10} | {ov} | {surv} | {dpr} |"
        )
    lines.append("")

    # Helper to index rows by (dataset, float(recall_target))
    by_ds_r = {}
    for r in all_ooc:
        key = (r["dataset"], float(r["recall_target"]))
        by_ds_r.setdefault(key, []).append(r)

    # 2. Near-Lossless / Strictly Lossless Speedup Table
    lines.append("## 2. Speedup Under Near-Lossless / Strictly Lossless Regime (`m=2048` / Exact Baselines)\n")
    lines.append("| Dataset | R | `TRIDENT (ub=1)` QPS | `TRIDENT (ub=0)` QPS | `CM-FP32-OOC` QPS | `CPU-24T` QPS | `IVFPQ (m=2048)` QPS [Rec@10 / Ov%] | `ub=1` vs `ub=0` | `ub=1` vs `CM-OOC` | `ub=1` vs `CPU-24T` | `ub=1` vs `m=2048` |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    ds_order = [("gist1m", 0.95), ("gist1m", 0.99), ("bge1024", 0.95), ("bge1024", 0.99),
                ("sift10m", 0.95), ("sift10m", 0.99), ("deep10m", 0.95), ("deep10m", 0.99),
                ("sift100m", 0.95), ("sift100m", 0.99)]

    for ds, rt in ds_order:
        rows = by_ds_r[(ds, rt)]
        ub1 = next(r for r in rows if r["method"].startswith("TRIDENT") and r["ub_seed"] == "1")
        ub0 = next(r for r in rows if r["method"].startswith("TRIDENT") and r["ub_seed"] == "0")
        cm = next(r for r in rows if r["method"] == "CM-FP32-OOC")
        cpu = next(r for r in rows if r["method"] == "Faiss-CPU-IVFFlat-24T")
        m2048 = next(r for r in rows if r["method"] == "IVFPQ+CPU-Refine" and r["m_cand"] == "2048")

        q_ub1 = float(ub1["qps_mean"])
        q_ub0 = float(ub0["qps_mean"])
        q_cm = float(cm["qps_mean"])
        q_cpu = float(cpu["qps_mean"])
        q_2048 = float(m2048["qps_mean"])
        r_2048 = float(m2048["recall_at_10"])
        ov_2048 = float(m2048["strict_overlap_pct"])

        lines.append(
            f"| {ds} | {rt:.2f} | {q_ub1:,.1f} | {q_ub0:,.1f} | {q_cm:,.1f} | {q_cpu:,.1f} | "
            f"{q_2048:,.1f} [{r_2048:.4f} / {ov_2048:.2f}%] | {q_ub1/q_ub0:.2f}x | {q_ub1/q_cm:.2f}x | "
            f"{q_ub1/q_cpu:.2f}x | {q_ub1/q_2048:.2f}x |"
        )
    lines.append("")

    # 3. Matched-Recall Target Speedup Table
    lines.append("## 3. Speedup Under Matched-Recall Target (`Recall@10 >= R`, Smallest Qualifying `m`)\n")
    lines.append("| Dataset | R Target | `TRIDENT (ub=1)` QPS [Rec@10 / Ov%] | Smallest `m` with `Rec@10 >= R` | `IVFPQ+Refine (m)` QPS | `IVFPQ+Refine (m)` Rec@10 | `IVFPQ+Refine (m)` Strict Overlap (%) | `TRIDENT (ub=1)` vs Matched-Recall `IVFPQ (m)` |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    matched_summary_json = {}
    for ds, rt in ds_order:
        rows = by_ds_r[(ds, rt)]
        ub1 = next(r for r in rows if r["method"].startswith("TRIDENT") and r["ub_seed"] == "1")
        q_ub1 = float(ub1["qps_mean"])
        r_ub1 = float(ub1["recall_at_10"])
        ov_ub1 = float(ub1["strict_overlap_pct"])

        pq_rows = [r for r in rows if r["method"] == "IVFPQ+CPU-Refine"]
        pq_rows.sort(key=lambda x: int(x["m_cand"]))
        qualifying = [r for r in pq_rows if float(r["recall_at_10"]) >= rt]

        if qualifying:
            best_pq = qualifying[0]
            m_val = int(best_pq["m_cand"])
            q_pq = float(best_pq["qps_mean"])
            r_pq = float(best_pq["recall_at_10"])
            ov_pq = float(best_pq["strict_overlap_pct"])
            sp = q_ub1 / q_pq
            lines.append(
                f"| {ds} | {rt:.2f} | {q_ub1:,.1f} [{r_ub1:.4f} / {ov_ub1:.2f}%] | `m={m_val}` | "
                f"{q_pq:,.1f} | {r_pq:.4f} | {ov_pq:.2f}% | **{sp:.2f}x** |"
            )
            matched_summary_json[f"{ds}_R{rt:.2f}"] = {
                "status": "reached",
                "smallest_m": m_val,
                "trident_ub1_qps": q_ub1,
                "trident_ub1_recall": r_ub1,
                "trident_ub1_overlap_pct": ov_ub1,
                "ivfpq_qps": q_pq,
                "ivfpq_recall": r_pq,
                "ivfpq_overlap_pct": ov_pq,
                "speedup_vs_matched_ivfpq": round(sp, 2),
            }
        else:
            max_pq = pq_rows[-1]
            m_val = int(max_pq["m_cand"])
            q_pq = float(max_pq["qps_mean"])
            r_pq = float(max_pq["recall_at_10"])
            ov_pq = float(max_pq["strict_overlap_pct"])
            sp = q_ub1 / q_pq
            lines.append(
                f"| {ds} | {rt:.2f} | {q_ub1:,.1f} [{r_ub1:.4f} / {ov_ub1:.2f}%] | not reached (at `m={m_val}`: {r_pq:.4f}) | "
                f"{q_pq:,.1f} (`m={m_val}`) | {r_pq:.4f} | {ov_pq:.2f}% | not reached (**{sp:.2f}x** vs `m={m_val}`) |"
            )
            matched_summary_json[f"{ds}_R{rt:.2f}"] = {
                "status": "not_reached",
                "max_m_tested": m_val,
                "trident_ub1_qps": q_ub1,
                "trident_ub1_recall": r_ub1,
                "trident_ub1_overlap_pct": ov_ub1,
                "ivfpq_max_m_qps": q_pq,
                "ivfpq_max_m_recall": r_pq,
                "ivfpq_max_m_overlap_pct": ov_pq,
                "speedup_vs_max_m": round(sp, 2),
            }
    lines.append("")

    # 4. NCU Per-Pass PCIe Traffic Table
    lines.append("## 4. Nsight Compute Per-Pass PCIe Traffic (`ncu_perpass_1m10m_A100.csv` & `ncu_perpass_100m_RTX4080.csv`)\n")
    lines.append("| Dataset | R / nprobe | `CM-FP32-OOC` Total (MB) | `TRIDENT (ub=0)` Pass-1 / Pass-2 / Total (MB) | `TRIDENT (ub=1)` Pass-1 / Pass-2 / Total (MB) | Pass-1 Reduction (`ub=0 -> ub=1`) | Total Reduction (`ub=0 -> ub=1`) | Total Reduction vs `CM-FP32-OOC` |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for ds in ("gist1m", "bge1024", "sift10m", "deep10m"):
        r_cm = next(r for r in ncu_1m10m if r["dataset"] == ds and r["method"] == "CM-FP32-OOC")
        r_ub0 = next(r for r in ncu_1m10m if r["dataset"] == ds and r["method"] == "TRIDENT-OOC" and r["ub_seed"] == "0")
        r_ub1 = next(r for r in ncu_1m10m if r["dataset"] == ds and r["method"] == "TRIDENT-OOC" and r["ub_seed"] == "1")
        npb = r_ub1["nprobe"]
        cm_mb = float(r_cm["total_mb_per_1024q"])
        p1_0, p2_0, tot_0 = float(r_ub0["pass1_mb_per_1024q"]), float(r_ub0["pass2_mb_per_1024q"]), float(r_ub0["total_mb_per_1024q"])
        p1_1, p2_1, tot_1 = float(r_ub1["pass1_mb_per_1024q"]), float(r_ub1["pass2_mb_per_1024q"]), float(r_ub1["total_mb_per_1024q"])
        p1_red_pct = (1.0 - p1_1 / p1_0) * 100.0
        lines.append(
            f"| {ds} | 0.99 (np={npb}) | {cm_mb:,.2f} | {p1_0:,.2f} / {p2_0:,.2f} / {tot_0:,.2f} | "
            f"{p1_1:,.2f} / {p2_1:,.2f} / {tot_1:,.2f} | -{p1_red_pct:.1f}% ({p1_0/p1_1:.2f}x) | "
            f"{tot_0/tot_1:.2f}x | {cm_mb/tot_1:,.2f}x ({(1.0 - tot_1/cm_mb)*100.0:.2f}%) |"
        )

    for rt in (0.95, 0.99):
        r_cm = next(r for r in ncu_100m if float(r["recall_target"]) == rt and r["method"] == "CM-FP32-OOC")
        r_ub0 = next(r for r in ncu_100m if float(r["recall_target"]) == rt and r["method"] == "TRIDENT-OOC" and r["ub_seed"] == "0")
        r_ub1 = next(r for r in ncu_100m if float(r["recall_target"]) == rt and r["method"] == "TRIDENT-OOC" and r["ub_seed"] == "1")
        npb = r_ub1["nprobe"]
        cm_mb = float(r_cm["total_mb_per_1024q"])
        p1_0, p2_0, tot_0 = float(r_ub0["pass1_mb_per_1024q"]), float(r_ub0["pass2_mb_per_1024q"]), float(r_ub0["total_mb_per_1024q"])
        p1_1, p2_1, tot_1 = float(r_ub1["pass1_mb_per_1024q"]), float(r_ub1["pass2_mb_per_1024q"]), float(r_ub1["total_mb_per_1024q"])
        p1_red_pct = (1.0 - p1_1 / p1_0) * 100.0
        lines.append(
            f"| sift100m | {rt:.2f} (np={npb}) | {cm_mb:,.2f} | {p1_0:,.2f} / {p2_0:,.2f} / {tot_0:,.2f} | "
            f"{p1_1:,.2f} / {p2_1:,.2f} / {tot_1:,.2f} | -{p1_red_pct:.1f}% ({p1_0/p1_1:.2f}x) | "
            f"{tot_0/tot_1:.2f}x | {cm_mb/tot_1:,.2f}x ({(1.0 - tot_1/cm_mb)*100.0:.2f}%) |"
        )
    lines.append("")

    # 5. E3 & Task A GPU Baselines Status Table
    lines.append("## 5. Step E3 & Task A 100M GPU Baselines Status (`gpu_baselines_100m_RTX4080.jsonl`)\n")
    lines.append("| Mode | Status | Elapsed (s) | Build (s) | Details / Results / Error |")
    lines.append("| :--- | :---: | :---: | :---: | :--- |")
    for rec in e3_recs:
        m = rec.get("mode", "")
        st = rec.get("status", "")
        el = rec.get("elapsed_s", "—")
        bs = rec.get("build_s", "—")
        if "results" in rec and rec["results"]:
            det = json.dumps(rec["results"])
        else:
            det = rec.get("error", f"timeout_s={rec.get('timeout_s', '')}")
        lines.append(f"| `{m}` | `{st}` | {el} | {bs} | `{det}` |")
    lines.append("")

    out_md = os.path.join(P11_DIR, "SUMMARY_V11.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    out_json = os.path.join(P11_DIR, "matched_recall_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(matched_summary_json, f, indent=2)

    print(f"Wrote {out_md} and {out_json}")


if __name__ == "__main__":
    main()
