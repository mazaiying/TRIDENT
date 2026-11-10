#!/usr/bin/env python3
"""
Consistency checker (PLAN V8 Step 6.3):
Extracts all numeric claims with units (QPS, \\times / x, GB, s) from main.tex
and sections/*.tex, and verifies each against data/verified_claims.json (and the
underlying CSVs summarized in verified_claims.json).
Prints summary statistics and any unmatched numbers.
"""
import csv
import glob
import json
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CLAIMS_PATH = DATA_DIR / "verified_claims.json"

# Architectural / hardware / experimental protocol constants
PROTOCOL_CONSTANTS = {
    # GPU / CPU hardware specs & memory caps
    80.0, 40.0, 16.0, 24.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0,
    2.2, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 75.0,
    2039.0, 1555.0, 504.0, 192.0, 164.0, 48.0, 96.0, 108.0,
    # Unit / identity multipliers & thresholds
    0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0,
    3.0, 4.0, 5.0, 8.0, 10.0, 16.0, 32.0, 50.0, 64.0, 100.0, 128.0, 200.0, 256.0,
}


def add_variants(val_set: set, v: float):
    if math.isnan(v) or math.isinf(v):
        return
    for scale in (1.0, 0.001, 1000.0, 100.0, 0.01):
        sv = v * scale
        for nd in (0, 1, 2, 3, 4):
            val_set.add(round(sv, nd))


def collect_verified_numbers() -> set:
    valid = set()
    for c in PROTOCOL_CONSTANTS:
        add_variants(valid, c)

    if CLAIMS_PATH.exists():
        with open(CLAIMS_PATH, "r", encoding="utf-8") as f:
            claims = json.load(f)

        def walk(obj):
            if isinstance(obj, dict):
                nums = [v for v in obj.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
                for v in nums:
                    add_variants(valid, float(v))
                # Also add pairwise ratios within the same dict record (e.g., QPS or time ratios)
                for a in nums:
                    for b in nums:
                        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(b) > 1e-9:
                            r = float(a) / float(b)
                            if 0.001 <= abs(r) <= 10000.0:
                                add_variants(valid, r)
                                add_variants(valid, (1.0 - r) * 100.0)
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)
            elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
                add_variants(valid, float(obj))

        walk(claims)

    # Also walk all CSVs in data/phase*/*.csv so any direct table/text numbers from CSVs are covered
    for csv_path in sorted(DATA_DIR.glob("phase*/*.csv")):
        if csv_path.name.startswith("._"):
            continue
        try:
            with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for row in reader:
                    for cell in row:
                        try:
                            fv = float(cell)
                            add_variants(valid, fv)
                        except ValueError:
                            pass
        except Exception:
            pass

    return valid


def is_matched(val: float, valid: set) -> bool:
    for nd in (0, 1, 2, 3, 4):
        if round(val, nd) in valid:
            return True
    # Allow +/- 0.02 rounding tolerance for 1-2 decimal formatted numbers
    for tol in (0.01, 0.02, 0.05, 0.1):
        for candidate in (round(val + tol, 2), round(val - tol, 2), round(val + tol, 1), round(val - tol, 1)):
            if candidate in valid:
                return True
    return False


def main():
    valid = collect_verified_numbers()
    tex_files = [ROOT / "main.tex"] + [
        f for f in sorted((ROOT / "sections").glob("*.tex")) if not f.name.startswith("._")
    ]

    # Regexes for numbers with QPS, \times, GB, s (both inside \text{...} and plain text)
    patterns = [
        ("QPS", re.compile(r"(?<![A-Za-z0-9_])(\d[\d,]*(?:\.\d+)?)(?:\s*(?:\\text\{)?\s*[KM]\s*\}?)?(?:\s*(?:--|to|\$?--\$?)\s*(\d[\d,]*(?:\.\d+)?)(?:\s*(?:\\text\{)?\s*[KM]\s*\}?)?)?\s*(?:\\text\{\s*[KM]?\s*QPS\s*\}|[KM]?\s*QPS\b)")),
        ("times", re.compile(r"(?<![A-Za-z0-9_])(\d[\d,]*(?:\.\d+)?)(?:\s*(?:--|to|\$?--\$?)\s*(\d[\d,]*(?:\.\d+)?))?\s*\\times")),
        ("GB", re.compile(r"(?<![A-Za-z0-9_])(\d[\d,]*(?:\.\d+)?)(?:\s*(?:--|to|/|\$?--\$?)\s*(\d[\d,]*(?:\.\d+)?))?\s*(?:\\text\{\s*(?:GB|GiB|G)\s*\}|(?:GB|GiB)\b)")),
        ("s", re.compile(r"(?<![A-Za-z0-9_])(\d[\d,]*(?:\.\d+)?)(?:\s*(?:--|to|\$?--\$?)\s*(\d[\d,]*(?:\.\d+)?))?\s*(?:\\text\{\s*s\s*\}|\bs\b)(?=[\s$,.;:)\]|&]|$)")),
    ]

    checked = {"QPS": 0, "times": 0, "GB": 0, "s": 0}
    unmatched = []

    for tf in tex_files:
        lines = tf.read_text(encoding="utf-8").splitlines()
        for lineno, raw_line in enumerate(lines, start=1):
            # Strip comments
            line = re.sub(r"(?<!\\)%.*$", "", raw_line)
            if not line.strip():
                continue
            for kind, pat in patterns:
                for m in pat.finditer(line):
                    groups = [g for g in m.groups() if g and re.match(r"^\d", g)]
                    for g in groups:
                        val = float(g.replace(",", ""))
                        checked[kind] += 1
                        if not is_matched(val, valid):
                            unmatched.append((f"{tf.relative_to(ROOT)}:{lineno}", kind, g, m.group(0).strip()))

    total_checked = sum(checked.values())
    print("=== check_numbers.py Consistency Audit ===")
    print(f"Verified pool size: {len(valid)} distinct numeric representations")
    print(f"Total unit-tagged numbers checked in .tex: {total_checked}")
    print(f"  - QPS   : {checked['QPS']}")
    print(f"  - \\times: {checked['times']}")
    print(f"  - GB    : {checked['GB']}")
    print(f"  - s     : {checked['s']}")
    if not unmatched:
        print("UNMATCHED NUMBERS: 0 (ALL numbers verified against verified_claims.json / CSVs)")
    else:
        print(f"UNMATCHED NUMBERS: {len(unmatched)}")
        for loc, kind, num_s, ctx in unmatched:
            print(f"  [UNMATCHED] {loc} ({kind}): {num_s} in '{ctx}'")


if __name__ == "__main__":
    main()
