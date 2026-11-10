#!/usr/bin/env python3
"""
Convert numpy datasets into standard fvecs/ivecs for ADSampling, Tribase, and PDX.
"""

import os
import sys
import numpy as np

DATA_ROOT = os.path.expanduser("~/trident/data")

def write_fvecs(filename, array):
    n, d = array.shape
    d_col = np.full((n, 1), d, dtype=np.int32)
    merged = np.hstack([d_col, array.astype(np.float32).view(np.int32)])
    merged.tofile(filename)
    print(f"Saved {filename} ({os.path.getsize(filename)} bytes)")

def write_ivecs(filename, array):
    n, d = array.shape
    d_col = np.full((n, 1), d, dtype=np.int32)
    merged = np.hstack([d_col, array.astype(np.int32)])
    merged.tofile(filename)
    print(f"Saved {filename} ({os.path.getsize(filename)} bytes)")

def convert_dataset_to_fvecs(ds_name):
    ds_dir = os.path.join(DATA_ROOT, ds_name)
    base_npy = os.path.join(ds_dir, "base.npy")
    query_npy = os.path.join(ds_dir, "query.npy")
    gt_npy = os.path.join(ds_dir, "gt.npy")

    if not os.path.exists(base_npy) or not os.path.exists(query_npy) or not os.path.exists(gt_npy):
        print(f"Skipping {ds_name}, npy files not found.")
        return

    xb = np.load(base_npy, mmap_mode="r")
    xq = np.load(query_npy)
    gt = np.load(gt_npy)

    out_dir = os.path.join(DATA_ROOT, ds_name, "fvecs")
    os.makedirs(out_dir, exist_ok=True)

    base_fvecs = os.path.join(out_dir, f"{ds_name}_base.fvecs")
    query_fvecs = os.path.join(out_dir, f"{ds_name}_query.fvecs")
    gt_ivecs = os.path.join(out_dir, f"{ds_name}_groundtruth.ivecs")

    if not os.path.exists(base_fvecs):
        print(f"Exporting {ds_name} base.npy to {base_fvecs}...")
        write_fvecs(base_fvecs, np.ascontiguousarray(xb, dtype=np.float32))
    if not os.path.exists(query_fvecs):
        print(f"Exporting {ds_name} query.npy to {query_fvecs}...")
        write_fvecs(query_fvecs, np.ascontiguousarray(xq, dtype=np.float32))
    if not os.path.exists(gt_ivecs):
        print(f"Exporting {ds_name} gt.npy to {gt_ivecs}...")
        write_ivecs(gt_ivecs, gt[:, :100].astype(np.int32))
    print(f"[{ds_name}] fvecs ready.")

if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["sift1m", "glove100", "gist1m"]
    for t in targets:
        convert_dataset_to_fvecs(t)
