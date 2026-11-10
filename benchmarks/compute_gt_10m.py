#!/usr/bin/env python3
"""
Compute Exact GPU Ground Truth for 10M Scale Datasets (PLAN_V5 Phase 4B-1)
Computes exact top-100 L2 ground truth using Faiss GpuIndexFlatL2 on A100.
Saves ground truth as gt_top100.ivecs and gt.npy.
"""

import os
import sys
import time
import struct
import numpy as np
import faiss
import torch

DATA_ROOT = os.path.expanduser("~/trident/data")

def read_u8bin(filepath):
    print(f"Reading {filepath}...")
    with open(filepath, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        print(f"  Header: N={n}, D={d}")
        data = np.fromfile(f, dtype=np.uint8, count=n * d).reshape((n, d))
    return data

def read_fbin(filepath):
    print(f"Reading {filepath}...")
    with open(filepath, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        print(f"  Header: N={n}, D={d}")
        data = np.fromfile(f, dtype=np.float32, count=n * d).reshape((n, d))
    return data

def write_ivecs(filename, array):
    n, d = array.shape
    d_col = np.full((n, 1), d, dtype=np.int32)
    merged = np.hstack([d_col, array.astype(np.int32)])
    merged.tofile(filename)
    print(f"Saved {n} vectors to {filename} ({os.path.getsize(filename)} bytes)")

def compute_gt_gpu(xb: np.ndarray, xq: np.ndarray, k: int = 100, dev_id: int = 0, block_size: int = 2500000):
    N, D = xb.shape
    Q = xq.shape[0]
    print(f"Computing exact GT for Q={Q}, N={N}, D={D}, k={k} on GPU {dev_id}...")

    # For 10M scale, if N fits directly into GPU VRAM (e.g. A100 80GB):
    # 10M * 128 * 4 = 5.12 GB float32 -> fits easily!
    res = faiss.StandardGpuResources()
    t0 = time.time()
    
    # Check if we can fit entire base in GPU VRAM
    base_bytes = N * D * 4
    if base_bytes < 30 * 1024 * 1024 * 1024: # < 30 GB fits easily on A100 80GB
        print(f"Adding all {N} vectors ({base_bytes / 1024**3:.2f} GB) to GpuIndexFlatL2...")
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = dev_id
        cfg.useFloat16 = False # Exact float32
        gpu_index = faiss.GpuIndexFlatL2(res, D, cfg)
        gpu_index.add(xb)
        print(f"Index built in {time.time()-t0:.2f}s. Performing search...")
        
        t_search = time.time()
        D_all = []
        I_all = []
        batch_q = 1000
        for i in range(0, Q, batch_q):
            q_chunk = xq[i : i + batch_q]
            D_c, I_c = gpu_index.search(q_chunk, k)
            D_all.append(D_c)
            I_all.append(I_c)
        D_all = np.vstack(D_all)
        I_all = np.vstack(I_all)
        print(f"Search completed in {time.time()-t_search:.2f}s. Total time: {time.time()-t0:.2f}s")
        return D_all, I_all
    else:
        # Sharded / Blocked search if base is massive (>30GB)
        print(f"Base size ({base_bytes/1024**3:.2f} GB) using blocked search with block_size={block_size}...")
        # Heap to keep track of overall top-k
        import heapq
        # Initialize with infinity
        all_dists = np.full((Q, k), np.inf, dtype=np.float32)
        all_indices = np.full((Q, k), -1, dtype=np.int64)

        for b_start in range(0, N, block_size):
            b_end = min(b_start + block_size, N)
            print(f"  Searching block [{b_start}:{b_end}]...")
            cfg = faiss.GpuIndexFlatConfig()
            cfg.device = dev_id
            cfg.useFloat16 = False
            gpu_sub = faiss.GpuIndexFlatL2(res, D, cfg)
            gpu_sub.add(xb[b_start:b_end])
            
            for i in range(0, Q, 1000):
                q_chunk = xq[i : i + 1000]
                D_sub, I_sub = gpu_sub.search(q_chunk, k)
                I_sub += b_start # Offset
                
                # Merge into all_dists, all_indices
                cur_d = np.hstack([all_dists[i:i+1000], D_sub])
                cur_i = np.hstack([all_indices[i:i+1000], I_sub])
                sort_idx = np.argsort(cur_d, axis=1)[:, :k]
                all_dists[i:i+1000] = np.take_along_axis(cur_d, sort_idx, axis=1)
                all_indices[i:i+1000] = np.take_along_axis(cur_i, sort_idx, axis=1)
            del gpu_sub

        return all_dists, all_indices

def process_dataset(ds_name):
    ds_dir = os.path.join(DATA_ROOT, ds_name)
    gt_ivecs = os.path.join(ds_dir, "gt_top100.ivecs")
    gt_npy = os.path.join(ds_dir, "gt.npy")
    base_npy = os.path.join(ds_dir, "base.npy")
    query_npy = os.path.join(ds_dir, "query.npy")

    if os.path.exists(gt_ivecs) and os.path.exists(gt_npy) and os.path.exists(base_npy) and os.path.exists(query_npy):
        print(f"[{ds_name}] All files already prepared and GT computed.")
        return

    if ds_name == "sift10m":
        base_bin = os.path.join(ds_dir, "base.10M.u8bin")
        query_bin = os.path.join(ds_dir, "query.public.10K.u8bin")
        if not os.path.exists(base_bin) or not os.path.exists(query_bin):
            print(f"[{ds_name}] Raw bin files not found in {ds_dir}!")
            return
        
        if not os.path.exists(base_npy):
            xb_u8 = read_u8bin(base_bin)
            xb = xb_u8.astype(np.float32)
            np.save(base_npy, xb)
            print(f"[{ds_name}] Saved {base_npy} ({xb.shape})")
        else:
            xb = np.load(base_npy, mmap_mode="r")

        if not os.path.exists(query_npy):
            xq_u8 = read_u8bin(query_bin)
            xq = xq_u8.astype(np.float32)
            np.save(query_npy, xq)
            print(f"[{ds_name}] Saved {query_npy} ({xq.shape})")
        else:
            xq = np.load(query_npy)

    elif ds_name == "deep10m":
        base_bin = os.path.join(ds_dir, "base.10M.fbin")
        query_bin = os.path.join(ds_dir, "query.public.10K.fbin")
        if not os.path.exists(base_bin) or not os.path.exists(query_bin):
            print(f"[{ds_name}] Raw bin files not found in {ds_dir}!")
            return

        if not os.path.exists(base_npy):
            xb = read_fbin(base_bin)
            np.save(base_npy, xb)
            print(f"[{ds_name}] Saved {base_npy} ({xb.shape})")
        else:
            xb = np.load(base_npy, mmap_mode="r")

        if not os.path.exists(query_npy):
            xq = read_fbin(query_bin)
            np.save(query_npy, xq)
            print(f"[{ds_name}] Saved {query_npy} ({xq.shape})")
        else:
            xq = np.load(query_npy)

    else:
        print(f"Unknown dataset {ds_name}")
        return

    # Compute GT on GPU 0 (or GPU from CUDA_VISIBLE_DEVICES)
    # Note: CUDA_VISIBLE_DEVICES will remap GPU to index 0
    D_gt, I_gt = compute_gt_gpu(np.ascontiguousarray(xb, dtype=np.float32), np.ascontiguousarray(xq, dtype=np.float32), k=100, dev_id=0)
    
    # Save gt_top100.ivecs and gt.npy
    write_ivecs(gt_ivecs, I_gt)
    np.save(gt_npy, I_gt.astype(np.int64))
    print(f"[{ds_name}] GT successfully computed and saved to {gt_ivecs} and {gt_npy}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target in ["all", "sift10m"]:
        process_dataset("sift10m")
    if target in ["all", "deep10m"]:
        process_dataset("deep10m")
