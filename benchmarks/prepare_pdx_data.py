#!/usr/bin/env python3
"""
Prepare raw binary data for PDX BenchmarkEndToEnd.
PDX expects:
  data_<dataset>.bin: raw N x D float32 bytes
  data_<dataset>_test.bin: raw N_q x D float32 bytes
Datasets:
  sift: N=1000000, D=128, N_q=1000
  gist: N=1000000, D=960, N_q=1000
Target directory: ~/trident/baselines/SuperKMeans/benchmarks/data/
"""

import os
import struct
import numpy as np

DATA_DIR = os.path.expanduser("~/trident/baselines/SuperKMeans/benchmarks/data")
os.makedirs(DATA_DIR, exist_ok=True)

def convert_fvecs_to_raw_bin(fvecs_path, out_bin_path, max_vectors=None):
    if os.path.exists(out_bin_path):
        print(f"Target {out_bin_path} already exists ({os.path.getsize(out_bin_path)} bytes), skipping.")
        return
        
    print(f"Converting {fvecs_path} -> {out_bin_path} (max_vectors={max_vectors})...")
    with open(fvecs_path, "rb") as f_in, open(out_bin_path, "wb") as f_out:
        count = 0
        while True:
            if max_vectors is not None and count >= max_vectors:
                break
            dim_bytes = f_in.read(4)
            if not dim_bytes or len(dim_bytes) < 4:
                break
            dim = struct.unpack("i", dim_bytes)[0]
            vec_bytes = f_in.read(dim * 4)
            if len(vec_bytes) < dim * 4:
                break
            f_out.write(vec_bytes)
            count += 1
            if count % 200000 == 0:
                print(f"  Processed {count} vectors...", flush=True)
                
    print(f"Done: {count} vectors written to {out_bin_path} ({os.path.getsize(out_bin_path)} bytes)")

def main():
    # SIFT
    sift_base = os.path.expanduser("~/trident/data/sift1m/fvecs/sift1m_base.fvecs")
    sift_query = os.path.expanduser("~/trident/data/sift1m/fvecs/sift1m_query.fvecs")
    convert_fvecs_to_raw_bin(sift_base, os.path.join(DATA_DIR, "data_sift.bin"), 1000000)
    convert_fvecs_to_raw_bin(sift_query, os.path.join(DATA_DIR, "data_sift_test.bin"), 1000)

    # GIST
    gist_base = os.path.expanduser("~/trident/data/gist1m/fvecs/gist1m_base.fvecs")
    gist_query = os.path.expanduser("~/trident/data/gist1m/fvecs/gist1m_query.fvecs")
    convert_fvecs_to_raw_bin(gist_base, os.path.join(DATA_DIR, "data_gist.bin"), 1000000)
    convert_fvecs_to_raw_bin(gist_query, os.path.join(DATA_DIR, "data_gist_test.bin"), 1000)
    
if __name__ == "__main__":
    main()
