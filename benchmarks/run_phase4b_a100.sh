#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda-13
export PATH="/usr/local/cuda-13/bin:$PATH"
export TORCH_CUDA_ARCH_LIST='8.0;8.9'
cd ~/trident
mkdir -p logs results/phase4b

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 4B (10M Scale Suite) on A100..."

# 1. Compute exact GPU Ground Truth (if not already done)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 1. Exact GPU Ground Truth Computation"
/home/mazaiying/trident/.venv/bin/python scripts/compute_gt_10m.py all > logs/phase4b_gt_compute.log 2>&1

# 2. Main 10M Benchmark Suite (SIFT10M, Deep10M)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 2. Running 10M Benchmark Suite"
/home/mazaiying/trident/.venv/bin/python scripts/phase4b_suite.py --gpu_tag A100 --datasets sift10m deep10m --seeds 0 1 2 --nlists 4096 > logs/phase4b_main_A100.log 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 4B benchmark completed successfully!"
