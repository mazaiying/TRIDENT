#!/usr/bin/env bash
set -euo pipefail

cd ~/trident
mkdir -p ~/trident/logs ~/trident/results/phase2b
rm -f ~/trident/results/phase2b/phase2b_A100_smoke.csv ~/trident/results/phase2b/phase2b_A100_v2check.csv

PY="/home/mazaiying/trident/.venv/bin/python"
CACHE_DIR="$HOME/trident/results/phase2b/.ivf_cache"

echo "=== [1/3] Checking GPU 1 (A100) idle status ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv

echo "=== [1/3] Running Full Phase 2B on GPU 1 (A100) ==="
CUDA_VISIBLE_DEVICES=1 "$PY" ~/trident/scripts/phase2b_benchmark.py \
  --datasets gist1m bge1024 sift1m msong glove100 nuswide \
  --seeds 0 1 2 \
  --batches 1 32 1024 \
  --gpu_tag A100 \
  --cache_dir "$CACHE_DIR" \
  2>&1 | tee ~/trident/logs/phase2b_A100.log

echo "=== [2/3] Checking GPU 0 (RTX 4080) idle status ==="
GPU0_UTIL=$(nvidia-smi --id=0 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
echo "GPU 0 utilization: ${GPU0_UTIL}%"
if [ "$GPU0_UTIL" -gt 10 ]; then
  echo "WARNING: GPU 0 busy (${GPU0_UTIL}%), waiting until idle..."
  while [ "$(nvidia-smi --id=0 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')" -gt 10 ]; do
    sleep 30
  done
fi

echo "=== [2/3] Running Full Phase 2B on GPU 0 (RTX 4080) ==="
CUDA_VISIBLE_DEVICES=0 "$PY" ~/trident/scripts/phase2b_benchmark.py \
  --datasets gist1m bge1024 sift1m msong glove100 nuswide \
  --seeds 0 1 2 \
  --batches 1 32 1024 \
  --gpu_tag RTX4080 \
  --cache_dir "$CACHE_DIR" \
  2>&1 | tee ~/trident/logs/phase2b_RTX4080.log

echo "=== [3/3] Running Nsight Compute (ncu) HBM Read Bytes Profiling on GPU 1 (A100) ==="
CUDA_VISIBLE_DEVICES=1 "$PY" ~/trident/scripts/run_phase2b_ncu.py \
  2>&1 | tee ~/trident/logs/phase2b_ncu.log

rm -rf "$CACHE_DIR"
echo "=== PHASE2B_ALL_COMPLETE ==="
