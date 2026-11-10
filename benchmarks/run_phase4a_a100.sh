#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda-13
export PATH="/usr/local/cuda-13/bin:$PATH"
export TORCH_CUDA_ARCH_LIST='8.0;8.9'
cd ~/trident
mkdir -p logs results/phase4a

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 4A Suite on A100..."

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 1. Ablation (5 seeds)"
/home/mazaiying/trident/.venv/bin/python scripts/phase4a_suite.py --mode ablation --gpu_tag A100 > logs/phase4a_ablation_A100.log 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 2. Recall 0.99 (5 seeds)"
/home/mazaiying/trident/.venv/bin/python scripts/phase4a_suite.py --mode recall99 --gpu_tag A100 > logs/phase4a_recall99_A100.log 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 3. QPS-Recall Pareto (3 seeds)"
/home/mazaiying/trident/.venv/bin/python scripts/phase4a_suite.py --mode qps_recall --gpu_tag A100 > logs/phase4a_qps_recall_A100.log 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 4. Prune Stats (A100)"
/home/mazaiying/trident/.venv/bin/python scripts/phase4a_suite.py --mode prune_stats --gpu_tag A100 > logs/phase4a_prune_A100.log 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 5. Build Time & Memory Footprint (A100)"
/home/mazaiying/trident/.venv/bin/python scripts/phase4a_suite.py --mode build_mem --gpu_tag A100 > logs/phase4a_build_A100.log 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 6. TRIDENT-Auto (A100)"
/home/mazaiying/trident/.venv/bin/python scripts/phase4a_suite.py --mode auto --gpu_tag A100 > logs/phase4a_auto_A100.log 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 7. NCU Hardware Profiling (A100)"
/home/mazaiying/trident/.venv/bin/python scripts/phase4a_ncu_runner.py > logs/phase4a_ncu_A100.log 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All Phase 4A tasks on A100 completed successfully!"
