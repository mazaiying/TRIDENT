#!/usr/bin/env bash
set -euo pipefail

mkdir -p ~/trident/results/phase10

echo "[1/3] Compiling V10.1 CUDA extension for sm_80;sm_89 ..." > ~/trident/results/phase10/k2_v10_1.log
TORCH_CUDA_ARCH_LIST="8.0;8.9" CUDA_VISIBLE_DEVICES=1 /home/mazaiying/trident/.venv/bin/python -c "import sys; sys.path.insert(0, '/home/mazaiying/trident/scripts'); import phase4a_suite as s; s.get_cuda_ext(); print('COMPILE_OK')" >> ~/trident/results/phase10/k2_v10_1.log 2>&1

echo "[2/3] Running diag_e0_qps.py --kernel patched ..." >> ~/trident/results/phase10/k2_v10_1.log
cd ~/trident/scripts/plan_v10
/home/mazaiying/trident/.venv/bin/python diag_e0_qps.py --kernel patched > ~/trident/results/phase10/diag_patched_v10_1.log 2>&1

echo "[3/3] Running Step E0 (lossless_A100.csv) ..." >> ~/trident/results/phase10/k2_v10_1.log
mv -f ~/trident/results/phase10/lossless_A100.csv ~/trident/results/phase10/lossless_A100_v10_0.csv 2>/dev/null || true
/home/mazaiying/trident/.venv/bin/python phase10_ooc_1m10m.py --gpu 1 --part lossless > ~/trident/results/phase10/e0.log 2>&1

echo "ALL_V10_1_E0_DONE" >> ~/trident/results/phase10/k2_v10_1.log
