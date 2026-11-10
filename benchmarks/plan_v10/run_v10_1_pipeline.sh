#!/usr/bin/env bash
set -euo pipefail

LOG=~/trident/results/phase10/pipeline_v10_1.log
mkdir -p ~/trident/results/phase10
cd ~/trident/scripts/plan_v10

echo "[$(date '+%F %T')] STARTING STEP E1 (ooc_1m10m_A100.csv on GPU 1)..." > "$LOG"
rm -f ~/trident/results/phase10/ooc_1m10m_A100.csv
/home/mazaiying/trident/.venv/bin/python phase10_ooc_1m10m.py --gpu 1 --part ooc > ~/trident/results/phase10/e1.log 2>&1
echo "[$(date '+%F %T')] FINISHED STEP E1" >> "$LOG"

echo "[$(date '+%F %T')] STARTING STEP N1 (ncu_perpass_1m10m_A100.csv on GPU 1)..." >> "$LOG"
rm -f ~/trident/results/phase10/ncu_perpass_1m10m_A100.csv
/home/mazaiying/trident/.venv/bin/python phase10_ooc_1m10m.py --gpu 1 --part ncu > ~/trident/results/phase10/n1.log 2>&1
echo "[$(date '+%F %T')] FINISHED STEP N1" >> "$LOG"

echo "[$(date '+%F %T')] STARTING STEP E2 (ooc_100m_RTX4080_v10.csv on GPU 0)..." >> "$LOG"
rm -f ~/trident/results/phase10/ooc_100m_RTX4080_v10.csv
/home/mazaiying/trident/.venv/bin/python phase10_ooc_100m.py --part main > ~/trident/results/phase10/e2.log 2>&1
echo "[$(date '+%F %T')] FINISHED STEP E2" >> "$LOG"

echo "[$(date '+%F %T')] STARTING STEP N2 (ncu_perpass_100m_RTX4080.csv on GPU 0)..." >> "$LOG"
rm -f ~/trident/results/phase10/ncu_perpass_100m_RTX4080.csv
/home/mazaiying/trident/.venv/bin/python phase10_ooc_100m.py --part ncu > ~/trident/results/phase10/n2.log 2>&1
echo "[$(date '+%F %T')] FINISHED STEP N2" >> "$LOG"

echo "[$(date '+%F %T')] STARTING STEP E3 (gpu_baselines_100m_RTX4080.jsonl on GPU 0)..." >> "$LOG"
rm -f ~/trident/results/phase10/gpu_baselines_100m_RTX4080.jsonl
chmod +x ~/trident/scripts/plan_v10/run_rapids_env.sh
~/trident/scripts/plan_v10/run_rapids_env.sh /home/mazaiying/trident/.venv/bin/python phase10_gpu_baselines_100m.py > ~/trident/results/phase10/e3.log 2>&1
echo "[$(date '+%F %T')] FINISHED STEP E3 — ALL PHASE 10 EXPERIMENTS COMPLETE" >> "$LOG"
