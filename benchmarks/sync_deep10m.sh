#!/bin/bash
SRC="/Volumes/Elements SE/科研/icde0117/data_10m/deep10m/base.10M.fbin"
DEST="mazaiying@10.77.110.167:/home/mazaiying/trident/data/deep10m/base.10M.fbin"

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Syncing base.10M.fbin with rsync resume..."
    sshpass -e rsync -avP --partial --inplace \
        -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3" \
        "$SRC" "$DEST"
    RET=$?
    if [ $RET -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Transfer finished successfully!"
        break
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] rsync exited with $RET, retrying in 3s..."
        sleep 3
    fi
done
