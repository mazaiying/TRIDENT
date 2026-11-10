#!/usr/bin/env python3
"""
Fast 32-thread HTTP Range downloader for SIFT100M (first 100,000,000 vectors of BIGANN 1B)
from https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin
Reuses the first 10M vectors already present in ~/trident/data/sift10m/base.10M.u8bin,
downloads vectors 10M..100M in 8 MB chunks across 32 threads with automatic retry,
and then computes exact top-100 L2 ground truth on A100 (CUDA_VISIBLE_DEVICES=1).
"""
import os, sys, time, struct, shutil, urllib.request, concurrent.futures
import numpy as np

# Bypass broken campus HTTP proxy
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(k, None)

URL_1B = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin"
DIR_10M = os.path.expanduser("~/trident/data/sift10m")
DIR_100M = os.path.expanduser("~/trident/data/sift100m")
os.makedirs(DIR_100M, exist_ok=True)

TARGET_U8BIN = os.path.join(DIR_100M, "base.100M.u8bin")
QUERY_U8BIN = os.path.join(DIR_100M, "query.public.10K.u8bin")
QUERY_NPY = os.path.join(DIR_100M, "query.npy")
GT_NPY = os.path.join(DIR_100M, "gt.npy")

N_100M = 100_000_000
D = 128
TOTAL_BYTES = 8 + N_100M * D  # 12,800,000,008 bytes
PREFIX_10M_BYTES = 8 + 10_000_000 * D  # 1,280,000,008 bytes
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB per chunk

def download_sift100m():
    if not os.path.exists(QUERY_U8BIN):
        shutil.copyfile(os.path.join(DIR_10M, "query.public.10K.u8bin"), QUERY_U8BIN)
    if not os.path.exists(QUERY_NPY):
        shutil.copyfile(os.path.join(DIR_10M, "query.npy"), QUERY_NPY)

    # Preallocate or resume TARGET_U8BIN
    need_prefix = True
    if os.path.exists(TARGET_U8BIN) and os.path.getsize(TARGET_U8BIN) == TOTAL_BYTES:
        need_prefix = False
    else:
        print(f"[SIFT100M] Initializing {TARGET_U8BIN} ({TOTAL_BYTES / (1024**3):.2f} GiB)...", flush=True)
        with open(TARGET_U8BIN, "wb") as f:
            f.truncate(TOTAL_BYTES)
            # Write header N=100,000,000, D=128
            f.seek(0)
            f.write(struct.pack("<II", N_100M, D))
            # Copy first 10M vectors from sift10m/base.10M.u8bin
            src_10m = os.path.join(DIR_10M, "base.10M.u8bin")
            if os.path.exists(src_10m) and os.path.getsize(src_10m) == PREFIX_10M_BYTES:
                print("[SIFT100M] Copying first 10M vectors (1.28 GB) from sift10m/base.10M.u8bin...", flush=True)
                with open(src_10m, "rb") as sf:
                    sf.seek(8)
                    shutil.copyfileobj(sf, f, length=64 * 1024 * 1024)

    # Track completed chunks in a state file
    state_file = os.path.join(DIR_100M, "chunks_done.txt")
    done_chunks = set()
    if os.path.exists(state_file):
        with open(state_file, "r") as sf:
            for line in sf:
                if line.strip():
                    done_chunks.add(int(line.strip()))

    # Build list of byte ranges from PREFIX_10M_BYTES to TOTAL_BYTES
    ranges = []
    pos = PREFIX_10M_BYTES
    cid = 0
    while pos < TOTAL_BYTES:
        end = min(pos + CHUNK_SIZE, TOTAL_BYTES) - 1
        if cid not in done_chunks:
            ranges.append((cid, pos, end))
        pos = end + 1
        cid += 1

    total_chunks = cid
    print(f"[SIFT100M] Total chunks (10M..100M): {total_chunks}, remaining: {len(ranges)}", flush=True)

    if ranges:
        t0 = time.time()
        bytes_dl = 0
        fd = os.open(TARGET_U8BIN, os.O_RDWR)
        state_fp = open(state_file, "a", buffering=1)

        def fetch_chunk(item):
            c_idx, start_b, end_b = item
            expected_len = end_b - start_b + 1
            for attempt in range(8):
                try:
                    req = urllib.request.Request(URL_1B, headers={"Range": f"bytes={start_b}-{end_b}"})
                    with urllib.request.urlopen(req, timeout=25) as resp:
                        data = resp.read()
                    if len(data) == expected_len:
                        os.pwrite(fd, data, start_b)
                        return c_idx, expected_len
                except Exception as e:
                    time.sleep(0.5 * (attempt + 1))
            raise RuntimeError(f"Failed chunk {c_idx} ({start_b}-{end_b})")

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
            futs = [ex.submit(fetch_chunk, r) for r in ranges]
            done_cnt = 0
            for fut in concurrent.futures.as_completed(futs):
                c_idx, b_len = fut.result()
                bytes_dl += b_len
                done_cnt += 1
                state_fp.write(f"{c_idx}\n")
                if done_cnt % 50 == 0 or done_cnt == len(ranges):
                    dt = max(time.time() - t0, 1e-3)
                    spd = (bytes_dl / (1024**2)) / dt
                    pct = 100.0 * (len(done_chunks) + done_cnt) / total_chunks
                    print(f"[SIFT100M] Progress: {pct:.1f}% ({done_cnt}/{len(ranges)} chunks), Speed: {spd:.2f} MB/s, Elapsed: {dt:.0f}s", flush=True)

        os.close(fd)
        state_fp.close()
    print("[SIFT100M] base.100M.u8bin download complete!", flush=True)

def compute_gt_100m_on_a100():
    if os.path.exists(GT_NPY):
        gt = np.load(GT_NPY)
        if gt.shape == (10000, 100):
            print("[SIFT100M] gt.npy already exists and verified:", gt.shape, flush=True)
            return
    print("[SIFT100M] Computing exact top-100 L2 Ground Truth on A100 80GB...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use A100 80GB for fast exact GT
    import faiss
    xq = np.ascontiguousarray(np.load(QUERY_NPY).astype(np.float32))
    Q, d = xq.shape
    k = 100
    xb_u8 = np.memmap(TARGET_U8BIN, dtype=np.uint8, mode="r", offset=8, shape=(N_100M, D))

    res = faiss.StandardGpuResources()
    all_dists = np.full((Q, k), np.inf, dtype=np.float32)
    all_indices = np.full((Q, k), -1, dtype=np.int64)

    block_size = 20_000_000  # 20M vectors = 10.24 GB FP32 per block on A100 80GB
    t0 = time.time()
    for b_start in range(0, N_100M, block_size):
        b_end = min(b_start + block_size, N_100M)
        tb = time.time()
        xb_blk = np.ascontiguousarray(xb_u8[b_start:b_end], dtype=np.float32)
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = 0
        cfg.useFloat16 = False
        gpu_sub = faiss.GpuIndexFlatL2(res, d, cfg)
        gpu_sub.add(xb_blk)
        D_sub, I_sub = gpu_sub.search(xq, k)
        I_sub += b_start

        cur_d = np.hstack([all_dists, D_sub])
        cur_i = np.hstack([all_indices, I_sub])
        sort_idx = np.argsort(cur_d, axis=1)[:, :k]
        all_dists = np.take_along_axis(cur_d, sort_idx, axis=1)
        all_indices = np.take_along_axis(cur_i, sort_idx, axis=1)
        del gpu_sub, xb_blk
        print(f"  [GT] Block [{b_start}:{b_end}] done in {time.time()-tb:.2f}s", flush=True)

    np.save(GT_NPY, all_indices.astype(np.int64))
    print(f"[SIFT100M] Saved exact GT {all_indices.shape} to {GT_NPY} in {time.time()-t0:.2f}s!", flush=True)

if __name__ == "__main__":
    download_sift100m()
    compute_gt_100m_on_a100()
