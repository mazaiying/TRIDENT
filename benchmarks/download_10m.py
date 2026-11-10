import os
import sys
import struct
import urllib.request
import time

DATA_DIR = "/Volumes/Elements SE/科研/icde0117/data_10m"
os.makedirs(os.path.join(DATA_DIR, "sift10m"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "deep10m"), exist_ok=True)

def download_file(url, out_path):
    print(f"Downloading {url} -> {out_path}...")
    t0 = time.time()
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(out_path, 'wb') as f:
        total = int(resp.headers.get('content-length', 0))
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0 and downloaded % (10 * 1024 * 1024) < 1024 * 1024:
                sys.stdout.write(f"\r{downloaded / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB ({downloaded*100.0/total:.1f}%)")
                sys.stdout.flush()
    print(f"\nDone in {time.time()-t0:.1f}s")

def download_range(url, out_path, num_points, dim, bytes_per_elem, start_byte=8):
    total_data_bytes = num_points * dim * bytes_per_elem
    total_expected_size = 8 + total_data_bytes
    print(f"Target file: {out_path}, expected size: {total_expected_size} bytes ({total_expected_size/1024/1024:.1f} MB)")

    # Write header if file does not exist or is too short
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 8:
        with open(out_path, 'wb') as f:
            f.write(struct.pack('<II', num_points, dim))

    while True:
        curr_size = os.path.getsize(out_path)
        if curr_size >= total_expected_size:
            print(f"File {out_path} already complete ({curr_size} bytes).")
            break
        
        # We need data from curr_size to total_expected_size - 1
        req_start = curr_size
        req_end = total_expected_size - 1
        print(f"\nResuming download from byte {req_start} to {req_end} (remaining: {(req_end - req_start + 1)/1024/1024:.1f} MB)...")
        
        t0 = time.time()
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)',
            'Range': f'bytes={req_start}-{req_end}'
        })
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, open(out_path, 'ab') as f:
                downloaded_session = 0
                while True:
                    chunk = resp.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    f.flush()
                    downloaded_session += len(chunk)
                    curr_total = curr_size + downloaded_session
                    pct = (curr_total * 100.0) / total_expected_size
                    speed = (downloaded_session / (1024 * 1024)) / max(time.time() - t0, 1e-3)
                    sys.stdout.write(f"\rTotal: {curr_total / 1024 / 1024:.1f} / {total_expected_size / 1024 / 1024:.1f} MB ({pct:.1f}%) - Speed: {speed:.1f} MB/s")
                    sys.stdout.flush()
        except Exception as e:
            print(f"\n[Warning] Connection dropped: {e}. Retrying in 3 seconds...")
            time.sleep(3)
            continue
            
    print(f"\nFinished: {out_path} ({os.path.getsize(out_path)} bytes)")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target in ["all", "sift_query"]:
        download_file("https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin",
                      os.path.join(DATA_DIR, "sift10m", "query.public.10K.u8bin"))
    if target in ["all", "sift_base"]:
        download_range("https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin",
                       os.path.join(DATA_DIR, "sift10m", "base.10M.u8bin"),
                       num_points=10000000, dim=128, bytes_per_elem=1)
    if target in ["all", "deep_query"]:
        download_file("https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin",
                      os.path.join(DATA_DIR, "deep10m", "query.public.10K.fbin"))
    if target in ["all", "deep_base"]:
        download_range("https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin",
                       os.path.join(DATA_DIR, "deep10m", "base.10M.fbin"),
                       num_points=10000000, dim=96, bytes_per_elem=4)
    if target in ["all", "t2i_query"]:
        os.makedirs(os.path.join(DATA_DIR, "t2i10m"), exist_ok=True)
        download_file("https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin",
                      os.path.join(DATA_DIR, "t2i10m", "query.public.100K.fbin"))
    if target in ["all", "t2i_base"]:
        os.makedirs(os.path.join(DATA_DIR, "t2i10m"), exist_ok=True)
        download_range("https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin",
                       os.path.join(DATA_DIR, "t2i10m", "base.10M.fbin"),
                       num_points=10000000, dim=200, bytes_per_elem=4)
