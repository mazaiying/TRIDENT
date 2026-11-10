#!/usr/bin/env python3
"""
PLAN_V8 Step 1:
  Properly enforced physical-VRAM-cap CAGRA evaluation using two separate processes:
    - Process A (--stage build): Unconstrained build & save of CAGRA indices (cagra.save)
      at the exact R=0.99 configuration from cuvs_qps_recall.csv, plus exact IVF-Flat
      reference top-10 results for strict overlap evaluation.
    - Process B (--stage worker): Fresh process per (dataset, cap_gb, mode) that:
      1. Pre-warms CAGRA search cubins and cudaMallocAsync mempool, then frees temporary buffers.
      2. Allocates physical HBM blockers in chunks via CudaMemoryResource until free <= cap_gb.
      3. Sets ManagedMemoryResource() and calls cagra.load().
      4. Verifies free_after_load_gb <= cap_gb.
      5. Measures cold_qps (1 run), warmup (1 run), steady_qps_mean & steady_qps_std (5 runs),
         recall_at_10, and strict_overlap_pct against exact IVF-Flat.
    - Also evaluates UVM_uncapped_control (cap_gb = 75.0) and VPQ_hardcap (CAGRA-Q / VPQ
      under 2.2 GB and 2.5 GB LimitingResourceAdaptor).
Outputs:
  - ~/trident/results/phase7/cagra_uvm_capped_A100.csv
"""
import os, sys, time, csv, gc, json, math, ctypes, argparse, subprocess, statistics
import numpy as np
import torch, faiss, rmm
from cuvs.neighbors import cagra

SCRIPT_DIR = os.path.expanduser("~/trident/scripts")
sys.path.insert(0, SCRIPT_DIR)
from phase4a_suite import eval_accuracy_v4, load_or_train_clustering
from phase4b_suite import load_or_train_clustering_10m

DATA_ROOT = os.path.expanduser("~/trident/data")
IDX_DIR = os.path.expanduser("~/trident/indices/phase7_cagra")
RES_DIR = os.path.expanduser("~/trident/results/phase7")
os.makedirs(IDX_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
OUT_CSV = os.path.join(RES_DIR, "cagra_uvm_capped_A100.csv")

# Exact R=0.99 configs from cuvs_qps_recall.csv and ooc.csv caps
DATASET_SPECS = {
    "gist1m":  {"dim": 960,  "nlist": 1024, "nprobe_r99": 128, "gdeg": 64, "itopk_r99": 512, "caps": [2.2, 2.5, 75.0]},
    "bge1024": {"dim": 1024, "nlist": 1024, "nprobe_r99": 128, "gdeg": 32, "itopk_r99": 256, "caps": [2.2, 2.5, 75.0]},
    "sift10m": {"dim": 128,  "nlist": 4096, "nprobe_r99": 96,  "gdeg": 64, "itopk_r99": 128, "caps": [2.5, 3.0, 75.0]},
    "deep10m": {"dim": 96,   "nlist": 4096, "nprobe_r99": 96,  "gdeg": 64, "itopk_r99": 128, "caps": [2.5, 75.0]},
}


class RMMArray:
    def __init__(self, arr_np: np.ndarray, mr):
        self.shape = arr_np.shape
        self.dtype = arr_np.dtype
        self.buf = rmm.DeviceBuffer(size=arr_np.nbytes, mr=mr)
        self.buf.copy_from_host(arr_np.view(np.uint8).ravel())

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": self.shape,
            "typestr": self.dtype.str,
            "data": (self.buf.ptr, False),
            "version": 3,
        }


def build_and_save_all():
    dev = torch.device("cuda:0")
    for ds, spec in DATASET_SPECS.items():
        idx_path = os.path.join(IDX_DIR, f"cagra_{ds}_deg{spec['gdeg']}.bin")
        ref_path = os.path.join(IDX_DIR, f"ref_ivfflat_{ds}.npz")
        ds_dir = os.path.join(DATA_ROOT, ds)
        xb = np.ascontiguousarray(np.load(os.path.join(ds_dir, "base.npy"), mmap_mode="r")[:10000000].astype(np.float32))
        xq = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy"))[:1000].astype(np.float32))
        N, D = xb.shape

        if not os.path.exists(ref_path):
            print(f"[Process A] Computing exact IVF-Flat reference for {ds} (nprobe={spec['nprobe_r99']})...", flush=True)
            if N >= 10000000:
                centroids, _, _, _, _, _ = load_or_train_clustering_10m(ds, xb, spec["nlist"], seed=0)
            else:
                centroids, _, _, _, _, _ = load_or_train_clustering(ds, xb, spec["nlist"], seed=0)
            res = faiss.StandardGpuResources()
            q = faiss.IndexFlatL2(D)
            cpu_flat = faiss.IndexIVFFlat(q, D, spec["nlist"], faiss.METRIC_L2)
            cpu_flat.quantizer.add(centroids)
            cpu_flat.is_trained = True
            gpu_flat = faiss.index_cpu_to_gpu(res, 0, cpu_flat)
            gpu_flat.add(xb)
            gpu_flat.nprobe = spec["nprobe_r99"]
            ref_d, ref_i = gpu_flat.search(xq, 10)
            np.savez(ref_path, ref_d=ref_d, ref_i=ref_i)
            del gpu_flat, cpu_flat, res
            gc.collect()
            torch.cuda.empty_cache()

        if not os.path.exists(idx_path):
            print(f"[Process A] Building & saving CAGRA index for {ds} (deg={spec['gdeg']})...", flush=True)
            xb_g = torch.from_numpy(xb).to(dev, dtype=torch.float32).contiguous()
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            ip = cagra.IndexParams(
                metric="sqeuclidean",
                graph_degree=spec["gdeg"],
                intermediate_graph_degree=spec["gdeg"] * 2,
                build_algo="nn_descent",
            )
            idx = cagra.build(ip, xb_g)
            torch.cuda.synchronize(dev)
            build_s = time.perf_counter() - t0
            cagra.save(idx_path, idx, include_dataset=True)
            fsize_gb = os.path.getsize(idx_path) / (1024 ** 3)
            print(f"[Process A] Saved {idx_path} ({fsize_gb:.3f} GiB) in {build_s:.2f}s", flush=True)
            del idx, xb_g
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print(f"[Process A] Reusing existing {idx_path} ({os.path.getsize(idx_path)/(1024**3):.3f} GiB)", flush=True)


def run_worker_uvm(ds: str, cap_gb: float, mode: str):
    spec = DATASET_SPECS[ds]
    D = spec["dim"]
    gdeg = spec["gdeg"]
    idx_path = os.path.join(IDX_DIR, f"cagra_{ds}_deg{gdeg}.bin")
    ref_path = os.path.join(IDX_DIR, f"ref_ivfflat_{ds}.npz")
    ds_dir = os.path.join(DATA_ROOT, ds)

    xq = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy"))[:1000].astype(np.float32))
    gt10 = np.load(os.path.join(ds_dir, "gt.npy"))[:1000, :10]
    ref_npz = np.load(ref_path)
    ref_d, ref_i = ref_npz["ref_d"], ref_npz["ref_i"]
    xq_1024 = np.concatenate([xq, xq[:24]], axis=0)

    torch.cuda.init()
    uvm_mr = rmm.mr.ManagedMemoryResource()
    rmm.mr.set_current_device_resource(uvm_mr)

    # 1. Warmup CAGRA single_cta and multi_cta cubins + cudaMallocAsync default pool before locking HBM
    dx = RMMArray(np.random.randn(512, D).astype(np.float32), uvm_mr)
    d_idx = cagra.build(
        cagra.IndexParams(metric="sqeuclidean", graph_degree=gdeg, intermediate_graph_degree=gdeg * 2, build_algo="nn_descent"),
        dx,
    )
    for bsz in [64, 128, 1024]:
        dq = RMMArray(np.random.randn(bsz, D).astype(np.float32), uvm_mr)
        for itk in sorted(list(set([128, spec["itopk_r99"]]))):
            _ = cagra.search(cagra.SearchParams(itopk_size=itk), d_idx, dq, 10)
    torch.cuda.synchronize()
    del d_idx, dx, dq
    gc.collect()
    torch.cuda.empty_cache()

    libcudart = ctypes.CDLL("libcudart.so")
    mempool = ctypes.c_void_p()
    libcudart.cudaDeviceGetDefaultMemPool(ctypes.byref(mempool), ctypes.c_int(0))
    thresh = ctypes.c_uint64(0xFFFFFFFFFFFFFFFF)
    libcudart.cudaMemPoolSetAttribute(mempool, ctypes.c_int(4), ctypes.byref(thresh))
    ptr = ctypes.c_void_p()
    libcudart.cudaMallocAsync(ctypes.byref(ptr), ctypes.c_size_t(64 * 1024 * 1024), ctypes.c_void_p(0))
    libcudart.cudaFreeAsync(ptr, ctypes.c_void_p(0))
    torch.cuda.synchronize()

    # 2. Lock physical HBM via CudaMemoryResource chunks so total free HBM (including 48 MiB ws_guard) <= cap_gb
    cuda_mr = rmm.mr.CudaMemoryResource()
    ws_guard_bytes = 48 * (1024 ** 2)
    ws_guard = rmm.DeviceBuffer(size=ws_guard_bytes, mr=cuda_mr)
    chunks = []
    target_free_bytes = max(0, int(cap_gb * (1024 ** 3)) - ws_guard_bytes)
    for step_mb in [1024, 256, 64, 16]:
        step_b = step_mb * (1024 ** 2)
        while True:
            f_b, _ = torch.cuda.mem_get_info()
            if f_b - step_b < target_free_bytes:
                break
            try:
                b = rmm.DeviceBuffer(size=step_b, mr=cuda_mr)
                chunks.append(b)
            except Exception:
                break

    f_before, _ = torch.cuda.mem_get_info()
    total_cap_effective_gb = (f_before + ws_guard_bytes) / (1024 ** 3)
    print(f"[Worker {ds} mode={mode} cap={cap_gb}GB] Effective free before load={total_cap_effective_gb:.3f} GiB", flush=True)

    # 3. Load index into ManagedMemoryResource
    results = []
    itopk_list = sorted(list(set([128, spec["itopk_r99"]])))
    try:
        idx = cagra.load(idx_path)
        del ws_guard
        f_after, _ = torch.cuda.mem_get_info()
        free_after_gb = f_after / (1024 ** 3)
        assert free_after_gb <= cap_gb + 0.05, f"Self-check failed: free_after_gb={free_after_gb} > cap_gb={cap_gb}"
        print(f"[Worker {ds} cap={cap_gb}GB] Loaded index via UVM! free_after_load={free_after_gb:.3f} GiB", flush=True)
    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
        print(f"[Worker {ds} cap={cap_gb}GB] UVM_LOAD_FAILED: {err_msg}", flush=True)
        for itk in itopk_list:
            results.append([
                ds, cap_gb, mode, gdeg, itk, -1.0, f"UVM_LOAD_FAILED ({err_msg})",
                0.0, 0.0, 0.0, 0.0, 0.0, 1
            ])
        return results

    # 4. Run search timing (1 cold, 1 warmup, 5 steady runs)
    xq_1024_uvm = RMMArray(xq_1024, uvm_mr)
    xq_128_uvm = RMMArray(xq[:128], uvm_mr)
    xq_64_uvm = RMMArray(xq[:64], uvm_mr)
    for itk in itopk_list:
        sp = cagra.SearchParams(itopk_size=itk)
        try:
            if cap_gb < 10.0:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                d_eval, i_eval = cagra.search(sp, idx, xq_128_uvm, 10)
                torch.cuda.synchronize()
                cold_s = time.perf_counter() - t0
                if cold_s < 2.0:
                    # Fast enough to run full B=1024
                    t0 = time.perf_counter()
                    d_eval, i_eval = cagra.search(sp, idx, xq_1024_uvm, 10)
                    torch.cuda.synchronize()
                    cold_s = time.perf_counter() - t0
                    b_cold = 1024
                    n_eval_q = 1000
                else:
                    b_cold = 128
                    n_eval_q = 128
            else:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                d_eval, i_eval = cagra.search(sp, idx, xq_1024_uvm, 10)
                torch.cuda.synchronize()
                cold_s = time.perf_counter() - t0
                b_cold = 1024
                n_eval_q = 1000

            cold_qps = float(b_cold) / max(cold_s, 1e-6)
            print(f"[Worker {ds} cap={cap_gb}GB itopk={itk}] Cold (B={b_cold}) took {cold_s:.3f}s -> cold_qps={cold_qps:.1f}", flush=True)

            if b_cold == 128 and cold_s > 8.0:
                b_steady = 64
                q_steady_arr = xq_64_uvm
            elif b_cold == 128:
                b_steady = 128
                q_steady_arr = xq_128_uvm
            else:
                b_steady = 1024
                q_steady_arr = xq_1024_uvm

            i_np = torch.as_tensor(i_eval, device="cuda").to(torch.int64).cpu().numpy()[:n_eval_q]
            d_np = torch.as_tensor(d_eval, device="cuda").cpu().numpy()[:n_eval_q]

            # Warmup 1 run
            _ = cagra.search(sp, idx, q_steady_arr, 10)
            torch.cuda.synchronize()

            n_steady = 5
            steady_qps_list = []
            for s_idx in range(n_steady):
                ts = time.perf_counter()
                _ = cagra.search(sp, idx, q_steady_arr, 10)
                torch.cuda.synchronize()
                dt_s = time.perf_counter() - ts
                q_s = float(b_steady) / max(dt_s, 1e-6)
                steady_qps_list.append(q_s)
                print(f"  steady[{s_idx+1}/{n_steady}] (B={b_steady}) dt={dt_s:.3f}s -> qps={q_s:.1f}", flush=True)

            rec, strict_ov, _ = eval_accuracy_v4(i_np, d_np, ref_i[:n_eval_q], ref_d[:n_eval_q], gt10[:n_eval_q])
            s_mean = statistics.mean(steady_qps_list)
            s_std = statistics.stdev(steady_qps_list) if len(steady_qps_list) > 1 else 0.0
            row = [
                ds, cap_gb, mode, gdeg, itk, round(free_after_gb, 3), "OK",
                round(cold_qps, 1), round(s_mean, 1), round(s_std, 1),
                round(rec, 4), round(strict_ov, 2), 1
            ]
            print("UVM_RESULT:", row, flush=True)
            results.append(row)
        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
            print(f"[Worker {ds} cap={cap_gb}GB itopk={itk}] UVM_SEARCH_FAILED: {err_msg}", flush=True)
            results.append([
                ds, cap_gb, mode, gdeg, itk, round(free_after_gb, 3), f"UVM_SEARCH_FAILED ({err_msg})",
                0.0, 0.0, 0.0, 0.0, 0.0, 1
            ])
    return results


def run_worker_vpq(ds: str, cap_gb: float):
    spec = DATASET_SPECS[ds]
    D = spec["dim"]
    gdeg = spec["gdeg"]
    ref_path = os.path.join(IDX_DIR, f"ref_ivfflat_{ds}.npz")
    ds_dir = os.path.join(DATA_ROOT, ds)

    xb = np.ascontiguousarray(np.load(os.path.join(ds_dir, "base.npy"), mmap_mode="r")[:10000000].astype(np.float32))
    xq = np.ascontiguousarray(np.load(os.path.join(ds_dir, "query.npy"))[:1000].astype(np.float32))
    gt10 = np.load(os.path.join(ds_dir, "gt.npy"))[:1000, :10]
    ref_npz = np.load(ref_path)
    ref_d, ref_i = ref_npz["ref_d"], ref_npz["ref_i"]
    xq_1024 = np.concatenate([xq, xq[:24]], axis=0)

    # Build VPQ CAGRA index first and save to disk, then load under hard LimitingResourceAdaptor(cap_gb)
    pq_dim = max(8, (D // 4 // 8) * 8)
    vpq_path = os.path.join(IDX_DIR, f"cagra_vpq_{ds}_deg{gdeg}_pq{pq_dim}.bin")
    dev = torch.device("cuda:0")
    if not os.path.exists(vpq_path):
        print(f"[VPQ {ds}] Building CAGRA-VPQ (pq_dim={pq_dim}, pq_bits=8)...", flush=True)
        xb_g = torch.from_numpy(xb).to(dev, dtype=torch.float32).contiguous()
        comp = cagra.CompressionParams(pq_bits=8, pq_dim=pq_dim, vq_n_centers=1024)
        ip = cagra.IndexParams(
            metric="sqeuclidean",
            graph_degree=gdeg,
            intermediate_graph_degree=gdeg * 2,
            build_algo="nn_descent",
            compression=comp,
        )
        idx_vpq = cagra.build(ip, xb_g)
        cagra.save(vpq_path, idx_vpq, include_dataset=True)
        del idx_vpq, xb_g
        gc.collect()
        torch.cuda.empty_cache()

    # Now enforce hard cap via LimitingResourceAdaptor(cap_gb)
    base_mr = rmm.mr.CudaMemoryResource()
    lim_mr = rmm.mr.LimitingResourceAdaptor(base_mr, allocation_limit=int(cap_gb * (1024 ** 3)))
    rmm.mr.set_current_device_resource(lim_mr)

    results = []
    itopk_list = sorted(list(set([128, spec["itopk_r99"]])))
    try:
        f0, _ = torch.cuda.mem_get_info()
        idx_vpq = cagra.load(vpq_path)
        f1, _ = torch.cuda.mem_get_info()
        used_gb = (f0 - f1) / (1024 ** 3)
        free_after_gb = cap_gb - used_gb
        xq_g = torch.from_numpy(xq_1024).to(dev, dtype=torch.float32).contiguous()
        for itk in itopk_list:
            sp = cagra.SearchParams(itopk_size=itk)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            d_eval, i_eval = cagra.search(sp, idx_vpq, xq_g, 10)
            torch.cuda.synchronize()
            cold_qps = 1024.0 / max(time.perf_counter() - t0, 1e-6)
            _ = cagra.search(sp, idx_vpq, xq_g, 10)
            torch.cuda.synchronize()
            steady_list = []
            for _ in range(5):
                ts = time.perf_counter()
                _ = cagra.search(sp, idx_vpq, xq_g, 10)
                torch.cuda.synchronize()
                steady_list.append(1024.0 / max(time.perf_counter() - ts, 1e-6))
            i_np = torch.as_tensor(i_eval, device="cuda").to(torch.int64).cpu().numpy()[:1000]
            d_np = torch.as_tensor(d_eval, device="cuda").cpu().numpy()[:1000]
            rec, strict_ov, _ = eval_accuracy_v4(i_np, d_np, ref_i, ref_d, gt10)
            row = [
                ds, cap_gb, f"VPQ_hardcap_M{pq_dim}", gdeg, itk, round(free_after_gb, 3), "OK",
                round(cold_qps, 1), round(statistics.mean(steady_list), 1),
                round(statistics.stdev(steady_list), 1), round(rec, 4), round(strict_ov, 2), 1
            ]
            print("VPQ_RESULT:", row, flush=True)
            results.append(row)
    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
        for itk in itopk_list:
            results.append([
                ds, cap_gb, f"VPQ_hardcap_M{pq_dim}", gdeg, itk, -1.0, f"VPQ_FAILED ({err_msg})",
                0.0, 0.0, 0.0, 0.0, 0.0, 1
            ])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "build", "worker"], default="all")
    parser.add_argument("--ds", type=str, default="")
    parser.add_argument("--cap", type=float, default=2.2)
    parser.add_argument("--mode", type=str, default="UVM_capped")
    args = parser.parse_args()

    if args.stage == "build":
        build_and_save_all()
        return

    if args.stage == "worker":
        if args.mode.startswith("VPQ"):
            rows = run_worker_vpq(args.ds, args.cap)
        else:
            rows = run_worker_uvm(args.ds, args.cap, args.mode)
        with open(OUT_CSV, "a", newline="") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow(r)
        return

    # Stage == "all": orchestrate Process A then Process B for every (ds, cap, mode)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "cap_gb", "mode", "graph_degree", "itopk",
            "free_after_load_gb", "status", "cold_qps", "steady_qps_mean",
            "steady_qps_std", "recall_at_10", "strict_overlap_pct", "measured"
        ])

    subprocess.run([sys.executable, "-u", __file__, "--stage", "build"], check=True)

    for ds, spec in DATASET_SPECS.items():
        for cap_gb in spec["caps"]:
            mode = "UVM_uncapped_control" if cap_gb >= 50.0 else "UVM_capped"
            print(f"\n>>> Launching fresh Process B: ds={ds}, cap={cap_gb}GB, mode={mode}", flush=True)
            subprocess.run([
                sys.executable, "-u", __file__,
                "--stage", "worker", "--ds", ds, "--cap", str(cap_gb), "--mode", mode
            ], check=False)

    for ds in ["gist1m", "bge1024"]:
        for cap_gb in [2.2, 2.5]:
            print(f"\n>>> Launching fresh Process B: ds={ds}, cap={cap_gb}GB, mode=VPQ_hardcap", flush=True)
            subprocess.run([
                sys.executable, "-u", __file__,
                "--stage", "worker", "--ds", ds, "--cap", str(cap_gb), "--mode", "VPQ_hardcap"
            ], check=False)


if __name__ == "__main__":
    main()
