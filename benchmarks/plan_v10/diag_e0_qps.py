#!/usr/bin/env python3
"""
PLAN_V10 diagnostic for Gate G1(b): why is ub=0 in-VRAM QPS 22-31% below the main table?
A/B on the A100 (physical GPU 1), same process, same index, same queries:
  kernel  : 'patched' (current trident_v4_kernels.cu, TRIDENT_UB_SEED=0) vs 'v9' (trident_v4_kernels.cu.bak_plan_v9)
  timing  : 'p4a' = phase4a measure_latency_qps (warmup 12, 50 iters, sync only at ends)  -> how the main table was measured
            'p10' = phase10 measure_runs        (warmup 3, 10 iters, sync every iter)
Output: ~/trident/results/phase10/diag_e0_qps.csv
Run: cd ~/trident/scripts/plan_v10 && /home/mazaiying/trident/.venv/bin/python diag_e0_qps.py --kernel patched
     cd ~/trident/scripts/plan_v10 && /home/mazaiying/trident/.venv/bin/python diag_e0_qps.py --kernel v9
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase10_common import setup_env  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kernel", choices=["patched", "v9"], required=True)
ap.add_argument("--datasets", default="gist1m,sift1m,msong,bge1024")
args = ap.parse_args()
setup_env(1)
os.environ["TRIDENT_UB_SEED"] = "0"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

import phase4a_suite as s  # noqa: E402
from phase10_common import DATA_ROOT, RES_DIR, measure_runs, write_rows  # noqa: E402

if args.kernel == "v9":
    src = os.path.expanduser("~/trident/scripts/trident_v4_kernels.cu.bak_plan_v9")
    tmp = os.path.expanduser("~/trident/scripts/plan_v10/_v9_kernels_copy.cu")
    with open(src) as f_in, open(tmp, "w") as f_out:
        f_out.write(f_in.read())
    s._CUDA_EXT = load(name="trident_v4_ext_v9ref", sources=[tmp],
                       extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"], verbose=False)
else:
    s.get_cuda_ext()

NP99 = {"gist1m": 128, "sift1m": 48, "msong": 24, "bge1024": 128, "glove100": 384, "nuswide": 8}
MAIN = {"gist1m": 24523.4, "sift1m": 226381.9, "msong": 143636.7, "bge1024": 43504.4,
        "glove100": 28772.2, "nuswide": 526784.2}
dev = torch.device("cuda:0")
print("device:", torch.cuda.get_device_name(dev), "kernel:", args.kernel, flush=True)
rows = []
for ds in args.datasets.split(","):
    d = os.path.join(DATA_ROOT, ds)
    xb = np.ascontiguousarray(np.load(os.path.join(d, "base.npy")).astype(np.float32))
    xq = np.ascontiguousarray(np.load(os.path.join(d, "query.npy")).astype(np.float32))
    centroids, list_ids, r_vec, mean64, R64, _ = s.load_or_train_clustering(ds, xb, 1024, 0)
    storage = s.SharedClusterStorage(xb, list_ids, r_vec, 1024, device="cuda:0")
    t = s.choose_tile(xb.shape[1])
    idx = s.TridentV4Index(storage, centroids, mean64, R64, tile_size=t, per_cluster_sq8=1, use_pca=1,
                           var_block=0, max_batch=1024, device="cuda:0")
    _, _, b_max, _, xq_batch = s.prepare_query_tensors(xq, ds, dev)
    xq_b = xq_batch[:b_max].contiguous()
    fn = lambda: idx.search(xq_b, NP99[ds], mode=2, use_l1=0, rerank_m=0, tile_override=t, warp_refine=1)  # noqa: E731
    e2e_ms = s.measure_latency_qps(fn, warmup=12, iters=50)
    q_p4a = b_max * 1000.0 / e2e_ms
    q_p10, sd = measure_runs(fn, warmup=3, iters=10, b_size=b_max)
    print(f"[{ds} kernel={args.kernel} tile={t}] p4a={q_p4a:.1f}  p10={q_p10:.1f}±{sd:.1f}  main={MAIN[ds]}  "
          f"p4a/main={q_p4a / MAIN[ds]:.3f}", flush=True)
    rows.append([ds, args.kernel, t, NP99[ds], b_max, round(q_p4a, 1), round(q_p10, 1), round(sd, 1), MAIN[ds],
                 round(q_p4a / MAIN[ds], 3)])
    del idx, storage
    torch.cuda.empty_cache()
write_rows(os.path.join(RES_DIR, "diag_e0_qps.csv"),
           ["dataset", "kernel", "tile", "nprobe", "batch", "qps_p4a_timing", "qps_p10_timing", "p10_std",
            "main_table_qps", "p4a_over_main"], rows)
