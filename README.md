# TRIDENT: A GPU-Native Lossless Vector Query Engine for High-Throughput Inverted-File Search via Progressive Dimension-Block Pruning

This repository contains the source code, CUDA kernels, benchmark drivers, raw experimental logs (CSV/JSONL), and figure generation scripts for **TRIDENT**.

---

## 1. Overview

**TRIDENT** is a GPU-native, mathematically lossless inverted-file (IVF) vector query engine that combines:
1. **Isometric PCA Energy Compaction & Per-Cluster Directional SQ8 Interval Lower Bounds**: Rotates vectors via an orthogonal PCA basis ($\mathbf{R}^\top \mathbf{R} = \mathbf{I}$) and computes tight per-cluster 8-bit interval lower bounds $[\ell_{c,d}, u_{c,d}]$ across 32-dimension blocks (`B_blk = ceil(D / 32)`), pruning **61.9%–92.5%** of dimensions on high-work workloads with **zero false dismissals** (`100.00%` identical top-$K$ results to exhaustive FP32 `IVF-Flat`, verified against a `float64` oracle).
2. **Online Workload-Aware Cost Router ($W_0 = 9.0 \times 10^8$)**: Dynamically routes low-work queries ($W = B \cdot \mathit{nprobe} \cdot \bar{L} \cdot D < W_0$) to single-pass cluster-major exact scanning (`CM-FP32`) and high-work queries ($W \ge W_0$) to the three-stage progressive pruning cascade.
3. **Divergence-Free Three-Stage Cluster-Major GPU Pipeline**:
   - **Stage X**: Warm-up threshold seeding (`m_warm` closest clusters in-VRAM, or in-VRAM SQ8 upper-bound seeding `TRIDENT_UB_SEED=1` in Out-of-Core mode).
   - **Stage Y**: 32-dimension blocked SQ8 lower-bound cascade with `__all_sync` warp early exit.
   - **Stage Z**: Warp-cooperative FP32 refinement via `__ballot_sync`, `__ffs`, `__shfl_sync`, and `__shfl_xor_sync`, converting divergent single-thread survivor reads into 128-byte coalesced warp transactions.
4. **Zero-Copy Out-of-Core (OOC) Execution**: Keeps only compact 32-dim blocked SQ8 codes (`N * D` bytes, `12.74 GiB` for `SIFT100M`) in GPU VRAM and streams only surviving FP32 candidates (`0.0068%–0.0618%` at `Recall@10 = 0.99`) from `cudaHostAllocMapped` page-locked host RAM over PCIe Gen4, cutting Nsight Compute measured PCIe traffic by **99.92%–99.99%**.

---

## 2. Repository Structure

```text
TRIDENT/
├── README.md                          # Documentation and artifact reproducibility guide
├── LICENSE                            # MIT License
├── src/                               # C++17 / CUDA 13.0 Kernels & PyBind11 Extensions
│   ├── trident_v4_kernels.cu          # Main TRIDENT 3-stage pipeline, UB seeding, & Zero-Copy OOC kernels
│   ├── trident_v4_kernels_inmem.cu    # In-memory A100 multi-K and ablation kernel suite
│   ├── trident_kernels_v1.cu          # Baseline cluster-major & prototype kernels
│   └── phase2a_cuda.cu                # Active warp survival curve profiling kernel
├── benchmarks/                        # Benchmark Drivers & Baseline Runners
│   ├── phase4a_suite.py               # 1M-scale in-memory benchmark suite (6 datasets)
│   ├── phase4b_suite.py               # 10M-scale in-memory benchmark suite (SIFT10M, Deep10M, T2I10M)
│   ├── phase8_ooc_100m_4080.py        # SIFT100M True OOC benchmark on 16 GB RTX 4080
│   ├── plan_v10/                      # Phase 10 OOC UB-seeding, matched-recall IVFPQ+Refine, & NCU per-pass suite
│   └── plan_v11/                      # Phase 11 CAGRA-Device 100M OOM diagnosis & summary aggregator
├── data/                              # All Raw Experimental CSVs / JSONLs & Verified Claims Database
│   ├── SUMMARY_V11.md                 # Complete summary of OOC, 100M, NCU per-pass, and lossless audit results
│   ├── verified_claims.json           # Automated verification database for all paper numbers
│   ├── phase1/ .. phase8/             # In-memory A100, ablation, NCU, RaBitQ, Tribase-GPU, cuVS-CAGRA CSVs
│   ├── phase10/                       # OOC UB-seeding, per-pass NCU PCIe counters, 100M RTX 4080 results
│   └── phase11/                       # Matched-recall summary & CAGRA 100M Device OOM diagnostic logs
├── scripts/                           # Paper Figure Generation & Automated Consistency Audit
│   ├── generate_all_figures.py        # Generates all 11 vector PDF figures (fig1..fig11) directly from data/
│   └── check_numbers.py               # Audits all unit-tagged numbers against verified_claims.json / CSVs
└── figures/                           # Generated Vector PDF Figures (fig1_intro_comparison.pdf .. fig11)
```

---

## 3. Mapping Paper Tables & Figures to Raw Data (`data/`)

| Paper Table / Figure | Description | Backing CSV / JSONL Files in `data/` |
| :--- | :--- | :--- |
| **Table I** (`tab:paradigm_comp`) | Paradigm comparison on `GIST1M` ($R=0.99$) | `data/phase4a/phase4a_qps_recall_A100.csv`, `data/phase6/ncu_v2.csv`, `data/phase6/tribase_gpu.csv`, `data/phase6/cuvs_qps_recall.csv` |
| **Table II** (`tab:datasets`) | 10 real-world datasets ($268\text{K}$ to $100\text{M}$) | `data/verified_claims.json` |
| **Table III** (`tab:main_qps`) | End-to-end QPS, speedup, & dims pruned across 9 datasets on A100 | `data/phase4a/phase4a_qps_recall_A100.csv`, `data/phase6/sift10m_rerun.csv`, `data/phase4b/phase4b_main_A100.csv`, `data/phase6/phase6c_main_A100.csv`, `data/phase6/tribase_gpu.csv`, `data/phase6/cuvs_qps_recall.csv` |
| **Table IV** (`tab:ooc`) | True `100M` OOC on 16 GB RTX 4080 & `2.2–3.0 GB` cap on A100 | `data/phase10/ooc_100m_RTX4080.csv`, `data/phase10/ooc_1m10m_A100.csv`, `data/phase10/ncu_perpass_RTX4080.csv`, `data/phase10/ncu_perpass_A100.csv`, `data/phase11/matched_recall_summary.json`, `data/phase11/gpu_baselines_100m_RTX4080_v11.jsonl` |
| **Table V** (`tab:sens_k_rabitq`) | Top-$K$ scaling ($K \in \{1, 10, 50, 100\}$) & vs. 24T AVX-512 `RaBitQ` | `data/phase6/sens_k.csv`, `data/phase6/rabitq_cpu.csv` |
| **Figure 1** (`fig1_intro_comparison.pdf`) | Overview comparison on `GIST1M` ($R=0.99$) | `data/verified_claims.json`, `data/phase6/ncu_v2.csv` |
| **Figure 2** (`fig2_motivation.pdf`) | Scalar bound collapse & active warp survival curves | `data/phase6/tribase_gpu.csv`, `data/phase2a/active_tile_curves.csv` |
| **Figure 6** (`fig6_overall_qps_recall_10panels.pdf`) | 10-panel QPS vs. Recall@10 Pareto frontiers on A100 | `data/phase4a/phase4a_qps_recall_A100.csv`, `data/phase6/cuvs_qps_recall.csv`, `data/phase6/tribase_gpu.csv` |
| **Figure 7** (`fig7_batch_and_10m_throughput.pdf`) | Locked $R=0.99$ QPS & batch scaling $B \in [1, 4096]$ | `data/phase6/auto_v2.csv`, `data/verified_claims.json` |
| **Figure 8** (`fig8_latency_traffic_breakdown_10panels.pdf`) | Nsight Compute FFMA, L2/HBM traffic, and kernel time | `data/phase6/ncu_v2.csv` |
| **Figure 9** (`fig9_ablation_study_10panels.pdf`) | Additive (B0 $\rightarrow$ B4) and leave-one-out ablation study | `data/phase3a/phase3a_ablation_A100.csv` |
| **Figure 10** (`fig10_build_amortization_memory.pdf`) | OOC QPS (`1M–100M`), NCU PCIe traffic, VRAM, & GPU build time | `data/phase10/ooc_100m_RTX4080.csv`, `data/phase10/ooc_1m10m_A100.csv`, `data/phase6/gpu_build_mem_A100.csv` |
| **Figure 11** (`fig11_heatmap_scalability.pdf`) | Sensitivity ($K$, `nlist`), `RaBitQ` / `Tribase-GPU`, & RTX 4080 | `data/phase6/sens_k.csv`, `data/phase6/rabitq_cpu.csv`, `data/phase6/sens_nlist.csv`, `data/phase6/main_4080.csv` |

---

## 4. Quick Start

### 4.1 Regenerate All 11 Paper Figures from Raw CSVs
```bash
python3 scripts/generate_all_figures.py
```

### 4.2 Run the Automated Paper Number Audit
When run inside the paper directory (or with `.tex` files present in `sections/`), `check_numbers.py` verifies every unit-tagged number against `data/verified_claims.json` and raw CSV files:
```bash
python3 scripts/check_numbers.py
```

### 4.3 Build and Run the CUDA Kernels & Benchmark Suites
Requirements:
- Linux (Ubuntu 22.04+), Python 3.10+, PyTorch 2.x (for PyBind11 / `torch.utils.cpp_extension`), CUDA Toolkit 12.x / 13.0 (`sm_80` for NVIDIA A100, `sm_89` for NVIDIA RTX 4080)
- `faiss-gpu` / `faiss-cpu`, `cuvs` (for RAFT/cuVS baselines), `scikit-learn`, `numpy`, `matplotlib`

To run the in-memory A100 benchmark suite or the 100M out-of-core RTX 4080 benchmark suite:
```bash
# In-memory 1M & 10M benchmarks (compiles src/trident_v4_kernels.cu JIT via PyBind11)
python3 benchmarks/phase4a_suite.py
python3 benchmarks/phase4b_suite.py

# Out-of-Core 1M–10M (A100) and True 100M OOC (16 GB RTX 4080)
bash benchmarks/plan_v10/run_all_v10.sh
```
