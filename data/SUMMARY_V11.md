# SUMMARY_V11 — Auto-Generated from CSV / JSONL (Zero Hand-Copied Numbers)

## 0. Step E0 Lossless Audit (`lossless_A100.csv`)

- Total rows: 18
- Min `strict_overlap_pct`: `100.0000%`
- Max `max_abs_dist_diff_vs_cm`: `0.000e+00`

| Dataset | Placement | R | nprobe | ub_seed | QPS (mean ± std) | Recall@10 | Strict Overlap (%) | Surv (%) | Dims Pruned (%) | Max Abs Dist Diff |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| sift1m | InVRAM | 0.99 | 48 | 0 | 229,910.3 ± 210.6 | 0.9924 | 100.00% | 0.6368% | 70.47% | 0.000e+00 |
| sift1m | InVRAM | 0.99 | 48 | 1 | 196,341.4 ± 775.1 | 0.9924 | 100.00% | 0.1097% | 71.01% | 0.000e+00 |
| sift1m | InVRAM | 0.99 | 48 | 2 | 82,630.1 ± 43.6 | 0.9924 | 100.00% | 0.1097% | 71.01% | 0.000e+00 |
| glove100 | InVRAM | 0.99 | 384 | 0 | 29,099.1 ± 8.6 | 0.9931 | 100.00% | 0.0876% | 52.32% | 0.000e+00 |
| glove100 | InVRAM | 0.99 | 384 | 1 | 28,265.6 ± 10.7 | 0.9931 | 100.00% | 0.0237% | 52.36% | 0.000e+00 |
| glove100 | InVRAM | 0.99 | 384 | 2 | 10,516.5 ± 3.3 | 0.9931 | 100.00% | 0.0234% | 52.36% | 0.000e+00 |
| msong | InVRAM | 0.99 | 24 | 0 | 146,039.8 ± 59.4 | 0.9952 | 100.00% | 2.0992% | 89.36% | 0.000e+00 |
| msong | InVRAM | 0.99 | 24 | 1 | 109,421.5 ± 72.3 | 0.9952 | 100.00% | 0.1334% | 91.20% | 0.000e+00 |
| msong | InVRAM | 0.99 | 24 | 2 | 36,343.6 ± 7.6 | 0.9952 | 100.00% | 0.1333% | 91.20% | 0.000e+00 |
| nuswide | InVRAM | 0.99 | 8 | 0 | 711,096.8 ± 3229.1 | 0.995 | 100.00% | 17.0191% | 76.60% | 0.000e+00 |
| nuswide | InVRAM | 0.99 | 8 | 1 | 689,331.5 ± 2278.2 | 0.995 | 100.00% | 7.4106% | 84.28% | 0.000e+00 |
| nuswide | InVRAM | 0.99 | 8 | 2 | 467,640.8 ± 863.2 | 0.995 | 100.00% | 7.4106% | 84.28% | 0.000e+00 |
| gist1m | InVRAM | 0.99 | 128 | 0 | 24,381.6 ± 8.3 | 0.9937 | 100.00% | 0.5319% | 92.49% | 0.000e+00 |
| gist1m | InVRAM | 0.99 | 128 | 1 | 22,289.5 ± 5.7 | 0.9937 | 100.00% | 0.0618% | 92.91% | 0.000e+00 |
| gist1m | InVRAM | 0.99 | 128 | 2 | 5,510.9 ± 0.7 | 0.9937 | 100.00% | 0.0615% | 92.91% | 0.000e+00 |
| bge1024 | InVRAM | 0.99 | 128 | 0 | 43,970.5 ± 25.2 | 0.992 | 100.00% | 0.6361% | 91.48% | 0.000e+00 |
| bge1024 | InVRAM | 0.99 | 128 | 1 | 38,582.7 ± 39.2 | 0.992 | 100.00% | 0.0437% | 91.99% | 0.000e+00 |
| bge1024 | InVRAM | 0.99 | 128 | 2 | 6,758.2 ± 1.6 | 0.992 | 100.00% | 0.0435% | 91.99% | 0.000e+00 |

## 1. Full OOC Benchmark Tables (1M/10M on A100 & 100M on RTX 4080)

| Dataset | GPU / Cap | R | nprobe | Method | Config / ub / m | QPS (mean ± std) | Recall@10 | Strict Overlap (%) | Surv (%) | Dims Pruned (%) |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| gist1m | A100 (2.2 GB cap) | 0.95 | 56 | `TRIDENT` | `ub_seed=0` | 3,506.8 ± 1.3 | 0.9622 | 100.00% | 1.1804% | 90.66% |
| gist1m | A100 (2.2 GB cap) | 0.99 | 128 | `TRIDENT` | `ub_seed=0` | 3,340.5 ± 0.7 | 0.9937 | 100.00% | 0.5319% | 92.49% |
| gist1m | A100 (2.2 GB cap) | 0.95 | 56 | `TRIDENT` | `ub_seed=1` | 23,894.1 ± 20.0 | 0.9622 | 100.00% | 0.1311% | 91.60% |
| gist1m | A100 (2.2 GB cap) | 0.99 | 128 | `TRIDENT` | `ub_seed=1` | 17,852.6 ± 5.2 | 0.9937 | 100.00% | 0.0618% | 92.91% |
| gist1m | A100 (2.2 GB cap) | 0.95 | 56 | `TRIDENT` | `ub_seed=2` | 9,754.2 ± 6.6 | 0.9622 | 100.00% | 0.1306% | 91.60% |
| gist1m | A100 (2.2 GB cap) | 0.99 | 128 | `TRIDENT` | `ub_seed=2` | 5,193.6 ± 1.1 | 0.9937 | 100.00% | 0.0615% | 92.91% |
| gist1m | A100 (2.2 GB cap) | 0.95 | 56 | `CM-FP32-OOC` | `exact` | 111.0 ± 1.5 | — | 100.00% | 100.0000% | 0.00% |
| gist1m | A100 (2.2 GB cap) | 0.99 | 128 | `CM-FP32-OOC` | `exact` | 81.1 ± 2.0 | — | 100.00% | 100.0000% | 0.00% |
| gist1m | A100 (2.2 GB cap) | 0.95 | 56 | `Faiss-CPU-IVFFlat-24T` | `faiss.IndexIVFFlat` | 210.5 ± 8.9 | 0.9622 | 100.00% | 100.0000% | 0.00% |
| gist1m | A100 (2.2 GB cap) | 0.99 | 128 | `Faiss-CPU-IVFFlat-24T` | `faiss.IndexIVFFlat` | 103.8 ± 9.6 | 0.9937 | 100.00% | 100.0000% | 0.00% |
| gist1m | A100 (2.2 GB cap) | 0.95 | 56 | `IVFPQ+CPU-Refine` | `M=48, m=100` | 46,710.1 ± 204.8 | 0.7603 | 50.51% | — | — |
| gist1m | A100 (2.2 GB cap) | 0.95 | 56 | `IVFPQ+CPU-Refine` | `M=48, m=200` | 33,938.1 ± 656.6 | 0.8578 | 69.39% | — | — |
| gist1m | A100 (2.2 GB cap) | 0.95 | 56 | `IVFPQ+CPU-Refine` | `M=48, m=400` | 22,757.4 ± 268.8 | 0.9188 | 85.29% | — | — |
| gist1m | A100 (2.2 GB cap) | 0.95 | 56 | `IVFPQ+CPU-Refine` | `M=48, m=800` | 13,054.5 ± 134.5 | 0.9497 | 95.11% | — | — |
| gist1m | A100 (2.2 GB cap) | 0.95 | 56 | `IVFPQ+CPU-Refine` | `M=48, m=1600` | 7,206.3 ± 33.0 | 0.9602 | 99.14% | — | — |
| gist1m | A100 (2.2 GB cap) | 0.95 | 56 | `IVFPQ+CPU-Refine` | `M=48, m=2048` | 5,900.0 ± 79.6 | 0.9614 | 99.67% | — | — |
| gist1m | A100 (2.2 GB cap) | 0.99 | 128 | `IVFPQ+CPU-Refine` | `M=48, m=100` | 23,050.9 ± 213.0 | 0.7659 | 48.54% | — | — |
| gist1m | A100 (2.2 GB cap) | 0.99 | 128 | `IVFPQ+CPU-Refine` | `M=48, m=200` | 19,745.4 ± 75.6 | 0.8682 | 66.60% | — | — |
| gist1m | A100 (2.2 GB cap) | 0.99 | 128 | `IVFPQ+CPU-Refine` | `M=48, m=400` | 14,805.7 ± 121.4 | 0.9357 | 82.53% | — | — |
| gist1m | A100 (2.2 GB cap) | 0.99 | 128 | `IVFPQ+CPU-Refine` | `M=48, m=800` | 9,606.7 ± 96.8 | 0.9725 | 92.22% | — | — |
| gist1m | A100 (2.2 GB cap) | 0.99 | 128 | `IVFPQ+CPU-Refine` | `M=48, m=1600` | 5,389.2 ± 26.7 | 0.9876 | 97.37% | — | — |
| gist1m | A100 (2.2 GB cap) | 0.99 | 128 | `IVFPQ+CPU-Refine` | `M=48, m=2048` | 4,587.2 ± 34.2 | 0.9907 | 98.65% | — | — |
| bge1024 | A100 (2.5 GB cap) | 0.95 | 14 | `TRIDENT` | `ub_seed=0` | 5,052.9 ± 0.7 | 0.952 | 100.00% | 4.3046% | 82.12% |
| bge1024 | A100 (2.5 GB cap) | 0.99 | 128 | `TRIDENT` | `ub_seed=0` | 3,277.7 ± 1.3 | 0.992 | 100.00% | 0.6361% | 91.48% |
| bge1024 | A100 (2.5 GB cap) | 0.95 | 14 | `TRIDENT` | `ub_seed=1` | 62,329.1 ± 41.1 | 0.952 | 100.00% | 0.2839% | 85.70% |
| bge1024 | A100 (2.5 GB cap) | 0.99 | 128 | `TRIDENT` | `ub_seed=1` | 27,426.5 ± 22.3 | 0.992 | 100.00% | 0.0437% | 91.99% |
| bge1024 | A100 (2.5 GB cap) | 0.95 | 14 | `TRIDENT` | `ub_seed=2` | 36,345.9 ± 24.9 | 0.952 | 100.00% | 0.2816% | 85.71% |
| bge1024 | A100 (2.5 GB cap) | 0.99 | 128 | `TRIDENT` | `ub_seed=2` | 6,311.2 ± 0.7 | 0.992 | 100.00% | 0.0435% | 91.99% |
| bge1024 | A100 (2.5 GB cap) | 0.95 | 14 | `CM-FP32-OOC` | `exact` | 484.4 ± 4.2 | — | 100.00% | 100.0000% | 0.00% |
| bge1024 | A100 (2.5 GB cap) | 0.99 | 128 | `CM-FP32-OOC` | `exact` | 92.6 ± 0.0 | — | 100.00% | 100.0000% | 0.00% |
| bge1024 | A100 (2.5 GB cap) | 0.95 | 14 | `Faiss-CPU-IVFFlat-24T` | `faiss.IndexIVFFlat` | 1,328.1 ± 44.5 | 0.952 | 100.00% | 100.0000% | 0.00% |
| bge1024 | A100 (2.5 GB cap) | 0.99 | 128 | `Faiss-CPU-IVFFlat-24T` | `faiss.IndexIVFFlat` | 104.0 ± 0.6 | 0.992 | 100.00% | 100.0000% | 0.00% |
| bge1024 | A100 (2.5 GB cap) | 0.95 | 14 | `IVFPQ+CPU-Refine` | `M=32, m=100` | 109,135.5 ± 2964.9 | 0.8041 | 72.52% | — | — |
| bge1024 | A100 (2.5 GB cap) | 0.95 | 14 | `IVFPQ+CPU-Refine` | `M=32, m=200` | 67,757.9 ± 999.5 | 0.8436 | 78.32% | — | — |
| bge1024 | A100 (2.5 GB cap) | 0.95 | 14 | `IVFPQ+CPU-Refine` | `M=32, m=400` | 37,043.6 ± 1946.5 | 0.8864 | 83.86% | — | — |
| bge1024 | A100 (2.5 GB cap) | 0.95 | 14 | `IVFPQ+CPU-Refine` | `M=32, m=800` | 19,171.6 ± 663.1 | 0.9387 | 95.37% | — | — |
| bge1024 | A100 (2.5 GB cap) | 0.95 | 14 | `IVFPQ+CPU-Refine` | `M=32, m=1600` | 9,522.2 ± 262.6 | 0.9473 | 98.03% | — | — |
| bge1024 | A100 (2.5 GB cap) | 0.95 | 14 | `IVFPQ+CPU-Refine` | `M=32, m=2048` | 7,749.5 ± 71.0 | 0.9491 | 98.65% | — | — |
| bge1024 | A100 (2.5 GB cap) | 0.99 | 128 | `IVFPQ+CPU-Refine` | `M=32, m=100` | 28,665.9 ± 375.7 | 0.8101 | 70.29% | — | — |
| bge1024 | A100 (2.5 GB cap) | 0.99 | 128 | `IVFPQ+CPU-Refine` | `M=32, m=200` | 24,926.7 ± 194.4 | 0.8534 | 75.43% | — | — |
| bge1024 | A100 (2.5 GB cap) | 0.99 | 128 | `IVFPQ+CPU-Refine` | `M=32, m=400` | 18,364.7 ± 206.8 | 0.9 | 79.83% | — | — |
| bge1024 | A100 (2.5 GB cap) | 0.99 | 128 | `IVFPQ+CPU-Refine` | `M=32, m=800` | 11,740.2 ± 235.7 | 0.9588 | 91.03% | — | — |
| bge1024 | A100 (2.5 GB cap) | 0.99 | 128 | `IVFPQ+CPU-Refine` | `M=32, m=1600` | 6,665.9 ± 75.3 | 0.974 | 94.46% | — | — |
| bge1024 | A100 (2.5 GB cap) | 0.99 | 128 | `IVFPQ+CPU-Refine` | `M=32, m=2048` | 5,546.0 ± 56.3 | 0.9783 | 95.47% | — | — |
| sift10m | A100 (3.0 GB cap) | 0.95 | 32 | `TRIDENT` | `ub_seed=0` | 47,439.9 ± 85.7 | 0.9523 | 100.00% | 0.4278% | 72.00% |
| sift10m | A100 (3.0 GB cap) | 0.99 | 96 | `TRIDENT` | `ub_seed=0` | 35,319.5 ± 27.5 | 0.9908 | 100.00% | 0.1488% | 73.56% |
| sift10m | A100 (3.0 GB cap) | 0.95 | 32 | `TRIDENT` | `ub_seed=1` | 86,478.3 ± 184.4 | 0.9523 | 100.00% | 0.0695% | 72.54% |
| sift10m | A100 (3.0 GB cap) | 0.99 | 96 | `TRIDENT` | `ub_seed=1` | 53,179.4 ± 65.4 | 0.9908 | 100.00% | 0.0251% | 73.75% |
| sift10m | A100 (3.0 GB cap) | 0.95 | 32 | `TRIDENT` | `ub_seed=2` | 40,170.6 ± 61.6 | 0.9523 | 100.00% | 0.0695% | 72.54% |
| sift10m | A100 (3.0 GB cap) | 0.99 | 96 | `TRIDENT` | `ub_seed=2` | 18,262.8 ± 6.2 | 0.9908 | 100.00% | 0.0250% | 73.75% |
| sift10m | A100 (3.0 GB cap) | 0.95 | 32 | `CM-FP32-OOC` | `exact` | 523.9 ± 0.7 | — | 100.00% | 100.0000% | 0.00% |
| sift10m | A100 (3.0 GB cap) | 0.99 | 96 | `CM-FP32-OOC` | `exact` | 260.4 ± 4.0 | — | 100.00% | 100.0000% | 0.00% |
| sift10m | A100 (3.0 GB cap) | 0.95 | 32 | `Faiss-CPU-IVFFlat-24T` | `faiss.IndexIVFFlat` | 1,582.1 ± 53.2 | 0.9523 | 100.00% | 100.0000% | 0.00% |
| sift10m | A100 (3.0 GB cap) | 0.99 | 96 | `Faiss-CPU-IVFFlat-24T` | `faiss.IndexIVFFlat` | 568.7 ± 4.3 | 0.9908 | 100.00% | 100.0000% | 0.00% |
| sift10m | A100 (3.0 GB cap) | 0.95 | 32 | `IVFPQ+CPU-Refine` | `M=16, m=100` | 128,203.6 ± 1408.7 | 0.9322 | 92.63% | — | — |
| sift10m | A100 (3.0 GB cap) | 0.95 | 32 | `IVFPQ+CPU-Refine` | `M=16, m=200` | 87,520.5 ± 1107.1 | 0.9472 | 97.80% | — | — |
| sift10m | A100 (3.0 GB cap) | 0.95 | 32 | `IVFPQ+CPU-Refine` | `M=16, m=400` | 56,518.3 ± 1152.1 | 0.9516 | 99.75% | — | — |
| sift10m | A100 (3.0 GB cap) | 0.95 | 32 | `IVFPQ+CPU-Refine` | `M=16, m=800` | 31,769.0 ± 311.0 | 0.952 | 99.93% | — | — |
| sift10m | A100 (3.0 GB cap) | 0.95 | 32 | `IVFPQ+CPU-Refine` | `M=16, m=1600` | 17,232.4 ± 329.2 | 0.9522 | 99.97% | — | — |
| sift10m | A100 (3.0 GB cap) | 0.95 | 32 | `IVFPQ+CPU-Refine` | `M=16, m=2048` | 14,943.1 ± 162.6 | 0.9522 | 99.97% | — | — |
| sift10m | A100 (3.0 GB cap) | 0.99 | 96 | `IVFPQ+CPU-Refine` | `M=16, m=100` | 38,234.5 ± 235.9 | 0.9619 | 90.85% | — | — |
| sift10m | A100 (3.0 GB cap) | 0.99 | 96 | `IVFPQ+CPU-Refine` | `M=16, m=200` | 34,961.5 ± 293.2 | 0.9823 | 97.23% | — | — |
| sift10m | A100 (3.0 GB cap) | 0.99 | 96 | `IVFPQ+CPU-Refine` | `M=16, m=400` | 25,936.2 ± 841.8 | 0.9892 | 99.54% | — | — |
| sift10m | A100 (3.0 GB cap) | 0.99 | 96 | `IVFPQ+CPU-Refine` | `M=16, m=800` | 18,005.7 ± 197.1 | 0.9903 | 99.90% | — | — |
| sift10m | A100 (3.0 GB cap) | 0.99 | 96 | `IVFPQ+CPU-Refine` | `M=16, m=1600` | 10,469.7 ± 45.4 | 0.9906 | 99.93% | — | — |
| sift10m | A100 (3.0 GB cap) | 0.99 | 96 | `IVFPQ+CPU-Refine` | `M=16, m=2048` | 9,295.5 ± 44.9 | 0.9906 | 99.93% | — | — |
| deep10m | A100 (2.5 GB cap) | 0.95 | 32 | `TRIDENT` | `ub_seed=0` | 58,393.2 ± 71.7 | 0.9623 | 100.00% | 0.4227% | 58.91% |
| deep10m | A100 (2.5 GB cap) | 0.99 | 96 | `TRIDENT` | `ub_seed=0` | 39,408.0 ± 41.3 | 0.9908 | 100.00% | 0.1484% | 61.91% |
| deep10m | A100 (2.5 GB cap) | 0.95 | 32 | `TRIDENT` | `ub_seed=1` | 98,341.9 ± 158.8 | 0.9623 | 100.00% | 0.0693% | 59.44% |
| deep10m | A100 (2.5 GB cap) | 0.99 | 96 | `TRIDENT` | `ub_seed=1` | 54,317.8 ± 36.3 | 0.9908 | 100.00% | 0.0249% | 62.10% |
| deep10m | A100 (2.5 GB cap) | 0.95 | 32 | `TRIDENT` | `ub_seed=2` | 47,066.3 ± 36.5 | 0.9623 | 100.00% | 0.0691% | 59.44% |
| deep10m | A100 (2.5 GB cap) | 0.99 | 96 | `TRIDENT` | `ub_seed=2` | 20,739.4 ± 16.2 | 0.9908 | 100.00% | 0.0249% | 62.10% |
| deep10m | A100 (2.5 GB cap) | 0.95 | 32 | `CM-FP32-OOC` | `exact` | 696.9 ± 6.0 | — | 100.00% | 100.0000% | 0.00% |
| deep10m | A100 (2.5 GB cap) | 0.99 | 96 | `CM-FP32-OOC` | `exact` | 359.4 ± 3.5 | — | 100.00% | 100.0000% | 0.00% |
| deep10m | A100 (2.5 GB cap) | 0.95 | 32 | `Faiss-CPU-IVFFlat-24T` | `faiss.IndexIVFFlat` | 1,784.0 ± 35.9 | 0.9623 | 100.00% | 100.0000% | 0.00% |
| deep10m | A100 (2.5 GB cap) | 0.99 | 96 | `Faiss-CPU-IVFFlat-24T` | `faiss.IndexIVFFlat` | 639.3 ± 26.1 | 0.9908 | 100.00% | 100.0000% | 0.00% |
| deep10m | A100 (2.5 GB cap) | 0.95 | 32 | `IVFPQ+CPU-Refine` | `M=16, m=100` | 137,091.0 ± 2867.2 | 0.9176 | 85.92% | — | — |
| deep10m | A100 (2.5 GB cap) | 0.95 | 32 | `IVFPQ+CPU-Refine` | `M=16, m=200` | 99,124.0 ± 732.4 | 0.9489 | 95.30% | — | — |
| deep10m | A100 (2.5 GB cap) | 0.95 | 32 | `IVFPQ+CPU-Refine` | `M=16, m=400` | 63,323.6 ± 1074.6 | 0.9588 | 98.87% | — | — |
| deep10m | A100 (2.5 GB cap) | 0.95 | 32 | `IVFPQ+CPU-Refine` | `M=16, m=800` | 37,090.4 ± 1514.3 | 0.9613 | 99.74% | — | — |
| deep10m | A100 (2.5 GB cap) | 0.95 | 32 | `IVFPQ+CPU-Refine` | `M=16, m=1600` | 20,551.5 ± 207.4 | 0.9621 | 99.93% | — | — |
| deep10m | A100 (2.5 GB cap) | 0.95 | 32 | `IVFPQ+CPU-Refine` | `M=16, m=2048` | 17,855.3 ± 103.8 | 0.9621 | 99.93% | — | — |
| deep10m | A100 (2.5 GB cap) | 0.99 | 96 | `IVFPQ+CPU-Refine` | `M=16, m=100` | 51,240.9 ± 133.6 | 0.9367 | 84.77% | — | — |
| deep10m | A100 (2.5 GB cap) | 0.99 | 96 | `IVFPQ+CPU-Refine` | `M=16, m=200` | 43,520.6 ± 447.0 | 0.9729 | 94.43% | — | — |
| deep10m | A100 (2.5 GB cap) | 0.99 | 96 | `IVFPQ+CPU-Refine` | `M=16, m=400` | 33,142.9 ± 493.5 | 0.986 | 98.52% | — | — |
| deep10m | A100 (2.5 GB cap) | 0.99 | 96 | `IVFPQ+CPU-Refine` | `M=16, m=800` | 20,828.1 ± 392.7 | 0.9896 | 99.72% | — | — |
| deep10m | A100 (2.5 GB cap) | 0.99 | 96 | `IVFPQ+CPU-Refine` | `M=16, m=1600` | 12,886.5 ± 86.5 | 0.9906 | 99.93% | — | — |
| deep10m | A100 (2.5 GB cap) | 0.99 | 96 | `IVFPQ+CPU-Refine` | `M=16, m=2048` | 11,440.9 ± 158.1 | 0.9906 | 99.93% | — | — |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.95 | 48 | `TRIDENT-OOC` | `ub_seed=0` | 30,119.7 ± 21.8 | 0.954 | 100.00% | 0.1227% | 73.05% |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.99 | 160 | `TRIDENT-OOC` | `ub_seed=0` | 18,100.7 ± 22.3 | 0.9922 | 100.00% | 0.0389% | 74.07% |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.95 | 48 | `TRIDENT-OOC` | `ub_seed=1` | 40,449.1 ± 50.7 | 0.954 | 100.00% | 0.0207% | 73.29% |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.99 | 160 | `TRIDENT-OOC` | `ub_seed=1` | 21,244.6 ± 42.7 | 0.9922 | 100.00% | 0.0068% | 74.14% |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.95 | 48 | `TRIDENT-OOC` | `ub_seed=2` | 16,939.4 ± 70.1 | 0.954 | 100.00% | 0.0206% | 73.29% |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.99 | 160 | `TRIDENT-OOC` | `ub_seed=2` | 6,281.4 ± 7.0 | 0.9922 | 100.00% | 0.0068% | 74.14% |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.95 | 48 | `CM-FP32-OOC` | `exact` | 108.9 ± 0.3 | — | 100.00% | 100.0000% | 0.00% |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.99 | 160 | `CM-FP32-OOC` | `exact` | 42.9 ± 0.1 | — | 100.00% | 100.0000% | 0.00% |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.95 | 48 | `Faiss-CPU-IVFFlat-24T` | `faiss.IndexIVFFlat` | 322.9 ± 7.9 | 0.954 | 100.00% | 100.0000% | 0.00% |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.99 | 160 | `Faiss-CPU-IVFFlat-24T` | `faiss.IndexIVFFlat` | 123.5 ± 2.3 | 0.9922 | 100.00% | 100.0000% | 0.00% |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.95 | 48 | `IVFPQ+CPU-Refine` | `M=16, m=100` | 41,270.7 ± 451.4 | 0.9051 | 83.24% | — | — |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.95 | 48 | `IVFPQ+CPU-Refine` | `M=16, m=200` | 35,638.3 ± 702.4 | 0.9401 | 94.53% | — | — |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.95 | 48 | `IVFPQ+CPU-Refine` | `M=16, m=400` | 28,659.4 ± 236.7 | 0.95 | 98.57% | — | — |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.95 | 48 | `IVFPQ+CPU-Refine` | `M=16, m=800` | 19,970.2 ± 249.3 | 0.9535 | 99.81% | — | — |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.95 | 48 | `IVFPQ+CPU-Refine` | `M=16, m=1600` | 11,260.9 ± 143.8 | 0.9538 | 99.90% | — | — |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.95 | 48 | `IVFPQ+CPU-Refine` | `M=16, m=2048` | 9,361.4 ± 103.6 | 0.9538 | 99.94% | — | — |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.99 | 160 | `IVFPQ+CPU-Refine` | `M=16, m=100` | 10,923.3 ± 87.9 | 0.9313 | 81.63% | — | — |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.99 | 160 | `IVFPQ+CPU-Refine` | `M=16, m=200` | 10,456.5 ± 339.2 | 0.9713 | 93.00% | — | — |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.99 | 160 | `IVFPQ+CPU-Refine` | `M=16, m=400` | 9,356.0 ± 254.7 | 0.9859 | 97.89% | — | — |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.99 | 160 | `IVFPQ+CPU-Refine` | `M=16, m=800` | 8,246.1 ± 59.1 | 0.9911 | 99.64% | — | — |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.99 | 160 | `IVFPQ+CPU-Refine` | `M=16, m=1600` | 5,397.1 ± 12.2 | 0.9917 | 99.90% | — | — |
| sift100m | RTX 4080 (16 GB, No Cap) | 0.99 | 160 | `IVFPQ+CPU-Refine` | `M=16, m=2048` | 4,981.8 ± 17.1 | 0.9918 | 99.91% | — | — |

## 2. Speedup Under Near-Lossless / Strictly Lossless Regime (`m=2048` / Exact Baselines)

| Dataset | R | `TRIDENT (ub=1)` QPS | `TRIDENT (ub=0)` QPS | `CM-FP32-OOC` QPS | `CPU-24T` QPS | `IVFPQ (m=2048)` QPS [Rec@10 / Ov%] | `ub=1` vs `ub=0` | `ub=1` vs `CM-OOC` | `ub=1` vs `CPU-24T` | `ub=1` vs `m=2048` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| gist1m | 0.95 | 23,894.1 | 3,506.8 | 111.0 | 210.5 | 5,900.0 [0.9614 / 99.67%] | 6.81x | 215.26x | 113.51x | 4.05x |
| gist1m | 0.99 | 17,852.6 | 3,340.5 | 81.1 | 103.8 | 4,587.2 [0.9907 / 98.65%] | 5.34x | 220.13x | 171.99x | 3.89x |
| bge1024 | 0.95 | 62,329.1 | 5,052.9 | 484.4 | 1,328.1 | 7,749.5 [0.9491 / 98.65%] | 12.34x | 128.67x | 46.93x | 8.04x |
| bge1024 | 0.99 | 27,426.5 | 3,277.7 | 92.6 | 104.0 | 5,546.0 [0.9783 / 95.47%] | 8.37x | 296.18x | 263.72x | 4.95x |
| sift10m | 0.95 | 86,478.3 | 47,439.9 | 523.9 | 1,582.1 | 14,943.1 [0.9522 / 99.97%] | 1.82x | 165.07x | 54.66x | 5.79x |
| sift10m | 0.99 | 53,179.4 | 35,319.5 | 260.4 | 568.7 | 9,295.5 [0.9906 / 99.93%] | 1.51x | 204.22x | 93.51x | 5.72x |
| deep10m | 0.95 | 98,341.9 | 58,393.2 | 696.9 | 1,784.0 | 17,855.3 [0.9621 / 99.93%] | 1.68x | 141.11x | 55.12x | 5.51x |
| deep10m | 0.99 | 54,317.8 | 39,408.0 | 359.4 | 639.3 | 11,440.9 [0.9906 / 99.93%] | 1.38x | 151.13x | 84.96x | 4.75x |
| sift100m | 0.95 | 40,449.1 | 30,119.7 | 108.9 | 322.9 | 9,361.4 [0.9538 / 99.94%] | 1.34x | 371.43x | 125.27x | 4.32x |
| sift100m | 0.99 | 21,244.6 | 18,100.7 | 42.9 | 123.5 | 4,981.8 [0.9918 / 99.91%] | 1.17x | 495.21x | 172.02x | 4.26x |

## 3. Speedup Under Matched-Recall Target (`Recall@10 >= R`, Smallest Qualifying `m`)

| Dataset | R Target | `TRIDENT (ub=1)` QPS [Rec@10 / Ov%] | Smallest `m` with `Rec@10 >= R` | `IVFPQ+Refine (m)` QPS | `IVFPQ+Refine (m)` Rec@10 | `IVFPQ+Refine (m)` Strict Overlap (%) | `TRIDENT (ub=1)` vs Matched-Recall `IVFPQ (m)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| gist1m | 0.95 | 23,894.1 [0.9622 / 100.00%] | `m=1600` | 7,206.3 | 0.9602 | 99.14% | **3.32x** |
| gist1m | 0.99 | 17,852.6 [0.9937 / 100.00%] | `m=2048` | 4,587.2 | 0.9907 | 98.65% | **3.89x** |
| bge1024 | 0.95 | 62,329.1 [0.9520 / 100.00%] | not reached (at `m=2048`: 0.9491) | 7,749.5 (`m=2048`) | 0.9491 | 98.65% | not reached (**8.04x** vs `m=2048`) |
| bge1024 | 0.99 | 27,426.5 [0.9920 / 100.00%] | not reached (at `m=2048`: 0.9783) | 5,546.0 (`m=2048`) | 0.9783 | 95.47% | not reached (**4.95x** vs `m=2048`) |
| sift10m | 0.95 | 86,478.3 [0.9523 / 100.00%] | `m=400` | 56,518.3 | 0.9516 | 99.75% | **1.53x** |
| sift10m | 0.99 | 53,179.4 [0.9908 / 100.00%] | `m=800` | 18,005.7 | 0.9903 | 99.90% | **2.95x** |
| deep10m | 0.95 | 98,341.9 [0.9623 / 100.00%] | `m=400` | 63,323.6 | 0.9588 | 98.87% | **1.55x** |
| deep10m | 0.99 | 54,317.8 [0.9908 / 100.00%] | `m=1600` | 12,886.5 | 0.9906 | 99.93% | **4.22x** |
| sift100m | 0.95 | 40,449.1 [0.9540 / 100.00%] | `m=400` | 28,659.4 | 0.9500 | 98.57% | **1.41x** |
| sift100m | 0.99 | 21,244.6 [0.9922 / 100.00%] | `m=800` | 8,246.1 | 0.9911 | 99.64% | **2.58x** |

## 4. Nsight Compute Per-Pass PCIe Traffic (`ncu_perpass_1m10m_A100.csv` & `ncu_perpass_100m_RTX4080.csv`)

| Dataset | R / nprobe | `CM-FP32-OOC` Total (MB) | `TRIDENT (ub=0)` Pass-1 / Pass-2 / Total (MB) | `TRIDENT (ub=1)` Pass-1 / Pass-2 / Total (MB) | Pass-1 Reduction (`ub=0 -> ub=1`) | Total Reduction (`ub=0 -> ub=1`) | Total Reduction vs `CM-FP32-OOC` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| gist1m | 0.99 (np=128) | 705,800.70 | 4,774.74 / 235.14 / 5,009.88 | 333.08 / 242.34 / 575.42 | -93.0% (14.34x) | 8.71x | 1,226.58x (99.92%) |
| bge1024 | 0.99 (np=128) | 557,507.71 | 5,219.19 / 65.01 / 5,284.20 | 339.62 / 66.89 / 406.51 | -93.5% (15.37x) | 13.00x | 1,371.45x (99.93%) |
| sift10m | 0.99 (np=96) | 157,251.48 | 278.03 / 17.75 / 295.78 | 40.73 / 11.76 / 52.49 | -85.4% (6.83x) | 5.63x | 2,995.84x (99.97%) |
| deep10m | 0.99 (np=96) | 109,749.06 | 203.20 / 8.91 / 212.11 | 32.80 / 8.53 / 41.33 | -83.9% (6.20x) | 5.13x | 2,655.43x (99.96%) |
| sift100m | 0.95 (np=48) | 163,974.28 | 199.51 / 15.23 / 214.74 | 35.59 / 19.60 / 55.19 | -82.2% (5.61x) | 3.89x | 2,971.09x (99.97%) |
| sift100m | 0.99 (np=160) | 471,090.07 | 195.22 / 23.04 / 218.26 | 35.64 / 25.26 / 60.90 | -81.7% (5.48x) | 3.58x | 7,735.47x (99.99%) |

## 5. Step E3 & Task A 100M GPU Baselines Status (`gpu_baselines_100m_RTX4080.jsonl`)

| Mode | Status | Elapsed (s) | Build (s) | Details / Results / Error |
| :--- | :---: | :---: | :---: | :--- |
| `faiss_ivfflat_device` | `OOM` | 59.4 | — | `RuntimeError: Error in virtual void* faiss::gpu::StandardGpuResourcesImpl::allocMemory(const faiss::gpu::AllocRequest&) at /project/faiss/faiss/gpu/StandardGpuResources.cpp:5` |
| `cagra_device_ivfpq` | `OK` | 640.6 | 640.5 | `[{"itopk": 128, "error": "CuvsException: RAFT failure at file=/__w/cuvs/cuvs/cpp/src/neighbors/cagra.cuh line=339: Number of query dimensions should equal number"}, {"itopk": 256, "error": "CuvsException: RAFT failure at file=/__w/cuvs/cuvs/cpp/src/neighbors/cagra.cuh line=339: Number of query dimensions should equal number"}, {"itopk": 512, "error": "CuvsException: RAFT failure at file=/__w/cuvs/cuvs/cpp/src/neighbors/cagra.cuh line=339: Number of query dimensions should equal number"}]` |
| `cagra_uvm_ivfpq` | `TIMEOUT` | 21603.3 | — | `timeout_s=21600` |
| `cagra_uvm_nn_descent` | `TIMEOUT` | 21629.6 | — | `timeout_s=21600` |
| `faiss_ivfflat_uvm` | `OK` | 738.2 | 241.3 | `[{"recall_target": 0.95, "nprobe": 48, "qps_mean": 43.9, "qps_std": 0.3, "recall_at_10": 0.9541, "strict_overlap_pct": 100.0}, {"recall_target": 0.99, "nprobe": 160, "qps_mean": 13.6, "qps_std": 0.0, "recall_at_10": 0.9923, "strict_overlap_pct": 100.0}]` |
| `cagra_device_ivfpq_v11` | `OOM` | 633.0 | 632.5 | `1M self-check succeeded (idx.dim=128, output shape (1000,10)); on 100M, graph (100M*32*4B = 12.80 GB) + dataset (100M*128*1B = 12.80 GB) = 25.60 GB > 16 GB VRAM (used 14626/16376 MiB after build, peak 14626 MiB), so cuVS drops the dataset attachment (idx.dim=0) and fails at search with: CuvsException: RAFT failure at file=/__w/cuvs/cuvs/cpp/src/neighbors/cagra.cuh line=339: Number of query dimensions should equal number of dimensions in the index.
Obtained 27 stack frames
#1 in /home/mazaiying/trident/.venv/lib/python3.10/site-packages/libcuvs/lib64/libcuvs.so: raft::logic_error::logic_error(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) +0x91 [0x708526f5ea91]
#2 in /home/mazaiying/trident/.venv/lib/python3.10/site-packages/libcuvs/lib64/libcuvs.so: void cuvs::neighbors::cagra::search<unsigned char, unsigned int, unsigned int>(raft::resources const&, cuvs::neighbors::cagra::search_params const&, cuvs::neighbors::cagra::index<unsigned char, unsigned int> const&, cuda::std::__4::mdspan<unsigned char const, cuda::std::__4::extents<long, 18446744073709551615ul, 18446744073709551615ul>, cuda::std::__4::layout_right, raft::host_device_accessor<cuda::std::__4::default_accessor<unsigned char const>, (raft::memory_type)2> >, cuda::std::__4::mdspan<unsigned int, cuda::std::__4::extents<long, 18446744073709551615ul, 18446744073709551615ul>, cuda::std::__4::layout_right, raft::host_device_accessor<cuda::std::__4::default_accessor<unsigned int>, (raft::memory_type)2> >, cuda::std::__4::mdspan<float, cuda::std::__4::extents<long, 18446744073709551615ul, 18446744073709551615ul>, cuda::std::__4::layout_right, raft::host_device_accessor<cuda::std::__4::default_accessor<float>, (raft::memory_type)2> >, cuvs::neighbors::filtering::base_filter const&) +0xfc2 [0x708528573932]
#3 in /home/mazaiying/trident/.venv/lib/python3.10/site-packages/libcuvs/lib64/libcuvs_c.so(+0xb28f3) [0x7085266b28f3]
#4 in /home/mazaiying/trident/.venv/lib/python3.10/site-packages/libcuvs/lib64/libcuvs_c.so(+0xb3b44) [0x7085266b3b44]
#5 in /home/mazaiying/trident/.venv/lib/python3.10/site-packages/libcuvs/lib64/libcuvs_c.so: cuvsCagraSearch +0x3f [0x7085266b55bf]
#6 in /home/mazaiying/trident/.venv/lib/python3.10/site-packages/cuvs/neighbors/cagra/cagra.cpython-310-x86_64-linux-gnu.so(+0x2a5b0) [0x70862dbe95b0]
#7 in /home/mazaiying/trident/.venv/bin/python: PyObject_Call +0xba [0x58147244651a]
#8 in /home/mazaiying/trident/.venv/bin/python: _PyEval_EvalFrameDefault +0x2d09 [0x58147242b659]
#9 in /home/mazaiying/trident/.venv/bin/python: _PyFunction_Vectorcall +0x6c [0x581472438e7c]
#10 in /home/mazaiying/trident/.venv/bin/python: PyVectorcall_Call +0xc7 [0x5814724468a7]
#11 in /home/mazaiying/trident/.venv/lib/python3.10/site-packages/cuvs/common/resources.cpython-310-x86_64-linux-gnu.so(+0xb63d) [0x70863c47063d]
#12 in /home/mazaiying/trident/.venv/bin/python: _PyObject_MakeTpCall +0x273 [0x581472431a63]
#13 in /home/mazaiying/trident/.venv/bin/python: _PyEval_EvalFrameDefault +0x519b [0x58147242daeb]
#14 in /home/mazaiying/trident/.venv/bin/python: _PyFunction_Vectorcall +0x6c [0x581472438e7c]
#15 in /home/mazaiying/trident/.venv/bin/python: _PyEval_EvalFrameDefault +0x313 [0x581472428c63]
#16 in /home/mazaiying/trident/.venv/bin/python(+0x1d268c) [0x5814724cf68c]
#17 in /home/mazaiying/trident/.venv/bin/python: PyEval_EvalCode +0x85 [0x5814724cf5d5]
#18 in /home/mazaiying/trident/.venv/bin/python(+0x2042aa) [0x5814725012aa]
#19 in /home/mazaiying/trident/.venv/bin/python(+0x1febf3) [0x5814724fbbf3]
#20 in /home/mazaiying/trident/.venv/bin/python(+0x97160) [0x581472394160]
#21 in /home/mazaiying/trident/.venv/bin/python: _PyRun_SimpleFileObject +0x1bb [0x5814724f649b]
#22 in /home/mazaiying/trident/.venv/bin/python: _PyRun_AnyFileObject +0x44 [0x5814724f61f4]
#23 in /home/mazaiying/trident/.venv/bin/python: Py_RunMain +0x371 [0x5814724f31e1]
#24 in /home/mazaiying/trident/.venv/bin/python: Py_BytesMain +0x37 [0x5814724c1ec7]
#25 in /lib/x86_64-linux-gnu/libc.so.6(+0x29d90) [0x7086f0629d90]
#26 in /lib/x86_64-linux-gnu/libc.so.6: __libc_start_main +0x80 [0x7086f0629e40]
#27 in /home/mazaiying/trident/.venv/bin/python: _start +0x2e [0x5814724c1dde]` |
