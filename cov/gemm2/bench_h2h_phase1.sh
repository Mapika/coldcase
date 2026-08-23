#!/bin/bash
# Phase 1 official paired benchmarks (idle GPU). Run from cov/gemm2.
# Output: logs/phase1_paired.log   ("bench_h2h" in name: guard-invisible)
set -u
cd "$(dirname "$0")"
LOG=logs/phase1_paired.log
{
echo "=== PHASE 1 PAIRED BENCHES $(date -u) ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "--- 1. ball kernels: baseline libdct.so vs libdct_hc.so ---"
GPUDCT_SO=libdct.so    python3 bench_h2h_hc.py bench_ball 8 9 4 940
GPUDCT_SO=libdct_hc.so python3 bench_h2h_hc.py bench_ball 8 9 4 940
GPUDCT_SO=libdct.so    python3 bench_h2h_hc.py bench_ball 8 10 4 11776
GPUDCT_SO=libdct_hc.so python3 bench_h2h_hc.py bench_ball 8 10 4 11776
echo "--- 2. ox table format (fmt=1, hc lib) ---"
GPUDCT_SO=libdct_hc.so python3 bench_h2h_hc.py bench_ball 8 9 4 940 1
GPUDCT_SO=libdct_hc.so python3 bench_h2h_hc.py bench_ball 8 10 4 11776 1
echo "--- 3. transforms: HBM baseline / HBM hc / ATS / managed ---"
GPUDCT_SO=libdct.so    python3 bench_h2h_hc.py bench_transform 8 10 4
GPUDCT_SO=libdct_hc.so python3 bench_h2h_hc.py bench_transform 8 10 4 0 0
GPUDCT_SO=libdct_hc.so python3 bench_h2h_hc.py bench_transform 8 10 4 1 0
GPUDCT_SO=libdct_hc.so python3 bench_h2h_hc.py bench_transform 8 10 4 2 0
GPUDCT_SO=libdct_hc.so python3 bench_h2h_hc.py bench_transform 9 10 5 1 0 3
echo "--- 4. owner-loss vs transform-loss ---"
GPUDCT_SO=libdct_hc.so python3 bench_h2h_hc.py bench_loss 8 9 4 940
GPUDCT_SO=libdct_hc.so python3 bench_h2h_hc.py bench_loss 8 10 4 11776
echo "--- 5. end-to-end greedy paired ---"
GPUDCT_SO=libdct.so    python3 bench_h2h_hc.py bench_greedy 8 9 4
GPUDCT_SO=libdct_hc.so python3 bench_h2h_hc.py bench_greedy 8 9 4
echo "=== PHASE 1 DONE $(date -u) ==="
} 2>&1 | tee -a "$LOG"
