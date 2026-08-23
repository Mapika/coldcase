# Post-queue runbook (2026-08-22, GPU-engine agent)

Trigger: "GPU QUEUE COMPLETE" in cov/gemm2/logs/gpu_queue.log (~19:30 UTC).
All commands from cov/gemm2 with GPUDCT_SO=libdct_hc.so unless noted.

## Phase 1 — official paired benchmarks (idle GPU, ~40 min)

Paired = same cell, same seed, sequential runs, idle GPU. Baseline lib is
the untouched libdct.so; HC lib is libdct_hc.so.

1. Ball kernels (Mission B geometry win), K8(9,4) M=940 and K8(10,4) M=11776:
   GPUDCT_SO=libdct.so    python3 bench_h2h_hc.py bench_ball 8 9 4 940
   GPUDCT_SO=libdct_hc.so python3 bench_h2h_hc.py bench_ball 8 9 4 940
   GPUDCT_SO=libdct.so    python3 bench_h2h_hc.py bench_ball 8 10 4 11776
   GPUDCT_SO=libdct_hc.so python3 bench_h2h_hc.py bench_ball 8 10 4 11776
2. ox table format (fmt arg = 1), HC lib only, same cells:
   python3 bench_h2h_hc.py bench_ball 8 9 4 940 1
   python3 bench_h2h_hc.py bench_ball 8 10 4 11776 1
3. Transforms HBM vs LPDDR (Mission A table), K8(10,4) fits both:
   GPUDCT_SO=libdct.so    python3 bench_h2h_hc.py bench_transform 8 10 4
   python3 bench_h2h_hc.py bench_transform 8 10 4 0 0
   python3 bench_h2h_hc.py bench_transform 8 10 4 1 0
   python3 bench_h2h_hc.py bench_transform 8 10 4 2 0   # managed, for the record
   also at scale: python3 bench_h2h_hc.py bench_transform 9 10 5 1 0 3
4. Owner-loss vs transform-loss (honest even if a loss):
   python3 bench_h2h_hc.py bench_loss 8 9 4 940
   python3 bench_h2h_hc.py bench_loss 8 10 4 11776
5. End-to-end greedy paired (old vs new lib), K8(9,4):
   GPUDCT_SO=libdct.so    python3 bench_h2h_hc.py bench_greedy 8 9 4
   GPUDCT_SO=libdct_hc.so python3 bench_h2h_hc.py bench_greedy 8 9 4

Log every number in SIEGE.md.

## Phase 2 — K10(10,5) host-coherent measurement + siege (primary target)

Keri ub=7106, lb=632; redundancy 11.6; greedy estimate ~7.9*612 = 4834.
Memory: layers 6x40 GB ATS (mbind node0), cnt 20 GB HBM; check free -g
first (need 240 GB + >=60 GB slack; verifiers add ~2x30 GB later).

1. Feasibility number for the paper (first 1e10 transform ever here):
   python3 bench_h2h_hc.py bench_transform 10 10 5 1 0 3
2. Siege (records go through record_gate = verify_cov re-read from disk):
   GPUDCT_SO=libdct_hc.so setsid nohup nice -n 5 python3 siege.py \
     -q 10 -n 10 -R 5 --hours 8 --scratch --step0 64 --notch-budget 1800 \
     > logs/queue_k10105_hc.log 2>&1 &
   (step0 large: expect greedy ~4.8k, huge slack to 7106 - descend fast.)
3. On every VERIFIED+RECORDED file additionally run (niced, background):
   python3 ../verify_independent.py ../results/K10_10_5_M*.txt \
     -q 10 -n 10 -R 5 --method dilate

## Phase 3 — K10(10,6) (bonus target, R=6 path gated PASS)

Keri ub=826, lb=122; redundancy 10.6; greedy estimate ~7.9*78 = 618.
Memory: layers 7x40=280 GB ATS + cnt 20 GB HBM; ball table 1.28e8 pats
= 1 GB HBM. RAM check mandatory (280+60 rule; do NOT run concurrent with
K10(10,5) engine still resident).
   python3 bench_h2h_hc.py bench_transform 10 10 6 1 0 2
   then siege as above with -R 6 --step0 16 --notch-budget 2400.

## Phase 4 — K10(10,4) (honest long shot)

ub=45900, redundancy 6.74 < greedy reach (~7.9); expect greedy ~53k FAIL.
Run one greedy build for the delimitation datapoint (cheap after Phase 2
numbers exist; skip if time is short):
   python3 bench_h2h_hc.py bench_greedy 10 10 4 7200

## Standing rules
- >=60 GB host RAM free AT ALL TIMES (check free -g before each phase).
- Verify gate: record_gate.py only; nothing is a record until
  verify_cov.py + verify_independent.py pass on the file re-read from disk.
- Log everything in SIEGE.md, including failures.
- If the user takes the GPU: Engine.guard yields automatically; do not
  relaunch until idle.
