# Journal hardening log — covering-codes paper (DCC submission)

Agent: journal-hardening, started 2026-08-23 ~10:5x UTC.
Scope: referee-demanded computational work. Tasks:
1. GPU K10(10,6) rematch (retuned, 10h) then K10(10,5) wall-test.
2. CPU: run remaining "↓" cells to stagnation (K6(10,5), K8(8,5), K8(10,6), K7(9,5)).
3. K8(4,2) exact: settle 22 vs 23 (ILP).
4. Extend LB primal self-test to n=6..10 (referee M10).
5. External baselines: plain ILP + plain tabu vs engine (referee M11).

Non-negotiables honored: record gate = verify_cov.py numpy (re-read from disk)
+ verify_independent.py; >=60 GB host RAM free; nice CPU; setsid long runs.

## Machine state at start
- GPU: idle, 0 MiB used / 97871 MiB.
- Host RAM: 525 GB total, 444 GB free.
- CPU: 64 cores.
- scipy 1.8.0 (no scipy.optimize.milp — need highspy/pulp for Task 3).

## Event log
- `08-23 12:34` TASK 1a LAUNCH: K10(10,6) rematch siege. Cmd (cwd cov/gemm2):
  `GPUDCT_SO=libdct_hc.so setsid nohup nice -n 5 python3 siege.py -q 10 -n 10 -R 6 --hours 10 --scratch --step0 16 --notch-budget 2400 > logs/jh_k10106_rematch.log 2>&1 &`
  PID 1455486. Retune vs failed 6h run: 10h budget (greedy endgame + repair
  were deadline-starved at 6h), step0=16 wider notches, notch-budget 2400s.
  Memory: ~273 GB RSS (layers in LPDDR via ATS+mbind), cnt 20 GB HBM.
  RAM check at launch: 444 GB free before, need >=60 free during — OK.
  ETA end: ~22:35 UTC. K10(10,5) wall-test queued AFTER this ends (sequential).
- `08-23 12:36` TASK 2 LAUNCH (pair 1, 20 workers each, nice 10):
  K6(10,5): `campaign.py -q 6 -n 10 -R 5 --descend-from 485 --seed-code cov/results/K6_10_5_M486.txt --workers 20 -t 3600 --rounds 2 --extra --preset p5b`, PID 1455948, log logs/t2_K6_10_5.log
  K8(8,5): same pattern, --descend-from 78, seed K8_8_5_M79.txt, PID 1456121, log logs/t2_K8_8_5.log
  Semantics = task spec: two consecutive 3600s x20-worker rounds at M-1;
  success -> campaign.record (verify_cov gate + sidecar) and continue down;
  first unsolved M stops. verify_independent will be run on any new record.
  K7(9,5) and K8(10,6) queued as pair 2 after these finish (~2h/notch).
  RAM after launch: 163 GB free (GPU siege resident) — above the 60 GB floor.
- `08-23 12:47` TASK 3 LAUNCH: K8(4,2) ILP, highspy 1.15.1 (installed).
  Model: 4096 bin vars, 4096 covering rows (323 nnz each), |B_2|=323 verified.
  Symmetry breaking: x_0000 = 1 only. VALIDITY: Hamming distance is
  translation-invariant, so C -> C - c (any c in C) maps covers to covers of
  equal size containing 0; checked numerically (random translation of a
  greedy cover stays a cover). No other fixing used.
  Runs (nice 10): opt mode tl=21600s threads=12 warm-start=known 23-word code
  translated to contain 0000 (accepted, primal=23 from start), PID 1457136,
  log logs/t3_k842_opt.log; feas22 mode (cardinality row <=22) tl=21600s
  threads=8, PID 1457295, log logs/t3_k842_feas.log.
  Note: 30/60s smoke runs show root LP not yet solved (dual bound 1);
  LP relaxation value ~4096/323=12.68, so B&B must climb 12.7 -> 22. Prior
  evidence (6x1.4e9-iter tabu lottery all stalled at 36 uncovered at M=22)
  suggests infeasibility at 22 is the likely verdict.
- `08-23 13:00` TASK 4 COMPLETE — EXTENDED SELFTEST PASSED (all 7 cases OK),
  original --selftest also re-PASSED (11 cases). check_primal complexity:
  code_primal_point is Theta(|C|^3 * n) exact-integer orbit counting, plus
  exact-Fraction linear (O(#lin * nnz)) and PSD checks (orbit-sized blocks).
  Battery re-runnable: cov/journal_hardening/selftest_extended.py.
  Coverage table (primal feed-in of verified record codes):
  | n | case | |C| | vars | lin | psd | build | check | result |
  | 6 | K8(6,4) | 19 | 64 | 703 | 48 | 0.3s | 0.1s | OK obj=|C|^3 |
  | 6 | K12(6,4) | 39 | 64 | 703 | 48 | 0.3s | 0.1s | OK |
  | 7 | K6(7,3) | 227 | 95 | 1099 | 60 | 0.8s | 19.6s | OK |
  | 8 | K8(8,5) | 79 | 136 | 1640 | 75 | 1.8s | 1.1s | OK |
  | 8 | K6(8,4) | 166 | 136 | 1640 | 75 | 1.0s | 6.2s | OK |
  | 9 | K6(9,5) | 119 | 189 | 2356 | 90 | 3.8s | 4.8s | OK |
  | 10 | K8(10,6) | 335 | 256 | 3282 | 108 | 8.7s | 74.1s | OK |
  With the original battery (n=2..5), the SDP transcription's primal check now
  covers every shape n=5..10 demanded by referee M10 (n=5 via K6(5,3) etc. in
  the original; n=6..10 via our record codes). No case exceeded 83s.
- `08-23 13:05` TASK 5a ILP size audit (plain set-cover ILP, one row per word,
  one column per word, nnz = q^n * |B_R|):
  K8(6,4): 262,144 vars/rows, |B_4|=43,653, nnz=1.14e10 (~137 GB matrix)
  K12(6,4): 2.99M, |B_4|=248,117, nnz=7.4e11 (~8.9 TB)
  K13(6,4): 4.8M, nnz=1.7e12; K14(6,4): 7.5M, nnz=3.6e12; K15(6,4): 11.4M,
  nnz=7.2e12 — all unbuildable in 525 GB host RAM, let alone solvable in 1h.
  K6(7,3): 279,936 vars/rows, |B_3|=4,936, nnz=1.38e9 (~17 GB) — the one
  attemptable cell; ILP run scheduled when the K8(4,2) ILPs release cores.
  Honest table entry for the unbuildable cells: "model exceeds memory".
- `08-23 13:2x` TASK 2 RESULT: K6(10,5) <= 485 (was 486). Solved in round 1
  of the M=485 attack (20 workers x 3600s, preset p5b, seeded from M486).
  Gate: campaign.record -> verify_cov PASS, stored
  cov/results/K6_10_5_M485.txt + sidecar; verify_independent dilate PASS
  (all 60,466,176 words covered at radius 5). Descent continues at M=484.
- `08-23 14:0x` TASK 5: K8(6,4) M=19 (Keri-1 = our record): base(plain tabu)
  SOLVED+verified, p5b SOLVED+verified within 8x3600s each. The smallest
  cell is baseline-reachable. Next: K12(6,4) M=40.
- `08-23 14:40` TASK 2 WALLED: K8(8,5) at M=78. Two consecutive 3600s x 20-worker
  rounds (preset p5b, seeded from the verified M=79 record), best uncovered
  7 then 6, throughput ~37 moves/s/worker. Per protocol the "79 ↓" marker
  becomes: "79, direct attack at 78 walled (2x 20h CPU-hours)".
  Launching pair 2: K7(9,5) descend-from 315 and K8(10,6) descend-from 334.
- `08-23 14:44` TASK 2 pair 2 launched: K7(9,5) --descend-from 315 (PID
  1478662, log t2_K7_9_5.log), K8(10,6) --descend-from 334 --st 2 (PID
  1478723, log t2_K8_10_6.log); both 20 workers x 2 x 3600s, p5b, seeded
  from the current record files. RAM 117 GB free after launch.
  CONTENTION NOTE: ~70 niced single-thread workers + 20 ILP threads on 64
  cores (~1.5x oversubscribed). t3 ILP budgets are wall-clock; effective
  CPU share reduced — will be stated with the attempted-budget record.
  t5 pairing unaffected (both configs run under identical load).
- `08-23 15:0x` TASK 5: K12(6,4) M=40 (Keri-1): base SOLVED+verified,
  p5b SOLVED+verified. Now pairing at M=39 (our record).
- `08-23 15:5x` TASK 2 RESULT: K6(10,5) <= 484 (round 2 of the M=484 attack).
  Gate: verify_cov PASS (campaign), stored K6_10_5_M484.txt + sidecar;
  verify_independent dilate PASS. Descent continues at M=483.
- `08-23 16:1x` TASK 5: K12(6,4) M=39 (our record): base SOLVED+verified,
  p5b SOLVED+verified. Plain tabu reproduces this record at 8x3600s.
  Next: K13(6,4) M=45 (Keri-1 = our record).
- `08-23 17:1x` TASK 5: K13(6,4) M=45 (Keri-1 = our record): base + p5b both
  SOLVED+verified. Next: K14(6,4) M=51.
- `08-23 17:2x` TASK 2 BREAKTHROUGH: K7(9,5) direct attack is cascading.
  Campaign (20 workers, p5b, seeded from monotonicity-derived M316) has
  solved and gate-verified EVERY M from 315 down to 247 (files
  K7_9_5_M*.txt + sidecars in cov/results, each verify_cov PASS), currently
  attacking M=246. Keri 2011 ub for the cell is 323; the paper's v2 value
  was 316 (monotonicity extension, never directly attacked). The referee
  intuition was right: direct attack improves it massively.
  verify_independent dilate launched on M247 (running, background).
- `08-23 17:2x` TASK 2 RESULT: K8(10,6) <= 334 (was 335; Keri 342). Gate:
  verify_cov PASS, stored K8_10_6_M334.txt + sidecar. Descent at M=333.
  verify_independent dilate launched (background).
- `08-23 17:2x` TASK 2: K6(10,5) M=483 round 1 FAILED at best uncovered=1;
  round 2 running.
- `08-23 17:3x` verify_independent dilate PASS on both interim records:
  K7(9,5) <= 247 (all 40,353,607 words of Z_7^9 covered at radius 5) and
  K8(10,6) <= 334 (all 1,073,741,824 words of Z_8^10 covered at radius 6).
  Both cells still descending; the final settled M of each will be
  independently verified again.
- `08-23 17:5x` TASK 2 CELL DONE: K6(10,5). New record 484 (was 486; two
  verified steps 485, 484, both verify_cov + verify_independent PASS).
  WALLED at M=483: two consecutive 3600s x 20-worker rounds failed
  (best uncovered 1 and 1). Honest statement for the journal: descent
  stagnated at 484 under 2x20 CPU-hours at 483.
- `08-23 18:2x` TASK 5: K14(6,4) M=51 (Keri-1): base + p5b both
  SOLVED+verified. Now at M=50 (our record).
- `08-23 18:5x` TASK 3: feas22 run ENDED (tl 21600s wall, 8 threads, nice 10,
  CONTENDED box — effective CPU share reduced, see 14:44 note). Outcome:
  UNDECIDED. No 22-cover found; dual bound reached 14.0 (root LP ~12.68);
  proving infeasibility needs >22. Log: logs/t3_k842_feas.log.
- `08-23 18:5x` TASK 3: opt run ENDED (tl 21600s, 12 threads, contended).
  UNDECIDED: primal stayed 23 (warm start; no 22 found), dual bound 14.0
  after root cuts (root LP 12.68 -> 13.56 -> 14.0). HiGHS presolve kept
  3774 rows x 4095 cols, 1.22M nnz. Logs: t3_k842_opt.log, t3_k842_feas.log.
  VERDICT SO FAR: 12h of HiGHS (opt+feas) cannot close 14 vs 22-23.
  ESCALATION: SAT decision attempt at M<=22 (pysat 1.9 + CaDiCaL):
  covering clauses (4096 x |B_2|=323), seqcounter/totalizer cardinality
  <=22, x_0=1 (translation-invariance, as in the ILP), plus lex-leader
  symmetry breaking restricted to generators of the stabilizer of word 0
  (coordinate perms + symbol perms fixing 0) — sound for SAT and UNSAT:
  the solution class containing word 0 is stabilizer-closed, so its
  lex-min member satisfies every added constraint.
- `08-23 19:2x` TASK 5: K14(6,4) M=50 (our record): base + p5b both
  SOLVED+verified. Next: K15(6,4) M=58 (Keri-1).
- `08-23 19:5x` TASK 1b PLAN (queued): after the K10(10,6) siege exits,
  wall-test K10(10,5) at M=5798: siege.py -q 10 -n 10 -R 5 --hours 4
  --step0 1 --notch-budget 1800 (auto-escalation gives 1800/3600/7200s at
  step=1, i.e. the escalating-budget single-step wall-test). Seeds from
  recorded K10_10_5_M5799.txt automatically.
- `08-23 20:3x` TASK 5: K15(6,4) M=58 (Keri-1): base + p5b both
  SOLVED+verified. Next: K15(6,4) M=57 (our record), then K6(7,3).
- `08-23 21:3x` TASK 5 SEPARATION: K15(6,4) M=57 (our record): plain tabu
  (base) FAILED with best uncovered = 3648; p5b SOLVED+verified. Identical
  8x3600s budgets, same seeds, simultaneous runs. First cell where the
  production configuration separates from the plain-tabu baseline.
  Next: K6(7,3) M=245 (Keri-1), then M=227 (our record).
- `08-23 22:4x` TASK 1a: K10(10,6) COVER ESTABLISHED at M=802 (Keri ub=826,
  -24). The 10h retune did what the 6h run could not: greedy completed
  (plus repair/peel) essentially at the deadline. Gate submission running
  (verify_cov numpy re-read from disk, ~20 min at 1e10 words).
  verify_independent dilate will follow on the recorded file.
- `08-23 23:0x` TASK 5 SEARCH BASELINES COMPLETE (t5_results.jsonl; all runs
  8 workers x 3600s per config, same seeds, paired-simultaneous, fresh
  starts, solves re-verified by verify_cov numpy):
  | cell | Keri | target | plain tabu (base) | engine (p5b) |
  | K8(6,4) | 20 | 19 (K-1 = ours) | SOLVED | SOLVED |
  | K12(6,4) | 41 | 40 | SOLVED | SOLVED |
  | K12(6,4) | 41 | 39 (ours) | SOLVED | SOLVED |
  | K13(6,4) | 46 | 45 (K-1 = ours) | SOLVED | SOLVED |
  | K14(6,4) | 52 | 51 | SOLVED | SOLVED |
  | K14(6,4) | 52 | 50 (ours) | SOLVED | SOLVED |
  | K15(6,4) | 59 | 58 | SOLVED | SOLVED |
  | K15(6,4) | 59 | 57 (ours) | FAIL (uncov 3648) | SOLVED |
  | K6(7,3) | 246 | 245 | SOLVED | SOLVED |
  | K6(7,3) | 246 | 227 (ours) | FAIL (uncov 22) | FAIL (uncov 21) |
  Reading: plain tabu reproduces the small n=6 records given 8 CPU-hours,
  separates at the K15(6,4) frontier, and neither config reaches the
  K6(7,3)=227 record from scratch at this budget (that record came from
  the seeded descending campaign) — an honest account of where the
  engine's margin lives. ILP column: all cells except K6(7,3) unbuildable
  (see 13:05 size audit); K6(7,3) ILP attempt queued for after the GPU
  siege frees host RAM.
- `08-23 ~21:22` TASK 1a RECORD: **K10(10,6) <= 802 VERIFIED+RECORDED**
  (Keri ub 826, -24, -2.9%). Gate = record_gate/campaign.record/verify_cov
  numpy re-read from disk; stored cov/results/K10_10_6_M802.txt + sidecar.
  verify_independent dilate launched (PID 1565977, log
  journal_hardening/logs/vi_k10106_802.log; expect hours at 1e10 words).
  Siege continues notching until its 22:34 deadline (M=786 failed at
  uncov=176, step halved).
- `08-23 23:31` SESSION NOTE: agent session hit a usage limit ~23:00 and was
  reset; all setsid jobs survived (coordinator-verified). Entries below
  reconstruct the boundary events.
- `08-23 22:39` TASK 1a SIEGE ENDED cleanly: descent ended at M=802 (stats:
  267 transforms, 870 placements, 4 LNS rounds; notch ladder 786@uncov176 /
  794@uncov77 FAILED at 2400s — greedy is the record-maker at 1e10).
  K10(10,6): Keri 826 -> **802**.
- `08-23 22:40` TASK 1b LAUNCH: K10(10,5) wall-test at M=5798. Cmd (cwd
  cov/gemm2): `GPUDCT_SO=libdct_hc.so setsid nohup nice -n 5 python3
  siege.py -q 10 -n 10 -R 5 --hours 4 --step0 1 --notch-budget 1800 >
  logs/jh_k10105_walltest.log 2>&1 &` — PID 1576020. Seeded from
  K10_10_5_M5799.txt, cover re-established at 5799; single-step notches,
  budgets escalate 1800/3600/7200s; three step-1 failures => walled.
- `08-23 23:3x` TASK 1a: verify_independent dilate PASS on K10_10_6_M802.txt
  — all 10,000,000,000 words of Z_10^10 within distance 6.
  **K10(10,6) <= 802 DOUBLE-VERIFIED** (verify_cov numpy + independent
  dilation). Record complete per the standard gate.
- `08-23 23:5x` TASK 2 CELL DONE: K7(9,5). Direct attack cascaded 315 -> 240
  (all 76 intermediate covers gate-verified + stored with sidecars).
  **K7(9,5) <= 240** (paper v2: 316 via monotonicity; Keri ub 323).
  verify_independent dilate PASS on the settled M=240 file (all 40,353,607
  words). WALLED at M=239: two consecutive 3600s x 20-worker rounds failed
  (best uncov 2, then 1). Cell closed.
- `08-24 00:0x` TASK 5a LAUNCH: K6(7,3) plain ILP baseline (k673_ilp.py),
  wrapper PID 1586300, log logs/t6_k673_ilp.log. Three sequential 1h HiGHS
  runs, 8 threads, deliberately plain (no symmetry, no warm start):
  min (what plain ILP achieves), feas245 (Keri-1), feas227 (our record).
  Model: 279,936 cols/rows, nnz 1.382e9, built as numpy CSC -> passModel.
- `08-24 01:2x` TASK 5a: K6(7,3) plain ILP, mode min: TIME LIMIT at 3600s
  solve (after 266s build+pass): primal=inf, dual=-inf, 0 vars set —
  HiGHS produced neither a feasible cover nor any dual bound in 1h on the
  1.382e9-nnz model. feas245 running next.
- `08-24 01:0x` TASK 1b: wall-test notch M=5798 FAILED #1 (uncov 3,
  1800s); escalates to 3600s.
- `08-24 01:1x` TASK 2: K8(10,6) descended to 326 (verified+stored),
  attacking 325.
