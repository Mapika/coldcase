# SIEGE — GPU transform-engine record hunt on the monster cells

## Design (2026-08-21, re-scoped from the chain port)

Engine: **exact full-grid distance-count transforms** (`dct.cu` + `dctlib.py`
+ `gpuchain.py`), not ported local search. One axis-DP pass sequence
(n axes, recurrence `newA_d = A_d + T_{d-1} - A_{d-1}` per length-q fiber)
computes, for EVERY x in Z_q^n at once, the exact number of S-members at each
distance d<=R. Three uses per round:

- S = code multiset  -> cnt(x) coverage multiplicity map (uint16, exact)
- S = [cnt==0]       -> gain(x) = exact #uncovered a codeword at x would cover
- S = [cnt==1]       -> loss(c) = exact private coverage of every codeword

Search = LNS (ruin & recreate): ruin low-loss/random/clustered subsets,
recreate by **lazy greedy** placement (submodularity makes stale gains upper
bounds, so pop/re-evaluate-exactly/commit reproduces true greedy); plus
zero-loss peeling after each solve. Every "solved" is re-counted from scratch
on device, and every record claim goes through `record_gate.py` =
`campaign.record()` = `cov/verify_cov.py` re-read from disk. Base-36 digits.

Correctness gates passed (`test_dct.py`): transform vs brute force on 8 small
grids x 3 reps (cnt/gain/loss/ball ops all exact, incl. multiplicities);
K6(6,3) full cnt map + 200 gain probes exact; engine uncovered == verifier
uncovered on a real unsolved code (130 == 130).

Memory plan (int32 layers A_0..A_R in place, sum folded into A_0 on the last
axis; uint16 exact cnt): bytes/word = 4(R+1)+2.
- K7(10,5): 7.4 GB;  transform 63 ms
- K8(10,4): 23.7 GB; transform 143 ms
- K9(10,4): ~77 GB (fits 97 GB HBM)
- K10(10,5): 1e10 words -> host-coherent path needed; DEFERRED.

Chain engine (`chainsolve.cu`) demoted to baseline/fallback; kept intact.

Resource rules honored: foreign-GPU guard (60 s poll, free all device memory,
resume on idle), <=8 CPU cores niced for driver+verify, host RAM use ~small.

## Event log
- `08-21 13:32` K8(10,4): exact-greedy build from scratch
- `08-21 13:35` K8(10,4): cover established at M=14946 (recording via gate)
- `08-21 13:37` K8(10,4): gate submissions only below M=11776 (Keri ub=11776, best recorded=None)
- `08-21 13:37` K8(10,4): seeding from seed_K8_10_4_M11776.txt
- `08-21 13:37` K8(10,4): cover established at M=11776
- `08-21 13:38` K8(10,4): gate submissions only below M=11776 (Keri ub=11776, best recorded=None)
- `08-21 13:38` K8(10,4): seeding from seed_K8_10_4_M11776.txt
- `08-21 13:38` K7(10,5): gate submissions only below M=981 (Keri ub=1225, best recorded=981)
- `08-21 13:38` K8(10,4): cover established at M=11776
- `08-21 13:38` K7(10,5): seeding from K7_10_5_M981.txt
- `08-21 13:38` K7(10,5): cover established at M=981
- `08-21 13:42` K8(10,4): notch M=11744 FAILED (best uncov=313641, step=32, budget=240s)
- `08-21 13:43` K7(10,5): notch M=977 FAILED (best uncov=8, step=4, budget=300s)
- `08-21 13:44` K8(10,4): gate submissions only below M=11776 (Keri ub=11776, best recorded=14946)
- `08-21 13:44` K7(10,5): gate submissions only below M=981 (Keri ub=1225, best recorded=981)
- `08-21 13:44` K8(10,4): seeding from seed_K8_10_4_M11776.txt
- `08-21 13:45` K8(10,4): cover established at M=11776
- `08-21 13:45` K7(10,5): seeding from K7_10_5_M981.txt
- `08-21 13:45` K7(10,5): cover established at M=981
- `08-21 13:46` K9(9,5): gate submissions only below M=729 (Keri ub=729, best recorded=729)
- `08-21 13:46` K10(9,5): gate submissions only below M=1088 (Keri ub=1088, best recorded=None)
- `08-21 13:46` K9(9,5): exact-greedy build from scratch
- `08-21 13:46` K10(9,5): exact-greedy build from scratch

## Strategy notes

- `13:5x` Scratch-greedy lands at ~7.8x the sphere-packing bound q^n/ball on
  every cell tried (K7(10,5): 981 = 7.85x125; K8(10,4): 14946 = 7.6x1961).
  Consequence: greedy beats incumbents with redundancy >~ 8 outright and
  cannot touch direct-sum incumbents at redundancy <= 6.
- `13:5x` K8(10,4) direct sum 23x512 is LOCALLY RIGID: every codeword
  privately covers ~9800 words (product structure is private-uniform), so a
  32-word notch instantly owes ~313k words and swap-LNS cannot repay it.
  Same expected for K9(10,4) 27x729. These need a structured (factor-slice)
  ruin or better factors, not word-level LNS.
- `13:5x` Task's "K9(10,4): 481-3969" is Keri's K9(10,5) (K9(10,4) is
  3872-19683). K9(10,5) is loose (10.5) and greedy-attackable.
- BLITZ queue by incumbent redundancy (est = 7.9 q^n/ball):
  K8(10,5) inc 2461(ours) est 1773 | K9(10,5) inc 3969 est 2999 (91 GB, must
  run alone) | K10(9,5) inc 1088 est 948 | K9(9,5) inc 729 est 652 |
  skipped as greedy-proof: K10(9,4) 8500/est 8867, K9(9,4), K8(9,4), K8(9,5).
- K10(10,5) (space 1e10, redundancy 11.6): needs 64-bit/host-coherent
  variant; deferred until the <=4.2e9 cells are harvested.
- `08-21 13:49` K8(10,4): notch M=11744 FAILED (best uncov=368148, step=32, budget=240s)
- `08-21 13:52` K7(10,5): notch M=977 FAILED (best uncov=8, step=4, budget=420s)
- `08-21 13:52` K9(9,5): cover established at M=685
- `08-21 13:52` K8(10,5): gate submissions only below M=2461 (Keri ub=2461, best recorded=2461)
- `08-21 13:52` K8(10,5): exact-greedy build from scratch
- `08-21 13:57` K9(9,5): notch M=681 FAILED (best uncov=13, step=4, budget=300s)
- `08-21 13:57` K9(9,5) M=685: VERIFIED+RECORDED
- `08-21 13:59` K7(10,5): notch M=979 FAILED (best uncov=1, step=2, budget=420s)
- `08-21 14:01` K10(9,5): cover established at M=1055
- `08-21 14:02` K9(9,5): notch M=683 FAILED (best uncov=4, step=2, budget=300s)
- `08-21 14:04` K9(9,5): SOLVED M=684 in 105s (step=1); gate submitted
- `08-21 14:06` K7(10,5): notch M=980 FAILED (best uncov=1, step=1, budget=420s)
- `08-21 14:06` K7(10,5): gate submissions only below M=981 (Keri ub=1225, best recorded=981)
- `08-21 14:06` K7(10,5): seeding from K7_10_5_M981.txt
- `08-21 14:06` K7(10,5): cover established at M=981
- `08-21 14:06` K10(9,5): notch M=1051 FAILED (best uncov=13, step=4, budget=300s)
- `08-21 14:06` K10(9,5) M=1055: VERIFIED+RECORDED
- `08-21 14:07` K8(10,5): cover established at M=1884
- `08-21 14:09` K9(9,5): notch M=683 FAILED (best uncov=3, step=1, budget=300s)
- `08-21 14:09` K9(9,5) M=684: VERIFIED+RECORDED
- `08-21 14:12` K10(9,5): notch M=1053 FAILED (best uncov=4, step=2, budget=300s)
- `08-21 14:12` K8(10,5): notch M=1880 FAILED (best uncov=9, step=4, budget=300s)
- `08-21 14:13` K7(10,5): notch M=977 FAILED (best uncov=11, step=4, budget=420s)
- `08-21 14:17` K10(9,5): notch M=1054 FAILED (best uncov=1, step=1, budget=300s)
- `08-21 14:17` K8(10,5): notch M=1882 FAILED (best uncov=3, step=2, budget=300s)
- `08-21 14:17` K8(10,5) M=1884: VERIFIED+RECORDED
- `08-21 14:19` K9(10,5): gate submissions only below M=3969 (Keri ub=3969, best recorded=None)
- `08-21 14:19` K9(10,5): exact-greedy build from scratch
- `08-21 14:36` K9(10,5): cover established at M=3393
- `08-21 14:44` K9(10,5): notch M=3389 FAILED (best uncov=11, step=4, budget=420s)
- `08-21 14:51` K9(10,5): notch M=3391 FAILED (best uncov=10, step=2, budget=420s)
- `08-21 14:59` K9(10,5): notch M=3392 FAILED (best uncov=2, step=1, budget=420s)
- `08-21 14:59` K9(10,5) M=3393: VERIFIED+RECORDED
- `08-21 15:02` K9(9,5): gate submissions only below M=684 (Keri ub=729, best recorded=684)
- `08-21 15:02` K9(9,5): seeding from K9_9_5_M684.txt
- `08-21 15:02` K8(10,5): gate submissions only below M=1884 (Keri ub=2461, best recorded=1884)
- `08-21 15:02` K9(9,5): cover established at M=684
- `08-21 15:02` K7(10,5): gate submissions only below M=981 (Keri ub=1225, best recorded=981)
- `08-21 15:02` K10(9,5): gate submissions only below M=1055 (Keri ub=1088, best recorded=1055)
- `08-21 15:02` K7(10,5): seeding from K7_10_5_M981.txt
- `08-21 15:02` K8(10,5): seeding from K8_10_5_M1884.txt
- `08-21 15:02` K10(9,5): seeding from K10_9_5_M1055.txt
- `08-21 15:02` K7(10,5): cover established at M=981
- `08-21 15:02` K8(10,5): cover established at M=1884
- `08-21 15:02` K10(9,5): cover established at M=1055

## Results summary (2026-08-21 window)

Five NEW verified upper bounds K_q(n,R) < Keri incumbent, all through
record_gate.py = campaign.record() = verify_cov.py re-read from disk
(sidecars in cov/results/):

| cell | Keri 2011 | NEW | delta | how |
|---|---|---|---|---|
| K_7(10,5) | 160-1225 | **981** | -244 (-19.9%) | scratch greedy (130 s) |
| K_8(10,5) | 287-2461 | **1884** | -577 (-23.5%) | scratch greedy |
| K_9(9,5)  | 120-729  | **684** | -45 (-6.2%) | greedy 685 + LNS notch |
| K_9(10,5) | 481-3969 | **3393** | -576 (-14.5%) | scratch greedy (91 GB cell) |
| K_10(9,5) | 174-1088 | **1055** | -33 (-3.0%) | scratch greedy |

Walls (best uncovered at one notch below, hundreds of LNS rounds):
K7@980: 1, K9(9,5)@683: 4->..., K10(9,5)@1054: 1, K8(10,5)@1883: 9,
K9(10,5)@3392: 2. The engine plateaus 0-2 notches below its greedy build;
the greedy construction is the record-maker, LNS polishes 1 notch sometimes.

Rigid cells (NOT cracked): K8(10,4) 11776 = 23x512 and K9(10,4) 19683 =
27x729 direct sums are private-uniform (~9800 privately-covered words per
codeword); word-level notches instantly owe ~300k words. Need factor-level
attack or structured slice ruin. K10(9,4) 8500, K9(9,4) 5103, K8(9,4) 2944,
K8(9,5) 384 are below the greedy-reachable redundancy (~7.9x packing bound).

Throughput, K8(9,4)@940 60 s head-to-head:
- transform engine: 5.86M uncovered (971 exact-greedy placements + LNS,
  69 transforms) — WHILE co-tenant with a 91 GB siege
- chain kernel (this dir, idle, 256 chains): 11.2M best, 926 agg moves/s
- CPU engine portfolio (cov/engine NOTES §4.2, 6 cores): ~6.44M best ever
The transform engine beats the best recorded number for this cell in half
the budget under contention.
- `08-21 15:09` K9(9,5): notch M=683 FAILED (best uncov=3, step=1, budget=400s)
- `08-21 15:09` K10(9,5): notch M=1054 FAILED (best uncov=3, step=1, budget=400s)
- `08-21 15:09` K7(10,5): notch M=980 FAILED (best uncov=1, step=1, budget=400s)
- `08-21 15:10` K8(10,5): notch M=1883 FAILED (best uncov=5, step=1, budget=400s)
- `08-21 15:11` K8(10,5): SOLVED M=1883 in 67s (step=1); gate submitted
- `08-21 15:13` K7(10,5): SOLVED M=980 in 193s (step=1); gate submitted
- `08-21 15:18` K8(10,5): notch M=1881 FAILED (best uncov=6, step=2, budget=400s)
- `08-21 15:20` K7(10,5): notch M=978 FAILED (best uncov=5, step=2, budget=400s)
- `08-21 15:23` K10(9,5): notch M=1054 FAILED (best uncov=1, step=1, budget=800s)
- `08-21 15:23` K9(9,5): notch M=683 FAILED (best uncov=3, step=1, budget=800s)
- `08-21 15:25` K8(10,5): notch M=1882 FAILED (best uncov=3, step=1, budget=400s)
- `08-21 15:25` K8(10,5) M=1883: VERIFIED+RECORDED
- `08-21 15:26` K7(10,5): notch M=979 FAILED (best uncov=1, step=1, budget=400s)
- `08-21 15:38` K8(10,5): notch M=1882 FAILED (best uncov=1, step=1, budget=800s)
- `08-21 15:40` K7(10,5): notch M=979 FAILED (best uncov=2, step=1, budget=800s)
- `08-21 15:40` K7(10,5) M=980: VERIFIED+RECORDED
- `08-21 15:49` K9(9,5): notch M=683 FAILED (best uncov=3, step=1, budget=1600s)
- `08-21 15:49` K9(9,5): descent ended at M=684 (floor=1, wall_fails=3, stats={'transforms': 2647, 'placements': 6118, 'removals': 6121, 'rounds': 294, 'recounts': 5}, paused=0s)
- `08-21 15:50` K10(9,5): notch M=1054 FAILED (best uncov=1, step=1, budget=1600s)
- `08-21 15:50` K10(9,5): descent ended at M=1055 (floor=1, wall_fails=3, stats={'transforms': 1408, 'placements': 4232, 'removals': 4235, 'rounds': 141, 'recounts': 2}, paused=0s)
- `08-21 15:56` K7(10,5): notch M=979 FAILED (best uncov=1, step=1, budget=965s)
- `08-21 15:56` K7(10,5): descent ended at M=980 (floor=1, wall_fails=3, stats={'transforms': 4174, 'placements': 13180, 'removals': 13187, 'rounds': 472, 'recounts': 8}, paused=0s)
- `08-21 15:56` K8(10,5): notch M=1882 FAILED (best uncov=2, step=1, budget=1065s)
- `08-21 15:56` K8(10,5): descent ended at M=1883 (floor=1, wall_fails=3, stats={'transforms': 1899, 'placements': 9413, 'removals': 9420, 'rounds': 186, 'recounts': 4}, paused=0s)
- `08-21 16:00` K9(10,5): gate submissions only below M=3393 (Keri ub=3969, best recorded=3393)
- `08-21 16:00` K9(10,5): seeding from K9_10_5_M3393.txt
- `08-21 16:00` K9(10,5): cover established at M=3393
- `08-21 16:00` K6(6,3): gate submissions only below M=41 (Keri ub=41, best recorded=41)
- `08-21 16:00` K6(6,3): exact-greedy build from scratch
- `08-21 16:00` K6(6,3): cover established at M=55

## STAND-DOWN (2026-08-21 ~17:00) — GPU released to the user

Per coordinator order: all GPU work stopped. Verified: 0 siege processes,
GPU memory 3 MiB, no compute apps. All monitors/watchers stopped — nothing
will auto-relaunch GPU work. Do not resume until explicitly told.

Final verified record tally of this siege (all: campaign.record gate PASS
= verify_cov.py numpy (+pure where q^n<=3e8) on the file re-read from disk;
plus verify_independent.py dilation PASS on every one):

| cell | Keri 2011 | NEW | files |
|---|---|---|---|
| K_7(10,5) | 160-1225 | **980** | cov/results/K7_10_5_M980.txt |
| K_8(10,5) | 287-2461 | **1883** | cov/results/K8_10_5_M1883.txt |
| K_9(9,5)  | 120-729  | **684** | cov/results/K9_9_5_M684.txt |
| K_9(10,5) | 481-3969 | **3393** | cov/results/K9_10_5_M3393.txt |
| K_10(9,5) | 174-1088 | **1055** | cov/results/K10_9_5_M1055.txt |

(981/1884/685 predecessor files also gated+stored; verify_independent PASS
confirmed for 980/1883/981/1884/684/1055/3393.)

Open items, exact state:
- **K8(10,4) long siege: NOT YET RUN.** Deficit-aware notch is implemented
  and smoke-tested (siege.py --ruin-extra/--kmin/--kmax; ruin past the
  deficit so repair has slots — fixes the pure-swap flaw of the first
  attempt). Suggested launch when GPU returns:
  `python3 siege.py -q 8 -n 10 -R 4 --hours 8 --seed-file
   <scratch>/seed_K8_10_4_M11776.txt --step0 32 --notch-budget 2400
   --ruin-extra 512 --kmin 64 --kmax 768`
  Seed file lives in the session scratchpad (gpc/seed_K8_10_4_M11776.txt);
  rebuildable in seconds: lincov K8(6,2)@512 x covsearch2 K8(4,2)@23,
  constructions.direct_sum. Caveat logged above: the direct sum is
  private-uniform (~9800/word) and greedy repair reconstructs it; treat as
  experimental. Higher-EV side bet running: K8(4,2)=22 would give
  K8(10,4) <= 11264 outright.
- **K9(10,5) long wall-test: PENDING.** Was launched (step0=1,
  notch-budget 1600s, escalating 1600/3200/6400) and killed ~15 min in by
  the stand-down before any notch resolved. 3393 stands, wall-test
  incomplete; relaunch:
  `python3 siege.py -q 9 -n 10 -R 5 --hours 3.5 --step0 1 --notch-budget 1600`
- **CPU-only lottery still running** (allowed core budget, no GPU): 6x
  covsearch2 nice-15 single-thread on K8(4,2)@22 (open Keri gap 22-23),
  4 h timeout, outputs in scratchpad gpc/K8_4_2_M22_s*.txt. If one solves:
  gate it, then K8(10,4) <= 22*512 = 11264 via direct sum.
- K10(10,5)/K10(10,4)/K9(10,4): out of engine's current reach (1e10 space
  needs host-coherent variant; K9(10,4)/K8(10,4) direct sums rigid).
- `08-22 08:26` K9(10,5): gate submissions only below M=3393 (Keri ub=3969, best recorded=3393)
- `08-22 08:26` K9(10,5): seeding from K9_10_5_M3393.txt
- `08-22 08:26` K9(10,5): cover established at M=3393
- `08-22 08:52` K9(10,5): notch M=3392 FAILED (best uncov=3, step=1, budget=1600s)
- `08-22 09:46` K9(10,5): notch M=3392 FAILED (best uncov=3, step=1, budget=3200s)
- `08-22 11:33` K9(10,5): notch M=3392 FAILED (best uncov=2, step=1, budget=6400s)
- `08-22 11:33` K9(10,5): descent ended at M=3393 (floor=1, wall_fails=3, stats={'transforms': 6173, 'placements': 48599, 'removals': 48602, 'rounds': 553, 'recounts': 11}, paused=0s)
- `08-22 11:33` K8(10,4): gate submissions only below M=11776 (Keri ub=11776, best recorded=14946)
- `08-22 11:33` K8(10,4): seeding from seed_K8_10_4_M11776.txt
- `08-22 11:33` K8(10,4): cover established at M=11776

2026-08-22 relaunch (user handed GPU back):
- K9(10,5) wall-test COMPLETE: notch M=3392 failed 3x at step0=1,
  budgets 1600/3200/6400s (best uncov 3/2/2). **3393 is WALLED** —
  added to scripts/paper_tables.py WALLED set.
- K8(4,2)=22 lottery LOST: all 6 seeds stalled at exactly 36 uncovered
  after 1.4e9 iters each. No direct-sum shortcut; gap 22-23 stays open.
- K8(10,4) deficit-notch siege (8h, --ruin-extra 512) started 11:33 UTC,
  log: logs/queue_k8104_notch.log.
- `08-22 12:13` K8(10,4): notch M=11744 FAILED (best uncov=827296, step=32, budget=2400s)
- `08-22 12:53` K8(10,4): notch M=11760 FAILED (best uncov=88959, step=16, budget=2400s)
- `08-22 13:33` K8(10,4): notch M=11768 FAILED (best uncov=327344, step=8, budget=2400s)
- `08-22 14:10` HC ENGINE: dct.cu evolved into dctcov core (dct_hc.cu +
  dctcore_core.cuh/_hamming.cuh/_torus.cuh -> libdct_hc.so; libdct.so and the
  running K8(10,4) siege untouched). New: (1) per-array memory modes
  HBM/ATS-malloc/managed for GH200 LPDDR (Mission A), 64-bit extraction,
  q^n cap lifted to 2^40; (2) ball kernels split blocks/word (old: ONE block
  per word) + ox mask+nibble table as switchable FMT + owner-trick losspass
  (Mission B); (3) per coordinator directive: problem-plugin seams — state
  core problem-blind, Hamming (incl. mixed-radix) and torus-L-inf-domination
  plugins, search core (gpuchain.Engine) untouched and problem-blind.
- `08-22 14:10` HC GATES ALL PASS on libdct_hc.so: original gate1 (8 small
  grids x3 reps exhaustive) + gate2 (K6(6,3)); layers/cnt in ATS and managed
  == HBM bit-identical (5 mode combos); owner-trick loss == transform loss
  (K6(6,3), K7(8,4)); ox-format walks == v1; mixed-radix axes=[3,4,5,3,4]
  R=2 exact vs brute force; torus plugin exact vs brute force on [7,7]R1,
  [5,5,5]R1, [9,8]R2; PLUS full-stack reuse proof: problem-blind search core
  found a 9-vertex dominating set of the 9x9 king torus (known optimum
  ceil(9/3)^2=9), verified by independent brute force.
- `08-22 14:14` K8(10,4): notch M=11772 FAILED (best uncov=410006, step=4, budget=2400s)
- `08-22 14:34` K6(9,5): gate submissions only below M=119 (Keri ub=144, best recorded=119)
- `08-22 14:34` K6(9,5): seeding from K6_9_5_M119.txt
- `08-22 14:34` K6(9,5): cover established at M=119
- `08-22 14:35` K6(9,5): notch M=115 FAILED (best uncov=85, step=4, budget=30s)
- `08-22 14:35` K6(9,5): notch M=117 FAILED (best uncov=29, step=2, budget=30s)
- `08-22 14:36` K6(9,5): notch M=118 FAILED (best uncov=9, step=1, budget=30s)
- `08-22 14:37` K6(9,5): notch M=118 FAILED (best uncov=14, step=1, budget=58s)
- `08-22 14:37` K6(9,5): descent ended at M=119 (floor=1, wall_fails=2, stats={'transforms': 1597, 'placements': 1704, 'removals': 1713, 'rounds': 201, 'recounts': 2}, paused=0s)
- `08-22 15:00` HC: GH200 HMM pitfall found and fixed — GPU-hot plain-malloc
  pages get MIGRATED into HBM by access counters (86 GB ATS cell filled HBM
  to 96.8/97.9 GB and thrashed; small "LPDDR" benches were secretly HBM).
  Fix: posix_memalign + mbind(MPOL_BIND,node0) in MEM_ATS. Verified: HBM
  flat, outputs bit-identical. Honest pinned numbers (CONTENDED, official
  idle pairs tonight): K8(9,4) transform HBM 31ms / ATS 0.60s (~19x) /
  managed 0.96s; K9(10,5) at scale: 91 GB cell, layers 84 GB LPDDR, cnt HBM,
  transform 36-40 s, uncov arithmetic exact.
- `08-22 15:00` Ball-kernel geometry rehearsal (CONTENDED): K8(9,4) M=940
  single-word update 2.67 -> 0.36 ms (7.3x), gather64 2.81 -> 1.98 ms,
  gather1024 11.1 -> 7.8 ms (old libdct.so vs libdct_hc.so). Owner-loss
  rehearsal: transform 17 ms vs owner 41 ms — owner trick LOSES at this
  M*ball/(nRq^n) ratio; kept for sparse regimes.
- `08-22 15:00` R=6 layer extension gated (K3(7,6) generic + K10(7,6)
  template exact vs brute force); full 7-group gate suite re-PASSED on the
  MAXR=6 build. siege.py end-to-end smoke on new lib OK (K6(9,5), seeded,
  notched, clean exit). RUNBOOK_HC.md written for the 19:30 handover;
  paper draft at paper/covering/section_gpu_hc_draft.tex with PENDING-IDLE
  markers for tonight's paired numbers.
- `08-22 14:40` K6(9,5) M=100: gate submission superseded (never claimed); file kept at a
- `08-22 14:40` K6(9,5) M=99: gate submission superseded (never claimed); file kept at b
- `08-22 14:54` K8(10,4): notch M=11774 FAILED (best uncov=402717, step=2, budget=2400s)
- `08-22 15:05` Gate ETA calibrated: verify_cov.verify_numpy on a
  K10(10,5)-scale M=4800 code = 1237 s (~21 min, niced). Recorder (now
  non-blocking, 3 lanes + superseding queue) absorbs descent submissions.
  K10 pattern tables: R=5 16.3M pats 131 MB 0.2s; R=6 128M pats 1 GB 1.8s.
- `08-22 15:34` K8(10,4): notch M=11775 FAILED (best uncov=405175, step=1, budget=2400s)
- `08-22 16:54` K8(10,4): notch M=11775 FAILED (best uncov=394895, step=1, budget=4800s)
- `08-22 19:33` K8(10,4): notch M=11775 FAILED (best uncov=77229, step=1, budget=9504s)
- `08-22 19:33` K8(10,4): descent ended at M=11776 (floor=1, wall_fails=3, stats={'transforms': 42176, 'placements': 1165272, 'removals': 1165337, 'rounds': 2794, 'recounts': 52}, paused=0s)
- `08-22 19:33` K8(10,4) 8h deficit-notch siege: CONCLUSIVELY WALLED. Full
  ladder failed: deficits 32/16/8/4/2/1 all FAILED (step=1 twice; best
  uncov 405175@2400s, 394895@4800s, and final M=11775 attempt 77229@9504s;
  2794 rounds, 1.17M placements, 42176 transforms). The 23x512 direct sum
  is private-uniform rigid — word-level LNS cannot un-knit it even one
  codeword below the incumbent. GPU queue COMPLETE; GPU handed to HC program.
- `08-22 19:43` PHASE 1 OFFICIAL PAIRED BENCHES (idle GPU, logs/
  phase1_paired.log). Ball kernels old libdct.so -> libdct_hc.so:
  K8(9,4) update1 0.798->0.027 ms (29.6x), gather64 0.99->0.23 ms (4.3x),
  gather1024 5.3->4.1 ms; K8(10,4) update1 1.335->0.036 ms (37x),
  gather64 1.68->0.45 ms (3.7x), gather1024 11.3->9.1 ms. ox mask+nibble
  table: parity within noise (kept as switchable option, v1 default).
  Transforms K8(10,4): baseline HBM 0.143s == hc HBM 0.143s (no
  regression); ATS+mbind LPDDR 3.39-3.43s (23.7x); managed 8.6-8.8s (60x).
  K9(10,5) at scale idle: recount 22.1s / gain 24.9s / loss 26.1s
  (91 GB cell, 84 GB layers LPDDR). Owner-trick losspass HONEST NEGATIVE:
  K8(9,4) 17 vs 28 ms, K8(10,4) 145 vs 669 ms (transform wins; owner only
  viable when M*|ball| << nRq^n). End-to-end greedy K8(9,4): 40.4 -> 28.7s
  (1.41x), M 2758 vs 2759.
- `08-22 20:02` K10(10,5) FEASIBILITY MEASURED (first 1e10-word exact
  transform here): 260.2 GB arrays (240 GB layers ATS+mbind LPDDR, cnt
  20 GB HBM), init 10.7 s, transforms gain 73.0 s / loss 71.9 s /
  recount 109.5 s (idle GPU, nice-5). Host-coherent verdict: VIABLE —
  LNS round ~3-4 min. Draft tex numbers COMPLETE for the v2 freeze.
- `08-22 20:03` K10(10,5) HC siege LAUNCHING: scratch exact-greedy seed
  (est ~4.8k vs Keri ub 7106, redundancy 11.6), step0=64,
  notch-budget 1800, 8h, log logs/queue_k10105_hc.log.
- `08-22 19:59` K10(10,5): gate submissions only below M=7106 (Keri ub=7106, best recorded=None)
- `08-22 19:59` K10(10,5): exact-greedy build from scratch
