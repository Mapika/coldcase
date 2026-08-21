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
