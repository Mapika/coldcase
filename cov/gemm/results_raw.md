# Raw measurements, 2026-08-20, GH200 480GB (shared with another tenant)

Host: GH200, 64 Neoverse-V2 cores, load average > 100 throughout (production
sweep).  CPU runs `nice -n 12`, `OMP_NUM_THREADS=1`, one thread, process CPU
time.  GPU runs wall clock; the other tenant held 39–53 GB at 100% utilisation
and 179–656 W for the whole session.

torch 2.7.0 / CUDA 12.8.  All GPU codes re-verified by `cov/verify_cov.py`.

---

## 1. Dense-GEMM calibration (`bench_kernel.py` and inline probes)

```
fp16 8192x8192x8192            333.9 TFLOPS
fp16 4096x4096x4096            310.5 TFLOPS
int8 8192x8192x8192 (_int_mm)   55.6 TOPS
int8 4096x4096x4096 (_int_mm)   53.2 TOPS

thin-K scan, fp16 65536 x K x 8192:
  K=  48   57.7 TFLOPS      (n*q for K6(8,4))
  K=  72   83.4 TFLOPS      (n*q for K8(9,4))
  K= 128  152.6 TFLOPS
  K= 512  267.1 TFLOPS
  K=2048  344.5 TFLOPS
```

## 2. Score kernel, K6(8,4), N = 1 679 616 rows (Gpair/s)

```
B      eager fp16   eager int8   compiled fp16 (fused epilogue)
16        22.0         18.7        -
64        40.8         27.4        160.9
256       73.4         41.2        129.7
1024      83.1         42.1        336.7
4096      83.4         42.9        -
```
Compiled and eager results checked bit-identical.

## 3. Launch / synchronisation latency

```                              heavy contention   light contention
async kernel launch (queued)          6.6 us              -
launch + cuda.synchronize()        2458.8 us          589.1 us
nonzero + .item() on 2M elements   2896.4 us          889.3 us
```

## 4. K6(8,4), M = 169, seed file cov/opt/seeds_K6_8_4_M169.txt (854 uncovered)

### 4.1 Iterations to level (4 seeds each, medians)

```
level   CPU p5b+wide   GPU focused GEMM --cand 32
          iters  CPUs      iters
 100        75   0.107        43
  50      127.5  0.167        84
  20      400    0.541       338
  10     1497    2.03        697
   5     4711    6.31       1666
   3     9256   13.0        2760
```
CPU per-seed iterations to level 20: 420, 533, 380, 323.
GPU per-seed iterations to level 20: 308, 516, 367, 290.
Moves per iteration: CPU ~1 088 (27.2 codewords x n(q-1)=40);
GPU 51 491 x 32 = 1 647 712.  Ratio 1 514x.

### 4.2 GPU per-run (45 s wall, --cand 32, --warmup 12)

```
seed  best  iters   ms_med   ttt_wall(target 20)  ttt_it
 1      2    9187    1.919      6.949             308
 2      3   10991    4.062      2.596             516
 3      3   11156    4.358      1.621             367
 4      2    3438   20.659      0.606             290
```
(ms_med varies 10x with the other tenant's load; see section 3.)

### 4.3 CPU quality at 20 CPU s (target 1, so the full budget is used)

```
p5b --wide : uncovered 3, 1, 1, 3    (iters 14096, 5930, 11790, 14176)
p5b        : uncovered 4, 1, 1, 2    (iters 14528, 8357, 5492, 14672)
```

### 4.4 Matched-budget harness run (results_K6_8_4.csv, 20 s budget)

Every output verified by `verify_cov.py --method both`; all agreed.
Note the CPU rows stop at the target, so their "best" is the target, not a
quality figure -- use 4.3 for quality.

```
engine                   n  med best   med ttt   solved  med iters
cpu --preset p5b         4      20.0     1.039    4/4          772
cpu --preset p5b --wide  4      20.0     0.556    4/4          400
gpu cand=32              4       9.0     8.532    4/4          800
```
(The GPU rows here were taken during the heavy-contention phase.)

## 5. K8(9,4), M = 2944, seed cov/opt/seeds_K8_9_4_M2944.txt (11 640 uncovered)

### 5.1 Iterations to level

```
level    CPU iters  CPU s     GPU iters   GPU s (iters x 40.7 ms)
10800       22.5     2.43        11        1.00  (measured ttt_wall)
 8000      131      14.9         62        2.5
 5000      334      35.2        174        7.1
 3000      599      58.0        347       14.1
 2000      862      80.0        578       23.5
 1600     1044     101.6        824       33.5
 1200     not reached in 110 CPU s        1788   72.8
```
CPU: 2 seeds per level.  GPU: seed 2 trace, --cand 32.

### 5.2 Quality at a matched 100 s budget

```
CPU p5b --wide, 100 CPU s : 1466, 1429, 1398, 1543   (median 1448, ~1150 iters)
GPU focused GEMM, 100 s   : 1137, 1137               (median 1137, ~2100 iters)
```

GPU RESULT lines:
```
seed 1: uncovered=1137 iters=1978 time=100.018 ttt_wall=0.9645 ttt_it=10 ms_med=40.741
seed 2: uncovered=1137 iters=2218 time=100.017 ttt_wall=0.9960 ttt_it=11 ms_med=40.693
```

### 5.3 Independent verification of the GPU code

```
$ python3 cov/verify_cov.py g894.txt -q 8 -n 9 -R 4 --method numpy
parameters: q=8 n=9 R=4   q^n=134217728   ball volume=333166
codewords : M=2944
method numpy: uncovered=1137   (45.01s)
```
Exactly the number the solver reported.

## 6. Formulation B (whole-space relocation, `gemmsolve.py`) -- negative

K6(8,4) from the same seed, 400 iterations: descends 854 -> ~150 and then
oscillates between 160 and 220.  Best 150 after 400 iterations, against 20 for
the CPU in 400 iterations.  Cause: at a local optimum the best position for a
codeword is where it already is, so forbidding the no-op turns the search into
a random walk and allowing it wastes a GEMM.
