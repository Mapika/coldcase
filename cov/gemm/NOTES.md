# Covering codes as dense integer GEMM on tensor cores

Can the GH200's low-precision matrix units beat the CPU at `K_q(n,R)` local
search?  The premise is attractive: the first CUDA port (`cov/NOTES.md` §7) lost
because every move was a latency-bound scattered walk over a `q^n` counter
array, and a GEMM formulation replaces that with dense arithmetic at a claimed
~2000 INT8 TOPS.

**Answer, measured: it beats one CPU core on the larger cell and loses to the
CPU socket everywhere — and it loses on energy by 1.5 to 3 orders of magnitude.**

| | `K_6(8,4)` (`q^n` = 1.7 M) | `K_8(9,4)` (`q^n` = 134 M) |
|---|---|---|
| GEMM/GPU vs **one** Grace core, wall clock | ~1× (level) | **~3× faster** |
| GEMM/GPU vs the **64-core** Grace | ~60× slower | ~21× slower |
| per joule | ~100–1000× worse | ~50× worse |

The trend with instance size is in the GPU's favour and is not an artefact —
the CPU evaluator slows down when its counter array leaves cache while the GEMM
does not — but two ceilings cap how far it can go, and neither is an
implementation detail that more engineering would remove:

1. **The one-hot encoding forces an inner dimension of `K = n·q ≈ 50–70`, and
   tensor cores at that `K` run at 6% of this GPU's own GEMM asymptote.**
   Measured: a `65536 × K × 8192` fp16 GEMM does 57.7 TFLOPS at `K = 48` and
   344.5 TFLOPS at `K = 2048`.  The 2000 TOPS in the premise is unreachable *by
   construction*, off by ~30× before a line of solver code is written.
2. **Move-evaluation breadth has brutally diminishing returns on this problem.**
   The GEMM engine evaluates a neighbourhood **1514× wider** than the CPU's, and
   *exactly*, and it buys only **1.2× to 3.4× fewer iterations** to reach the
   same uncovered count.  Even a free, instantaneous move evaluator would win
   at most a small constant factor over `covsearch2 --preset p5b --wide`.

Caveat stated once and meant throughout: **the GPU was shared with another
tenant's job for this entire session** (39–53 GB, 100% utilisation, 179–656 W).
§3.4 measures the damage directly — one host synchronisation cost 589–2 459 µs
against ~10–20 µs on an idle device, and an iteration contains ~9–11 of them.
Every GPU timing here is therefore a lower bound; the iteration-count results
(§5.1, §5.2), which carry the argument, are unaffected.

Everything here is exact — no sampling, no approximation, no tolerance.  Every
count the GPU produces is checked against an integer CPU reference, and every
code file either engine writes is re-verified by `cov/verify_cov.py` with both
of its exhaustive methods (§6).

---

## 1. The identity, and why fp16 is exact integer arithmetic

Encode `w ∈ Z_q^n` as the concatenation of its `n` one-hot digit vectors,

```
phi(w) ∈ {0,1}^{n q},   phi(w)[i·q + d] = 1  iff  w_i = d.
```

For any two words,

```
<phi(a), phi(b)> = #{ i : a_i = b_i } = n − d_H(a,b),
```

so one dense product `Phi(A) @ Phi(W)^T` carries the exact Hamming distance
between every row of `A` and every row of `W`, and ball membership is the
threshold `>= n − R`.

**Exactness of the fp16 path.**  The inputs are 0/1, so every product is 0 or 1
and exact in any format with at least one mantissa bit.  Every partial sum is a
non-negative integer bounded by the full inner product, which is at most `n`.
IEEE binary16 represents every integer in `[0, 2048]` exactly (11-bit
significand), so for `n <= 2048` — every cell in this project has `n <= 16` — no
partial sum and no result is ever rounded, in any accumulation order, whether
the accumulator is fp16 or fp32.  `allow_fp16_reduced_precision_reduction` is
therefore irrelevant here.  `test_gemm.py` checks this against `numpy` int64 up
to `n = 1024` rather than trusting the argument (§6).

The *second* GEMM of the solver (§4) accumulates counts that can reach `|U|`,
which is not bounded by 2048, so that one is done in int8 → int32
(`torch._int_mm`), exact for any magnitude.

---

## 2. Choosing the formulation: the arithmetic-intensity argument

Four candidate formulations, costed on `K_6(8,4)` (`q^n = 1 679 616`,
`nq = 48`, `M = 169`, `|B_R| = 51 491`, mean `|U| ≈ 30–850`, `|U_1| ≈ 6 600`).
"Ops" counts multiply–adds ×2; a *move delta* is the exact new uncovered count
of one single-move neighbour of the current code.

| # | formulation | cost per iteration | exact move deltas produced | ops / move delta |
|---|---|---|---|---|
| A | recompute `cnt` over `Z_q^n` from the whole code | `M·q^n·nq` = 13.6 GMAC | 0 (bookkeeping only) | — |
| B | relocate one codeword anywhere: score all of `Z_q^n` against `U_{-c}` | `q^n·|U_{-c}|·nq` ≈ 5.6 GMAC | `q^n` = 1.68 M | ≈ 6 700 |
| C | **focused relocation** — score `B_R(u)` against `U ∪ S1(c)` for `k` codewords | `|B_R|·C·(nq + k)` ≈ 5.8 GMAC | `|B_R|·k` = 1.65 M | ≈ 7 000 |
| D | greedy insertion: score all of `Z_q^n` against `U` | `q^n·|U|·nq` | `q^n` insertions | ≈ 6 700 |
| — | *CPU reference*: `covsearch2 --preset p5b --wide` | `C(n,R)(q−1)^R` 2-bit reads per candidate codeword | `n(q−1)` per codeword | **≈ 1 094** |

Two things fall out immediately.

* **A is the wrong target.**  `cov/opt/METHODS.md` §A measures candidate
  evaluation at 93–99.6% of CPU run time and *rising* with instance size;
  bookkeeping is under 1%.  A GEMM that only maintains `cnt` optimises the part
  that is already free.  (It is still used, once, at start-up.)
* **The GEMM route is ~6× *less* efficient per move delta than the CPU
  evaluator**, in raw elementary operations.  That is not a bug: the CPU's
  shared-sphere identity (`METHODS.md` §B) amortises one sphere walk over all
  `n(q−1)` moves of a codeword and reads a 2-bit state array, so it spends
  ~1 100 operations per exact move delta.  The GEMM spends ~7 000, because every
  (position, witness) pair costs `2·nq = 96` operations regardless.  The GEMM
  bet is therefore entirely that those operations run >6× faster on tensor
  cores.

**Why C over B.**  B was implemented first (`gemmsolve.py`) and is a measured
negative result.  Relocating a *random* codeword to the globally best position
is a strictly wider neighbourhood than the CPU's, but at a local optimum the
best position for most codewords is the one they already occupy; forbidding the
no-op turns the search into a random walk, and allowing it wastes a full GEMM.
On `K_6(8,4)` from the fixed seed it descends from 854 uncovered to ~150 and
then oscillates between 160 and 220 indefinitely (best 150 after 400
iterations, against 20 for the CPU in 400 iterations).  Recorded, not fixed.

C fixes this the way the CPU solver does: pick a random *uncovered word* `u`
and restrict target positions to `B_R(u)`, the complete set of positions from
which a codeword covers `u`.  Every move in the neighbourhood then covers `u`,
so the search always has traction, and the neighbourhood is still
`|B_R| × k = 1.65 M` moves against the CPU's ~1 088.

**Arithmetic intensity of C.**  Writing `C` for the number of witness columns:

```
ops   = |B_R|·C·(2nq + 2k)          = |B_R|·C·160        (nq=48, k=32)
bytes = |B_R|·C (write Ind) + |B_R|·C (read Ind) + 4·|B_R|·k
      ≈ |B_R|·C·2.07
intensity ≈ 77 ops/byte
```

The int8 ridge point of a GH200 is `1979 TOPS / 4.0 TB/s ≈ 495 ops/byte`, so C
sits at 16% of the ridge — memory-bound, ceiling ≈ 308 TOPS.  This is *after*
the two design decisions that matter:

* **the threshold is fused into the GEMM epilogue** so the indicator is written
  once as **1 byte per pair**, never as a 4-byte int32 agreement.  Materialising
  the agreement matrix instead gives 96 ops per 8 bytes moved = 12 ops/byte, a
  41× worse intensity.  Measured cost of getting this wrong: 83 Gpair/s eager
  versus 337 Gpair/s with the fused `torch.compile` epilogue (§3).
* **the ownership reduction is itself a GEMM.**  Turning "sum the indicator
  columns belonging to each candidate codeword" into a second dense product
  `Ind @ O` keeps the whole iteration on tensor cores instead of on a segmented
  scatter-reduce.

---

## 3. Measured kernel throughput

GH200 480GB, torch 2.7 / CUDA 12.8.  **The GPU was shared with another
tenant's ML job throughout this session** (39–53 GB, 100% utilisation,
179–656 W); every GPU number below is therefore a lower bound, and the
variability is large.  Contention is quantified directly at the end of this
section.

### 3.1 Dense-GEMM calibration (what this GPU can actually do right now)

| shape | rate |
|---|---|
| fp16 `8192³` | **333.9 TFLOPS** |
| fp16 `4096³` | 310.5 TFLOPS |
| int8 `8192³` (`torch._int_mm`) | **55.6 TOPS** |
| int8 `4096³` | 53.2 TOPS |

Two findings worth recording. First, contended, we reach ~34% of the 989 TFLOPS
fp16 dense peak. Second, **`torch._int_mm` is 6× slower than fp16 on this
host** — 55.6 TOPS against a 1979 TOPS int8 peak, i.e. 2.8%.  The int8 path also
carries shape constraints (`M` a multiple of 32, `K`, `N` multiples of 8, probed
empirically in `covgemm.py:int_mm`).  So on torch 2.7/sm90 **fp16 is the fast
integer path**, not int8; int8 is used only where exactness demands more than
2048 (§1).

### 3.2 The thin-`K` penalty, which is the structural problem

`65536 × K × 8192` fp16, varying only the inner dimension:

| `K` | TFLOPS | note |
|---|---|---|
| 48 | **57.7** | `n·q` for `K_6(8,4)` |
| 72 | 83.4 | `n·q` for `K_8(9,4)` |
| 128 | 152.6 | |
| 512 | 267.1 | |
| 2048 | 344.5 | asymptote |

The one-hot encoding pins `K` at `n·q`, which for every cell in this project is
between 24 and about 100.  **That costs a factor of 6 against the same GPU's own
asymptotic GEMM rate, and a factor of ~30 against the nominal 2000 TOPS.**  No
amount of solver engineering changes it: it is a property of the encoding.
Packing several *independent* instances along `K` is not possible — the inner
product must be exactly the agreement count of one word pair.

### 3.3 The score kernel

`score[x] = #{w ∈ W : d(x,w) <= R}` for all `x ∈ Z_q^n`, `K_6(8,4)`,
`N = 1 679 616` rows.  "Gpair/s" counts (row, witness) pairs, each of which is
one exact Hamming distance:

| witnesses `B` | eager fp16 | eager int8 | **fused fp16 (`torch.compile`)** |
|---|---|---|---|
| 16 | 22.0 | 18.7 | — |
| 64 | 40.8 | 27.4 | **160.9** |
| 256 | 73.4 | 41.2 | 129.7 |
| 1024 | 83.1 | 42.1 | **336.7** |
| 4096 | 83.4 | 42.9 | — |

Eager int8 is exactly half of eager fp16 because both are bound by the width of
the materialised intermediate (4 bytes versus 2).  Fusing the threshold and the
reduction into the matmul epilogue removes the intermediate entirely and gives
4×; the compiled and eager results are checked bit-identical.  337 Gpair/s is
32 TOPS, i.e. 1.6% of the int8 peak — consistent with §3.2 plus contention.

### 3.4 Launch and synchronisation latency

The solver's iteration contains ~9 host synchronisations that cannot be removed,
because the shapes are data-dependent (`nonzero` of the counter array, the size
of the witness set, the arg-max of the objective).  Measured cost of one:

| operation | heavy contention | light contention | idle GPU (typical) |
|---|---|---|---|
| async kernel launch (queued, no sync) | 6.6 µs | — | ~5 µs |
| launch + `cuda.synchronize()` round trip | **2 459 µs** | **589 µs** | ~10–20 µs |
| `nonzero` + `.item()` on a 2 M array | 2 896 µs | 889 µs | ~30 µs |

Nine round trips at 2 459 µs is 22.1 ms, which is *exactly* the 22.4 ms median
iteration measured while the other tenant was drawing 656 W.  When it dropped to
179 W the median iteration fell to 1.9 ms.  **The GPU solver's iteration time in
this session is dominated by the other tenant, not by its own kernels**, and
the honest statement is a range, not a number.

---

## 4. The iteration

For codeword `c`, let `S1(c) = {w : cnt[w] = 1 and c covers w}` — the words `c`
alone covers.  Moving `c` to `x` leaves uncovered exactly

```
(|U| + |S1(c)|) − #{ w ∈ U ∪ S1(c) : d(w,x) <= R }.                      (*)
```

The `S1` sets are disjoint (a word with `cnt = 1` has a unique owner), so with
the column set `W = U ++ S1(c_1) ++ … ++ S1(c_k)` and the 0/1 ownership matrix
`O[j,i] = 1` iff column `j ∈ U` or `j ∈ S1(c_i)`:

```
Ind[x,j] = [ <phi(x), phi(w_j)> >= n−R ]      GEMM 1 (fp16, fused threshold → int8)
P[x,i]   = (Ind @ O)[x,i]                     GEMM 2 (int8 → int32)
objective to maximise:  P[x,i] − |S1(c_i)|,   resulting uncovered = |U| − max
```

One iteration is then:

1. `nonzero(cnt <= 1)` once, split into `U` and `U_1` (one host sync, not two);
2. a random `u ∈ U`; rows = `B_R(u)`, generated by pure arithmetic from a
   precomputed weight-`<= R` pattern table (no gather over `Z_q^n`);
3. candidate codewords = the `k` nearest to `u` (one tiny GEMV);
4. owners of `U_1` = one `|U_1| × nq @ nq × k` GEMM, thresholded;
5. GEMM 1 and GEMM 2, tiled over the rows of `B_R(u)`;
6. arg-max with a uniform random tie-break (`obj + U(0,1)`, no host round trip
   per tie);
7. commit: `cnt.index_add_` over `B_R(c_old)` and `B_R(x_new)`.

Step 7 is the only scatter left, and it is `2|B_R|` fully parallel increments
(103 k on `K_6(8,4)`), against the `2S` dependent counter walks per *candidate*
that made the first CUDA port latency-bound.

Engineering notes that mattered:

* **Shape bucketing.**  `torch.compile(dynamic=False)` specialises per shape, and
  the witness count changes every iteration.  Rounding the column count to a
  power of two and zero-padding (a zero column is the all-zero one-hot vector,
  whose agreement with everything is `0 < n−R`, so it can never be counted)
  cut the number of compilations from hundreds to about five.  Before this fix
  the measured time to target was 17.8 s; after, 7.5 s — the difference was
  entirely JIT.
* **A digit table for `Z_q^n`** (`N × n` uint8, 13 MB on `K_6(8,4)`) turns
  "digits of these word indices", needed three times per iteration, from `n`
  kernel launches into one gather.  In a latency-bound loop this is worth more
  than any flop saving.
* **`--warmup`** runs a few steps on a saved copy of the state and restores it
  (state *and* RNG), so reported times exclude JIT compilation.  Reported
  separately, never silently folded in.

---

## 5. Results

### 5.1 `K_6(8,4)`, `M = 169`, from the fixed seed `cov/opt/seeds_K6_8_4_M169.txt`

Both engines start from the identical code (854 uncovered).  CPU is
`covsearch2 --preset p5b --wide`, one thread, process CPU time (the host runs a
production sweep at load average > 100, so wall clock would measure the sweep —
this is the `METHODS.md` §F protocol).  GPU is wall clock on the shared GH200.
4 seeds each, medians.

**Iterations to reach a given uncovered count** — hardware-independent, and the
core result:

| target `|U|` | CPU iters (≈1 088 moves/iter) | GPU iters (1 647 712 moves/iter) | CPU/GPU |
|---|---|---|---|
| 100 | 75 | 43 | 1.74× |
| 50 | 128 | 84 | 1.52× |
| 20 | 400 | 338 | 1.19× |
| 10 | 1 497 | 697 | 2.15× |
| 5 | 4 711 | 1 666 | 2.83× |
| 3 | 9 256 | 2 760 | 3.35× |

**A 1 514× wider exactly-evaluated neighbourhood buys 1.2×–3.4× fewer
iterations.**  The trend is the same one `METHODS.md` §C2 already saw at small
scale — widening from `R+1` to `n(q−1)` moves per codeword (7.5×) bought 1.6×
fewer iterations — extrapolated three more orders of magnitude and still
holding.  Move *quality* saturates; the returns are roughly logarithmic in
neighbourhood width.

**Time and energy:**

| | CPU `p5b --wide`, 1 core | GPU focused GEMM, `--cand 32` |
|---|---|---|
| per iteration | **1.35 ms** | 1.9 / 4.1 / 4.4 / 20.7 ms (4 seeds; contention) |
| time to `|U| <= 20` | **0.541 CPU s** | 0.61 / 1.62 / 2.60 / 6.95 s wall |
| best `|U|` in 20 s | **2** (20 CPU s) | 9 (20 wall s, heavy contention) |
| best `|U|` in 45 s | — | 2, 2, 3, 3 (45 wall s) |
| power | ~5 W (one Grace core, module TDP split) | 179–656 W observed on the GPU |
| **energy to `|U| <= 20`** | **≈ 3 J** | **≈ 400–4 900 J** |

Even taking the GPU's single best observed run (0.61 s) and pretending it had
the GPU to itself, one Neoverse-V2 core matches it in wall clock and beats it by
**two orders of magnitude per joule**.  Against the whole 64-core Grace running
64 independent chains — which is what the project's campaign actually does, and
which needs no code changes — the GPU is behind by a further ~64×.

### 5.2 `K_8(9,4)`, `M = 2944` — the larger cell, where it turns round

`q^n = 134 217 728`, `nq = 72`, `|B_R| = 333 166`, seed
`cov/opt/seeds_K8_9_4_M2944.txt` (11 640 uncovered).  I expected this cell to be
worse for the GEMM route and **it is markedly better** — the prediction was
wrong and the measurement is what stands.

| target `|U|` | CPU iters | CPU s (1 core) | GPU iters | GPU s (wall) |
|---|---|---|---|---|
| 10 800 (the `METHODS.md` target) | 22.5 | 2.43 | **11** | **1.00** |
| 8 000 | 131 | 14.9 | 62 | ~2.5 |
| 5 000 | 334 | 35.2 | 174 | ~7.1 |
| 3 000 | 599 | 58.0 | 347 | ~14.1 |
| 2 000 | 862 | 80.0 | 578 | ~23.5 |
| 1 600 | 1 044 | 101.6 | 824 | ~33.5 |
| 1 200 | not reached in 110 CPU s | — | 1 788 | ~72.8 |

CPU: `p5b --wide`, 2 seeds per level, medians, process CPU time.  GPU: median
iteration 40.7 ms, `--cand 32`; the "GPU s" column is iterations × 40.7 ms
except the first row, which is the measured `ttt_wall`.

**Quality at a matched 100 s budget** — CPU 100 CPU s on one core against GPU
100 wall s on the shared GPU, 4 and 2 seeds:

| | best `|U|` reached | iterations |
|---|---|---|
| CPU `p5b --wide` | 1 398, 1 429, 1 466, 1 543 (median **1 448**) | ~1 150 |
| GPU focused GEMM | 1 137, 1 137 (median **1 137**) | ~2 100 |

Two things flipped relative to `K_6(8,4)`:

* **The CPU's iteration got 72× more expensive** (1.35 ms → ~97 ms) while the
  GPU's got only 21× more expensive (1.9 → 40.7 ms).  The reason is in
  `METHODS.md` §A: the CPU walk is a strided gather whose cost per pattern rises
  from 1.9 ns (L2-resident counters) to 7.4 ns once the counter array is 268 MB
  and lives in DRAM, and the sphere itself grows from 1 250 to 168 070 patterns.
  The GPU's per-iteration cost grew only with `|B_R|` because the witness count
  `C` stayed around 2 300 — `|U|` collapses in the first few dozen iterations
  and never returns.
* **The GEMMs finally became big enough to amortise the launch latency.**  At
  `K_6(8,4)` an iteration is ~9 host round trips around ~0.4 ms of kernel; at
  `K_8(9,4)` it is ~11 round trips around ~3 ms of kernel over six row tiles.

Net: on this cell the GEMM engine is ~**2.4× faster per iteration and needs
~1.3–2.0× fewer iterations** than one Neoverse-V2 core, i.e. ~**3× faster in
wall clock than one core** — and it reaches a strictly better code at a matched
100 s budget.  It is still ~21× behind the full 64-core CPU running 64
independent chains, and ~50× behind it per joule.

One exact optimisation not implemented, which would help most on cells with
small `R/n`: only witnesses within distance `2R` of `u` can be covered by any
`x ∈ B_R(u)`, so the columns may be filtered to `B_{2R}(u)` provided `|U|` and
`|S1(c)|` stay unfiltered in the constant term of (*).  On `K_8(9,4)`
(`2R = 8 < n = 9`) that removes ~30% of columns; on `K_6(8,4)` (`2R = 8 = n`)
nothing.

### 5.3 What the result is *not*

* Not a correctness problem.  Every code either engine emitted was re-verified
  (§6); the GPU's exact move deltas agree with an integer CPU reference on every
  randomised test.
* Not explained away by contention.  Contention explains why the GPU iteration
  is 1.9–20 ms rather than perhaps 0.4 ms on `K_6(8,4)`, and §3.4 pins that down
  directly — so **all GPU timings here are lower bounds and the size-dependent
  verdict of §7 can only move in the GPU's favour**.  It does not explain the
  iteration-count tables (5.1, 5.2), which are hardware-free, nor the thin-`K`
  ceiling (3.2), nor the 6× ops-per-move deficit (§2).
* Not the whole GPU case.  `cov/NOTES.md` §7 makes an orthogonal argument:
  cells with `q^n ≈ 10^10` do not fit in host memory at all.  That is a capacity
  argument, not a throughput one, and it stands independently.

---

## 6. Correctness

`./selftest.sh` — all pass.

1. **`test_gemm.py`** — every count the GEMM formulation produces against an
   independent integer computation in numpy int64: the digit/index bijection
   against the verifier's own convention; both one-hot builders; the agreement
   identity `<phi(a),phi(b)> = n − d_H(a,b)` in fp16 *and* int8 over
   `q ∈ {2,3,6,7,8,11,12}`, `n ∈ {2,…,16}`; chunked ball counts at three
   different chunk sizes; full-space `cnt` from the GEMM against exhaustive ball
   marking; the relocation identity (*) against a from-scratch recount of the
   moved code; and fp16 exactness at `n = 1024` (`K = 2048`, values to 535).
2. **`--selftest`** — after *every* move, the incrementally maintained `cnt`
   array is compared element-wise against a from-scratch recount, and the cached
   one-hot code matrix against a rebuild.  Any drift aborts.
3. **`verify_cov.py --method both`** — every emitted code file is re-read from
   disk and its uncovered count recomputed by two independent exhaustive
   methods, which must agree with each other *and* with the number the solver
   printed, including when that number is not zero.  Checked over 10
   configurations spanning `R = 0`, `R = n−1`, `q` from 2 to 7, and sizes above
   and below the optimum, ending at 0, 16, 33, 226, 442, 727 and 2 908
   uncovered; and on all 16 benchmark runs of §5.1.  The `K_8(9,4)` code the
   GPU wrote after 100 s was verified separately by the meet-in-the-middle
   method over all 134 217 728 words: **`uncovered=1137`, exactly the number the
   solver reported** (45 s of verifier time; the `pure` method is not usable at
   this size).

`run_bench.py` runs the verifier on every benchmark output and exits non-zero on
any disagreement; the §5.1 numbers were produced with it.

---

## 7. Verdict

**The answer is size-dependent, and it is negative where it matters for a
production campaign.**

| | `K_6(8,4)` (`q^n` = 1.7 M) | `K_8(9,4)` (`q^n` = 134 M) |
|---|---|---|
| vs **one** CPU core, wall clock | ~1× (level, best case) | **~3× faster** |
| vs the **64-core Grace**, throughput | ~60× slower | ~21× slower |
| per joule | ~100–1000× worse | ~50× worse |
| quality at a matched budget | CPU better (2 vs 9 in 20 s) | **GPU better (1 137 vs 1 448 in 100 s)** |

* **The trend is in the GPU's favour with instance size**, for a reason that
  will keep holding: the CPU evaluator's cost per pattern triples once the
  counter array leaves cache (`METHODS.md` §A: 1.9 → 7.4 ns), and its sphere
  grows as `C(n,R)(q−1)^R`, while the GEMM's witness count `C` is set by `|U|`,
  which the search itself drives down.  On `K_8(9,4)` the CPU iteration got 72×
  more expensive than on `K_6(8,4)` and the GPU's only 21×.
* **But the GPU never catches the CPU *chip*.**  The campaign runs independent
  workers on 64 cores at nearly perfect scaling; the GEMM engine is one chain.
  A 3× advantage over one core is a 21× deficit against the socket, and the
  energy gap is worse still.
* **Two ceilings are structural, not implementation defects.**  `K = n·q ≈ 50`
  costs 6× against this GPU's own GEMM asymptote and ~30× against the nominal
  2000 TOPS (§3.2); and neighbourhood breadth pays only logarithmically — 1 514×
  wider buys 1.2–3.4× fewer iterations (§5.1).  Together they cap what any
  amount of further kernel work can recover.
* **Every GPU number here is a lower bound.**  The GPU was shared throughout;
  §3.4 shows one host synchronisation cost 589–2 459 µs against ~10–20 µs idle,
  and the iteration contains ~9–11 of them.  An idle-machine re-run should be
  expected to improve the GPU rows by something between 2× and 8×, which would
  make `K_8(9,4)` a clear win against one core and a near-tie against ~10 cores
   — still not against 64.

The formulation itself is a success even though the bet is not: it is exact,
it is entirely free of the scattered read-modify-write traffic that sank the
first CUDA port (`cov/NOTES.md` §7), and the only scatter left is `2|B_R|`
independent increments at commit time.  If this family of problems ever presents
a large inner dimension (`n·q` in the thousands), or a cell whose counter array
does not fit in host memory at all, the arithmetic changes and this code is the
right starting point.

### If someone wants to push it further

In rough order of expected value:

1. **Batch-parallel chains.**  The one change with a plausible path to beating
   64 cores.  The bottleneck is latency, not flops (§3.4): the iteration is
   ~9–11 host round trips wrapped around ~0.4–3 ms of kernel.  Running `T`
   independent searches and stacking their witness sets along the column
   dimension turns `T` latency-bound small GEMMs into one large one and
   amortises the round trips over `T` chains — and independent chains are what
   the campaign already wants.  It also raises `K`-utilisation indirectly by
   making every GEMM bigger.  It does not fix §3.2.
2. **Get an uncontended GPU and re-run §5.1 and §5.2.**  It will not change the
   shape of the verdict but it replaces a 10× range with a number, and §5.2 is
   close enough to interesting that the number matters.
3. **Push on the larger cells, since that is where the trend points.**
   `K_8(10,5)`, `K_8(10,6)` and the `q^n ≈ 10^9–10^10` corner are where the CPU
   evaluator is furthest out of cache; the GEMM engine's `dig_all` table is the
   only `O(q^n · n)` structure in it and can be dropped for arithmetic digit
   generation if memory gets tight.
4. **A CUTLASS kernel** fusing GEMM 1, the threshold, and the `Ind @ O`
   reduction into one pass, so the indicator never reaches HBM.  Worth ~2× on
   intensity (77 → ~160 ops/byte); does not fix the `K = 48` ceiling.
5. **1-bit tensor cores.**  The natural fit for one-hot data (XOR-popcount) and
   4× the int8 rate on Turing/Ampere, but the `b1` MMA is deprecated on Hopper
   and torch exposes no path to it; and it would not fix `K = 48` either.

---

## 8. Files

| file | what |
|---|---|
| `covgemm.py` | the identity, encodings, chunked space, `int_mm` shape wrapper, CPU references |
| `test_gemm.py` | randomised exactness tests against integer numpy |
| `gemmsolve.py` | formulation **B** (whole-space relocation) — the recorded negative result |
| `focused.py` | formulation **C** (focused relocation), the working solver |
| `bench_kernel.py` | score-kernel throughput and roofline numbers (§3.3) |
| `run_bench.py` | matched-budget CPU-vs-GPU harness, verifies every output |
| `selftest.sh` | the correctness gate (§6) |
| `results_raw.md` | every number quoted above, as measured |
| `results_K6_8_4.csv` | raw rows from the matched-budget harness |

Reproduce the headline:

```
./selftest.sh                                       # correctness
python3 bench_kernel.py --cells "K6(8,4)"           # kernel roofline
python3 run_bench.py --cell K6_8_4 --seeds 4 --budget 20 --cand 32

# the larger cell, where the GEMM engine wins against one core (sec. 5.2)
python3 focused.py -q 8 -n 9 -R 4 -M 2944 \
        --in ../opt/seeds_K8_9_4_M2944.txt \
        --time 100 --iters 100000000 --target 10800 -s 1 --cand 32 --warmup 3 \
        --out /tmp/g894.txt
python3 ../verify_cov.py /tmp/g894.txt -q 8 -n 9 -R 4 --method numpy
```

The power figures are nameplate estimates, not measurements: the GPU draw is
from `nvidia-smi` (whole device, shared), and the per-core CPU figure is the
Grace module TDP divided by its core count.  They are good to a factor of two,
which is enough for a conclusion whose smallest margin is 50x.
