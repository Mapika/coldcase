# Arena entry `lowlevel` — making the baseline inner loop fast

Angle: **low-level performance only**.  The search is deliberately the *same*
algorithm as `cov/search/covsearch.c` — focused local search: pick a uniformly
random uncovered word `u`, enumerate every codeword-move that could cover it,
evaluate them **all** exactly, commit the best with random tie-breaking, kick a
couple of codewords on stagnation.  No new heuristic, no new construction, no
portfolio, no parameter tuning.  Everything below is about how many of those
moves fit into a second.

| file | what |
|---|---|
| `covfast.c` | the solver |
| `run_entry.sh` | arena contract wrapper (6 threads, `nice -n 15`) |
| `covfast` | committed binary (a rebuild takes minutes on this box; `run_entry.sh` only builds if it is missing) |
| `baseline.c` | verbatim copy of `cov/search/covsearch.c` plus two instrumentation lines (`cands=`, a `CPU` line) — the A/B reference, never the entry |
| `bench.sh` | ablation harness |
| `quality.sh` | head-to-head on best-uncovered-in-T-seconds |
| `selftest.sh` | correctness gate |

## 1. What changed

**(1) Compile-time specialisation of the sphere walk.**
A move enumerates `C(n-1,R)` position subsets and, for each, `(q-1)^R` value
assignments.  The baseline runs a runtime odometer with a carry chain and
re-reads `q`, `n`, `R` from globals inside it; `sphere_walk` does not get
inlined and pays a stack-protector prologue, a `memset` of the odometer state
and an `R`-way switch ladder per subset.

Here the value enumeration is a **static loop nest of depth R**, macro-generated
for `R = 1..6`, with the offset row pointers hoisted out of the nest
(`NEST_1`…`NEST_6`).  `q-1` is additionally specialised to the literals 2, 5
and 7 — the values `q ∈ {3,6,8}` produce — so for those the innermost loop is
fully unrolled and the carry logic disappears entirely.  48 evaluation kernels
are generated and one is chosen by function pointer at start-up; `R > 6` or an
unusual `q` falls back to a generic odometer, so no shape is unsupported.

**(2) Branchless evaluation.**  Each element contributes
`(cnt[a]==1) - (cnt[b]==0)` with no control flow, so the loads stay
independent and the core can keep many misses in flight.

**(3) NEON.**  `combs` rows are sorted ascending, so the innermost coordinate
of a subset is the *largest* position, and it is `n-1` — stride 1 — for
`R/(n-1)` of all subsets.  Its `q-1` words then live in one `q`-aligned
contiguous block, so a single 128-bit load + `vceq` + lane mask + `vaddlv`
replaces `q-1` scalar load/compare/accumulate triples, for the leaving and the
entering word alike.  The lane mask (`v < q && v != c_{n-1}`) is precomputed
per digit value.  Enabled for `5 ≤ q ≤ 16` (uint8 counters) or `5 ≤ q ≤ 8`
(uint16).  This is the single biggest win on `K_6(8,4)`, where 57 % of the
subsets qualify.

**(4) uint8 counters, exactly, for any `M`.**
`cnt[w] ≤ M`, so for `M ≤ 255` an 8-bit counter is exact by construction.  For
larger `M` it is made exact instead of merely likely: a word's counter rises by
at most one per committed move (the entering set of a move contains each word
once), so latching a flag when any counter reaches 250 and widening the whole
array to uint16 *between* moves cannot lose a count.  In practice the flag
never fires — mean multiplicity is `M·|B_R|/q^n ≈ 2.3` on `K_8(9,4)` — but the
fallback is what makes it safe to use everywhere.  It halves the working set:
`K_6(6,3)` 93 KB → 47 KB (L1-resident), `K_6(8,4)` 3.4 MB → 1.7 MB,
`K_8(9,4)` 268 MB → 134 MB.  `--satat N` lowers the promotion threshold so the
widening path can be tested (see §3); `--force16` disables it.

**(5) Early exit that provably does not change the answer.**
A move's delta is `(#words that become uncovered) − (#uncovered words that get
covered)`.  The second term is at most `U`, the current number of uncovered
words, because every word it counts is uncovered right now and the sphere
enumeration visits each word at most once.  So a candidate whose running
partial delta exceeds `bd + U` — `bd` being the best *completed* delta any
thread has produced so far, kept as a relaxed atomic min — has final delta
`> bd` and cannot win.  Checked once per position subset.

This is exact pruning, not a heuristic cut: the move that gets committed is the
move an unpruned search would commit.  (The only observable difference is that
a pruned candidate no longer takes part in the *running* minimum during the
tie-break scan, so the number of random draws the tie-break consumes can shift
and trajectories can diverge later on.  Every individual decision is the same.)
It costs nothing when `U` is large and buys a lot in the endgame, which is
exactly when it is needed.

**(6) O(1) sampling of uncovered words, when that is the cheaper option.**
The baseline probes 64 random words and then falls back to a **full `q^n`
linear scan** — once per iteration, precisely in the endgame.  Here an exact
uncovered-word list plus an index array (`ulist`/`upos`, 8 bytes per word) is
maintained by the commit path, so a uniformly random uncovered word costs one
RNG draw and one load.  Maintaining it is *not* free when a large fraction of
the space is uncovered (on `K_8(9,4)` a single commit covers thousands of new
words, i.e. thousands of index updates into a 0.5 GB array, and that measured
64 % slower), so the list is switched on below `uncovered < q^n/32` and off
again above `q^n/16`, and its memory is allocated lazily on first use.

**(7) A measured, not assumed, parallel decision.**  See §2 — one OpenMP
fork/join per iteration is a 34× *loss* on the small cells when the box is
contended, and a 6× win on the big ones, so the solver benchmarks both against
each other while it runs.

**(8) Small stuff.**  Lemire's nearly-divisionless bounded RNG instead of `%`
(`--modrng` restores `%` so the baseline's RNG stream can be reproduced
exactly for A/B); per-codeword offset tables kept incrementally rather than
rebuilt inside every candidate evaluation; `-fno-stack-protector`; the wall
clock is checked every iteration rather than every 256, because one sphere walk
on `K_8(9,4)` takes most of a second and the coarse check overshoots `-t`
badly; the output file is de-duplicated and padded to exactly `M` **distinct**
codewords, since the judge counts distinct words and a full cover containing a
repeat would score 0 instead of 1000.

## 2. Measured speedup

Method.  Fixed iteration budget, single thread, best of 3 interleaved runs,
**CPU time** (`CLOCK_PROCESS_CPUTIME_ID`) — this box is shared with a `nice -10`
production campaign that keeps it 2–8× oversubscribed and wall clock on it is
not reproducible.  Rows 2–5 run with `--modrng`, which makes `covfast` consume
the RNG exactly like the baseline; the printed fingerprint
`uncovered/kicks/candidates-evaluated` is identical down the column, so those
rows do *identical work* and the ratio is a pure low-level speedup.  The last
row is the shipped configuration; the uncovered list changes which uncovered
word gets picked, so its trajectory differs and it is compared per candidate
evaluated rather than per iteration.

```
K6(6,3)@41   iters=40000                    K3(11,4)@81  iters=8000
  baseline               5.887s  x1.00        baseline               4.116s  x1.00
  specialised kernels    2.488s  x2.37        specialised kernels    1.540s  x2.67
    + uint8 counters     2.211s  x2.66          + uint8 counters     1.450s  x2.84
    + NEON               1.499s  x3.93          + NEON               1.463s  x2.81  (q=3: off)
    + early exit         1.180s  x4.99          + early exit         1.481s  x2.78  (U large: no-op)
    + uncovered list     1.366s                 + uncovered list     1.481s

K6(8,4)@169  iters=400                      K8(9,4)@940  iters=200 (idle box, init subtracted)
  baseline               4.041s  x1.00        baseline              48.93s  x1.00
  specialised kernels    3.298s  x1.23        specialised kernels   30.38s  x1.61
    + uint8 counters     1.572s  x2.57          + uint8 counters    26.58s  x1.84
    + NEON               1.171s  x3.45          + NEON              24.61s  x1.99
    + early exit         1.421s  x2.84          + early exit        24.54s  x1.99
    + uncovered list     1.274s                 + uncovered list  (auto-off on this cell)
```

Normalised per candidate evaluation (shipped build vs baseline), which is the
fair single number because the shipped build's trajectory differs slightly:

| cell | baseline | `covfast` | speedup |
|---|---|---|---|
| `K_6(6,3) @ 41`  | 2.53 µs/cand | 0.60 µs/cand | **4.3×** |
| `K_6(8,4) @ 169` | 75.9 µs/cand | 24.3 µs/cand | **3.1×** |
| `K_3(11,4) @ 81` | 8.96 µs/cand | 3.30 µs/cand | **2.7×** |
| `K_8(9,4) @ 940` | 2.96 ms/cand | 1.48 ms/cand | **2.0×** |

The shape of the table is the memory hierarchy.  `K_6(6,3)` fits in L1 once the
counters are bytes, and there the win is almost all instruction-level
(specialisation + NEON + pruning).  `K_8(9,4)` is 134 MB of counters touched in
a scattered pattern; it is DRAM-latency bound and no amount of instruction
selection fixes that, so halving the counter width is most of its 2×.
`K_3(11,4)` gets nothing from NEON (`q = 3`: two words per contiguous run) and
nothing from early exit (hundreds of words stay uncovered, so the pruning bound
`bd + U` is never tight) — its 2.7× is purely the static loop nest.

### Threads: the fork/join trap, and the fix

Parallelism is over candidates, one OpenMP task each, so it scales with how big
a single candidate is.  On a **quiet** box (load average ~40), best of 2:

| cell | 1 thr | 2 thr | 4 thr | 6 thr |
|---|---|---|---|---|
| `K_6(6,3)`, 20 000 it  | 0.67 s | 0.51 s | 0.38 s | 0.33 s (2.0×) |
| `K_3(11,4)`, 8 000 it  | 1.39 s | 0.83 s | 0.52 s | 0.41 s (3.4×) |
| `K_6(8,4)`, 400 it     | 1.13 s | 0.61 s | 0.31 s | 0.22 s (5.1×) |
| `K_8(9,4)`, 60 it      | 10.6 s | 5.87 s | 2.96 s | 1.63 s (6.5×) |

On a **loaded** box this reverses violently on the small cells.  A `K_6(6,3)`
candidate is ~0.6 µs, so an iteration is ~30 µs of work with one OpenMP
fork/join wrapped around it; when the box is 3× oversubscribed by other tenants
and this process is `nice -15`, the six workers are not all scheduled when the
barrier needs them.  Measured, 40 s wall on `K_6(6,3)` at load average ~190:

| | iterations | CPU used | best uncovered |
|---|---|---|---|
| fixed 6 threads (first version) | 23 628 | 23.4 s | 29 |
| self-calibrating | **748 055** | 22.7 s | **9** |

That is 34× more iterations for the *same* CPU seconds — all of the difference
was barrier wait.  The baseline has the same exposure (it hard-codes
`par_cand = NCOMB < 256`, which is true on all four benchmark cells).

So the parallel decision is not a constant and not a threshold: the solver
**measures it**.  It runs a block of parallel iterations, then a block of serial
iterations, keeps whichever was faster per iteration, and re-measures
periodically.  Parallel goes first because that lets the serial probe be
skipped entirely when an iteration turns out to cost more than 5 ms — an
OpenMP fork/join is at worst ~1 ms even on a contended box, so above that the
answer is already known, and the serial probe is exactly the expensive one on
those cells (three serial iterations of `K_8(9,4)` are ~2 s of a 60 s budget).
A probe block ends after 0.05 s or 128 iterations, whichever comes first, and
the next re-measure is scheduled at 20× what the last one cost (at least 10 s
later), which bounds calibration at ~5 % of the run.  Typical verdicts under
load:

```
K6(6,3)@41    parallel 1227 us/it   serial   37 us/it            -> serial
K6(8,4)@169   parallel  494 us/it   serial 2902 us/it            -> parallel
K3(11,4)@81   parallel   50 us/it   serial  177 us/it            -> parallel
K8(9,4)@940   parallel 107970 us/it serial probe skipped (>=5ms) -> parallel
```

It tracks the machine: on `K_6(6,3)` the same binary picks parallel on an idle
box and serial on a busy one.  Skipping the serial probe on `K_8(9,4)` alone
took that cell from 306 to 552 iterations in a 30 s budget.

## 3. Correctness

`selftest.sh` — **ALL PASS**.  Three independent checks:

* **Move-by-move.**  `--selftest` recomputes the uncovered count from scratch
  over the whole `q^n` array after *every committed move*, and also checks the
  uncovered-list length against it, aborting on the first mismatch.  Run on 20
  configurations, including all four benchmark cells and the corners `R = 0`,
  `R = n−1`, `q = 2`, `q = 10` (NEON must switch itself off), and with each
  optimisation individually disabled.
* **Independent verification.**  Every emitted code is handed to
  `cov/verify_cov.py`, which must report *exactly* the uncovered count the
  solver claimed — not only when that count is zero.  This is the check that
  catches incremental-bookkeeping drift, which is the failure mode that would
  otherwise produce a confident false claim.
* **Trajectory identity.**  `--modrng --nolist --noearly` must reproduce the
  baseline's `uncovered/iters/kicks/cands` fingerprint bit for bit, and the
  NEON, uint8 and generic kernels must all agree with each other.  Checked on
  5 cells.  Separately, `--satat {2,3,7,250}` — which forces the uint8→uint16
  widening to fire early and repeatedly — must reproduce the `--force16`
  trajectory exactly; checked on 3 cells, so the promotion path is genuinely
  exercised rather than merely present.

`run_entry.sh` additionally writes a trivially valid `M`-word file *before*
starting the solver and only overwrites it if the solver produced a non-empty
output, so a crash or an OOM cannot produce the unparseable output that scores
−10⁶.

## 4. Does the speed convert into results?

This is the question the ablation does not answer, so it was measured
separately: all configurations run **round-robin** on the same cell so they see
the same machine load, each with a wall-clock budget, and the number reported is
the best uncovered count reached (0 = full cover).  The baseline is shown at
both 6 threads (its shipped setting — `par_cand` is true on all four benchmark
cells) and 1 thread, so it is compared at its best.

```
K6(6,3)@41   t=30s, 1 thread, 10 seeds
  baseline                 solved 1/10  sum 129   0 8 14 8 9 19 16 16 25 14
  covfast, kernels only    solved 1/10  sum 102   14 8 0 14 9 9 14 9 16 9
  covfast, no uncov. list  solved 1/10  sum 116   14 8 0 9 21 9 14 9 18 14
  covfast, everything on   solved 0/10  sum 109   9 8 9 16 18 8 9 14 9 9

K6(8,4)@169  t=45s, 4 seeds
  baseline, 6 threads      solved 0/4   sum 9     4 2 2 1
  baseline, 1 thread       solved 0/4   sum 15    5 5 2 3
  covfast                  solved 3/4   sum 1     0 1 0 0

K3(11,4)@81  t=45s, 4 seeds          K8(9,4)@940  t=60s, 3 seeds
  baseline, 6 threads  sum 1647        baseline, 6 threads  sum 24 301 558
  baseline, 1 thread   sum 1753        baseline, 1 thread   sum 26 660 741
  covfast              sum 1667        covfast              sum 24 799 304
```

The honest reading: **the speedup buys a real result on one of the four public
cells and nothing on the other three.**  (Under the judge's amended scoring —
partials normalised to parts-per-million of `q^n` — those four sums translate
to roughly −193, −2, −2377 and −60 000 respectively, so `K_8(9,4)` dominates
the total and `K_6(8,4)`, the cell this entry actually wins, is worth about
1000 points when it lands.)

* `K_6(8,4) @ 169` is the win, and a large one — 3/4 full covers against 0/4.
  This is the cell where the search is genuinely converging when the clock runs
  out, so 3× the moves finishes the job.
* `K_6(6,3) @ 41` plateaus around 8–16 uncovered and 5× the moves does not break
  through; the four configurations are indistinguishable at 10 seeds (and the
  earlier 4-seed sample that looked like a regression was noise).
* `K_3(11,4) @ 81` is the known-hard control and behaves like one: everything
  lands at 400–460.
* `K_8(9,4) @ 940` is a draw under contention, which flatters the baseline:
  its `-t` clock starts *after* initialisation, so at `-t 60` it actually runs
  for ~70 s wall, whereas `covfast` counts initialisation against the budget
  and stops at 60 s.  Both leave ~8 × 10⁶ words uncovered out of 1.34 × 10⁸.

Which is exactly what §9 of `cov/NOTES.md` predicts: candidate *breadth* is what
this search is made of, and once it has saturated, more of the same samples buy
very little.  A faster inner loop is worth having — it is free once written, it
converts directly on the cells that are still descending, and it is what makes
the difference on `K_8(9,4)` initialisation — but it is not a substitute for a
better move.

## 5. Judge self-check

`scripts/arena_judge.py lowlevel --seeds 5 --time 60`.  Note the scoring
amendment made today: a partial now scores `-floor(uncovered·10⁶/q^n) - 1`,
i.e. **parts per million of the space left uncovered**, not the raw count.  So
`-193` on `K_6(6,3)` means 9 uncovered words out of 46 656, and `-2377` on
`K_3(11,4)` means 421 out of 177 147.  (That normalisation is worth reading
before interpreting any of these numbers — it cost an hour here chasing a
"regression" that was the judge's arithmetic changing under the entry.)

| run | load avg | `K_6(6,3)@41` | `K_6(8,4)@169` | `K_8(9,4)@940` | `K_3(11,4)@81` | solved | `sum_median` |
|---|---|---|---|---|---|---|---|
| 1, before the `K_8(9,4)` init fix | ~100 | 0/5, −193 (9 unc.) | **4/5, +1000** | 0/5, −66 924 (9.0 M) | 0/5, −2 377 (421) | 4/20 | −68 494 |
| 2, same build, another competitor's campaign running | ~180 | 0/5, −301 (14) | 0/5, −2 (3) | 0/5, −48 042 (6.4 M) | 0/5, −2 377 (421) | 0/20 | −50 722 |
| 3, shipped build | ~70 | 0/5, −193 (9) | 1/5, −2 (3) | 0/5, −50 110 (6.7 M) | 0/5, −2 377 (421) | 1/20 | −52 682 |

Median wall was 56.5–57.0 s against a 60 s budget on every cell, so the entry
honours the contract with a little headroom rather than running long.

Two things are visible.  `K_6(8,4)` is bimodal — the search either finishes its
descent within the budget or stops one or two words short, so its median is
either +1000 or about −2, and which one it is depends on how many cycles the
box hands out (the round-robin comparison in §4, where the baseline got 0/4 and
this entry 3/4, is the load-controlled version of the same measurement).  And
`K_8(9,4)` contributes ~95 % of `sum_median` even after the normalisation, so
that cell decides the ranking — and it is the cell where a faster inner loop
helps least in relative terms.

Judge run 1 exposed two real bugs, both fixed, both worth recording:

* On `K_8(9,4)` the *initialisation* — marking 940 balls of 333 166 words each
  — could outrun the whole 60 s budget on a loaded box, after which the process
  was killed with nothing written and the run scored the fallback file's
  1.07 × 10⁸ uncovered.  Marking one ball is now OpenMP-parallel over the first
  differing coordinate (safe without atomics: that loop partitions the position
  subsets, so threads touch disjoint words), initialisation gives up gracefully
  if the budget is gone anyway, and the solver **publishes the code to the
  output path as soon as it has one** and re-publishes it atomically (write to
  `.part`, `rename`) on every improvement, so a kill at any moment leaves the
  best code so far on disk.  Seeds 1000/1001 went from 9.47 × 10⁶ / 1.07 × 10⁸
  uncovered to 6.52 × 10⁶ / 7.58 × 10⁶.
* A fixed 6 OpenMP threads was a 34× *loss* on the small cells under load; see
  §2.

`run_entry.sh` also targets an *absolute* niceness of 15 rather than adding a
relative `nice -n 15`, which would compound to 19 if the judge itself is niced.

## 6. What is not claimed

* **No search-strategy improvement**, by design.  Where the baseline plateaus,
  this entry plateaus in the same place.
* **`K_8(9,4) @ 940` stays far from solved.**  Greedy initialisation already
  leaves ~10⁷ words uncovered and one move fixes a few thousand; 2× the moves
  moves that by a few per cent, not to zero.  Closing that cell needs a
  construction, not a faster inner loop — and under the normalised scoring it
  is still ~95 % of the total, so that is where the arena will actually be won.
* The obvious next low-level step, deliberately not taken because it changes
  the strategy: with the parallel decision so often coming out "serial", five
  of the six allowed threads sit idle on `K_6(6,3)` and `K_3(11,4)`.  Six
  independent chains with no synchronisation would use them, and on a plateaued
  cell a 6-way portfolio is worth more than 6× the moves in one chain.  That is
  a portfolio, not a faster inner loop, so it belongs to a different entry.
* All measurements were taken on a box shared with a `nice -10` production
  campaign (and, for part of the time, another competitor's benchmark run)
  whose load average ranged from 24 to 580.  CPU-time ratios (§2) were stable
  across that range to within ~15 %; wall-clock results (§4, §5) were not, and
  are only meaningful as round-robin comparisons taken at the same moment.
