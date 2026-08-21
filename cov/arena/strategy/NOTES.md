# arena entry `strategy` — search strategy over `covsearch`

Mandate: beat `cov/search/covsearch.c` on time-to-solution by changing *what the
search does*, not by making its inner loop faster.  The sphere-difference move,
the `cnt[]` bookkeeping and the candidate enumeration are taken from
`covsearch.c` unchanged; everything measured below is a change to which move is
made, which word is worked on, how the code is built, and how the time budget is
spent.

Files: `strategy.c` (solver), `run_entry.sh` (portfolio driver, the arena entry),
`baseline.c` (a copy of `cov/search/covsearch.c`, used only as the A/B control —
the production binary was never touched).

## How these numbers were measured

Everything was run on the shared 64-core box between 11:00 and 13:00 on
2026-08-20, while other arena entries and a production `covsearch` campaign were
running: the load average moved between 40 and 620 over the session.  Absolute
numbers therefore move by 2-3x between one hour and the next and **no result
below should be read as an absolute**.  Every comparison is a *paired* one: the
variants being compared were launched at the same moment, on the same seeds, and
ran side by side, so they saw the same machine.  Where a table mixes runs from
different times it says so.

`uncovered` is the count reported by the solver; it was cross-checked against
`cov/verify_cov.py` on every configuration used here and always agreed exactly
(the incremental counters do not drift).

---

## 1. The baseline's wall clock is a function of its iteration rate

`covsearch` checks the wall clock every 256 iterations and only writes its output
when it exits, so its overrun is `256 / (iterations per second)` in the worst
case.  That is a few percent when iterations are fast and unbounded when they are
not.  The judge kills an entry at `TIME_S + 60` and scores an empty or
unparseable file at **-10^6**.

**How bad it gets depends on the machine, and I initially overstated this.**
Measured both ways:

| condition | `covsearch` outcome |
|---|---|
| load ~430-630, `K_8(9,4)@940`, `TIME_S=60`, judge kill at 120 s | **empty file, both seeds → -10^6** |
| load ~430, `K_6(8,4)@169`, hard kill at 45 s, 4 seeds | **empty file on all 4 seeds** |
| load ~40-90, all four public cells, `TIME_S=60` (§10 run 1) | exits at 60.0-65.7 s, no lost files |
| load ~130, `K_8(9,4)@940`, `TIME_S=60`, 5 seeds (§10 run 2) | **-10^6 on 2 of 5 seeds**, median wall 118.7 s |

So it is a **tail risk whose probability rises with contention**, not a certainty
and not an artifact of one pathological hour.  I first measured it only at load
430-630 and wrote it up as unconditional; then run 1 landed on a quiet box and it
did not fire at all, and I corrected the other way; then run 2 at a very ordinary
load of ~130 lost two seeds out of five.  The correct statement is the mechanism
one: the overrun is `256 / (iterations per second)`, `K_8(9,4)@940` runs at a few
iterations per second, so whenever the box is busy enough the first clock check
lands past `TIME_S + 60` and the whole file is lost.  Two seeds in five is -2·10^6
on one cell.

The overrun itself is unconditional.  Across both runs `covsearch` finished at
60.0-65.7 s (quiet) and 118.7 s (busy) against `-t 60`, i.e. it exceeded the "at
most TIME_S seconds" contract on every cell of every run, while this entry
finished at 57.8-58.1 s throughout.  The judge's 60 s grace usually makes that
free, and some of `covsearch`'s quality is bought with the extra time.  This
entry does not take it.

Fixes in `strategy.c`: the clock is read every iteration; the best code is
written to `--out` whenever it improves (rate-limited to 1 Hz) and every 3 s
during construction; writes go through `rename()` so a kill can never leave a
half-written file; a count sidecar `<out>.cnt` lets the driver rank chains that
were killed.  `run_entry.sh` additionally drops a valid random code into
`OUTFILE` before any solver starts, so the worst case is a random-code score
rather than -10^6.

## 2. Threading: one chain on six threads is *slower* than one chain on one

`covsearch` parallelises over candidate moves.  Paired runs, `covsearch`, 30 s:

| cell | `--threads 1` | `--threads 6` |
|---|---|---|
| `K_6(6,3)@41` | 870/773/730 it/s, unc **8 / 14 / 33** | 111/145/96 it/s, unc 47 / 51 / 52 |
| `K_3(11,4)@81` | 306/232 it/s, unc **463 / 454** | 111/97 it/s, unc 483 / 485 |
| `K_6(8,4)@169` | 6/8 it/s, unc 40 / 33 | 26/30 it/s, unc **26 / 17** |

The OpenMP fork/join per iteration costs more than the sphere walk it is hiding
when `S = C(n-1,R)(q-1)^R` is small.  So the entry runs a **portfolio**: `P`
independent chains of `T` threads with `P·T ≤ 6`.  The rule is on the cost of one
construction pass, `M·|B_R|` touches, because every chain pays that in full:

* `M·|B_R| ≥ 1e8` → `P=1, T=6` (`K_8(9,4)@940`: 3.1e8)
* otherwise → `P=6, T=1` (`K_6(6,3)`: 1.2e5, `K_3(11,4)`: 5.5e5, `K_6(8,4)`: 8.7e6)
* `P` is additionally capped by `20 GB / 2·q^n`.

Paired check of the two policies on `K_6(8,4)@169`, 45 s, 3 seeds:
`P=6,T=1` → 32 / 24 / 62 (median 32); `P=1,T=6` → 44 / 32 / 43 (median 43).

The portfolio also converts run-to-run variance into a best-of-`P`, which is what
the judge's per-cell *median* rewards.

## 3. The word you work on: the baseline's biased pick is worth 15%

`covsearch`'s `pick_uncovered` returns *the first uncovered word at or after a
random index*.  That is **not** uniform — a word is returned with probability
proportional to the run of covered words in front of it, so isolated uncovered
words are strongly preferred.  It reads like an implementation detail; it is the
single largest quality knob found in this session.

I first replaced it with a uniform draw from a cached list of uncovered words
(cheaper in the endgame).  Paired, `K_3(11,4)@81`, 4 seeds, 30 s, same binary,
only the pick differing:

| selection | uncovered | mean |
|---|---|---|
| uniform (cache) | 508, 556, 534, 554 | 538 |
| **size-biased scan (covsearch)** | 450, 447, 467, 481 | **461** |

17% worse for uniform.  The scan is kept; its expected cost is `q^n/uncovered`
sequential reads, which is self-limiting, and it is capped at 4M reads with the
cache as the fallback so a nearly-solved huge instance cannot stall on it.

Also tried, since the mandate asks for WalkSAT-style weighted selection:
**hardest-first** — draw `b` uncovered words, keep the one whose nearest codeword
is furthest away.  Paired, 4 seeds, 30 s:

| `b` | `K_3(11,4)@81` mean | `K_6(6,3)@41` mean |
|---|---|---|
| 1 (off) | **467** | 31 |
| 2 | 478 | **15.5** |
| 4 | 480 | 21.5 |

Helps one cell, hurts the other, both within the spread of four seeds.  Left off
(`--hard 1`).  This is the trap `cov/NOTES.md` §6 warns about and I am not going to claim
it as a win on this evidence.

## 4. Construction: best-of-k max-uncovered-ball placement, time-throttled

`covsearch` places each initial codeword on a random uncovered word.  Scoring `k`
sampled uncovered words by how many uncovered words their whole radius-`R` ball
contains, and placing on the best, is strictly better.  `K_8(9,4)@940`, init
uncovered (all one machine-load epoch, so the *quality* column is comparable
across rows, the cost column only roughly):

| `k` probes per placement | init uncovered |
|---|---|
| `covsearch` greedy (random uncovered) | 10.39 M |
| 1 | 10.44 M |
| 2 | 9.35 M |
| 4 | 8.82 M |
| 8 | 8.36 M |
| 16 | 8.02 M |
| 32 | 7.83 M |

For scale: the local search recovered only 0.5-0.6 M on this cell in the
remaining ~33 s, so under contention **the construction, not the search, decides
the score** — a regime `cov/NOTES.md` §6 was not tuned in.

That statement is load-conditional too, and I nearly drew the wrong conclusion
from it.  On an idle box `covsearch` gets far more iterations and recovers ~3.5 M
from a cheap greedy start, reaching 6.91 M — better than the 7.28 M this entry
reached from a *better* start in an earlier, more contended run.  Comparing those
two numbers directly is invalid (different machines, hours apart), and doing so
would have argued for gutting the construction.  The paired §10 run settles it:
run side by side, construction-heavy wins on this cell by 167k.  The lesson is
that only simultaneous runs are evidence here; cross-epoch numbers are not.

Cost is `(k+1)` ball walks per placement and one pass of `M` walks costs 10-100 s
on this cell depending on load, so a fixed `k` is unsafe.  The entry projects the
construction's finish time every 16 placements and halves or doubles `k` to keep
it inside `cfrac` of the budget, dropping to `k=0` past 80% of the budget.

`cfrac = 0.55` and `kinit = 12` come from equating the two marginal rates on the
numbers above.  Measure everything in "passes" of `M` ball walks; with the `k`
probes for one placement run in parallel across `T=6` threads a placement costs
`ceil(k/6)+1` passes, and on `K_8(9,4)@940` one pass was ~9 s:

| step | init gain | passes | gain per pass |
|---|---|---|---|
| k 6 → 12 | 0.35 M | 1 | 350k |
| k 12 → 16 | 0.13 M | ~0.7 | 190k |
| k 16 → 32 | 0.19 M | 3 | 63k |
| local search (33 s) | 0.60 M | 3.7 | 162k |

so probing is worth more than searching up to `k ≈ 12-16` and less beyond it,
which puts the construction at ~4 passes ≈ 55% of a 60 s budget.

### 4b. CELF lazy greedy: implemented, measured, rejected

`gain(w) = |ball_R(w) ∩ uncovered|` is monotone non-increasing as codewords are
added, so any previously computed gain is a valid upper bound and lazy greedy
(CELF) over a pool of candidate centres is *exact* over that pool at a few ball
walks per placement.  That should dominate best-of-`k`.  It does not:

| construction | init uncovered | wall |
|---|---|---|
| best-of-8 | 8.36 M | 105 s |
| CELF, pool 1024 | 8.46 M | 436 s |

(both launched at the same moment, so the wall column is comparable)

Worse *and* 4x slower.  Reason: with `n=9, R=4` a single placement lowers the
gain of essentially every centre in the pool, so the heap top is stale on every
step and the "only a few re-evaluations per step" premise collapses.  Lazy greedy
needs each step to invalidate few candidates; covering codes at these radii
invalidate all of them.  Kept in the code behind `--pool` (default 0, off) as a
documented negative result.

## 5. Two operators, chosen by an online bandit

* **FINE** — `covsearch`'s focused single-coordinate move, unchanged.
* **TELEPORT** — ruin-and-recreate: delete the least-uniquely-covering of
  `k_rem=3` sampled codewords, re-insert it at the best of `k_add=6` sampled
  uncovered words scored by ball-gain, revert if the result is worse.

Selection is a two-armed bandit over 0.1 s blocks, scored by **improvement of the
record per second**, 3% exploration, and hysteresis: TELEPORT has to look 2x
better than FINE to take a block, because FINE is the known-good default.

Three things about the estimator that had to be measured rather than assumed:

* Scoring the *current* uncovered count is a trap.  TELEPORT reverts its own
  damage so it can never score negative, while FINE legitimately random-walks
  upward on a plateau, and the bandit drifts to TELEPORT for the wrong reason.
  Scoring the record fixes it: both arms are then non-negative and a plateau
  leaves both at 0, where the tie-break picks FINE.
* Exploration blocks must not be a single move.  With a one-move exploration
  block, `gain / elapsed` divides a whole number by a microsecond: one lucky
  teleport produced a rate estimate of ~10^5 and bought the arm full blocks it
  had not earned.  Caught on `K_5(7,3)@45`, where TELEPORT was taking 50% of the
  budget for a total record gain of 1 against FINE's 2474.  Fixed by
  accumulating per-arm time and gain until at least 20 ms has been observed
  before touching the estimate; TELEPORT's share on that cell fell to 2%.
* Even so the arms are noisy on small cells, hence the 2x hysteresis.

What it does:

| cell | outcome |
|---|---|
| `K_6(6,3)@41` | bandit abandons TELEPORT: 2056 teleports, total record gain **0**, 116k fine moves |
| `K_3(11,4)@81` | same, TELEPORT gain 0 |
| `K_8(9,4)@940`, lightly loaded | rate[FINE] 44.7k/s vs rate[TEL] 11.1k/s → uses FINE |
| `K_8(9,4)@940`, heavily loaded | rate[TEL] 30-37k/s vs rate[FINE] 7-8k/s → uses TELEPORT |

and the end-to-end paired check on `K_8(9,4)@940`, 60 s, 2 seeds, under load:

| ops | uncovered |
|---|---|
| `--ops fine` | 10.11 M, 10.05 M |
| **`--ops auto`** | **9.50 M, 9.69 M** |

The interesting part is that the ranking of the two operators *flips with machine
load* — FINE's advantage is 74 candidate evaluations run in parallel, and when
the box is oversubscribed that parallelism is not there, while TELEPORT's few
large ball walks degrade gracefully.  That is the argument for choosing online
rather than picking a winner offline: the arena box is shared.

**But do not read that table as "TELEPORT is good on the big cell".**  Re-measured
on an idle box (load ~55), `K_8(9,4)@940`, 60 s, 2 seeds:

| ops | uncovered | TELEPORT share of budget |
|---|---|---|
| fine-only | 6 941 026, 6 940 535 | — |
| auto | 6 960 429, 6 943 349 | 0.3 s / 60 s |

i.e. on an unloaded machine TELEPORT is not worth using on *any* of the four
public cells, and the +0.5 M it showed earlier was an artifact of FINE being
starved of CPU.  What it buys is insurance for the contended case, at a measured
cost of ~0.5% of the budget once §5b is in.  Given the arena box sat above load
400 for most of this session, that trade looks right, but the operator is not a
win in its own right and I am not claiming it as one.

Cost on the small cells, after the estimator fixes above (`--ops fine` vs
`--ops auto`, 4 seeds, 15 s, single-threaded, mean uncovered):

| cell | fine | auto |
|---|---|---|
| `K_6(6,3)@41` | 42.5 | 46.5 |
| `K_3(11,4)@81` | 538 | 526 |
| `K_6(5,2)@66` | 18 | 17.7 |
| `K_8(4,1)@192` | 33 | 37 |

i.e. a wash inside four-seed noise, against a clear +0.5 M on the big cell.
That is why `auto` ships.  It is also a confirmation of `cov/NOTES.md` §6: on
tight instances the extra stochastic operator buys nothing, and the bandit
discovers that in under a second instead of costing a tuning campaign.

### 5b. The bandit needed a long-horizon gate (found by losing a cell)

The first judge run (§10) lost `K_3(11,4)@81` to `covsearch` by 10 uncovered.
The cause was not the search: `covsearch`'s five scores there span 8 points, so
it simply converges, while this entry was handing 20-30% of the budget to
TELEPORT on a cell where TELEPORT has *never once* moved the record.  EMA noise
was enough to clear the 2x hysteresis every so often, and a quarter of the search
time is worth about 10 uncovered words.

Fix: an exploit block for TELEPORT now requires it to be ahead on the **whole run
so far** (`gains[tel]/time[tel] > 1.2 · gains[fine]/time[fine]`) as well as on
the EMA.  Cumulative statistics are far less noisy than a two-block EMA, and an
operator that has produced nothing all run can no longer win a block on a lucky
fluctuation.  Measured after the fix, 4 seeds, 60 s, single chain:

| cell | fine-only | gated auto | TELEPORT share | `covsearch --threads 1` |
|---|---|---|---|---|
| `K_3(11,4)@81` | 427.8 mean | **427.5 mean** | 0.4 s / 60 s | 435.8 mean |
| `K_6(6,3)@41` | 11.5 mean | 11.5 mean | 0.3 s / 60 s | 7.5 mean |
| `K_8(9,4)@940` | 6.941 M | 6.952 M | 0.3 s / 60 s | — |

The tax is gone (25% → 0.5%) and `auto` now tracks `fine-only` everywhere
instead of costing a cell.  Note the `K_6(6,3)` column: per *chain* `covsearch`
at `--threads 1` is as good as this solver or better; the entry's win on that
cell in §10 comes from the portfolio, not from the move set.

## 6. Candidate breadth is not universally dominant

`cov/NOTES.md` §6 measured breadth on `K_6(6,3)` and found it dominates.  On the big loose
cell it does not.  `K_8(9,4)@940`, fine-only, `kinit=8`, 60 s, 2 seeds, showing
uncovered words removed by the *search* (init held equal):

| `--cand` | search gain |
|---|---|
| 8 | 382k, 450k |
| **32** | **608k, 629k** |
| unlimited | 519k, 574k |

~10% in favour of a bounded breadth of 32, i.e. real but small, and it is inside
the seed-to-seed spread on the other three cells.  Default left at unlimited; the
knob is `--cand`.

## 7. Output hygiene (worth 1000 points on its own)

`arena_judge.py` awards 1000 only when the number of **distinct** words reaches
`M`; a genuine cover containing one repeated codeword scores 0 instead.
`covsearch` does not deduplicate.  `strategy.c` replaces repeats with fresh
unused words at write time — adding a codeword can never uncover anything, so the
cover survives — and never emits an uninitialised digit (the code array is
zeroed, so a partial code written mid-construction is still parseable).

## 8. What is *not* in here

* **Late-acceptance hill climbing / record-to-record travel.**  Not implemented.
  The bandit already handles "when is a different move worth it", and NOTES §6's
  SA result (33x the move rate, still loses) plus the measured `--ops auto`
  behaviour on the small cells say the acceptance criterion is not where the
  money is on these instances.  Honest status: untested by me.
* **Restart from perturbed elites.**  *Implemented but shipped off* (`--elite k`,
  default 0): every k-th fruitless kick it rebuilds `cnt[]` from the incumbent
  and kicks three times harder, so the perturbation lands on an elite instead of
  a plateau wanderer.  I could not get a usable measurement: under a load average
  of 435 a 30 s chain on `K_6(6,3)` completes ~4 000 iterations, the stall
  threshold is 5 000, and the mechanism never fires (`kicks=0` in most runs).
  `--elite 3` vs `--elite 0`, 4 seeds, 30 s: `K_6(6,3)` mean 30 vs 32.8,
  `K_3(11,4)` mean 514 vs 502 — noise around a feature that was not active.
  Default off rather than claim it.
* **Cross-chain sharing.**  The portfolio chains are fully independent.  Sharing
  an elite between the six would need a file or shared-memory channel in the
  driver; not attempted.
* **Sampled ball-gain probes.**  The construction's cost is dominated by exact
  ball walks (`|B_R|` touches each).  Estimating the gain from a few thousand
  uniform samples from the ball is ~10x cheaper per probe at ~4% relative error,
  which would buy a much larger `k`.  Sketched, not implemented: the ranking is
  only useful if the spread of true gains between candidate centres exceeds the
  sampling error, and I did not measure that spread.
* **Linear-code search.**  When `M = q^k` exactly (`K_3(11,4)@81 = 3^4`) the
  covering radius of a linear code is computable in one pass over `q^n` words by
  minimum coset weight, which is a search space several orders of magnitude
  smaller than the free code.  Kéri gives that cell upper-bound key `m` (a
  search) rather than `p` (a linear construction), and the best direct sum from
  his own table is `K_3(4,2)·K_3(7,2) = 3·34 = 102 > 81`, so there is no cheap
  seed either.  Not attempted.
* **Product/direct-sum seeding.**  For `K_8(9,4)` the best product available from
  Kéri's own table is `K_8(4,2)·K_8(5,2) = 23·128 = 2944` against a target of 940,
  so there is nothing to descend from on this cell.  It is the right move for
  cells whose target is near the tabulated bound (`cov/NOTES.md` §9) and the wrong one
  here.

## 9. What the entry actually is

`run_entry.sh` — writes a valid random code to `OUTFILE` immediately, picks
`(P,T)` from `M·|B_R|`, launches `P` chains with different seeds under
`timeout -s KILL`, stops the whole portfolio the instant one chain reports a full
cover (the judge breaks ties on median wall time), and copies the chain with the
smallest `.cnt` sidecar to `OUTFILE`.  Solver budget is `TIME_S - max(2, 4%)`
with the hard kill 0.3 s inside `TIME_S`.

`strategy.c` defaults: `--kinit 12 --cfrac 0.55` (construction),
`--ops auto --block 0.1 --eps 0.03` (bandit), `--krem 3 --kadd 6` (teleport),
size-biased pick, `--hard 1`, `--pool 0`, `--elite 0`, tabu as covsearch,
breadth unlimited.  Every one of those is a knob so the choices can be re-checked
rather than trusted.

## 10. Judge results

Both columns are `--seeds 5 --time 60`: `arena_judge.py strategy` against
`./cmp_baseline.py` (covsearch, identical protocol, identical verifier), the two
processes **launched together** so they share the machine.  Two full runs, before
and after the §5b bandit gate and early-stop-on-solve.

### Run 2 (final, 12:44, load ~130 throughout)

| cell | strategy | baseline (`covsearch --threads 6`) | delta |
|---|---|---|---|
| `K_6(6,3)@41` | **4/5 solved, median 1000, wall 4.8 s** | 1/5, median -8, wall 60.0 s | **+3 solves, 12x faster** |
| `K_6(8,4)@169` | **median -1**, wall 57.8 s | median -4, wall 60.9 s | **+3** |
| `K_8(9,4)@940` | median -8 899 639, wall 58.1 s | median **-8 834 755**, wall 118.7 s, **-10^6 on 2 of 5 seeds** | -64 884 |
| `K_3(11,4)@81` | **median -433**, wall 57.9 s | median -452, wall 60.1 s | **+19** |
| **total_solved** | **4** | 1 | **+3** |
| sum_median | -8 899 073 | **-8 835 219** | -63 854 |

### Run 1 (12:19, load falling 630 -> 40, pre-fix binary)

| cell | strategy | baseline | delta |
|---|---|---|---|
| `K_6(6,3)@41` | **1/5 solved, median -9**, wall 58.2 s | 0/5, median -46, wall 62.2 s | **+37, +1 solve** |
| `K_6(8,4)@169` | 1/5, median -1, wall 57.8 s | 1/5, median -1, wall 60.3 s | tie |
| `K_8(9,4)@940` | **median -6 747 828**, wall 57.8 s | median -6 914 984, wall 65.7 s | **+167 156** |
| `K_3(11,4)@81` | median -422, wall 57.8 s | median **-412**, wall 60.0 s | -10 |
| **total_solved** | **2** | 1 | **+1** |
| sum_median | **-6 748 260** | -6 915 443 | **+167 183** |

### What this actually says

* **`K_6(6,3)@41` is a clear, repeatable win, and it is the portfolio's.**  4/5
  solved at a **median wall of 4.8 s** against 1/5 at 60 s.  Six independent
  single-threaded chains (§2) plus stopping the portfolio the moment one reports
  a cover.  Per *chain* this solver is no better than `covsearch --threads 1`
  (§5b) — the win is entirely in how the six threads are spent and in not
  burning the clock after the answer exists.  The judge breaks ties on median
  wall time, so 4.8 s vs 60 s counts twice.
* **`K_6(8,4)@169` and `K_3(11,4)@81` are small, consistent wins** in run 2
  (+3, +19), and the `K_3(11,4)` sign flipped from -10 to +19 exactly when the
  §5b gate stopped TELEPORT stealing a quarter of that cell's budget.
* **`K_8(9,4)@940` is a wash.**  +167k in run 1, -65k in run 2, i.e. ±1-2% either
  way depending on the hour.  I am not claiming that cell.
* **`total_solved` 4 vs 1 and 2 vs 1** is the result I would stand behind; the
  `sum_median` column is dominated by the one cell that is a wash, and it goes to
  `covsearch` in run 2 by 0.7%.

### The scoring quirk, and what happened to it

Under the original rules a valid partial on `K_8(9,4)@940` scored about
**-8 900 000** while invalid or missing output was priced at **-10^6**, so
emitting garbage was ~9x better than emitting the best partial cover anyone had
found.  The baseline's two lost files in run 2 each scored -10^6 — better than
its own three successful seeds.  This entry deliberately did the *worse* thing
under that scoring (it writes a valid random code to `OUTFILE` before any solver
starts) rather than farm the penalty.

Flagged to the arena owner; fixed in `RULES.md` **amendment 1** (2026-08-20,
pre-judging, commit `01ae8d0`): partials are now
`-floor(uncovered · 10^6 / q^n) - 1`, i.e. the uncovered *fraction* in ppm, and
invalid output moved to `-2·10^6` so it is always worse than any valid partial.
The runs in the two tables above were scored under the *old* rules; §10b re-runs
both columns under the amended judge.

Two consequences for this entry, neither of which changed a design decision:

* Writing a valid fallback immediately, and never emitting an unparseable or
  duplicate-bearing file, went from "honest but costly" to "strictly correct".
* Cell weights are now uncovered *fractions*: `K_8(9,4)@940` sits at ~6.6%
  uncovered and still carries ~97% of the total magnitude, but it largely
  cancels between the two entries, so the differentiators are the `K_6(6,3)`
  solve (worth 1000 - (-172) = 1172) and `K_3(11,4)`.  For future work the
  arithmetic says a 10% cut in `K_8(9,4)`'s uncovered fraction is worth ~+6 700,
  more than solving `K_6(6,3)` outright — the construction is still where the
  money is.

## 10b. Run 2 re-scored under RULES amendment 1

The amended score is a deterministic, strictly monotone function of the uncovered
count and `q^n`, and neither column had a run whose *validity* is in question
beyond the two baseline files that were already recorded as lost.  So run 2 can
be re-scored exactly from its recorded per-seed outcomes — this is arithmetic on
the same measurements, not a new run:

| cell | strategy | baseline | delta |
|---|---|---|---|
| `K_6(6,3)@41` | **1 000** (4/5 solved) | -172 | **+1 172** |
| `K_6(8,4)@169` | **-1** | -3 | **+2** |
| `K_8(9,4)@940` | **-66 308** | -66 597 | **+289** |
| `K_3(11,4)@81` | **-2 445** | -2 552 | **+107** |
| **sum_median** | **-67 754** | -69 324 | **+1 570** |
| **total_solved** | **4** | 1 | **+3** |

Under the amended rules this entry wins **all four public cells**.  The
`K_8(9,4)@940` column flips from a 65k loss to a 289 win for a reason worth
stating plainly: nothing about either solver changed, only the price of losing
your output file.  Under the old rules the baseline's two lost seeds scored
-10^6 each, *better* than its three successful seeds at ~-8.9 M, so crashing
pulled its median up; at -2·10^6 they push it down instead.  That cell is still
genuinely a wash on search quality (§10, run 1 vs run 2), and I am not claiming
otherwise — the win there is the time handling of §1, now that it is priced.

Caveat on this table: the `--seeds 5 --time 60` re-run under the amended judge
was launched (13:20) but its `arena_judge` process was killed partway, and the
paired baseline was then unpaired and stopped.  Rather than report a
load-mismatched comparison I re-scored run 2, whose two columns did run side by
side.  §10's run 1 and run 2 tables are the raw measurements.
