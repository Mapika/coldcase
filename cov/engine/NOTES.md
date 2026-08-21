# cov/engine — the merged production engine

The arena judged four solvers against each other on the same cells under the
same contract. This directory is what the four of them are worth **together**,
and it is the thing the record hunt runs from now on.

The verdict the merge implements:

| lane | what it won | where it is here |
|---|---|---|
| `structure` | held-out judging, 11/28 solves. The algebraic front end is devastating on lattice-shaped cells | phase A: `lincov`, and `symsearch` inside the phase-B portfolio |
| `lowlevel` | the hardened free-search inner loop: 2–4.3x bit-identical speedup, deadline-safe init and publishing | `covfast` as a portfolio engine, and its deadline/publishing discipline transplanted into `covsearch2e` |
| `cov/opt` p5b | the best *evaluation algorithm*: shared sphere + uncovered-word list, 6–10x | `covsearch2e` = `covsearch2 --preset p5b` |
| `strategy` | portfolio scheduling: N single-threaded chains beat 1 multi-threaded one, stop-on-solve, time discipline | the whole of phase B |

Nothing that works was rewritten. `lincov.c`, `symsearch.c`, `covcount.c` and
`covfast.c` are **verbatim copies** of the arena entries' sources; `covengine`
is a driver that runs them.

---

## 1. Files

| file | what |
|---|---|
| `covengine` | the driver. Two interfaces: the arena's `Q N R M SEED TIME_S OUT`, and covsearch's `-q -n -R -M -t -s --in --out --threads --quiet`, so `COVSEARCH_BIN=cov/engine/covengine` makes `cov/campaign.py` work unchanged |
| `campaign2.py` | thin fork of `cov/campaign.py` for descend-from workflows; **imports `record()` from `campaign.py` unmodified**, so the verification gate is literally the same code |
| `covsearch2e.c` | `cov/opt/covsearch2.c` + the deadline/publishing hardening of `covfast` (§3). The search is bit-identical to `covsearch2`; `selftest.sh` proves it |
| `covfast.c`, `lincov.c`, `symsearch.c`, `covcount.c` | verbatim copies from `cov/arena/lowlevel` and `cov/arena/structure` |
| `selftest.sh` | the correctness gate (§3) |
| `bench_engines.sh`, `bench_big.sh`, `ab_cell.sh` | the paired benchmarks behind every policy choice below |
| `../arena/engine0/run_entry.sh` | arena wrapper, for judging the engine against its own parts |

## 2. What the engine does with a cell

**Phase A — algebra, seconds.**

* `lincov` over every feasible `k`: the code is a union of `M/q^k` cosets of a
  linear `[n,k]_q` code, which turns a covering question about `q^n` words into
  one about `q^(n-k)` syndromes. Applicable only when `q` is a prime power and
  `q^k | M`, and it exits in under a millisecond when it is not, so an
  inapplicable cell costs nothing. Where it applies it is decisive:
  `K_3(11,4) <= 81` — the judge's declared known-hard control, which the
  production solver plateaued on at 410 uncovered — comes out of the engine
  **verified in 0.18 s wall including process start-up**.
* Seeds that cost no search: a recorded code for this cell at `M` or just above
  it (handed to a chain for remove-and-repair descent), and direct sums
  `K_q(n1,R1) x K_q(n2,R2)` over every split with the factors taken from
  `cov/results/` or from perfect Hamming codes — the "factor attack" of
  `cov/NOTES.md` §9 that produced `K_6(9,4)`, `K_6(10,4)` and `K_6(10,5)`.
  A seed is only used when its remove-and-repair cost is affordable: each
  deletion re-scores every surviving codeword over its ball, so an excess of
  `E` words costs about `E · M' · |B_R|` touches.

**Phase B — a portfolio of free-search chains, the rest of the budget.**
Independent chains, one thread each unless the cell says otherwise, different
seeds, different engines, different amounts of imposed symmetry, different
seeds from phase A. The world stops the moment one chain reports a cover. The
best partial anyone has on disk is republished atomically every 2 s, so a kill
at any moment leaves the best code found so far in `--out`.

**Finish.** `covcount` re-reads the winner from disk, de-duplicates it, tops it
up to exactly `M` **distinct** words and recomputes coverage by direct ball
marking. No solver's own incremental counter is ever the evidence.

## 3. `covsearch2e`: p5b with lowlevel's clock discipline

`cov/opt/covsearch2 --preset p5b` is the fastest evaluator we have (2.03x /
6.12x / 8.07x bit-identical against the production solver on the three
benchmark cells, `cov/opt/METHODS.md` §G.2b). As shipped it is not safe under a
deadline: it starts its `-t` clock *after* initialisation, checks it every 16th
iteration, and writes `--out` exactly once, at exit. On `K_8(9,4)` one
iteration is most of a second and initialisation is ~28 s, so a 60 s budget
becomes a >90 s run — and a run that is killed has written nothing at all.
That is the single most expensive bug an entry can have, and both `structure`
and `lowlevel` were bitten by it before they fixed it.

`covsearch2e.c` is `covsearch2.c` with those three properties imported from
`covfast`, and nothing else:

1. the wall clock starts at process start, so the budget covers initialisation;
2. the clock is read on **every** iteration (a vDSO read of ~20 ns against a
   ≥30 µs iteration);
3. the incumbent is published to `--out` atomically (write `.part`, `rename`)
   as soon as one exists and on every improvement, throttled to 1 Hz; and what
   is written is always exactly `M` **distinct** codewords (a genuine cover
   containing a repeat is scored as a code of size `M-1`);
4. initialisation is itself deadline-bounded: if the budget dies while the `M`
   initial balls are being marked, the remaining codewords are placed without
   marking, a complete valid file is published, and the run reports
   `uncovered=-1` — unknown — rather than a number its counters no longer
   support.

None of that reads the clock inside a search decision, so the trajectory is
unchanged. **`selftest.sh` proves it**, and this is the gate that lets the
engine write to `cov/results/`:

```
ALL PASS  (40 trajectory comparisons, 40 solver/verifier cross-checks)
```

* **40 trajectory comparisons**, 10 cells × 2 presets × 2 seeds, including the
  corners `R = 0`, `R = n-1`, `q = 2`, `q = 7`: `covsearch2e` and
  `cov/opt/covsearch2` report the identical `uncovered/iters/kicks`
  fingerprint at a fixed iteration budget.
* **40 solver/verifier cross-checks**: `cov/verify_cov.py --method pure`,
  re-reading the file from disk, reports *exactly* the number the solver
  claimed — including when that number is not zero. This is the check that
  catches incremental-counter drift, which is the failure mode that would
  otherwise produce a confident false record.
* **distinctness**: every file written holds exactly `M` distinct codewords.
* **deadline**: on `K_8(9,4)@940`, `-t 0.05` returns in 0.1 s and `-t 2` in
  2.0 s, each with a complete 940-word file — where the unpatched binary would
  have spent ~28 s in initialisation before writing anything.
* **SIGKILL**: killed 6 s into a `K_6(8,4)` run, the output file holds a valid
  169-word code.

## 4. The policy choices, and the measurements behind them

Every table below is a **paired** measurement: all variants were launched at
the same moment on the same seeds, because this box is shared with a production
sweep whose load average moved between 78 and 125 during the session and
sequential wall-clock numbers on it are worthless.

### 4.1 Which free-search binary (`bench_engines.sh`, 45 s, 1 thread, 3 seeds)

Best uncovered reached, with iterations in brackets:

| cell | `covfast` | `covsearch2e` p5b | p5b `--wide` |
|---|---|---|---|
| `K_6(6,3)@41` | 9, 8, 9 (0.8–1.1 M it) | **0, 8, 14** (0.46–0.53 M it) | 9, 21, 16 |
| `K_3(11,4)@81` | 421, 427, 447 | 449, 411, 434 | **328, 320, 316** |
| `K_6(8,4)@169` | 2, 3, 3 (8.6–9.6 k it) | 4, 2, 2 (14–21 k it) | **2, 2, 2** |
| `K_8(9,4)@940` | 9.36, 9.41, 9.32 M | 9.25, 9.20, 9.21 M | **9.18, 9.12, 9.16 M** |

Reading it: `covfast` does ~2x the iterations of `covsearch2e` on the small
cache-resident `K_6(6,3)` (its NEON + specialised kernels are worth 4.3x there
against p5b's 2.03x), and `covsearch2e` does ~2x the iterations on everything
bigger, where the shared sphere and the uncovered-word list dominate. Quality
tracks that, weakly, so the portfolio runs **both** and the mix is decided by
`q^n`: below 3·10^5 half the free chains are `covfast`, above it a quarter.

`--wide` (the whole `n(q-1)` single-coordinate neighbourhood, free once the
uncovered list exists) is a per-cell decision. `METHODS.md` measured it as a
3.6x *loss* on `K_6(6,3)` (redundancy 2.55) and a large win on `K_6(8,4)` and
`K_8(9,4)` (redundancy 5.2, 7.3); the new number here is `K_3(11,4)` at
redundancy 3.13, where it is worth **~100 uncovered words out of 177 147**, the
largest single effect anyone has found on that cell short of `lincov`. So the
crossover is near redundancy 3 — but it is not sharp enough to bet a portfolio
on, so the minority setting always gets chains too.

### 4.2 How to spend cores on a big cell (`bench_big.sh`, `K_8(9,4)@940`, 55 s, 6 cores per variant, 2 seeds)

| variant | seed 1000 | seed 1001 |
|---|---|---|
| `symsearch`, 6 threads | **6 443 720** | **6 508 360** |
| `covsearch2e --wide`, 6 threads | 7 753 081 | 7 737 290 |
| `covfast`, 6 threads | 7 783 951 | 7 785 733 |
| 6 × 1-thread `covsearch2e --wide` portfolio | 8 906 542 | 9 069 727 |

Two policies fall out, and they are opposite to the small-cell ones:

* **the quotient search wins outright on a big cell**, by 17% over the best
  free search — as `structure` reported, and now measured against the *fast*
  free searches rather than against the baseline. So when
  `q^n > 5·10^7` or one initialisation would eat more than 35% of the budget,
  the portfolio is all-symmetric and `symsearch` keeps the whole budget instead
  of handing 55% of it to a free phase.
* **one wide chain beats a portfolio of narrow ones**, by 27%. This is the
  regime `lowlevel` identified where the OpenMP fork/join is cheap relative to
  a sphere walk (one `K_8(9,4)` iteration is most of a second), and it is the
  exact reverse of the small-cell measurement that gave `strategy` its
  `K_6(6,3)` win. The engine decides it from `M·|B_R|` and the budget, not from
  a constant.

### 4.3 Symmetry, and how much of it

`structure` measured on `K_6(8,4)@169` that neither end of the symmetry axis
solves the cell — a fully invariant code plateaus 1–2 quotient points short and
one spare codeword provably cannot repair an orbit, while a fully free search
plateaus too — but that "most of the code invariant plus a genuinely free
remainder" solved it 3 times out of 3 at several points in between. So on
cells small enough to restart cheaply, a third of the chains run
`symsearch` at `M/q, M/q − s, M/q − 2s, …` orbits for 45% of their slice and
then hand the invariant code to `covsearch2e` for the rest.

### 4.4 Restarts

`METHODS.md` §G.4 measured that at targets the search reaches routinely the
run-time distribution is light-tailed (cv 0.2–0.5) and restarting can only
throw progress away, while on a tight cell at a hard target it becomes heavier
than exponential and restarting cut expected time-to-target by 4x. The engine
restarts chains with a fresh seed only where the cell is tight
(redundancy < 3.5) and an initialisation is nearly free.

### 4.5 A bug worth recording

The first version seeded chain `i` with `1 + i`, ignoring the caller's seed.
Every run of a cell was therefore byte-identical, which an A/B on
`K_6(6,3)@41` made obvious — three different policies, two different judge
seeds, and all six runs returned exactly 8 uncovered. Under the judge that
would have turned 15 seeds into 1 seed reported 15 times. Chain seeds are now
`seed·1000003 + i·7919`.

## 5. The judge, all four entries in one window

`python3 scripts/arena_judge.py structure lowlevel strategy engine0 --seeds 5
--time 60`, public cells, run as one job so the four entries share the machine
(load average 76–180 throughout; the sweep and this engine's own record hunt
were the background). Raw log: `results/arena/public_4way_engine0.log`.

| cell | `structure` | `lowlevel` | `strategy` | **`engine0`** |
|---|---|---|---|---|
| `K_6(6,3)@41` | 2/5, −172, 57.6 s | 0/5, −193 | 1/5, −172 | **4/5, +1000, 20.2 s** |
| `K_6(8,4)@169` | 5/5, +1000, 57.5 s | 0/5, −2 | 0/5, −5 | **5/5, +1000, 26.7 s** |
| `K_8(9,4)@940` | 0/5, −49 638 | 0/5, −62 057 | 0/5, −69 688 | **0/5, −48 311** |
| `K_3(11,4)@81` | 5/5, +1000, 0.3 s | 0/5, −2 377 | 0/5, −2 456 | **5/5, +1000, 0.3 s** |
| **total solved** | 12 / 20 | 0 / 20 | 1 / 20 | **14 / 20** |
| **sum of medians** | −47 810 | −64 629 | −72 321 | **−45 311** |

The engine is **at least as good as every entry on every cell and strictly
better on three of the four**, and it wins both aggregate columns. Against the
reference figure in the brief — `structure`'s public 12/20, −50 613 — it is
+2 solves and +5 302.

Where each cell's win comes from is exactly the merge:

* `K_6(6,3)@41` — 2/5 → **4/5, and the median wall falls from 57.6 s to 20.2 s**
  (the judge breaks ties on median wall). Portfolio + stop-on-solve
  (`strategy`) + `covfast`'s inner loop on the one cell where it is fastest
  (`lowlevel`) + restarts on a tight cell (`METHODS.md` §G.4).
* `K_6(8,4)@169` — 5/5 for both, but **2.2x faster to the cover** (57.5 s →
  26.7 s): `structure`'s symmetry-axis portfolio with p5b's evaluator under it
  instead of the baseline's.
* `K_8(9,4)@940` — the best number anyone has produced on this cell,
  −48 311 against `structure`'s −49 638 and the baseline's −66 092: the
  quotient search, run the way `bench_big.sh` says to run it.
* `K_3(11,4)@81` — the declared known-hard control, 5/5 in 0.3 s, algebra.

Note `lowlevel` scoring 0/20 and `strategy` 1/20 in this window against the
4/20 and 4/20 their own NOTES report: both are throughput-bound on cells that
sit right at the edge of solvable, and this window was busier than theirs.
That is the load caveat `structure` §3.5 documents, and it is the reason all
four columns above were measured in one job rather than quoted from four.

## 6. Three of `cov/NOTES.md` §8.4's failures, settled in under a second each

`cov/NOTES.md` §8.4 ("what did not work") lists cells the production pipeline
could not reach *at Kéri's own upper bound*, let alone below it. Three of them
are algebraic cells, and phase A settles all three before the clock matters:

```
$ ./lincov -q 8 -n 8  -R 4 -M 512 -k 3 -t 60 -s 1 --out out.txt
LINRESULT q=8 n=8 R=4 M=512 k=3 c=1 m=5 syndromes=32768 uncovered_words=0 evals=1 time=0.00
$ ./lincov -q 5 -n 11 -R 5 -M 625 -k 4 -t 60 -s 1 --out out.txt
LINRESULT q=5 n=11 R=5 M=625 k=4 c=1 m=7 syndromes=78125 uncovered_words=0 evals=69 time=0.54
$ ./lincov -q 3 -n 12 -R 5 -M 54  -k 3 -t 60 -s 1 --out out.txt
LINRESULT q=3 n=12 R=5 M=54 k=3 c=2 m=9 syndromes=19683 uncovered_words=0 evals=706 time=0.45
```

All three verified independently by `cov/verify_cov.py --method numpy`:
`K_8(8,4) <= 512`, `K_5(11,5) <= 625`, `K_3(12,5) <= 54`.

Against `cov/NOTES.md` §8.4, which recorded: *"`K_8(8,4)` (Kéri 512) and
`K_10(8,5)` (Kéri 168) from scratch: neither reached the incumbent within the
time available. Descending from a loose start got `K_8(8,4)` to 740"*, and
*"`K_3(12,5)` at `M = 54`: plateaued at 432"*. `512 = 8^3`, `625 = 5^4` and
`54 = 2·3^3` are all `c · q^k`, i.e. unions of cosets of a linear code, and a
free search over subsets of `Z_q^n` was simply looking in the wrong place —
the same diagnosis `structure` made for `K_3(11,4) = 81 = 3^4`.

These are reproductions of Kéri's incumbents, not new records. What they buy is
a *real starting point*: the siege on `K_8(8,4) <= 511` and `K_5(11,5) <= 624`
now begins from a verified code at the incumbent instead of from 740 and from
nothing.

## 7. `campaign.py` compatibility, checked

```
$ COVSEARCH_BIN=$PWD/engine/covengine python3 campaign.py \
      -q 6 -n 6 -R 3 -M 30 --workers 1 --threads 2 -t 20
cell K_6(6,3)  q^n=46656  ball=2906
  Keri 2011: 24 - 41  (upper key q)
--- M = 30 ---
  workers: best uncovered=1014   throughput 0-0 moves/s each
  M=30 not solved; stopping descent
```

The unmodified `cov/campaign.py` drives the engine through `COVSEARCH_BIN` and
parses its `RESULT` line. Use `--workers 1`: covengine *is* the worker pool, so
`--workers W` would launch `W` portfolios of `--threads` cores each.
`campaign2.py` is the interface that says that in its own arguments.

## 8. What the engine found

All of it under the resource contract: ≤ 40 cores, `nice -n 12`, no GPU, on a
box shared with the production sweep (load average 76–180 throughout). Every
code below was written by `campaign2.py`, which means every one of them was
re-read from disk and passed `cov/verify_cov.py`'s **two exhaustive methods**
with the distinct-codeword count checked against the claimed `M`, before it was
allowed into `cov/results/`. Nothing here rests on a solver's own counters.

### 8.1 New upper bounds

| cell | Kéri 2011 | our previous best | **now** | vs Kéri | vs us |
|---|---|---|---|---|---|
| `K_6(8,4)` | 216 | 167 | **166** | −23.1% | −1 |
| `K_6(9,5)` | 144 | 123 | **119** | −17.4% | −4 |
| `K_6(10,5)` | 615 | 610 | **523** | −15.0% | −87 |
| `K_7(9,4)` | 1843 | 1743 | **1600** | −13.2% | −143 |
| `K_6(9,4)` | 738 | 703 | **660** | −10.6% | −43 |
| `K_7(8,4)` | 343 | 329 | **316** | −7.9% | −13 |
| `K_6(7,3)` | 246 | 232 | **227** | −7.7% | −5 |
| `K_6(8,3)` | 1080 | 1045 | **1005** | −6.9% | −40 |
| **`K_5(11,5)`** | 625 | *(none)* | **602** | −3.7% | new cell |
| `K_6(10,4)` | 2952 | 2951 | **2890** | −2.1% | −61 |
| **`K_8(8,4)`** | 512 | *(none — best was 740)* | **505** | −1.4% | new cell |

Every one of the nine cells we already held a record on was improved, and two
cells that `cov/NOTES.md` §8.4 recorded as outright failures became records.
`K_5(11,5)` is also the first record outside the `q ≥ 6` corner — and the one
cell where the post-2011 SDP lower bounds apply at all (Gijswijt–Polak give
`K_5(11,5) >= 100`, below Kéri's 103, so the gap there is 103–602).

The two new cells are the algebra doing the work. Kéri's `512 = 8^3` and
`625 = 5^4` are unions of cosets of a linear code; `lincov` produced both in
under a second (§6) and the descent then started from a verified incumbent
instead of from 740 and from nothing.

The headline codes were additionally put through `verify_independent.py`, whose
dilation method shares no code with `verify_cov.py` and never computes a
distance at all: `K_6(9,5) <= 122`, `K_6(9,5) <= 119` and `K_8(8,4) <= 505`
all pass, and `mindist` reports the exact covering radius as 5, 5 and 4.

**These are not converged.** Most of the descents were still taking a notch
every few minutes when the session's time ran out; each stopped at a floor set
in the attack list, not at a wall. `cov/results/final_records.json` carries the
current state with provenance per cell.

### 8.2 Where it stopped

* `K_6(6,3) <= 40` — **not reached**; best 7 uncovered out of 46 656.
  30 min × 12 cores and then 2 × 60 min × 16 cores, both starting from the
  verified 41-word code. The cell is tight (redundancy 2.55) and this is the
  third session to fail on it.

  Worth updating the reason it was the headline target, though. `cov/NOTES.md`
  §9 wanted it because `K_6(10,4) = K_6(4,1)·K_6(6,3) = 72·41` and
  `K_6(10,5) = K_6(6,3)·K_6(4,2) = 41·15`, so `40` would have given 2880 and
  600 for free. Direct descent has since taken `K_6(10,5)` to **530** and
  `K_6(10,4)` to **2890**, i.e. past *both* numbers the factor attack would
  have bought. `K_6(6,3) <= 40` is no longer worth anything to those two cells;
  it would now only be a record in its own right.
* `K_8(8,4) <= 504` — not reached in 900 s × 16 cores, then 2 × 2700 s × 12
  cores. 505 is a hard floor for this engine at this budget.
* `K_6(9,5) <= 118` — not reached in 2 × 2700 s × 16 cores. 119 stands.
* Walls, one notch below the table: `K_6(9,4) <= 659`, `K_6(8,3) <= 1004`,
  `K_7(8,4) <= 315`, `K_6(7,3) <= 226`, `K_6(8,4) <= 165`, `K_5(11,5) <= 601`,
  at 900–3600 s per notch.
* **`K_6(10,5)`, `K_7(9,4)` and `K_6(10,4)` have not hit a wall at all.** Each
  has been relaunched several times, every time because it reached the floor
  written into the attack list rather than because it stopped improving: `K_6(10,5)` went
  610 -> 575 -> 530 -> 523, `K_7(9,4)` 1743 -> 1680 -> 1609 -> 1600 and
  `K_6(10,4)` 2951 -> 2942 -> 2890, still taking a notch every few minutes.
  These are three of the four highest-redundancy cells in `cov/NOTES.md` §9's
  never-searched list (9.51, 8.35 and 7.20), and the honest reading is that
  nobody has ever looked at them and there is a lot more slack there than this
  session had time to take. The table in §8.1 is a *snapshot of an unconverged
  descent*, not a limit; `python3 cov/engine/tally.py` regenerates it.

## 9. Honesty about the arena comparison

`cov/arena/engine0/run_entry.sh` sets `COVENGINE_NO_RESULTS=1`, which switches
off everything the engine would otherwise take from `cov/results/`. Two of the
four public cells — `K_6(6,3)@41` and `K_6(8,4)@169` — are cells whose answer
is already recorded there, and seeding a search with its own stored answer
would make the comparison against `structure`, `lowlevel` and `strategy`
meaningless. What phase A keeps in arena mode is exactly what was available to
every other entry: the algebra, and `constructions.py`'s primitive recipes.

Production (`campaign2.py`) leaves it on. There, reusing what we have already
proved is the entire point.
