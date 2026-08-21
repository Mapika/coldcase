# arena entry `structure` — exploit the invariances, not the clock

Mandate: attack `K_q(n,R) <= M` through the problem's group structure rather
than through raw search throughput. Three engines, ordered by how much
structure they impose, and a portfolio across that ordering so that
over-constraining can never cost us the answer.

Everything here lives in `cov/arena/structure/`. Nothing outside it was
touched. `cov/search/covsearch.c` was **read** (as the rules allow) and copied
into `freesearch.c` for the symmetry-broken phase.

---

## 1. The three engines

### 1.1 `lincov` — the code is a union of cosets of a linear code

Fix a linear `[n,k]_q` code `C0` with parity check matrix `H` (`m = n-k` rows),
let

```
S = { H e : wt(e) <= R }
```

be the syndromes reachable by an error of weight at most `R`, and take the code
to be a union of `c` cosets `C = U_i (v_i + C0)` with `sigma_i = H v_i`. Then

```
w is covered   <=>   exists i : H w - sigma_i in S
C covers Z_q^n <=>   U_i (sigma_i + S) = GF(q)^m.
```

The covering question stops being about `q^n` words and becomes a covering
question about `q^m` syndromes. For `K_3(11,4)` at `M = 81` that is **2187
syndromes instead of 177147 words**, and the only free variables are the `m*k`
entries of `A` in the systematic form `H = [I_m | A]` — 28 ternary symbols —
plus the `c-1` coset syndromes (`sigma_0 = 0` WLOG, by translating the code).
Systematic form is itself a canonical-form reduction: row operations do not
change the code and a coordinate permutation does not change the covering
radius, so every linear code is reachable.

`k` is the symmetry knob. `M = c*q^k`: large `k` means few cosets and a rigid,
highly structured search; small `k` means many cosets and something closer to a
free search. The driver walks `k` downwards.

**This is the single biggest result in the entry.** `K_3(11,4)` at `M = 81` is
the judge's declared "known-hard control", the cell `cov/NOTES.md` §8.4 records
as *"8 workers x 90 s x 3 rounds plateaued at 410 uncovered out of 177147. We
cannot reproduce the incumbent here, let alone beat it."*

```
$ for s in 1000..1014; do ./lincov -q 3 -n 11 -R 4 -M 81 -t 30 -s $s; done
LINRESULT ... k=4 c=1 m=7 syndromes=2187 uncovered_syndromes=0 restarts=1 evals=19 time=0.00
... 15/15 identical: uncovered = 0, every one inside 10 ms, 9-83 objective evaluations.
```

Verified independently:

```
$ python3 cov/verify_cov.py b4.txt -q 3 -n 11 -R 4 --method both
method pure : uncovered=0   (1.47s)
method numpy: uncovered=0   (2.16s)
RESULT    : VERIFIED  -- K_3(11,4) <= 81
```

`K_3(11,4) = 81 = 3^4` is Östergård's 1991 bound and it is a **linear** code;
looking for it among 177147-word subsets is looking in the wrong place. The
same module also settles `K_3(12,5) <= 54 = 2*3^3` in 0.12 s — another cell
`cov/NOTES.md` §8.4 lists as a failure (plateaued at 432 uncovered).

It does not apply everywhere: it needs `q` a prime power and `q^k | M` with
`k >= 1`. On `K_6(6,3)`, `K_6(8,4)` (q not a prime power) and `K_8(9,4)` at
`M = 940` (`8 \nmid 940`) it exits with status 3 in under a millisecond, so an
inapplicable cell costs nothing.

### 1.2 `symsearch` — the code is invariant under `x -> x + 1^n`

`G = <x -> x + (1,...,1)>` acts on `Z_q^n` by isometries and freely, so every
orbit has exactly `q` elements. If `C` is a union of orbits then the coverage
multiplicity `cnt[w]` is constant on orbits and the whole problem descends to
the quotient.

Concretely, map `w` to `(w_0, d)` with `d_i = w_i - w_0`; `G` moves only `w_0`,
so **the quotient is `Z_q^{n-1}`**. Normalise an orbit of codewords by its
member with first coordinate `0`, i.e. by `e in Z_q^{n-1}`. With
`delta = j - w_0`,

```
d(w, c^{(j)}) = [delta != 0] + #{ i : d_i != e_i + delta }
```

so one orbit acts on the quotient exactly like **`q` ordinary Hamming balls**:
one of radius `R` centred at `e`, and `q-1` of radius `R-1` centred at
`e + delta*1`. Every piece of the baseline solver — incremental counters, the
sphere-difference move, greedy init, kicks — carries over one radius at a time.

The arithmetic is exact, not approximate:

```
sum_delta C(n-2, R_delta) (q-1)^{R_delta}  ==  C(n-1, R) (q-1)^R
```

(for `K_8(9,4)`: `84035 + 7*12005 = 168070 = C(8,4)*7^4`). So **one quotient
move relocates `q` codewords for the price of one baseline move**, on an array
`q` times smaller. Measured on `K_8(9,4)` at `M = 940`:

| | counters | init | search rate |
|---|---|---|---|
| baseline `covsearch` (full space) | 8^9 = 134 M cells, 262 MB | 27.5 s | 1 it/s |
| `symsearch` (quotient) | 8^8 = 16.7 M cells, 33 MB | 5.7 s | 49 it/s |

The one place the reduction is not exact is candidate *ranking*: a read-only
evaluation sums the `q` sub-moves independently, and the leaving set of one
sub-centre can meet the entering set of another. That only affects which move
is picked. The commit applies the `q` sub-moves in sequence against the shared
counter array, so the maintained `uncovered` is exact — confirmed against the
verifier: `symsearch` reported 12 uncovered on a 168-word code for `K_6(8,4)`
and `verify_cov.py --method pure` reported 12.

### 1.3 `freesearch` — the baseline, with the symmetry broken

`cov/search/covsearch.c` with three changes that matter for a timed judge:

1. **The wall-clock deadline is tested on every iteration, not every 256th.**
   On `K_8(9,4)` one iteration costs ~0.7 s, so the shipped baseline turned a
   60 s budget into a **188 s run**. At the judge's `TIME_S=120` that is a
   ~250 s run against a 180 s `subprocess` timeout: killed before writing, i.e.
   an empty file, i.e. **-10^6**. This is the single most expensive bug in the
   baseline as an arena entry, and it is invisible on the three small cells.
2. **The clock starts before initialisation.** Allocating and marking 940 balls
   into a 262 MB counter array costs 27.5 s on `K_8(9,4)`; timing only the
   search loop is what let the overrun happen.
3. **The incumbent is flushed to `--out` atomically on every improvement**
   (throttled to one write a second, via `rename`), so a kill can never lose
   the run, and `covcount` tops the answer up to exactly `M` **distinct** words
   — the judge counts distinct words and scores a short or duplicate-bearing
   file as a non-solution even when it covers.
4. **Every improvement prints a line containing `uncovered=`.** If the process
   is killed before it can print its `RESULT` line, that trail is what tells the
   driver what the run achieved, so a killed worker still competes in the
   portfolio with the code it actually left on disk.
5. **Initialisation is itself deadline-bounded.** If the budget runs out while
   the `M` initial balls are still being marked, the greedy loop stops where it
   is and keeps `M' < M` codewords with exactly consistent counters. A short
   code that exists beats a full one that does not, and the driver's finisher
   tops it back up to `M` words. `symsearch` does the same with its orbits.

### 1.4 The wall clock is a first-class part of the problem

The first judge self-test of this entry scored **-10^6 on `K_8(9,4)`**: five out
of five runs hit the judge's `TIME_S+60` kill at 120 s. Reproduced with
`bash -x`, a 60 s budget had become a 99 s run. Two causes, both about a loaded
machine rather than about covering codes:

* the cost model under-predicted. `freesearch`'s initialisation is 27.5 s of
  ball marking into 262 MB on an idle box; at load 400+ on 64 cores it ran past
  the entire budget and was killed having printed nothing at all.
* the finisher was in the critical path. `covcount`'s recount allocates `q^n`
  bytes and marks `M*|B_R|` cells — 3.7 s idle, tens of seconds loaded — and it
  ran *after* the deadline, not before it.

The fixes are structural, not tuning:

* **`covcount --fast`.** De-duplicate, top up to `M` distinct words, write. No
  coverage array, no ball marking, instant at any size. Used to publish whenever
  `q^n > 5*10^6`; the full recount is only used where it is genuinely cheap.
* **A legal answer is published before any search starts.** The first thing
  `run_entry.sh` does is write `M` distinct words to `OUTFILE`. It covers almost
  nothing, but from that line onwards there is no execution path — overrun,
  kill, OOM, a solver that never got past `malloc` — that leaves the judge an
  unparseable file. Every later write replaces it atomically.
* **Every child runs under `timeout` computed from the remaining budget**, and
  the publishing step is itself bounded, with a plain `cp` as its fallback.

Re-measured after the fix, `TIME_S=60`, same loaded machine:
`K_8(9,4)` 57.8 s, `K_6(8,4)` 58.5 s, `K_6(6,3)` 57.8 s, `K_3(11,4)` 0.5 s.

---

## 2. The symmetry-then-free schedule, measured

The risk in this lane is over-constraint, so the schedule was measured rather
than assumed. `K_6(8,4)` at `M = 169`, 55 s symmetric + 55 s free, 2 threads,
3 seeds each, uncovered words at the end:

| orbits | 28 | 27 | 26 | 24 | 20 | 0 (pure free) |
|---|---|---|---|---|---|---|
| invariant words | 168 | 162 | 156 | 144 | 120 | 0 |
| free words | 1 | 7 | 13 | 25 | 49 | 169 |
| **uncovered** | 9, 11, 6 | 0, 0, 6 | **0, 0, 0** | **0, 0, 0** | 3, 3, 2 | 4, 3, 8 |

Neither endpoint solves it. Both interiors solve it every time.

The failure at the symmetric end is not bad luck, it is a theorem. Six long
runs of the fully invariant search (150 s, 6 seeds) all converged to 1-2
uncovered quotient points, i.e. 6-12 uncovered words forming whole orbits — and
one spare codeword cannot repair an orbit:

```
sum_j #{ i : c_i = w_i + j }  =  n     for any c and any q-orbit {w + j*1}
```

so `#agree(c, w+j) >= n-R` can hold for at most `n / (n-R)` values of `j`
(2 of 6 for `K_6(8,4)`). Structure has to be given back, not topped up.

So on cells small enough to restart cheaply the driver runs a **portfolio along
the symmetry axis** — orbits `= floor(M/q)`, then progressively fewer, plus two
fully free runs — one thread per worker, instead of betting on one point of it.

On a big cell the axis collapses. A single `freesearch` initialisation eats 28 s
of a 112 s budget on `K_8(9,4)` — more than the whole budget when the machine is
busy — so there is no room for a free remainder. Running the two engines side by
side was measurably worse than running the invariant search alone: `freesearch`'s
262 MB counter array saturates memory bandwidth for both, and in the first judge
run it produced no output whatsoever. So on a big cell `symsearch` gets all six
threads and `freesearch` is kept only as the fallback for a cell the quotient
cannot represent (`M < q`, or `q^(n-1)` too large to hold).

`covcount` arbitrates: it re-reads each candidate file, de-duplicates, tops it
up to `M` distinct words (placing the extras on still-uncovered words, which is
free and occasionally helps), and recomputes coverage by direct ball marking.
No solver's own incremental counter is ever the evidence.

---

## 3. Measured results

Scored under **RULES.md amendment 1** (2026-08-20): a full cover is `+1000`, a
valid partial is `-floor(uncovered * 10^6 / q^n) - 1`, invalid output is
`-2*10^6`.

The box is a shared 64-core machine also running the production `cov_sweep`
campaign, so the load average moved between 50 and 630 over the session. Both
tables below come from the same regime (load ~55-130); the effect of the load is
quantified in §3.5, and it is large.

### 3.1 Judge self-test

`python3 scripts/arena_judge.py structure --seeds 5 --time 60` (`bench/judge60.log`):

```
structure    K6(6,3)@41:  solved 2/5 median_score     -65   median_wall 57.8s
structure    K6(8,4)@169: solved 5/5 median_score    1000   median_wall 57.9s
structure    K8(9,4)@940: solved 0/5 median_score  -52548   median_wall 57.8s
structure    K3(11,4)@81: solved 5/5 median_score    1000   median_wall  0.9s
{"structure": {"total_solved": 12, "sum_median": -50613}}
```

`K_6(6,3)`'s median `-65` is 3 uncovered words out of 46656; two of the five
seeds reached zero, so the median seed is not one of them.

### 3.2 Baseline for comparison

`cov/search/covsearch --threads 6 -t 60`, run with the same seeds, killed at
`TIME_S+60` exactly as the judge kills it, raw log in `bench/base60.log`
(`bench/base_scored.sh`) and remapped to amendment-1 scores by
`bench/rescore.py`:

```
baseline     K6(6,3)@41:  solved 1/5 median_score   -172   median_wall  60.0s
baseline     K6(8,4)@169: solved 0/5 median_score     -5   median_wall  63.0s
baseline     K8(9,4)@940: solved 0/5 median_score -66092   median_wall 108.3s
baseline     K3(11,4)@81: solved 0/5 median_score  -2349   median_wall  60.1s
{"baseline": {"total_solved": 1, "sum_median": -68618}}
```

Raw uncovered counts behind those, per seed:

```
K_6(6,3)@41    0,  8, 14,  8, 10
K_6(8,4)@169   4,  9,  7,  5,  7
K_8(9,4)@940   8938474, KILLED (empty file), 8836903, 8870595, 8834755
                                       walls 104.1 / 120.0 / 119.5 / 108.3 / 96.2 s
K_3(11,4)@81   452, 411, 408, 416, 452
```

### 3.3 Side by side

| cell | baseline solved | baseline median | `structure` solved | `structure` median |
|---|---|---|---|---|
| `K_6(6,3)@41` | 1/5 | -172 | **2/5** | **-65** |
| `K_6(8,4)@169` | 0/5 | -5 | **5/5** | **1000** |
| `K_8(9,4)@940` | 0/5 | -66 092 | 0/5 | **-52 548** (-20%) |
| `K_3(11,4)@81` | 0/5 | -2 349 | **5/5** | **1000** |
| **total solved** | **1 / 20** | | **12 / 20** | |
| **sum of medians** | **-68 618** | | **-50 613** | |

Reading the four cells:

* `K_3(11,4)@81` — the declared known-hard control — goes from *never* solved
  (the baseline plateaus at 408-452 uncovered, matching `cov/NOTES.md` §8.4's
  410) to **5/5, median wall 0.9 s**, because the incumbent is a linear code
  and `lincov` searches linear codes rather than 81-subsets of 177147 words.
* `K_6(8,4)@169` goes from 0/5 to **5/5**. Neither a fully invariant nor a fully
  free search does this (§2); the portfolio along the symmetry axis does.
* `K_6(6,3)@41` goes from 1/5 to 2/5 (3/5 in the run of §3.5).
* `K_8(9,4)@940` is not solvable by anyone (§4). The invariant search leaves
  ~20% fewer words uncovered than the baseline's best valid run, in 57.8 s
  median wall against the baseline's 108 s — the baseline's median run *exceeds
  the budget it was given by 80%*, and one of its five runs was killed outright
  and scored `-2*10^6`.

### 3.4 Verified against `cov/verify_cov.py`

The judge's own scorer is an independent numpy dilation and every solved run
above passed it. Spot checks straight out of `run_entry.sh` through the project
verifier, both exhaustive methods plus the exact radius:

```
$ bash run_entry.sh 6 8 4 169 1000 90 out.txt
$ python3 cov/verify_cov.py out.txt -q 6 -n 8 -R 4 --method both --radius
codewords : M=169
method pure : uncovered=0   (1.63s)
method numpy: uncovered=0   (0.15s)
covering radius (exact): 4
RESULT    : VERIFIED  -- K_6(8,4) <= 169

$ bash run_entry.sh 3 11 4 81 2000 30 out4.txt
$ python3 cov/verify_cov.py out4.txt -q 3 -n 11 -R 4 --method both
RESULT    : VERIFIED  -- K_3(11,4) <= 81
```

Partial answers were checked too, and the number the entry believes is the
number the verifier finds: a 30 s `K_6(6,3)` run reporting 8 uncovered verifies
at 8, a `K_6(8,4)` run reporting 1 verifies at 1, and the invariant search's
168-word `K_6(8,4)` code reporting 12 verifies at 12. That matters more than the
zeros do: it is the check that the incremental counters in the quotient are not
quietly drifting.

### 3.5 The load caveat, stated plainly

Both engines are throughput-bound and the machine is shared, so the same
self-test gives materially different answers at different times of day. Three
complete runs of `--seeds 5 --time 60` on the same code:

| load average | `K_6(6,3)@41` | `K_6(8,4)@169` | `K_8(9,4)@940` uncovered | `K_3(11,4)@81` |
|---|---|---|---|---|
| ~430-630 | 0/5 (-14 unc) | 0/5 (-21 unc) | 6 196 506 | **5/5** |
| ~50-130 | **3/5** | **5/5** | 6 202 499 | **5/5** |
| ~90-110 | **2/5** | **5/5** | ~7 052 000 | **5/5** |

The two `q = 6` cells sit right on the edge of solvable at this budget, so a
factor of ~5 in effective CPU decides them. `K_3(11,4)` is unaffected in every
regime — it is settled algebraically before the clock matters at all — and
`K_8(9,4)` moves by only a few percent because it is nowhere near a threshold.
Any comparison across entries needs to be run in one window, as §3.1 and §3.2
were. Raw logs for all three runs are kept: `bench/judge60.log` (the table in
§3.1), `bench/judge60_run2.log` (high load, original scoring),
`bench/judge60_oldscoring.log` (low load, original scoring).

---

## 4. Honest limits, and one thing the judge should know

**`K_8(9,4)` at `M = 940` is not a solvable cell, by anybody.** Kéri's upper
bound is 2944 and that is exactly the best direct sum over every split
(`K_8(4,2)*K_8(5,2) = 23*128 = 2944`); the sphere-covering bound is
`8^9 / |B_4| = 134217728 / 333166 = 403` and Kéri's lower bound is 409. At
`M = 940` the redundancy is `940*333166/8^9 = 2.33`, which is where the good
`q=6` cells sit — but no construction in the literature comes within a factor
of three of 940 here. So B3 is a "how small can you make `uncovered`" cell, and
that is what this entry optimises there.

**A scoring artifact, since fixed.** Under the rules as originally frozen, every
honest answer on B3 scored about `-6*10^6` while an *invalid* answer scored
`-10^6`: writing garbage on that cell beat writing the best code you could find,
by a factor of six. This entry never exploited it — it always publishes the best
valid partial it has, per the rules' own instruction ("write the best partial
code's words anyway") — and the measured baseline shows the artifact was live,
not hypothetical: one of five `covsearch` runs on B3 was killed by the judge's
`TIME_S+60` timeout and collected the *better* score of the two, `-10^6`, **for
crashing**, against `-8.8*10^6` for its four runs that worked.

RULES.md amendment 1 (2026-08-20, credit to competitor "strategy" for flagging
it) fixes exactly this: partial scores are now
`-floor(uncovered * 10^6 / q^n) - 1`, in roughly `[-10^6, -1]`, and invalid
output is `-2*10^6`, so a valid partial always beats a crash. Nothing in this
entry changes as a result; the design decision that guarantees a legal `OUTFILE`
on every path (§1.4) is simply now unambiguously the right one.

**Things tried that did not pay off.**

* *Direct-sum seeding for B3.* Build `K_8(4,2) <= 23` and `K_8(5,2) <= 128` and
  take a 940-word subset of their 2944-word direct sum. The full code covers
  with multiplicity ~7.3, so a 32% subset leaves roughly `0.68^7.3 = 5.5%` of
  the space uncovered, i.e. ~7.4 M — worse than the 4.6% the invariant search
  reaches. Not implemented.
* *Full invariance on B2.* Six 150 s runs of `symsearch` at 28 orbits (168 of
  the 169 words invariant), 6 seeds, all converged to 1-2 uncovered quotient
  points and stuck there. The identity above says why: one spare word can never
  repair an orbit. This is the concrete form of the "over-constraining makes
  good codes unreachable" risk, and the portfolio is the answer to it.
* *Tabu / noise inside the quotient search.* Not re-tested; `cov/NOTES.md`
  measured both to hurt in the full space and the quotient move is the same
  move, so the shipped defaults follow the baseline (`--tabu 0`, `--noise 0`,
  all candidates evaluated).

**Not done.**

* *Bigger groups.* The natural next group is the coordinate rotation
  `x -> shift(x) + c*1`, of order `n*q` (72 on `K_8(9,4)` against the 8 used
  here). The blocker is not the search, it is the quotient map: for the
  diagonal translation the quotient is `Z_q^{n-1}` with an `O(1)` index, while
  a rotation needs a canonicalisation and hence a `q^n`-sized orbit-id array,
  which throws away the memory saving that made the quotient worth having.
  Orbits also stop being regular (constant words are fixed points), so moves
  need a stabiliser-preserving variant.
* *Translation subgroups of order `L | q`.* `M = 41` and `M = 169` are both
  `1 mod 2` and `1 mod 3`, so the smaller subgroups do not fit those cells any
  better than the full one; the quotient becomes `Z_q^{n-1} x Z_{q/L}` and the
  ball decomposition has to be redone per `L`. Left out.
* *`lincov` for non-prime-power `q`.* `Z_6 = Z_2 x Z_3` gives subgroups of
  `Z_6^n` via CRT, but the Hamming metric does not factor through the CRT
  (agreement in `Z_6` is agreement in *both* components), so the syndrome
  collapse does not apply. `q = 6` cells get nothing from this engine.
* *Descent from a larger code (remove-and-repair).* `freesearch` inherits it
  from the baseline and the driver never uses it: at these `M` the seeds we can
  build are all *smaller* than `M`, not larger.

## 5. Files

| file | what |
|---|---|
| `run_entry.sh` | the arena contract; phase schedule, portfolio, arbitration |
| `lincov.c` | coset-of-a-linear-code search in syndrome space (GF(q) built at runtime for any prime power) |
| `symsearch.c` | translation-invariant search in the quotient `Z_q^{n-1}` |
| `freesearch.c` | `covsearch.c` + deadline/flush/distinctness fixes |
| `covcount.c` | independent recount, de-duplication, top-up to `M` distinct words |
| `bench/` | benchmark scripts and raw logs |
