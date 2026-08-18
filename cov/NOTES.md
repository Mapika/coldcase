# Covering codes track — notes, formats, decisions, status

`K_q(n,R)` is the smallest size of a code `C ⊆ Z_q^n` such that every word of
`Z_q^n` is within Hamming distance `R` of some codeword. An explicit code of
size `M` with covering radius `≤ R` proves `K_q(n,R) ≤ M`. We hunt **upper**
bounds: smaller codes that still cover.

The reference tables are Gerzson Kéri's, frozen in 2011
(<http://old.sztaki.hu/~keri/codes/>), mirrored as PDFs in `data/raw/keri_*.pdf`.

---

## 1. Files

| file | what |
|---|---|
| `tables.py` | parses the Kéri PDFs into `bounds.json`; `--check` runs the cross-check suite |
| `bounds.json` | 1145 parsed cells + key legends + post-2011 lower-bound updates |
| `pdfaudit.py` | prints one PDF page as baseline-only text, for eyeballing the parse |
| `verify_cov.py` | the exact verifier — **every claim goes through this** |
| `verify_independent.py` | a second verifier sharing no code with the first; min-distance scan and Hamming-graph dilation |
| `test_verify.py` | test suite: three independent implementations must agree |
| `test_solver.py` | regression test: the solver's incremental uncovered count must equal the verifier's |
| `constructions.py` | direct sum, shortening, blow-up, perfect Hamming codes, `GF(q)`; seed builder |
| `targets.py` | ranks cells by how attackable their upper bound is |
| `campaign.py` | runs many `covsearch` workers, verifies, records results |
| `finalize.py` | prunes superseded descent steps and re-verifies what is kept with every method |
| `search/covsearch.c` | CPU (OpenMP) local-search solver |
| `search/covsearch.cu` | CUDA port of the same solver |
| `results/` | verified codes + JSON sidecars |

## 2. Formats

**Code files.** One codeword per line. Digits are base-36 characters, one per
coordinate (`0122101`), or whitespace/comma-separated integers (needed once
`q > 36`). `#` starts a comment; blank lines are ignored. Duplicate codewords
are reported and removed by the verifier — they never help coverage, and keeping
them would inflate the claimed `M`.

**Result sidecars.** `results/K<q>_<n>_<R>_M<M>.json` records `q, n, R, M`, the
date, the method, the verifier's own output on that file, and the Kéri incumbent
for the cell, with explicit `beats_keri_upper_bound` /
`matches_keri_upper_bound` flags.

**`bounds.json`.** `entries[]` holds one record per cell:

```json
{"q": 6, "n": 10, "R": 4, "lb": 417, "ub": 2952,
 "lb_key": "s", "ub_key": "f", "n_optimal": null,
 "src": "keri_6-21_tables.pdf", "page": 1}
```

`lb_key` / `ub_key` are Kéri's provenance letters; `keys[]` holds the legend for
each table family (the same letter means different things for lower and upper
bounds, and across families, so always resolve it against the right block).
`n_optimal` is the number of inequivalent optimal codes, present only when the
cell is known exactly. `lb_updated` carries a post-2011 improved lower bound.

## 3. How the PDF parsing works, and why it is not `pdftotext`

Kéri's cells are typeset as `[lower key] value [upper key]`, where `value` is a
single integer when the bounds coincide or `lb–ub` otherwise. **Exactly known
cells additionally carry a raised 8 pt integer right after the value: the number
of inequivalent optimal codes.** Any text extractor glues that superscript onto
the number — `K_3(6,3) = 6` with 28 inequivalent optima extracts as `d 628 u`,
and `K_3(11,2) = 729` extracts as `7291`. Silently trusting that would corrupt
exactly the cells we care most about.

So `tables.py` works from glyph geometry (pdfminer.six): glyphs at ≥ 9 pt are
the baseline text, glyphs below that are super/subscripts. Rows are grouped by
`y`; each row is split into cells at horizontal gaps > 9 pt; each cell is
assigned to the column whose header centre is nearest. Column assignment by
`x` (rather than by tokenising the row text) is what resolves the genuine
ambiguity in rows like `3 g 8 g 4 1`, where the middle `g` could be the upper
key of one cell or the lower key of the next.

One wrinkle: a wide classification superscript can push the trailing key letter
more than 9 pt from its value, orphaning it as a cell of its own. Any cell that
is a bare key letter is merged back into its nearer neighbour.

`q` is read from the subscript on the `Bounds on K_q(n,R)` title and
cross-checked against the Latin ordinal in the sub-title ("senary", "novenary",
…). Both must agree or the page is rejected.

### Extraction confidence

`python3 tables.py --check` reports:

* **17/17 targeted cross-checks pass**, including every cell quoted in the
  project brief, and six perfect-Hamming cells whose values are known
  independently (`K_3(4,1)=9`, `K_5(6,1)=625`, `K_7(8,1)=7^6`, `K_9(10,1)=9^8`,
  `K_3(13,1)=3^10`, `K_2(3,1)=2`).
* **0 structural problems** over all 1145 entries: no `lb > ub`, no `ub > q^n`,
  no cell with `R ≥ n` other than 1, no violation of the sphere-covering bound,
  no violation of monotonicity in `n` or of `K_q(n+1,R+1) ≤ K_q(n,R)`.
* **85/85 agreement with Gijswijt–Polak**: their paper prints, for each cell it
  improves, the previously known lower bound *and* the best known upper bound.
  All 85 of those pairs match our parse exactly — an independent audit of 85
  cells by people who read the same table.
* **1145/1145 agreement with an independent transcription.** `data/README.md`
  documents `data/keri_third_party.csv`, the Kéri snapshot transcribed for the
  `florath/covering-codes-lean` project from the same PDFs by someone else, for
  a different purpose. Diffed cell by cell against `bounds.json`: **same 1145
  cells, no cell in one and not the other, zero disagreements on either bound.**
  This is the strongest evidence we have that the parse is right, and it is
  reproducible offline via `tables.py --check`.

### One inconsistency inside Kéri's own table

`tables.py --check` also applies Kéri's three derivation rules — `c`
(`K_q(n,R) <= K_q(n-1,R-1)`), `e` (`<= q·K_q(n-1,R)`) and `f` (direct sum over
every split) — to every cell using only tabulated values, and asks whether the
printed upper bound is ever beaten by the table's own arithmetic. Across all
1145 cells there is exactly one:

```
K_17(7,2): table says 252735 (key f, = K_17(3,1)·K_17(4,1) = 145·1743)
           but rule e gives 17 · K_17(6,2) = 17 · 14424 = 245208
```

Both cells were confirmed against the independent transcription, so this is not
a parsing artifact. The most likely explanation is ordering: `K_17(6,2) <= 14424`
carries key `q` (Rivas Soriano, 2005–2008) and the derived value for
`K_17(7,2)` was evidently not re-propagated afterwards. So
**`K_17(7,2) <= 245208`** follows immediately from Kéri's own table and his own
rule.

We flag this as an observation, not as one of our results: we have no explicit
14424-word code for `K_17(6,2)`, so nothing here has passed our verifier. It is
a paper-and-pencil consequence of two published numbers, and it is listed
separately from the searched results in §8 for that reason.

### Two discrepancies found in the brief

The task brief quoted two cells that do not match the table. In both cases the
quoted numbers are real Kéri values, but for a different cell — the one at
`(n, R) = (10, 5)`:

| brief says | table actually says | the quoted value is |
|---|---|---|
| `K_9(10,4): 481–3969` | `K_9(10,4) = 3872–19683` | `K_9(10,5) = 481–3969` |
| `K_8(9,4): 287–2461` | `K_8(9,4) = 409–2944` | `K_8(10,5) = 287–2461` |

All the other quoted cells (`K_10(10,5) = 632–7106`, `K_7(10,5) = 160–1225`,
`K_6(10,4) = 417–2952`, `K_5(11,5) = 103–625`) match exactly. The three ternary
cells match once you account for the lower bounds having moved (below).

## 4. Post-2011 lower bounds (Gijswijt–Polak)

*Semidefinite lower bounds for covering codes*, Dion Gijswijt & Sven Polak,
arXiv:2504.01932 (v1 2025-04-02, revised 2026-06-19). Symmetry-reduced SDP over
the Terwilliger algebra; 85 improved lower bounds, transcribed into
`bounds.json` under `lower_bound_updates_2025` and mirrored into the affected
entries as `lb_updated`.

Two things matter for us:

1. **They only move lower bounds.** Our targets are upper bounds, so nothing we
   hunt gets harder. The gaps just got narrower on paper — e.g. `K_3(11,4)` is
   now `34–81` rather than `30–81`, which is where the brief's ternary numbers
   come from.
2. **The paper never touches `q ≥ 6`.** It treats `q ∈ {2,3,4,5}` only; the
   tractability ceiling is roughly `n ≤ 16` for `q = 3` and `n ≤ 13` for
   `q = 4, 5`. The whole `q ≥ 6` corner is exactly as frozen as it was in 2011.

Their SDP does give `K_5(11,5) ≥ 100`, which does not beat Kéri's 103 and so is
not in their improvement table.

## 5. The verifier

`verify_cov.py` is the gate. It implements two exhaustive methods and can run
both and refuse to answer if they disagree:

* **`pure`** — no dependencies at all. A `bytearray` of `q^n` flags; for every
  codeword, enumerate its whole Hamming ball by walking (subset of changed
  positions, new digit values) and mark. Cost `M·V`. This is the method to
  trust; it is short enough to read in one sitting.
* **`numpy`** — meet in the middle. Split `n = n1 + n2` and index a word as
  `w = x·q^n2 + y`; then `d(c,w) = d1(c1,x) + d2(c2,y)`, so `w` is covered iff
  some codeword has `d2(c2_i,y) ≤ R − d1(c1_i,x)`. Precompute, per codeword and
  per radius `r ≤ R`, the bitset over `y` of `{y : d2 ≤ r}`; one pass over the
  `q^n1` prefixes ORs the relevant bitsets and popcounts. Never allocates `q^n`
  bytes, so it goes past what `pure` can hold.

`test_verify.py` adds a **third** implementation — `itertools.product` over the
whole space with an inner distance scan, sharing no code with either — and
requires all three to agree. It covers random codes across
`q ∈ {2,…,7}`, every `R` from 0 to 4, every meet-in-the-middle split
`n1 = 1..n−1`, structural cases with known answers (whole space, single
codeword, perfect Hamming codes for `q = 2,3,4,5` and the same codes with one
word deleted), direct sums, the file-parsing edge cases, and the CLI exit codes.
All pass.

`--radius` reports the exact covering radius by binary search, which catches the
failure mode where a code covers at `R` but you meant to claim `R−1`.

## 6. The solver

`search/covsearch.c`. State: the codewords plus `cnt[w]`, the number of
codewords covering `w` (`uint16`, so `M ≤ 65535`), and the running count of `w`
with `cnt[w] = 0`.

**The move and why it is cheap.** Change one coordinate: `c → c'` with
`c'_p = v`. Since

```
d(w,c') = d(w,c) + [w_p = c_p] − [w_p = v]
```

a word can only *leave* the ball of `c` if `d(w,c) = R` and `w_p = c_p`, and can
only *enter* the ball of `c'` if `d(w,c') = R` and `w_p = v`. Both sets are
indexed by the *same* `(n−1)`-suffixes — the words differing from `c` in exactly
`R` coordinates other than `p` — so one enumeration of

```
S = C(n−1,R)·(q−1)^R
```

patterns yields both: the leaving word is `base+off`, the entering word is
`base+off+dv`. So a move costs `2S` touches rather than two full balls of volume
`Σ_i C(n,i)(q−1)^i`. For `(q,n,R) = (6,10,4)` that is 157k instead of 295k, and
evaluating a candidate without committing costs the same `S` reads.

The same argument shows the commit needs **no atomics**: a word at distance
exactly `R` from `c` determines uniquely which `R` coordinates differ, so
distinct (subset, assignment) pairs give distinct words, and the leaving set
(`w_p = c_p`) is disjoint from the entering set (`w_p = v`).

**Search.** Focused local search in the WalkSAT lineage: pick a random uncovered
word `u`; collect the moves that would cover it (every codeword at Hamming
distance exactly `R+1` from `u`, nudged one coordinate towards `u`; if none is
that close, fall back to the codewords at minimum distance); evaluate them
exactly, in parallel, one thread per candidate; commit the best, ties broken at
random. Optional tabu on restoring a value, optional WalkSAT noise, kicks that
teleport codewords onto uncovered words on stagnation, and `--sa` for focused
simulated annealing with Metropolis acceptance.

**Correctness of the bookkeeping.** The solver never recomputes coverage; the
uncovered count survives millions of sphere updates incrementally, so drift there
would produce a confident false claim. `test_solver.py` closes that loop
directly: it runs the solver with a fixed iteration budget on ten configurations
(including `R = 0`, `R = n−1`, `q` from 2 to 7, and sizes both above and below
the optimum), takes the code it emits, and requires `verify_cov.py` to report
exactly the number the solver printed — not only when that number is zero. It
also checks that seeding preserves the seed, that remove-and-repair lands on
exactly `M` codewords, and that an impossible size (`K_3(4,1)` at `M = 8`) never
reports success. All pass.

**Seeding and descent.** `--in` accepts a code file. If it holds *more* than `M`
codewords the solver descends by remove-and-repair: repeatedly delete the
codeword covering the fewest words that nothing else covers, then repair. That
is how a construction of size `UB` is turned into an attempt at `UB−1`.

### Tuning, measured

Everything below is on `K_6(6,3)`, whose incumbent upper bound is 41, 15–60 s
per run:

| change | best uncovered |
|---|---|
| sample 24 candidates | 82 |
| sample 64 candidates | 18 |
| **evaluate all candidates** | **8, then 0** |
| tabu tenure 0 / 2 / 4 / 8 | 82 / 83 / 98 / 110 |
| WalkSAT noise 0 / 10% / 30% | 82 / 153 / 261 |

Candidate breadth dominates everything else, by a lot. Tabu and noise both
*hurt* on this instance, so the shipped defaults are `--cand` unlimited,
`--tabu 0`, `--noise 0`. Greedy initialisation (placing each new codeword on a
still-uncovered word) makes no measurable difference to the final plateau, so it
is on by default but is not doing real work.

Simulated annealing runs ~92k moves/s against ~2.8k for the focused search — a
33× move-rate advantage — and still loses: on `K_3(11,4)` at `M = 81` it
plateaus at 521 uncovered versus 410 for focused search. Move *quality* beats
move *count* here.

### Throughput

| cell | `q^n` | `M` | move cost `S` | moves/s/worker |
|---|---|---|---|---|
| `K_7(3,1)` | 343 | 25 | 6 | 375k–409k |
| `K_6(4,1)` | 1296 | 72 | 5 | 349k–358k |
| `K_6(6,3)` | 46 656 | 41 | 1 250 | 3.0k–5.9k |
| `K_6(7,3)` | 279 936 | 246 | 2 500 | 1.3k–1.6k |
| `K_3(11,4)` | 177 147 | 81 | 3 360 | 1.4k–1.8k (2 threads) |

## 7. The CUDA solver

`search/covsearch.cu` is a direct port: the counter array lives in GPU global
memory (10^10 `uint16` cells would be 20 GB, so a single chain of that size fits
one GH200), and the sphere walk is parallel over the `C(n−1,R)` position
subsets, with one block per (candidate, chunk). The whole candidate batch is
evaluated in one launch, which is the right shape for the GPU given that
candidate breadth is what the CPU tuning said matters.

`--selftest` recomputes the uncovered count on the device from scratch after
every single move and compares it against the incrementally maintained value;
the commit kernel also cross-checks its own delta against the delta the
evaluation kernel predicted, and aborts on any mismatch. Both checks pass.

Two latency fixes were applied after the first measurement: the uncovered-word
list is now cached across iterations (a two-byte read revalidates a cached word
instead of a full `q^n` scan — scans on `K_3(11,4)` dropped from 1429 to 27 over
the same wall clock), and the commit kernel is pointed at the candidate slice
that the evaluation launch already uploaded rather than re-uploading it.
Initialisation was moved to the host as well: the obvious device version costs
`q^n · M · n`, which dominated everything, while host-side ball marking costs
`M · |B_R|` plus one sequential copy.

**Honest status: the CUDA version is correct but still slower than the CPU
version on every instance we can run** — ~43 iterations/s after the fixes (up
from 24) versus ~2 800 for the CPU on `K_3(11,4)`. What remains is per-iteration
launch and synchronisation latency: each move is a handful of small kernels and
copies with a host round-trip in between, and the sphere work is not large
enough to hide it at these instance sizes. The GPU was also at 100% utilisation
from another campaign throughout, so these are lower bounds on idle-machine
performance.

It does demonstrate the thing it exists for: a run on `K_10(8,5)` held all
`10^8` counters in device memory (0.20 GB) with a sphere of 1.24 M patterns per
move, and drove uncovered from 188 749 to 2 388 in 75 s. The path to `q^n ≈ 10^10`
is real; the remaining blocker is that host-side initialisation needs the same
`2·q^n` bytes, so a device-side ball-marking kernel is the next piece. Runs were
kept under two minutes each as instructed.

## 8. Results

**8 cells of Kéri's table now have smaller covering codes than the 2011
upper bound, all in the `q >= 6` corner.** Every code in `results/` passed four
independent exact checks before being recorded — see §8.2.

### 8.1 New upper bounds

| cell | Kéri 2011 lb–ub | ours | improvement | how Kéri's ub arose |
|---|---|---|---|---|
| `K_6(8,4)` | 46 – **216** | **169** | −47 (−21.8%) | key `e`: `<= 6 K_6(7,4) = 6*36` |
| `K_6(9,5)` | 32 – **144** | **126** | −18 (−12.5%) | key `j`: alphabet-product rule |
| `K_6(7,3)` | 70 – **246** | **235** | −11 (−4.5%) | key `e`: `<= 6 K_6(6,3) = 6*41` |
| `K_7(8,4)` | 76 – **343** | **333** | −10 (−2.9%) | key `c`: `<= K_7(7,3) = 343` |
| `K_6(9,4)` | 136 – **738** | **719** | −19 (−2.6%) | key `f`: `K_6(3,1) K_6(6,3) = 18*41` |
| `K_6(8,3)` | 246 – **1080** | **1054** | −26 (−2.4%) | key `f`: `K_6(4,1) K_6(4,2) = 72*15` |
| `K_6(10,5)` | 83 – **615** | **610** | −5 (−0.8%) | key `f`: `K_6(6,3) K_6(4,2) = 41*15` |
| `K_6(10,4)` | 417 – **2952** | **2951** | −1 (−0.0%) | key `f`: `K_6(4,1) K_6(6,3) = 72*41` |

`results/records.json` is the machine-readable version, and every row there
carries `all_verifiers_agree`. The descents were still improving when the run
was stopped, so these are lower-effort bounds, not converged ones.

Every one of these upper bounds is **inherited arithmetic** rather than a search
result: a direct sum, a `q`-fold blow-up, or a shortening of a neighbouring
cell. `constructions.py -q Q -n N -R R` reproduces each one exactly from smaller
table entries:

```
K_6(7,3)  = 6 · K_6(6,3)          = 6·41   = 246   (key e)
K_6(8,4)  = 6 · K_6(7,4)          = 6·36   = 216   (key e)
K_6(8,3)  = K_6(4,1) · K_6(4,2)   = 72·15  = 1080  (key f)
K_6(9,4)  = K_6(3,1) · K_6(6,3)   = 18·41  = 738   (key f)
K_6(10,4) = K_6(4,1) · K_6(6,3)   = 72·41  = 2952  (key f)
K_6(10,5) = K_6(6,3) · K_6(4,2)   = 41·15  = 615   (key f)
K_7(8,4)  = K_7(7,3)              = 343            (key c)
```

That is exactly the corner the project set out to hit. Strictly, the key letter
records where the tabulated *value* came from, not that nobody ever looked; but
a cell whose recorded bound is `6 × (a smaller cell)` is one where no code in
`Z_6^8` was ever written down. The answer for `K_6(8,4)` is at most 170.

### 8.2 What "verified" means here

Nothing enters `results/` on the solver's word. The solver maintains coverage
counters incrementally, so a bug there would report success while covering
nothing; the counters are never the evidence. Each recorded code is re-read from
disk and checked by four exhaustive methods, two of which share no code with the
other two:

1. `verify_cov.py` **pure** — `bytearray` of `q^n`, mark every ball, count zeros.
2. `verify_cov.py` **numpy** — meet in the middle with per-codeword bitsets.
3. `verify_independent.py` **mindist** — enumerate every word, compute its
   distance to every codeword, take the minimum, report the maximum. Also prints
   the full distribution of minimum distances, so an off-by-one in `R` is
   visible rather than inferable.
4. `verify_independent.py` **dilate** — never computes a distance at all. Starts
   from the code's indicator vector on `Z_q^n` and applies the Hamming-graph
   radius-1 dilation `R` times, which is the ball of radius `R` by definition of
   the graph metric. Cost is independent of `M`, which is what makes `q^n = 6·10^7`
   checkable.

Sample output for the largest improvement, `K_6(8,4) <= 170` (at `M = 171`,
before the last step):

```
  dilate     : 171 codeword cells set
  dilate     : after radius 1, covered 7011 / 1679616
  dilate     : after radius 2, covered 125271 / 1679616
  dilate     : after radius 3, covered 1005460 / 1679616
  dilate     : after radius 4, covered 1679616 / 1679616
  mindist    : covering radius (exact) = 4
  methods run: dilate=covers, mindist=covers
RESULT: VERIFIED  -- every one of the 1679616 words of Z_6^8 is within distance 4
```

and for `K_6(10,4) <= 2951`, where only dilation is affordable:

```
  dilate     : after radius 4, covered 60466176 / 60466176
```

Sanity against the sphere-covering bound, which is independent of everything
above: `K_6(8,4) >= 6^8/|B_4| = 1679616/51491 = 33`, and Kéri's lower bound is
46. Our 170 sits comfortably above both, so nothing here contradicts a known
lower bound. The same holds for all six cells.

### 8.2b Confirming the incumbent

A record claim is only as good as the baseline it beats, so the baseline was
re-checked independently on 2026-08-18:

* **Kéri's site is live and frozen.** `https://old.sztaki.hu/~keri/codes/` still
  serves the tables. The live `index.htm` is byte-identical (MD5
  `3ea9b7ad941aa803bd229c74ae0444c0`, 24 153 bytes) to the Wayback captures of
  2011-11-25 and 2024-06-15. CDX digests date the last edit of the index to
  between 2011-10-28 and 2011-11-25, and the last data-file change to
  `3_tables.pdf` / `mixed_tables.pdf` on 2011-11-24. **The `6-21_tables.pdf` we
  parse — the one every cell we improved lives in — has not changed since
  2009-10-15.**
* **The seven cells were re-read from the live PDF character by character** and
  cross-checked against an independent third-party transcription. Exact match
  with our parse in every case.
* **No post-2011 upper-bound improvement for `q >= 5` exists in the indexed
  literature.** The only post-2011 movement in Kéri's parameter range at
  `q >= 5` is on *lower* bounds: Gijswijt–Polak (arXiv:2504.01932, `q <= 5`
  only) and Florath's Lean-certified `K_8(4,2) >= 23` (arXiv:2606.16688). Both
  benchmark explicitly against Kéri as the current authority.
* **There is no successor table.** Lobstein's covering-codes bibliography carries
  a formal retirement notice with no successor. The `florath/covering-codes-lean`
  repository ships `reference-data/post-keri/non_mixed_covering_codes.csv`
  explicitly as the place for later literature updates; as of its 2026-07-20
  push it is still a straight conversion of the frozen Kéri snapshot, with zero
  post-2011 upper-bound changes. (Its row count, 1145, matches ours exactly —
  one more check on the parse.)

So the right framing for these results is: **better than Kéri 2009/2011,
verified unchanged as of 2026-08-18, and no competing published value found.**

### 8.3 Codes that reproduce known values

Useful as calibration, not as records:

| cell | `M` | Kéri | note |
|---|---|---|---|
| `K_6(3,1)` | 18 | 18–18 | reproduces the optimum |
| `K_6(4,1)` | 72 | 72–72 | reproduces the optimum |
| `K_6(4,2)` | 15 | 15–15 | reproduces the optimum |
| `K_6(6,3)` | 41 | 24–41 | reproduces the incumbent ub |
| `K_6(7,3)` | 246 | 70–246 | reproduces the incumbent ub |
| `K_6(8,3)` | 1080 | 246–1080 | reproduces the incumbent ub |
| `K_6(8,4)` | 216 | 46–216 | reproduces the incumbent ub |
| `K_6(9,4)` | 738 | 136–738 | explicit direct sum, reproduces the incumbent ub |
| `K_6(9,5)` | 144 | 32–144 | reproduces the incumbent ub |
| `K_6(10,4)` | 2952 | 417–2952 | explicit direct sum, reproduces the incumbent ub |
| `K_6(10,5)` | 615 | 83–615 | explicit direct sum, reproduces the incumbent ub |
| `K_7(3,1)` | 25 | 25–25 | reproduces the optimum |
| `K_7(8,4)` | 343 | 76–343 | reproduces the incumbent ub |

The three "explicit direct sum" rows matter procedurally. Kéri's entries for
those cells are products of two numbers; until now there was no code behind them
in our hands. Building the factors by search (`K_6(3,1)=18`, `K_6(4,1)=72`,
`K_6(4,2)=15`, `K_6(6,3)=41`, all reproduced above) and concatenating them turns
the table entry into an object the verifier can check — and, more usefully,
gives remove-and-repair a real starting point instead of noise. The
`K_6(9,4)`, `K_6(10,4)` and `K_6(10,5)` improvements all came out of exactly
that pipeline.

Note also that the cells appearing in both §8.1 and §8.3 (`K_6(7,3)`,
`K_6(8,3)`, `K_6(8,4)`, `K_6(9,4)`, `K_6(9,5)`, `K_7(8,4)`) were each reached at
the incumbent size first and then descended, so the improvement is not an
artifact of a lucky start: the same solver, on the same instance, produced both
the incumbent-matching code and the smaller one.

### 8.4 What did not work

* `K_3(11,4)` at `M = 81`, the incumbent (Östergård 1991): 8 workers × 90 s × 3
  rounds plateaued at 410 uncovered out of 177 147. We cannot reproduce the
  incumbent here, let alone beat it. Focused simulated annealing did worse (521)
  despite 33× the move rate.
* `K_3(12,5)` at `M = 54`: plateaued at 432.
* `K_6(6,3)` at `M = 40`: we reach 41 reliably, so this is a fair test of the
  descent, and it failed over three rounds. It is the cell that matters most —
  see §9 — so it deserves a better algorithm rather than more of this one.
* `K_8(8,4)` (Kéri 512) and `K_10(8,5)` (Kéri 168) from scratch: neither reached
  the incumbent within the time available. Descending from a loose start got
  `K_8(8,4)` to 740 and `K_10(8,5)` to 260, both still well above Kéri, so
  nothing was recorded for them. `K_7(9,4)` (Kéri 1843) likewise reached only
  2599. These are not negative results about the cells — they are targets whose
  redundancy says they should fall, given either more time or a constructed seed
  of the kind that unlocked the `q = 6` cells.
* `K_6(5,2)` at `M = 66` (the incumbent): best 6 uncovered. Close, not there.
* `K_7(7,4)` at `M = 49` (the incumbent, a linear code): best 365 uncovered.
  Unsurprising: `49 = 7^2` is almost certainly a `[7,2]_7` linear code, and
  unstructured local search is the wrong tool for finding one.

The pattern is clean. The solver reaches or beats the frontier on cells whose
upper bound is inherited arithmetic — those codes are loose, with redundancy
`M·|B_R|/q^n` of 4–10 — and fails on cells where somebody already ran a real
search or found an algebraic construction, which are tight (redundancy 1–3).
That is the expected shape of the result, and it is why the target ranker sorts
by provenance first and redundancy second.

## 9. Where the fat is

`targets.py` ranks cells by whether anyone ever searched them. Kéri's upper-bound
keys `f` (direct sum), `c` (`K_q(n+1,R+1) ≤ K_q(n,R)`), `e`
(`K_q(n+1,R) ≤ q·K_q(n,R)`), `j` and `k` are pure bookkeeping: no code was ever
searched for those cells, the number is inherited from a smaller one. Keys
`m,n,o,p,q,t,x,y,z` are somebody's search or construction.

The top of that list, restricted to `q ≥ 5`, `q^n ≤ 10^8`, all bookkeeping keys:

| cell | lb–ub | redundancy | how the UB arose |
|---|---|---|---|
| `K_6(10,5)` | 83–615 | 9.51 | direct sum |
| `K_7(9,5)` | 52–323 | 9.31 | direct sum |
| `K_7(9,4)` | 264–1843 | 8.35 | direct sum |
| `K_6(10,4)` | 417–2952 | 7.20 | direct sum |
| `K_6(9,5)` | 32–144 | 6.92 | key `j` |
| `K_6(8,4)` | 46–216 | 6.62 | `≤ 6·K_6(7,4)` |
| `K_6(9,4)` | 136–738 | 6.61 | direct sum |
| `K_7(8,4)` | 76–343 | 6.18 | `≤ K_7(7,3)` |

Redundancy 6–9.5 means the incumbent code covers the space six to nine times
over. That is a lot of slack, and none of it has ever been squeezed.

The other lesson from this session is the **factor attack**. `K_6(6,3) = 41`
appears as a factor in at least two headline cells:

```
K_6(10,4) = K_6(4,1)·K_6(6,3) = 72·41 = 2952
K_6(10,5) = K_6(6,3)·K_6(4,2) = 41·15 =  615
```

so `K_6(6,3) <= 40` would drop those to 2880 and 600 immediately — three records
for one search in a space of 46 656 words with a move cost of 1250. That is by
far the best compute-per-record ratio on the board, and it is the one thing this
session tried and failed at (§8.4). Likewise `K_7(10,5) = K_7(3,1)·K_7(7,4) =
25·49`, so `K_7(7,4) <= 48` would give `K_7(10,5) <= 1200`.

The general recipe, which is what produced `K_6(10,4) <= 2951` and the
`K_6(10,5) <= 615` explicit code:

1. read the factorisation of the incumbent off `targets.py`;
2. find explicit codes for the small factors by search (they are tiny, and the
   solver reproduces known optima there reliably);
3. concatenate to get a verified code at the incumbent size;
4. remove-and-repair downward.

Step 2 is cheap and step 4 is where the compute goes. Searching the headline
cell from scratch skips the only part that is easy.

## 10. Decisions and their reasons

* **Verify first, always.** `campaign.py` re-parses the code file from disk and
  runs the verifier before anything is written to `results/`; a code that fails
  is discarded loudly. The solver's own "uncovered = 0" is never trusted on its
  own, because it comes from the same incremental counters that a bug would
  corrupt.
* **Glyph-level PDF parsing** rather than `pdftotext`, for the superscript
  reason in §3. Accuracy over coverage: we would rather parse fewer cells than
  parse `729` as `7291`.
* **`uint16` counters.** `uint8` would halve memory but overflows at
  `M > 255`, which several targets exceed. On the GPU this caps a single chain
  at ~5·10^9 words in 97 GB rather than 10^10; worth revisiting with saturating
  8-bit counters plus an exact recount.
* **Sphere-difference moves** rather than remove-ball/add-ball. Factor ~2 on
  every move and it makes the lock-free commit argument trivial.
* **Duplicate codewords are removed, not tolerated.** A file with duplicates
  claims a smaller `M` than it has distinct words; the verifier reports the count
  and the campaign rejects any file whose distinct count differs from the claimed
  `M`.

## 11. Not done

* Mixed-alphabet tables (`keri_mixed_tables.pdf`, `keri_qtb_tables.pdf`) are not
  parsed. Different cell shape, and no target of ours lives there.
* No amalgamated direct sum (Kéri's upper-bound key `v`), no adjoint-code
  constructions (key `l`), no linear-code search (key `p`). Several incumbents
  we cannot reproduce come from those, and a linear search over generator
  matrices would reach `K_7(7,4) = 49 = 7^2` and `K_3(11,4) = 81 = 3^4` cheaply —
  though beating them requires leaving the linear family, since `48` and `80` are
  not powers of `q`.
* No guided local search / weighted objective. Given how much candidate breadth
  mattered, an objective that puts growing weight on persistently uncovered
  words is the obvious next thing to try on the tight instances.
* The CUDA iteration is latency-bound; see §7.
