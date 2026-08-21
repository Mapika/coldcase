# Lower-bound track for covering codes — design, validation, status

`K_q(n,R)` is the smallest size of a code `C ⊆ Z_q^n` whose radius-`R` Hamming
balls cover `Z_q^n`. The rest of `cov/` hunts **upper** bounds (explicit small
codes). This directory hunts **lower** bounds, and every number it produces is
backed by an *exact rational certificate* that a standalone, dependency-free
checker re-verifies from scratch.

The target is the nine cells where we hold new upper bounds
(`cov/results/final_records.json`), and more generally the whole `q ≥ 6` corner,
which the 2025 Gijswijt–Polak semidefinite work explicitly does not touch
(`cov/NOTES.md` §4: "the paper never touches `q ≥ 6`").

---

## 0. Headline

**Seven new certified lower bounds in the `q ≥ 6` corner** — a region the 2025
semidefinite literature does not cover at all (Gijswijt–Polak treat
`q ∈ {2,3,4,5}`), so the incumbents are Kéri's 2011 table.

| cell | new LB | previous best | Kéri key | sphere bound | known UB | certificate |
|---|---|---|---|---|---|---|
| `K_6(9,3)`  | **924**  | 921  | `y` | 881  | 4752 | `certs_all/cert_q6_n9_R3.json` |
| `K_6(10,4)` | **441**  | 417  | `s` | 411  | 2952 | `certs_all/cert_q6_n10_R4.json` |
| `K_7(8,3)`  | **471**  | 457  | `y` | 439  | 2337 | `certs_all/cert_q7_n8_R3.json` |
| `K_8(8,3)`  | **855**  | 829  | `y` | 813  | 4096 | `certs_all/cert_q8_n8_R3.json` |
| `K_8(9,4)`  | **428**  | 409  | `s` | 403  | 2944 | `certs_all/cert_q8_n9_R4.json` |
| `K_9(9,4)`  | **718**  | 703  | `s` | 690  | 5103 | `certs_all/cert_q9_n9_R4.json` |
| `K_10(9,4)` | **1145** | 1130 | `s` | 1123 | 8500 | `certs_all/cert_q10_n9_R4.json` |

`K_6(10,4) ≥ 441` (was 417, +24) is on **one of our nine record cells** — we
already hold the best known upper bound 2951 there, so that cell is now
`441 ≤ K_6(10,4) ≤ 2951` with both ends ours.

Every one of these is re-checkable in one command, e.g.

```
python3 cov/lb/certify.py cov/lb/certs_all/cert_q6_n10_R4.json
```

which rebuilds the SDP from `(q,n,R)` in exact integer arithmetic and verifies
the rational dual certificate; it prints the exact rational SDP value and
`K_6(10,4) >= 441`. Machine-readable list: `results/new_bounds.json`. The full
cell-by-cell table (including where we fall short) is `results/summary.md`.

---

## 1. Quick start

```bash
cd cov/lb

# verify the headline bound (rebuilds the SDP exactly, checks the certificate)
python3 certify.py certs_all/cert_q6_n10_R4.json

# the model is valid: push real covering codes through it in exact arithmetic
python3 certify.py --selftest

# step 1: the LP bound is exactly the sphere covering bound
python3 lp_bound.py --sweep

# re-verify every certificate and diff against cov/bounds.json
python3 report.py --certs certs_all

# reproduce Gijswijt-Polak's published values
python3 reproduce.py --min-n 5 --max-n 9 --jobs 10

# coefficient-level diff against their own Julia code (needs tools/, see .gitignore)
JULIA_DEPOT_PATH=tools/depot tools/julia-1.10.9/bin/julia gp_dump.jl 6 7 3 /tmp/d.txt
python3 xcheck.py 6 7 3 /tmp/d.txt

# solve a new cell
python3 solve_ipm.py 6 10 4 --out /tmp/c.json && python3 certify.py /tmp/c.json
```

`certify.py`, `lp_bound.py` and `report.py` need only the Python standard
library. `solve.py` / `solve_ipm.py` need numpy, scipy, and `scs` / `cvxopt`
(`pip install scs cvxopt`).

---

## 1b. Files

| file | what |
|---|---|
| `certify.py` | **the deliverable checker.** Standard library only. Rebuilds the SDP from `(q,n,R,λ,β)` in exact integer arithmetic, checks a rational dual certificate exactly, prints the certified bound. Also `--selftest`. |
| `lp_bound.py` | step 1: the fractional covering LP, its weight-class symmetry reduction, an exact rational checker, and the proof-by-computation that this LP *is* the sphere-covering bound |
| `solve.py` | model scalings + SCS (first-order) dual point + rounding/repair to an exact certificate |
| `solve_ipm.py` | same, with cvxopt's interior-point method (the workhorse) |
| `run_cells.py` | parallel driver over a list of cells |
| `reproduce.py` | reproduction gate against arXiv:2504.01932 Tables 7/8/9 |
| `report.py` | re-verifies every certificate with `certify.py` and cross-checks `cov/bounds.json` |
| `xcheck.py`, `gp_dump.jl`, `gp_funcs.jl` | coefficient-level diff against the original Gijswijt–Polak Julia code |
| `gp_julia/` | their upstream repository (github.com/CoveringCodes/Julia) |
| `certs_all/` | **the certificates** — best one per cell, 163 cells, all re-verified |
| `certs_geo/`, `certs_retry/`, `certs_hiq/`, `certs_repro/`, `certs/` | per-sweep raw output that `certs_all/` is merged from |
| `results/` | reproduction tables, sweep results, summary |
| `ref/` | the paper |

---

## 2. Step 1 — the LP bound, and why it is a dead end

The covering LP relaxation is

```
min  sum_c x_c    s.t.  sum_{c in B_R(w)} x_c >= 1  for all w,   x >= 0,
```

and weak duality says: any `y : Z_q^n → R_{≥0}` with
`sum_{w in B_R(c)} y_w ≤ 1` for every `c` certifies `K_q(n,R) ≥ sum_w y_w`.

`lp_bound.py` implements the symmetry reduction (restrict `y` to be invariant
under `Aut_0(q,n)`, the stabiliser of the zero word, whose orbits are the
weight classes), giving `n+1` inequalities in `n+1` unknowns, and an exact
rational checker `verify_rational_y`.

**The honest conclusion, and it is a theorem, not a numerical observation:**
`Aut(q,n) = S_q wr S_n` acts *transitively* on `Z_q^n` and preserves the dual
feasible set. Averaging any feasible `y` over the group gives a feasible
**constant** `y` with the same objective, and a constant `y` is feasible iff
`y ≤ 1/V` with `V = |B_R(0)|`. Hence

```
LP optimum = q^n / V      exactly.
```

The fractional covering LP is *precisely* the sphere-covering bound. No finer
symmetry class can beat it — refining the class only shrinks the feasible set.

Validation (`python3 lp_bound.py --sweep`, 144 cells `q ≤ 7, n ≤ 10`): the
exact uniform certificate always reproduces `ceil(q^n/V)`, and scipy/HiGHS on
the reduced LP agrees to numerical precision. Cross-checked against
`cov/bounds.json`: our LP certificate **never exceeds Kéri's lower bound** on
any of the 1145 cells (it must not), never exceeds any upper bound, and equals
Kéri's lower bound exactly on 83 cells — all the untagged (trivial) ones, all
12 perfect-code cells (key `h`), the Rodemich cell, and the single `q ≥ 6` cell
whose key is `a` = sphere-covering bound.

*Note on the brief:* it said Kéri's key for the sphere-covering bound is `s`.
In the `q ≥ 6` table the sphere-covering key is **`a`**; `y` is "improved
sphere-covering"; `s` is Lang–Quistorff–Schneider. (`s` does mean
Kamps–van Lint in the `q = 4,5` table.) The 8 record cells with key `m` are
Haas–Halupczok–Schlage-Puchta 2009, which are *far* stronger than
sphere-covering — see §6.

Everything better than `q^n/V` in the literature (van Wee, Zhang,
Habsieger–Plagne, Haas et al.) comes from adding *extra combinatorial
inequalities*, not from this LP. That is why the track goes straight to the
semidefinite relaxation.

---

## 3. Step 2 — the Gijswijt–Polak SDP, re-implemented exactly

### 3.1 What the relaxation is

Gijswijt–Polak (arXiv:2504.01932, Theorem 2.5) bound `K_q(n,R)^3` by an SDP in
matrices `M, M', M'', N` built from a code `C`, where
`M'_{u,v} = |Aut|^{-1}|{σ : 0,u,v ∈ σC}|`, together with Lasserre-style and
matrix-cut constraints derived from the sphere-covering inequalities
`(λ_0,…,λ_n)β`. Theorem 4.18 gives the symmetry-reduced form for `q ≥ 3`:
variables `x^{t,p}_{i,j}` indexed by orbits of triples `{0,u,v}` under
`Aut(q,n)`, `O(n^4)` of them, and `O(n^2)` psd blocks of size `O(n)` from the
block-diagonalisation of the Terwilliger algebra.

Crucially **the size of the reduced SDP depends only on `n`, not on `q`** — `q`
enters only through the coefficients. That is the opening: their `q ≤ 5`
ceiling was compute/precision, not model size.

### 3.2 Our implementation

`certify.build_model(q, n, R)` builds the entire Theorem 4.18 SDP in **exact
integer arithmetic** (Python `int`, no floats anywhere):

* orbit indexing by the canonical `S_3`-invariant key
  `(sorted(d(0,u), d(0,v), d(u,v)), t−p)`;
* objective `γ^{t,p}_{i,j}` (Theorem 4.18, times `q^n`);
* Prop. 4.11 order inequalities;
* Prop. 4.17 matrix-cut inequalities via the distribution numbers of
  Lemma 4.15;
* Prop. 4.12 psd blocks for `M'` and `M''` and Prop. 4.14 blocks for `N`, using
  the `α`/`β` block-diagonalisation coefficients in the *integral*
  normalisation of the reference Julia code (it differs from the paper's by the
  diagonal congruence `diag((q−1)^{i/2})`, which does not affect psd-ness but
  keeps every coefficient an integer);
* identically-true and duplicated linear rows are dropped — purely a
  conditioning measure, it cannot change the SDP value.

We use the same `(λ,β)` the paper uses for `q ≥ 3`: the sphere-covering
inequalities, `λ_0 = … = λ_R = 1`, `β = 1` (they use van Wee inequalities only
for `q = 2`).

### 3.3 Certificate format and the exact checker

The SDP is `min c^T x` over `x ≥ 0`, linear rows `L_k(x) = ⟨l_k,x⟩ + l_k^0 ≥ 0`
and blocks `A_b(x) = C_b + Σ_v x_v A_b^v ⪰ 0`. A certificate is a *dual* point:
rationals `y_k ≥ 0` and rational psd matrices `Y_b`, all written over one
common integer denominator `den`, such that with

```
d_v = Σ_k y_k l_k[v] + Σ_b ⟨Y_b, A_b^v⟩ ,   d_0 = Σ_k y_k l_k^0 + Σ_b ⟨Y_b, C_b⟩
```

we have `d_v ≤ c_v` for every variable. Then for every feasible `x`

```
c^T x − (−d_0) = Σ_v (c_v − d_v) x_v + Σ_k y_k L_k(x) + Σ_b ⟨Y_b, A_b(x)⟩ ≥ 0,
```

so `K_q(n,R)^3 ≥ OPT ≥ −d_0` and `K_q(n,R) ≥ ceil((−d_0)^{1/3})`, the cube root
taken by exact integer comparison `m^3 · den ≥ num`.

`certify.py` re-derives `c`, `l_k`, `A_b^v`, `C_b` itself from `(q,n,R,λ,β)`,
checks `y_k ≥ 0`, checks each `Y_b ⪰ 0` by an exact symmetric-pivot `LDL^T`
over `fractions.Fraction`, and checks `d_v ≤ c_v` in pure integer arithmetic.
Nothing produced by a floating point solver is trusted.

It also checks that the `(λ,β)` in the file really is a valid covering
inequality — `λ ≥ 0`, `β > 0`, `min_{d≤R} λ_d ≥ β`, which makes
`Σ_d λ_d A_d(u) ≥ β` follow from `Σ_{d≤R} A_d(u) ≥ 1`. Without that check a
certificate could smuggle in an unsound inequality and have arithmetically
correct nonsense verified. Tamper tests (`β := 2`, or zeroing `λ_R`) are
rejected with that reason.

### 3.4 How a dual point is found and made exactly feasible

1. Solve the (scaled) SDP numerically — cvxopt's interior-point method
   (`solve_ipm.py`), or SCS (`solve.py`).
2. Map the numerical dual back through the scalings. **All scalings are powers
   of two**, so this map is exact both in binary floating point and in
   rationals.
3. Project each `Y_b` to the psd cone (eigenvalue clip) and push it strictly
   inside by `ε·λ_max·I`, then round to integers over `den = 2^46`. Verify
   psd-ness exactly; if it fails, retry with a larger `ε`.
4. Compute `d_v` exactly and, if any `d_v > c_v`, shrink the whole dual by a
   dyadic `θ ≤ min_v c_v/d_v`. Since `c_v > 0` for every `v` when `q ≥ 3`, this
   always restores exact feasibility, at the cost of scaling the bound by `θ`.
   A good dual gives `θ = 1`.

### 3.5 Scaling — this was the whole ball game

Naive scaling makes the interior point method report a *spurious* "primal
infeasible" on every cell whose SDP value exceeds ~`5·10^6`, i.e. exactly the
interesting ones. Two separate effects:

* **`c` versus `b`.** The optimum is `|C|^3` (up to `10^13` for our cells) while
  every constraint constant is `O(1)`. In a homogeneous self-dual embedding the
  dual iterate then has to carry all `10^13`, and the method breaks down.
  Fix: normalise the objective by `(sphere covering bound)^3`.
* **The spread inside `c`.** The objective coefficients `q^n γ^{t,p}_{i,j}`
  span ~`10^10`–`10^14`. A solver's dual residual is measured relative to
  `‖c‖`, so it says nothing at all about the variables with small `c_v` — and
  those are precisely the ones whose `d_v ≤ c_v` gets violated. On `K_6(8,3)`
  that alone forced `θ = 0.449`, throwing away 55% of the bound.

The fix that works (`mode="geo"` in `solve.compute_scalings`) is
geometric-mean equilibration: scale variable `v` so that the scaled objective
coefficient and the scaled variable have the *same* magnitude profile
`sqrt(contribution_v)/M^{3/2}`, and normalise the objective by `M^3` with
`M` the sphere-covering bound. Both `c̃` and `z̃` then live in a range that is
the square root of the original spread. Effect on `K_6(8,3)`: `183 → 240`
(the true SDP value is `239.52`); on `K_6(10,4)`: `56 → 441`.

Per-block symmetric diagonal equilibration (damped Jacobi on `log2` of the
coefficient magnitudes) and per-row scaling are applied on top, also as powers
of two.

---

## 4. Validation — four independent gates

**(a) The relaxation is valid *as implemented*.** `certify.py --selftest`
feeds *actual* covering codes into the model and checks, in exact rational
arithmetic, that every linear inequality holds, every psd block is psd, and the
objective equals `|C|^3` exactly. Cases: the whole space, the ternary Hamming
`[4,2,3]` code, and randomly grown covering codes for `q = 6,7,8` with up to
200 codewords. This is the check that would catch a wrong sign or a wrong
combinatorial coefficient, and it caught a real bug in the *test harness*
(feeding a multiset instead of a set makes the psd constraints fail, correctly).

**(b) Coefficient-level identity with the original Julia code.**
`gp_dump.jl` runs Gijswijt and Polak's own functions (copied verbatim out of
`CoveringCodesQary.jl` into `gp_funcs.jl`) and dumps every coefficient;
`xcheck.py` diffs them against ours. All match, for
`(q,n,R) ∈ {(3,6,2), (4,6,3), (5,6,2), (6,5,2), (6,7,3), (7,6,3)}`:
the orbit partition, `γ'`, `α`, the `MakeDistrQary` distribution numbers and
the Lasserre `η`-expansion — tens of thousands of big-integer values each.

We could not *run* their solver: `SDPAFamily.jl` ships `sdpa_gmp` binaries for
`x86_64` only and the build fails on this `aarch64` machine. Julia 1.10.9 and
`Combinatorics` are installed under `tools/` and their generator runs fine; only
the solver is missing. The coefficient-level diff plus gate (c) is a stronger
check than re-running their pipeline would have been.

**(c) Reproduction of their published values.** `reproduce.py` runs our
pipeline on the cells of their Tables 7/8/9 and compares the *certified*
cube-root value with the published one. See `results/reproduce_n7.json` and
`results/reproduce_n89.json`. Representative agreement (published → ours):

| cell | published | our certified | their improved LB | ours |
|---|---|---|---|---|
| K_3(7,2) | 26.3830 | 26.3830 | 27 | **27** |
| K_3(8,1) | 402.9463 | 402.9253 | 403 | **403** |
| K_3(8,2) | 57.4972 | 57.4972 | 58 | **58** |
| K_3(8,3) | 15.5959 | 15.5959 | 16 | **16** |
| K_3(9,1) | 1063.9751 | 1063.5588 | 1064 | **1064** |
| K_3(9,2) | 131.8916 | 131.8909 | 132 | **132** |
| K_3(9,3) | 30.1035 | 30.1036 | 31 | **31** |
| K_4(6,2) | 32.9100 | 32.9104 | 33 | **33** |
| K_4(7,2) | 87.6300 | 87.6391 | 88 | **88** |
| K_4(8,2) | 250.8700 | 250.8717 | 251 | **251** |
| K_4(8,3) | 45.0200 | 45.0263 | 46 | **46** |
| K_4(9,2) | 774.4600 | 774.3131 | 775 | **775** |
| K_4(9,3) | 115.2800 | 115.2884 | 116 | **116** |
| K_4(9,4) | 26.6600 | 26.6698 | 27 | **27** |
| K_5(5,1) | 161.0300 | 161.0314 | 162 | **162** |
| K_5(7,2) | 235.3500 | 235.3490 | 236 | **236** |
| K_5(8,3) | 110.2400 | 110.2431 | 111 | **111** |
| K_5(9,3) | 353.3200 | 353.1899 | 354 | **354** |

**18 of the 22 improved lower bounds we attempted are reproduced exactly at the
integer level** (the four misses are `K_4(7,1)` by 1, `K_5(7,1)` by 7, and
`K_5(8,1)`/`K_5(8,2)`, all in double-precision-limited territory, see §5) —
far more than the three the brief asked for. Over all 49 published values we
ran, **44 match to the published display precision**
(`results/reproduce_geo.json`, table printed by `reproduce.py`). Small excesses
such as 87.6391 vs 87.63 are their two-decimal display truncation; our value is
always a certified lower bound on the same optimum.

**(d) Hard invariants on every certificate.** `report.py` re-verifies every
certificate with `certify.py` and cross-checks `cov/bounds.json`: a certified
bound that exceeded a known *upper* bound would mean our code is wrong. None
ever did.

---

## 5. Scaling analysis — what is and is not tractable

Model size depends only on `n`: `n = 6` → 64 variables / 48 psd blocks;
`n = 8` → 136 / 75; `n = 10` → 256 / 108; `n = 11` → 339 / 126. Our variable
counts and block-size sums match the paper's Table 4 exactly (64/50/210,
136/95/495, 339/203/1365).

Cost is dominated by (i) exact model generation, seconds to a few minutes, and
(ii) the interior point solve, seconds to ~10 minutes per cell at `n ≤ 11` on
one core. Whole sweeps of dozens of cells run in tens of minutes on 22 cores.
`q` costs nothing extra.

The real ceiling is **precision, not size**. Gijswijt–Polak used SDPA-GMP at
512 bits. We are in double precision, and after the geometric-mean
equilibration of §3.5 that is enough up to SDP values around `1.5·10^9`
(cube root ~`1150`; our largest good certificate is `K_10(9,4)` at
`1144.198`). Above that cvxopt breaks down — it reports a *spurious* "primal
infeasible" or a `math domain error` — and the best certificate we can round
out collapses far below even the sphere-covering bound.

`report.py` detects this automatically: the SDP is provably at least as tight
as the sphere-covering bound, so a certified value below `q^n/V` means the
floating point solver failed, and the row is labelled *solver failed* rather
than reported as a weak bound. In the `q ≥ 6` sweep, 31 of 64 cells are in
that state — and they are, frustratingly, mostly the `R = 2` cells where
Kéri's incumbent is only `1.00`–`1.05` times the sphere-covering bound, i.e.
where the head-room is largest. Retrying them with all four scaling modes
recovered none. This is the single most valuable thing to fix (§7).

---

## 6. Results on the `q ≥ 6` corner

See `results/summary.md` (generated by `report.py`) for the full,
certificate-by-certificate table with the Kéri incumbents.

All nine record cells, certified (values are cube roots of the exact rational
SDP bound):

| cell | our certified value | our LB | Kéri LB (key) | verdict |
|---|---|---|---|---|
| `K_6(7,3)`  | 67.1414  | 68  | 70  (`m`) | below |
| `K_6(8,3)`  | 239.5193 | 240 | 246 (`m`) | below |
| `K_6(8,4)`  | 39.5880  | 40  | 46  (`m`) | below |
| `K_6(9,4)`  | 126.5224 | 127 | 136 (`m`) | below |
| `K_6(9,5)`  | 25.6395  | 26  | 32  (`m`) | below |
| `K_6(10,4)` | 440.2854 | **441** | 417 (`s`) | **IMPROVES (+24)** |
| `K_6(10,5)` | 74.8554  | 75  | 83  (`m`) | below |
| `K_7(8,4)`  | 64.0214  | 65  | 76  (`m`) | below |
| `K_7(9,4)`  | 240.6407 | 241 | 264 (`m`) | below |

Eight of the nine carry Kéri lower bounds with key `m`
(Haas–Halupczok–Schlage-Puchta 2009), which are much stronger relative to the
sphere-covering bound (ratios 1.13–1.52) than what the triple-correlation SDP
delivers at `q = 6,7`. The SDP ratio decays steadily with `q` — for
`(n,R) = (8,4)` it is 1.43 at `q = 4`, 1.30 at `q = 5`, 1.21 at `q = 6`, 1.15
at `q = 7` — so at `q ≥ 6` the SDP loses to HHSP whenever HHSP applies. The one
record cell whose incumbent is *not* HHSP is `K_6(10,4)` (key `s`, only
`1.0146 ×` sphere-covering), and that is exactly the one we improve. The same
pattern explains all seven improvements: every one is on a cell whose incumbent
key is `y` (improved sphere-covering) or `s` (Lang–Quistorff–Schneider), never
`m`.

Beyond the nine record cells we swept every `q ≥ 6` cell with `n ≤ 11` whose
Kéri lower bound is at most `1.30 ×` the sphere-covering bound — 64 cells with
`q ∈ {6,…,10}` and 50 more with `q ∈ {11,…,21}`, `n ≤ 10`, `R ≥ 3`. Model
*size* is independent of `q`, so `q = 21` costs no more than `q = 6`; only
conditioning gets worse, and at `q ≥ 11` every cell with real head-room lands
above the double-precision ceiling (§5). Several came close —
`K_20(8,4)` certified 2615.70 against Kéri's 2760, `K_21(8,4)` 3135.14 against
3288 — close enough that a higher-precision solve is the obvious follow-up.

**Master table.** `results/summary.md` / `results/summary.json` cover all 163
cells for which we hold a verified certificate (`certs_all/`, one per cell,
best of all sweeps): **7 improve** the best known lower bound, **27 match** it
(these are the `q ≤ 5` reproduction cells — see §4c), the rest are below or hit
the precision ceiling. `report.py` re-verified every one of the 163 with
`certify.py` and confirmed that none exceeds the known upper bound.

---

## 7. Not done / next steps

* **Higher precision.** A quad-precision or MPFR interior point solver (or
  building `sdpa_gmp` for aarch64) would lift the `10^9` ceiling of §5 and open
  the large `R = 1,2` cells at `q ≥ 6`, where Kéri's bounds are barely above
  sphere-covering.
* **A smarter dual repair.** Instead of the global `θ`-shrink, re-optimising
  the linear multipliers `y_k` against the *fixed* rounded `Y_b` (a small LP)
  would recover most of what a slightly infeasible dual loses.
* **Van Wee inequalities.** For `q = 2` the paper adds a second `(λ,β)` from
  the van Wee inequalities and gains. `build_model` already takes `(λ,β)`, so
  adding a second family of `N` blocks and matrix cuts is mechanical, and no
  one has tried it for `q ≥ 6`.
* **Assuming a verified upper bound.** `M ≤ UB` may legitimately be added to
  the SDP (if every code of size `≤ UB` has `M ≥ L`, and a code of size `UB`
  exists, then `K ≥ L`). We hold verified codes for all nine record cells, so
  this is available and would both tighten the relaxation and improve
  conditioning. Not used in any certificate here — every bound in this
  directory is unconditional.
* **GPU (step 3b).** Not needed yet: the reduced SDPs are tiny (≤ 339
  variables, blocks ≤ 12) and are limited by precision, not throughput. The
  place a GH200 would pay off is a *dense first-order solver for a larger
  relaxation* (finer symmetry classes, or the 4-point/quadruple level), where
  the inner loop becomes dense GEMM. The plan sketched with the project owner —
  one-hot INT8 coordinate encoding so that `A @ B^T` yields exact
  coordinate-agreement counts on tensor cores, and Ozaki-style mantissa
  splitting to emulate fp64 GEMM through INT8 — applies there, and the
  invariant stays the same: floats only to *find* a dual point, exact rationals
  through `certify.py` to certify it. Prototyped kernels are kept small while
  the GPU is occupied by another job.
