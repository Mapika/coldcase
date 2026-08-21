# coldcase

Reopening cold cases in combinatorial record tables with a GH200.

Many entries in the standard record tables of combinatorics were set decades
ago on single CPUs and have not moved since. This project attacks them with
modern hardware and algorithm engineering, under a strict verification-first
doctrine: **no record is claimed without an independent, dependency-free
verifier passing**, and lower bounds are only claimed with exact rational
certificates.

**Paper:** [New upper bounds on covering codes K_q(n,R) for alphabets of size
six and seven](https://arxiv.org/abs/2608.19872) (arXiv:2608.19872, cs.IT).

## Results so far (August 2026)

- **Covering codes, upper bounds** — first improvements to any K_q(n,R)
  upper bound with q ≥ 5 since Kéri's 2011 tables: 18+ cells improved so far
  across q = 5…15, including K6(9,4): 692→660, K7(9,4): 1843→1480 (and
  descending), K6(10,4): 2952→2784 (and descending). Codes + verifier in
  `cov/results/` and `cov/verify_cov.py`; the arXiv ancillary files contain a
  standalone verifier.
- **Covering codes, lower bounds** — 7 certified improvements for q ≥ 6, the
  first since 2011, via semidefinite-programming duality with exact rational
  PSD certificates (`cov/lb/`, checker `cov/lb/certify.py`, pure stdlib).
- **Degree-diameter graphs** — 5 records on the Comellas table via Cayley
  graphs of metacyclic groups (`dd/`), live and credited on the table.
- **Constant-weight codes** — honestly closed: 0 improvements from 137
  attacked cells; the engine and negative result are kept for the record
  (`src/`, `verify/`).

## Method highlights

- Focused local search (WalkSAT lineage) on the uncovered set, with a
  shared-sphere incremental evaluation that amortizes the C(n,R)(q−1)^R
  neighbourhood enumeration across all coordinate moves (6–10× speedup).
- Structure-aware attacks: unions of cosets of linear codes (`lincov`,
  collapsing q^n to q^(n−k) syndromes), translation-quotient search
  (`symsearch`), direct-sum seeding.
- A frozen-rules multi-agent "solver arena" (`cov/arena/`) used to develop and
  select engine components on held-out cells; the winners are merged into the
  production portfolio engine (`cov/engine/`).
- Float SDP search → rational rounding → exact LDL^T certification for lower
  bounds; weak duality means the certificate alone proves the bound.

## Layout
- `cov/` — covering-code track: search engines, campaign drivers, results,
  lower-bound certificates, arena
- `dd/` — degree-diameter track (constructions + rebuild/verify script)
- `src/`, `verify/`, `scripts/` — constant-weight-code engine and verifiers
- `paper/` — LaTeX sources of the arXiv note
- `results/` — logs and per-track summaries

## Verification

Every claimed code in `cov/results/` can be re-checked with:

    python3 cov/verify_cov.py cov/results/K6_9_4_M660.txt

Every lower-bound certificate with:

    python3 cov/lb/certify.py cov/lb/certs_all/cert_q6_n10_R4.json

Both are pure-Python, dependency-free (the covering verifier uses numpy for
speed; an exact stdlib fallback is included in the paper's ancillary files).

## Credits

Idea and methodological guidance: Mark Marosi (BME MIT, Budapest).
The overwhelming majority of the design, code, search campaigns, and analysis
were done autonomously by Claude (Anthropic), running on a dedicated GH200.
