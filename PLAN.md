# Longshots — GPU record hunting on frozen mathematical record tables

Goal: new best-known constructions (exactly verifiable records) + an open-source,
verification-first GPU search engine + an arXiv paper.

## Why this direction
- Records in these tables are binary-novel: either you beat the table or you don't.
- The 2025–26 LLM-evolution wave (AlphaEvolve etc.) swept the celebrity geometry
  problems but runs unoptimized Python inner loops; the global-NLP crowd (SCIP/Xpress)
  owns small dense instances. Neither reaches the large frozen backfield.
- One GH200 (97 GB HBM, ~800 fp16 TFLOP/s, 64 ARM cores) brings ~1000× the search
  throughput of the hardware that set most standing records (1990s–2010s CPUs).

## Fronts
1. **Constant-weight codes A(n,d,w)** — Brouwer's live table. Proven yield: bit-swap
   tabu on CPU got 24 (Rosin) and ~126 (Echols, Aug 2026) records. We scale the same
   primitive to millions of GPU chains with seeded initialization.
2. **Unrestricted binary codes A(n,d)** — same kernel, colder table (frozen 2019).
3. **Covering codes K_q(n,R)** — Kéri tables frozen 2011; q≥6 corner never searched
   (upper bounds are trivial direct sums). Ball-marking kernel. Zero competition.
4. **Spherical codes / Tammes dims 4–5** — batched Riesz-energy descent + minimax
   polish; automated submission at spherical-codes.org.

## Architecture
- `data/` raw mirrored tables + parsed JSON bounds DBs with provenance.
- `src/` CUDA C++ kernels (compiled .so, ctypes) + Python orchestration.
- `verify/` independent exact verifiers (pure Python, no deps) — every claimed record
  gets a certificate file + verifier run before it is believed.
- `results/` found constructions, certificates, campaign logs.
- Campaign manager selects instances by (gap, record age, search-space fit), seeds
  from incumbent constructions, schedules GPU runs, re-checks the live table before
  any claim.

## Rules
- A record claim requires: explicit construction file + independent verifier pass +
  re-fetch of the live table to confirm the incumbent + provenance notes.
- Never submit without user visibility; batch submissions with clear attribution.
- Paper: ≥6 pages, cs.IT primary; ship constructions as ancillary files + verifier.
