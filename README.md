# Coldcase

GPU-accelerated hunting of best-known constructions on open combinatorial
record tables — starting with binary constant-weight codes A(n,d,w).

The record tables of combinatorics are full of entries set decades ago with a
single CPU. Recent work (Rosin 2026, Echols 2026) showed that plain seeded
bit-swap tabu search on a desktop still moves dozens of records in Brouwer's
constant-weight-code table. Coldcase (né Longshots) scales that primitive to a GH200: each
CUDA thread block runs one tabu chain with an incrementally maintained pairwise
distance matrix; thousands of chains run concurrently with seeded and
population-based restarts.

**Verification-first**: every candidate record must pass an independent,
dependency-free verifier (`verify/`) before it is recorded, and is re-checked
against the live table before any claim is made public.

## Layout
- `src/` — CUDA kernel (`cwc_tabu.cu`) + ctypes driver (`cwc.py`)
- `verify/` — standalone exact verifiers (pure Python)
- `scripts/` — table parsers, campaign runners (`hunt_cwc.py`, `spray_cwc.py`)
- `data/` — mirrored source tables + parsed bounds databases (with provenance)
- `results/` — found constructions with JSON certificates
- `dd/` — degree-diameter graph track
- `PLAN.md` — project plan and rules

## Status
Active hunt, August 2026. See `results/`.
