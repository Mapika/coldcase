# Covering-code solver arena — rules (FROZEN)

Goal: find the fastest solver for "given (q,n,R,M), find M codewords covering
Z_q^n" — the inner loop of the record hunt.

## Entry contract
Each competitor works ONLY in cov/arena/<name>/ and must provide:

    cov/arena/<name>/run_entry.sh  Q N R M SEED TIME_S OUTFILE

which runs its solver for at most TIME_S seconds (wall clock) using at most
6 CPU threads and 25 GB RAM, and writes the best code found to OUTFILE (one
codeword per line, digits 0..q-1; the file may have fewer than M words if no
full solution was found — write the best partial code's words anyway).
Deterministic per SEED where feasible. Exit 0.

## Scoring (frozen)
The judge runs scripts/arena_judge.py:
- Benchmark cells (public, for development):
    B1: q=6 n=6 R=3 M=41
    B2: q=6 n=8 R=4 M=169
    B3: q=8 n=9 R=4 M=940
    B4: q=3 n=11 R=4 M=81      (known-hard control)
- Held-out cells (revealed only at judging) of similar shapes.
- 15 seeds per cell, TIME_S=120. Score per run: 1000 if OUTFILE is a valid
  full cover with M words (verified by cov/verify_cov.py), else
  -floor(uncovered * 10^6 / q^n) - 1 (normalized uncovered fraction,
  range about [-10^6, -1]). Ties broken by median wall time to full cover.
- Any invalid output (bad weights, wrong q/n, unparseable) scores -2*10^6 for
  that run. The verifier is the law.
- AMENDMENT 1 (2026-08-20, pre-judging): partial scores normalized by q^n and
  invalid moved to -2e6, because the original absolute-uncovered penalty made
  garbage output outscore valid partials on large cells (credit: competitor
  "strategy" for flagging).

## Conduct
- Do not modify anything outside your own directory.
- Do not touch cov/search/covsearch (production, in use) or other entries.
- nice -n 15 everything; ≤6 threads; ≤25 GB.
- You may read cov/NOTES.md, cov/search/covsearch.c, cov/opt/ for ideas.
- Commit your work with git (your directory only).
