# `dd/` — the degree/diameter track

Hunting for graphs with maximum degree Δ and diameter D that have **more vertices**
than the current best known, per the live table maintained by Francesc Comellas
(<https://web.mat.upc.edu/francesc.comellas/delta-d/table_degree_diameter.html>).

## Status

**Five verified improvements** — certificates and reproduction instructions in
[`results/CLAIMS.md`](results/CLAIMS.md):

| cell | published record | found here | margin | engine |
|------|------------------|-----------|--------|--------|
| (11,5) | 20646 | **20952** | +306 | GPU (`dd_gpu`, via `nearmiss.py`) |
| (13,5) | 42680 | **42744** | +64 | GPU (`dd_gpu`) |
| (14,5) | 60390 | **60680** | +290 | GPU (`dd_gpu`) |
| (14,3) | 979 | **1032** | +53 | CPU (`dd_search2`, via `nearmiss.py`) |
| (9,5) | 8760 | **8802** | +42 | GPU (`dd_gpu`) |

Machinery validated by re-discovering current records from scratch:

| cell | record | engine | time | group |
|------|--------|--------|------|-------|
| (14,3) | 979 | **979 matched, then 1032 found** | 2.1 s / 15 min | `Z_89 ⋊ Z_11` / `Z_344 ⋊_337 Z_3` |
| (9,4) | 1640 | 1640 matched | < 60 s | `Z_41 ⋊ Z_40` (the record's own group) |
| (10,4) | 2485 | 2485 matched | 26 s | `Z_71 ⋊_15 Z_35` |

For (14,3) the published record is Rishabh Rajiv's July 2026 Cayley graph of
`Z_89 ⋊ Z_11` — the engine independently re-found a graph in that exact family
before sweeping upward.

## Quick start

```bash
cd dd
g++ -O3 -march=native -fopenmp -o src/dd_search  src/dd_search.cpp    # metacyclic
g++ -O3 -march=native -fopenmp -o src/dd_search2 src/dd_search2.cpp   # + affine type

python3 parse_table.py                       # refresh table_current.json from the mirror
python3 targets.py --delta 14 --D 3          # candidate orders + abelian ceiling

# search one cell adaptively (scan the range, then focus on the promising orders)
python3 focus.py --delta 14 --D 3 --lo 1027 --hi 1400 \
                 --scan-secs 300 --focus-secs 240 --topk 14

# verify everything found so far and print the claim table
python3 harvest.py
```

Self-test (builds, kernel cross-checks, known-value regressions, verifier negative controls):

```bash
./selftest.sh
```

Independent verification of any edge list:

```bash
python3 verify_dd.py  results/d14_D3_N1032_m344_n3_a337_nm.edges     # exhaustive, pure stdlib
python3 verify_vt.py  results/d14_D5_N60452_m2159_n28_a123_gpu.edges # exact, near-linear
python3 verify_big.py results/d14_D5_N60452_m2159_n28_a123_gpu.edges # exhaustive, scipy
```

`verify_dd.py` runs a BFS from every vertex, which is impractical past a few
thousand vertices. `verify_vt.py` stays exact there by *proving* vertex-transitivity
from the edge list — it checks two suggested permutations really are automorphisms
and that the group they generate is transitive — after which one BFS gives the
diameter.

## How it works

See [`NOTES.md`](NOTES.md) for the construction space, the vertex-transitivity
argument that makes one ball growth sufficient, the bitset kernel, the search
strategy, and the verification protocol.

Short version: Cayley graphs of non-abelian groups, evaluated by growing the ball
around the identity as a bit-matrix (rows = coset blocks, right multiplication =
row permutation + cyclic bit rotation), driven by iterated local search over the
connection set, swept in parallel over all orders `N` above the record and all
admissible group structures of each order.

## Layout

```
data/table_raw.html      mirrored source table (+ table_recheck.html for re-confirmation)
table_current.json       parsed table: value, Moore bound, provenance per cell
src/dd_search.cpp        metacyclic engine  (Z_m ⋊_a Z_n)
src/dd_search2.cpp       generalised engine (+ (Z_q × Z_q) ⋊_A Z_n)
src/dd_gpu.cu            CUDA engine (device-side ILS, one block per chain; even degrees)
parse_table.py           table → JSON
targets.py               candidate-order ranking, Moore / abelian bounds
focus.py                 two-phase (scan → focus) per-cell driver
run_cell.py              simple per-cell probe/attack driver
campaign*.sh             batch campaigns (campaign3 = CPU, campaign_gpu = GPU)
crosscheck.py            metacyclic kernel vs pure-Python BFS + vertex-transitivity
crosscheck2.py           affine kernel vs pure-Python BFS + vertex-transitivity
crosscheck_gpu.py        CUDA kernel vs C++ engine vs pure Python
emit_graph.py            hit → edge list + JSON sidecar (independent re-derivation)
verify_dd.py             standalone stdlib-only exact verifier (all-pairs BFS)
verify_vt.py             exact near-linear verifier via a checked vertex-transitivity proof
verify_big.py            exhaustive all-pairs verifier for large graphs (scipy)
harvest.py               keep best hit per cell, emit, verify, tabulate
nearmiss.py              mine the logs for orders with small residual f and pound on them
make_submission.py       adjacency list + description + verifier run for a claim
classify_cells.py        scrape which construction each standing record uses
selftest.sh              exact self-test suite
results/                 edge lists, sidecars, CLAIMS.md, campaign_log.jsonl, raw/*.jsonl
```
