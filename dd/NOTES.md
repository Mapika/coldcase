# Degree/diameter track — implementation notes

Target: the live table at
<https://web.mat.upc.edu/francesc.comellas/delta-d/table_degree_diameter.html>
(largest known order `N(Δ,D)` of a graph with maximum degree Δ and diameter D).
Mirrored raw HTML in `data/table_raw.html`, parsed to `table_current.json`
(value + Moore bound + inferred provenance per cell).

## Construction space

`G = Z_m ⋊_a Z_n`, metacyclic, order `N = m·n`:

```
element (i,j),  index = j*m + i,   i ∈ [0,m), j ∈ [0,n)
(i1,j1)·(i2,j2) = (i1 + a^{j1}·i2  mod m,  j1+j2  mod n),     a^n ≡ 1 (mod m)
identity (0,0);   (i,j)^{-1} = (−a^{−j}·i, −j)
n = 1  → cyclic Z_m        (circulant graphs)
a = 1  → abelian Z_m × Z_n
```

`Cay(G,S)` with `S = S^{-1}`, `e ∉ S`, `|S| = Δ`, is a simple undirected Δ-regular
graph on N vertices.

### Why one ball growth is enough (vertex-transitivity)

For `g ∈ G` the left translation `λ_g : x ↦ gx` is a graph automorphism:
`{x, xs}` maps to `{gx, gxs}`, still an edge, because adjacency is defined by *right*
multiplication by S while λ acts on the left. λ_g carries `e` to `g`, so the
automorphism group is transitive and **every vertex has the same eccentricity**.
Hence `diam(Cay(G,S)) = ecc(e) = min{ D : |B_D| = N }`, where
`B_0 = {e}`, `B_k = B_{k-1} ∪ B_{k-1}·S`.

This is used to make the inner loop a single ball growth from the identity.
It is checked empirically in `crosscheck.py` (all-vertex eccentricities computed in
pure Python on 40 random instances — all constant), and *never* relied on for a
reported result: `verify_dd.py` re-derives the diameter with a BFS from **every**
vertex of the emitted edge list.

### Why non-abelian groups are required

An abelian Cayley graph with k generator pairs is a quotient of the Cayley graph of
`Z^k`, so `|B_D| ≤ #{L1 ball of radius D in Z^k} = Σ_i 2^i C(k,i) C(D,i)`.
`targets.py` prints this "abelian ceiling". Examples:

| cell | record | Moore | abelian ceiling |
|------|--------|-------|-----------------|
| (14,3) | 979 | 2563 | 575 |
| (16,3) | 1610 | 3857 | 833 |
| (10,4) | 2485 | 8201 | 681 |

So circulants and abelian Cayley graphs cannot even reach the current records in
these cells — every hit must come from `a ≠ 1`.

## The fast kernel

A subset of `G` is a bit-matrix of `n` rows × `m` bits (`W = ⌈m/64⌉` words per row).
Right multiplication by `s = (i2,j2)` sends row `j` to row `(j+j2) mod n`, cyclically
rotated by `a^j·i2 mod m`. So applying one generator costs `n·O(W) ≈ N/64` word
operations instead of `N` scalar edge traversals — a 16–64× speedup over a
pointer-chasing BFS, and it is branch-free and cache-friendly.

```
per evaluation:  D · Δ · n · O(W)  word ops
```

Measured throughput (64 threads, GH200 ARM cores):

| instance | evals/s |
|----------|---------|
| N ≈ 20, Δ=3, D=3     | 2.8 × 10^8 |
| N ≈ 1000, Δ=14, D=3  | 3.0 × 10^7 |
| N = 2485, Δ=10, D=4  | 1.7 × 10^7 |

Restricting the group enumeration to non-abelian actions of full order
(`--nonab --faithful`) is worth much more than raw speed: it cut the time to
re-find the (14,3) record from 23 s to 2.1 s, an 11× reduction in wall time on top
of the same throughput.

The kernel was cross-checked against an independent pure-Python BFS on 800 random
(m, n, a, S, D) instances: **0 mismatches**.

## The search

Minimise `f(S) = N − |B_D(S)|` over inverse-closed Δ-element connection sets.
`f = 0` ⟺ a (Δ,D) graph of order N.

* items = inverse pairs `{g, g^{-1}}` (weight 2) and involutions (weight 1);
  all splits `2k + t = Δ` are tried
* move = replace one item with a random unused one
* accept if the score strictly improves, or sideways with prob 1/2
* score = `f·(N+1) + (N − |B_{D-1}|)` (ball size at D−1 as a tie-breaker)
* **iterated local search**: after `--stall` non-improving moves, restore the
  incumbent and kick it with 1–3 random item replacements; after `--kicks` kicks,
  abandon the restart and resample the group

The plain hill-climb stalls (it reached only `f = 2` on (10,4) at N=2485);
adding the ILS kick found `f = 0` there in 26 s.

## The CUDA engine

`src/dd_gpu.cu` runs the *whole* iterated local search on the device: one CUDA block
owns one search chain, holds its connection set and its two ball bitsets in shared
memory, and only talks to the host to report a hit. There are no per-iteration host
round trips.

The ball step is parallelised by **destination** word. Indexing by destination row,

```
nxt[r'] = cur[r'] | OR over s in S of rot( cur[src_s(r')], sh_s(r') )
src_s(r') = (r' - j + n) mod n      sh_s(r') = a^{src} * i mod m
```

`src_s` is a permutation of the rows, so every destination word `(r', w)` is an
independent reduction over `|S|` generators of at most four source words: no atomics,
no write conflicts, and exactly one `__syncthreads()` per BFS level. Threads are laid
out over the flat `R·W` destination words.

One optimisation matters a lot: `sh_s(r')` depends only on `(s, r')`, so computing it
inside the word loop repeats a modular multiply `W` times. Hoisting it into a shared
`shf[|S|·R]` table (when that fits) raised large-`N` throughput by 46 %.

Odd degrees are handled by giving the last connection-set slot an involution drawn
from a host-built table of the involutions of `Z_m ⋊_a Z_n` (elements `(i,j)` with
`2j ≡ 0 (mod n)` and `i(1+a^j) ≡ 0 (mod m)`).

Measured, against the 64-core CPU engine:

| instance | CPU (64 cores) | GPU | speedup |
|---|---|---|---|
| Δ=14, D=3, N≈1000  | 30 Mevals/s   | 40 Mevals/s   | 1.3× |
| Δ=9,  D=4, N=1640  | —             | 28 Mevals/s   | — |
| Δ=14, D=5, N≈79000 | 0.29 Mevals/s | 1.72 Mevals/s | 5.9× |

So the GPU is worth it exactly where the CPU's bitset stops fitting in L1 — the large
D=5 cells — and the two run concurrently on different cells. `crosscheck_gpu.py`
requires CUDA, C++ and pure Python to agree on `|B_k|`; 320 random instances, no
disagreement.

## Target selection: which cells this engine can actually win

`classify_cells.py` scrapes the per-diameter description pages (`desc_g2.html` …)
and records the construction behind each standing entry. That splits the table in
two:

* **Records that are themselves Cayley graphs of semidirect products** — our family
  provably reaches them, so a systematic sweep should beat the casual search that
  set them. With `D ≤ 5` these are
  `(16,2)=200, (6,3)=111, (7,3)=168, (8,3)=253, (14,3)=979, (5,4)=212, (6,4)=390,
  (7,4)=672, (8,4)=1100, (9,4)=1640, (10,4)=2485, (6,5)=1404, (7,5)=2756`.
  (14,3) fell first, and it is exactly this kind of cell: the incumbent was Rishabh
  Rajiv's July 2026 Cayley graph of `Z_89 ⋊ Z_11`.
* **Records from compound graphs / generalized-quadrangle-polarity quotients with
  vertex additions** — `(9,3), (10,3), (11,3), (12,3), (13,3), (15,3), (16,3),
  (11,4), (12,4), (13,4), (14,4), (15,4), (16,4)`. These beat what a Cayley graph
  of that order can do, so a pure Cayley search is structurally behind there. Our
  measured shortfalls agree: at the record order the best connection set still
  leaves `f = 77` uncovered vertices for (12,3), `f = 106` for (13,3), `f = 120`
  for (16,3) — while at (14,3) and (10,4) the engine reaches `f = 0`.

This classification is what campaign 3 is built around.

## Verification protocol

Nothing is reported without all three of:

1. `emit_graph.py` — re-derives the group law and the whole edge list in pure
   Python from `(m, n, a, S)`, re-checking `a^n ≡ 1`, inverse-closure of S,
   `e ∉ S`, and the realised degree. Writes `results/*.edges` + `*.json`.
2. `verify_dd.py` — standalone, stdlib-only. Reads only the edge list: rejects
   loops/multi-edges, checks `N`, `max deg ≤ Δ`, connectivity, and runs BFS from
   **every** vertex to get the exact diameter.
3. re-fetch of the live table to confirm the incumbent value before any claim.

## Files

| file | role |
|------|------|
| `parse_table.py` | fetch-parsed table → `table_current.json` (value, Moore bound, provenance) |
| `targets.py` | rank candidate orders `N > record` by metacyclic richness; abelian ceiling |
| `src/dd_search.cpp` | OpenMP sweep engine (bitset ball growth + ILS); `--eval` for exact single-set evaluation |
| `crosscheck.py` | C++ kernel vs pure-Python BFS + vertex-transitivity check |
| `emit_graph.py` | hit → edge list + JSON sidecar, independently re-derived |
| `verify_dd.py` | standalone exact verifier |
| `run_cell.py` | per-cell campaign driver (probe / attack), logs to `results/campaign_log.jsonl` |

## Campaign log (chronological)

| when | what | outcome |
|------|------|---------|
| campaign 1 | flat sweep (14,3) N=980..1100, 900 s | **N=1026** (also 1005, 999) |
| campaign 3 | CPU adaptive scan→focus over the cells with Cayley-graph records | no hits; left every soft cell at `f = 1…4` |
| GPU focus | (14,5) window 60391..60520, 1500 s | **N=60452** and **N=60450**, 7 independent hits |
| near-miss | (14,3) N=1032 at `f=1`, 900 s of concentrated ILS | **N=1032**, improving our own 1026 |

The shape of the result matters more than any single number: a broad scan gets a
soft cell to `f = 1…4` within minutes, and the last few uncovered vertices are what a
long concentrated run buys. Both records here came from that second step, not from
the scan. `nearmiss.py` automates it — it mines the campaign logs for orders with a
small residual `f` and re-runs them with a much larger budget.

Division of labour that worked: the GPU scans breadth (it is ~6x the CPU once the
bitset leaves L1, i.e. exactly the large D=5 cells), while the CPU pounds on the
specific near-miss orders the scans expose.

Budgeting note: `focus.py` costs `scan-secs + topk × focus-secs` per cell, so the
per-cell budget has to be chosen against the number of cells — 12 × 240 s of focus
is 48 min for one cell, which is the wrong allocation when 20 cells are untried.
