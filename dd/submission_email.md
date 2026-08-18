# DRAFT — submission to Comellas degree-diameter table (NOT SENT)

To: francesc.comellas@upc.edu
Subject: Five improved entries for the degree-diameter table (Δ,D) = (14,3), (9,5), (11,5), (13,5), (14,5)

Dear Professor Comellas,

We would like to submit five improved entries for your table of largest known
(Δ,D)-graphs. All five are Cayley graphs of metacyclic groups Z_m ⋊_a Z_n
(element (i,j), product (i1,j1)·(i2,j2) = (i1 + a^{j1}·i2 mod m, j1+j2 mod n)),
found by a large parallel search (millions of connection sets per cell,
bitset BFS evaluation, hill-climbing refinement on near-misses) on an NVIDIA
GH200 node.

| (Δ,D) | current table | new | group | connection set (element indices j*m+i) |
|-------|--------------|------|-------|----------------------------------------|
| (14,3) | 979 | **1032** | Z_344 ⋊_337 Z_3 | 564, 916, 563, 965, 449, 703, 533, 715, 431, 897, 387, 989, 536, 912 |
| (9,5) | 8760 | **8802** | Z_163 ⋊_141 Z_54 | 5394, 3462, 3227, 5781, 2350, 6535, 4859, 4179, 4429 |
| (11,5) | 20646 | **20952** | Z_873 ⋊_785 Z_24 | 17925, 4053, 17295, 4479, 11902, 9880, 16477, 5825, 42, 831, 11115 |
| (13,5) | 42680 | **42744** | Z_1781 ⋊_969 Z_24 | 4333, 39699, 34936, 8915, 24972, 17983, 24223, 20246, 33887, 9248, 42306, 2325, 22919 |
| (14,5) | 60390 | **60680** | Z_1517 ⋊_142 Z_40 | 7759, 53494, 8035, 53133, 7305, 54929, 49995, 12979, 47806, 13776, 9202, 52936, 23616, 38540 |

Each connection set is inverse-closed and generates the group; the graphs are
vertex-transitive, Δ-regular, and connected. Diameters were verified three
ways: (i) exhaustive all-pairs bit-parallel BFS over every ordered pair,
(ii) a vertex-transitivity proof from the edge list followed by a single-source
BFS, and (iii) an independent implementation of the all-pairs check
(sparse-matrix reachability on GPU). A short Python script that rebuilds any
of the graphs from the group data above and re-verifies it is attached
(rebuild_verify.py), along with the edge lists (zipped) should you prefer them.

Credit: [NAME(S) TO CONFIRM — suggested: "Mark Marosi and Claude (Anthropic)"],
August 2026. Method description for the table notes: "Cayley graphs of
metacyclic groups Z_m ⋊_a Z_n found by massively parallel generator-set search
with bitset BFS on a GPU node, plus local search refinement."

Best regards,
[SIGNATURE]

---
## Attachments to prepare before sending
- [ ] rebuild_verify.py — standalone rebuilder+verifier from (m, n, a, connection set)
- [ ] edges_5cells.zip — the five edge lists
- [ ] FINAL STEP before sending: re-fetch the live table and re-confirm the five
      incumbent values are still 979 / 8760 / 20646 / 42680 / 60390 (this table
      moved 4 of these 5 cells within the last week).
