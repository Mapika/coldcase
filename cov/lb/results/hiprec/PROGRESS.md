# High-precision lower-bound campaign — FINAL (2026-08-21)

## HEADLINE: 58 certified lower bounds beating the best previously known,
## all verified by the frozen checker `python3 cov/lb/certify.py <cert>`

`results/hiprec/status.md` / `status.json` hold the frozen-checker verdict for
every certificate (81 cells checked, 0 invalid, none exceeds a known upper
bound). 51 cells are NEW improvements on Kéri's 2011 incumbents beyond the
seven the double-precision pipeline already held, and 5 of those seven were
strengthened (K_6(9,3) 924→926, K_8(8,3) 855→856, K_8(9,4) 428→429,
K_9(9,4) 718→722, K_10(9,4) 1145→1160; K_7(8,3)=471 and K_6(10,4)=441 were
already optimal). Certificates: `cov/lb/certs_hp/` (and merged into
`cov/lb/certs_all/`, best per cell).

### All 58 frozen-verified improvements (new LB vs incumbent)

R=2 family (the double-precision "solver failed" wall, now fully broken):
| cell | new LB | was | cell | new LB | was |
|---|---|---|---|---|---|
| K_6(8,2) | 2367 | 2276 | K_13(8,2) | 199633 | 197563 |
| K_6(9,2) | 10965 | 10900 | K_14(8,2) | 307909 | 305294 |
| K_7(7,2) | 1081 | 1035 | K_15(8,2) | 461294 | 457584 |
| K_7(8,2) | 5631 | 5457 | K_16(8,2) | 673723 | 669207 |
| K_7(9,2) | 30562 | 29889 | K_17(8,2) | 962145 | 955978 |
| K_7(10,2) | 170632 | 168042 | K_18(8,2) | 1346931 | 1339650 |
| K_8(8,2) | 12033 | 11766 | K_19(8,2) | 1852296 | 1842639 |
| K_9(8,2) | 23642 | 23184 | K_20(8,2) | 2506759 | 2495614 |
| K_10(7,2) | 5824 | 5676 | K_21(8,2) | 3343629 | 3329193 |
| K_10(8,2) | 43423 | 42772 | K_11(7,2) | 9193 | 9106 |
| K_10(9,2) | 337394 | 333560 | K_13(7,2) | 20545 | 20189 |
| K_10(10,2) | 2699348 | 2676660 | K_14(7,2) | 29404 | 29273 |
| K_11(8,2) | 75448 | 74415 | K_16(7,2) | 56237 | 55600 |
| K_12(8,2) | 125156 | 123772 | K_17(7,2) | 75559 | 75429 |
| K_13(6,2) | 2233 | 2169 | K_19(7,2) | 130081 | 128972 |
| K_14(6,2) | 2964 | 2955 | K_20(7,2) | 167204 | 167165 |
| K_15(6,2) | 3862 | 3812 | K_16(6,2) | 4951 | 4848 |
| K_18(6,2) | 7804 | 7741 | K_19(6,2) | 9624 | 9479 |
| K_21(6,2) | 14201 | 14131 | | | |

R=3/R=4 families:
| cell | new LB | was | cell | new LB | was |
|---|---|---|---|---|---|
| K_7(9,3) | 2143 | 2077 | K_17(8,3) | 29652 | 29478 |
| K_9(8,3) | 1464 | 1413 | K_21(8,3) | 82595 | 82341 |
| K_9(9,3) | 8685 | 8544 | K_19(7,3) | 4282 | 4237 |
| K_9(10,3) | 54600 | 54144 | K_20(7,3) | 5215 | 5174 |
| K_6(10,3) | 3836 | 3815 | K_21(7,3) | 6294 | 6249 |
| K_12(8,3) | 5605 | 5577 | K_17(8,4) | 1464 | 1458 |
| K_13(8,3) | 8193 | 8086 | K_10(10,4) | 6915 | 6886 |
| K_10(9,4) | 1160 | 1130 | K_9(9,4) | 722 | 703 |
| K_8(9,4) | 429 | 409 | K_6(9,3) | 926 | 921 |
| K_8(8,3) | 856 | 829 | K_6(10,4) | 441 | 417 |
| K_7(8,3) | 471 | 457 | K_7(10,2) see above | | |

Validation anchor: K_5(8,2) fresh 200-bit solve certifies 861 — exactly the
value Gijswijt–Polak published from their own 512-bit SDPA-GMP runs (the one
overlap cell where our high-precision solver and theirs solve the same SDP).

### Cells where the SDP truly falls below the incumbent (certified optimum,
theta = 1, so these are now known losses at this relaxation level):
K_15(7,3) 1740<1745, K_16(7,3) 2222<2226, K_17(7,3) 2798<2806,
K_18(7,3) 3481<3492, K_14(8,3) 11666<11878, K_15(8,3) 16233<16380,
K_16(8,3) 22135<22167, K_18(8,3)? not run (pred. loss), K_20(8,3) 65151<65351,
K_18(8,4) 1817<1839, K_19(8,4) 2230<2256, K_20(8,4) 2710<2760,
K_21(8,4) 3264<3288, K_9(9,2) 165297<165310 (13 short!), K_8(10,2)
477830<478586, K_10(10,3) 110297<110867, K_17(6,2) 6256<6260,
K_20(6,2) 11746<11778, K_18(7,2) 99871<100579, K_21(7,2) 212365<213881,
K_4(9,1) 9363<9368, K_6(7,1) 7776=sphere-LP exactly (R=1 SDP adds nothing
here), K_6(8,1) 41561<41991.

## Method (what worked)

1. **SDPA-GMP native aarch64 build** (github.com/nakatamaho/sdpa-gmp,
   autotools + system GMP after `apt install libgmp-dev`; binary at
   scratchpad/sdpa-gmp/sdpa_gmp). The historical blocker was only the Julia
   wrapper's x86-64 binaries — the C++ source builds clean on this GH200.
2. **cov/lb/solve_hp.py**: exact model (certify.build_model) + §3.5
   geometric-mean power-of-two equilibration, exported in SDPA sparse format
   as EXACT decimal strings (int·2^e), solved at 200-bit GMP precision,
   epsilonStar 1e-30, 40-digit yMat output; dual parsed into exact Fractions.
3. **Round in the scaled coordinates** over 2^128, then unscale by absorbing
   the power-of-two exponents into the common denominator (rounding in
   unscaled coordinates destroys blocks whose equilibration exponents span
   ~2^80 — this bug caused every early "FAILED to round").
4. **Diagonal-shift ladder tied to the rounding grain** (start 2^-(bits-10)):
   a fixed 2^-58 shift leaked up to 6.8% into small-c_v variables and caused
   all theta<1 losses. With the fix, theta = 1.0 on every final certificate.
5. **Divergences (phase noINFO)** on ~10 cells: cured by shrinking the IPM
   initial-point radius lambdaStar to 1e2 (K_10(8,2): 1e1). run_hp.py retries
   the ladder automatically. No cell needed >200 bits or >200 iterations.
6. **LP dual-repair** (NOTES §7: fix rounded Y_b, re-optimize y_k, one scale
   per block, HiGHS + exact dyadic floor + theta mop-up) is implemented in
   solve_hp.py (lp_repair) and used as fallback; after fix (4) the plain
   rounding almost always achieves theta = 1 and beats it.

## Files
- solver/driver: cov/lb/solve_hp.py, cov/lb/run_hp.py
- verification: cov/lb/hp_check.py (subprocess-runs frozen certify.py),
  results/hiprec/status.{md,json}, final_check.log
- merge: cov/lb/merge_hp.py (certs_hp -> certs_all, best per cell; 81 copied)
- attempt log: results/hiprec/sweep.log; per-cell logs log_qQ_nN_RR.txt
- frozen master regen: report.py --certs certs_all (results/summary.md)
