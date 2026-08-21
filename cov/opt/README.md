# `cov/opt` — algorithm engineering for the covering-code local search

Optimisation and benchmarking workspace for `cov/search/covsearch.c`.
**Nothing here touches the production solver or the running sweep.**

| file | what |
|---|---|
| `covsearch_base.c` | byte-for-byte copy of `cov/search/covsearch.c` at the time of the fork; the reference implementation |
| `covsearch2.c` | the optimised solver. Every improvement is behind a flag and **defaults to off**, so `--preset base` reproduces `covsearch_base` exactly |
| `covsearch2` | the recommended engine (see below) |
| `covsearch2_prof` | same source built with `-DPROF`: per-phase `CLOCK_THREAD_CPUTIME_ID` accounting |
| `selftest.sh` | correctness gate: trajectory identity vs the original + agreement with `cov/verify_cov.py` |
| `micro.py` | fixed-iteration micro-benchmark (Table 1) |
| `ttt.py` | time-to-target benchmark, 20 seeds/config (Table 2) |
| `analyze.py` | medians, quartiles, bootstrap CIs, restart analysis; writes `table1_micro.csv`, `table2_ttt.csv` |
| `cnthist.py` | coverage-multiplicity histogram of a code file |
| `prof2.sh` | post-optimisation profile and early-exit pruning rate |
| `seeds_*.txt` | fixed starting codes so every benchmark run starts from the identical state |
| `METHODS.md` | the write-up: draft "Algorithm engineering" subsection with the measured tables |

## New command-line flags of `covsearch2`

Search-neutral (they change only how the same search is executed, and
`selftest.sh` proves the trajectory is bit-identical to the original):

```
--walk 1     peeled-innermost sphere walk
--hoist      compute the per-codeword offset table once per iteration
--st 1|2     side array min(cnt,2) as uint8 / 2-bit, used by the read path
--eval 1     shared distance-R sphere for the leaving side of all R+1 moves
--eval 2     ... and the uncovered-word list for the entering side
--huge       madvise(MADV_HUGEPAGE) the counter arrays
--pf         software prefetch of the next position subset
--wide       evaluate all n(q-1) moves of each candidate codeword
--preset X   base | p1 | p2 | p2b | p3 | p4 | p5 | p5b(=recommended)
```

Search-altering (they change what the search does, so they are measured by
time-to-target, not by fixed-iteration timing):

```
--fix0       translate the code so codeword 0 is 0^n, then freeze it
--ucache     serve pick_uncovered from a cached buffer of uncovered words
--upick      focus on the longest-uncovered word instead of a random one
--early      lower-bound early exit inside a candidate walk (per-candidate
             evaluator only)
```

Measurement support:

```
--target U   stop as soon as best_uncovered <= U; the RESULT line then carries
             ttt_cpu / ttt_wall / ttt_it, the CPU time, wall time and iteration
             at which the target was first reached
--cpu SEC    budget in CPU seconds (CLOCK_PROCESS_CPUTIME_ID) instead of wall
             clock; the RESULT line always reports cpu= and cpurate=
```

The `RESULT` line is a strict superset of the original one, so any existing
parser keeps working.

## Recommended engine

`./covsearch2 --preset p5b` (= `--hoist --st 2 --eval 2 --huge`), adding
`--wide` on loose, high-redundancy cells.

Measured against the current production solver, on a bit-identical search
(20 seeds per cell, paired per-seed CPU ratios):

| cell | `--preset p5b` | `--preset p5b --wide` |
|---|---|---|
| `K_6(6,3)` `M=41` | **2.03x** [1.99, 2.13] | 0.28x (do not use `--wide` here) |
| `K_6(8,4)` `M=169` | **6.12x** [6.06, 6.62] | **8.03x** (10.97x end to end) |
| `K_8(9,4)` `M=2944` | **8.07x** [6.19, 10.72] | **11.04x** (8.38x end to end) |

See `METHODS.md` §G for the full tables, §H for what did not work and §I for
the caveats. It is *not* wired into `cov/campaign.py`; the running sweep is
untouched.
