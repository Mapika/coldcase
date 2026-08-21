#!/usr/bin/env python3
"""Head-to-head on K_8(9,4) @ M=940, 60 s each: transform engine vs the
chain-batched kernel, against the recorded CPU numbers.

Reference points (cov/engine/NOTES.md §4.2/§5, 55-60 s budgets on the shared
socket): best free search ~7.75M uncovered, symsearch 6 threads ~6.44-6.51M,
engine0 median -48311 score ~ best-of-portfolio ~6.4M. Chain kernel: 256
chains sustained ~1492 aggregate moves/s on this cell (contended socket
comparison ~1200 moves/s for 64 cores).

Refuses to run when a foreign process owns the GPU.
"""
import subprocess
import sys
import time

import numpy as np

from gpuchain import Engine, foreign_gpu_pids
import chain


def main():
    if foreign_gpu_pids():
        print("YIELD: GPU busy")
        return 2
    q, n, R, M, budget = 8, 9, 4, 940, 60.0

    # ---- transform engine
    e = Engine(q, n, R, seed=5, out=None, log=lambda *a: None,
               deadline=time.time() + budget)
    t0 = time.time()
    e.load(np.zeros(0, dtype=np.int64))
    unc = e.lns(M, round_budget_s=budget - (time.time() - t0))
    unc = e.verify_solved_on_device() if unc == 0 else unc
    dt = time.time() - t0
    print(f"transform-engine: uncovered={unc} M={len(e.code)} "
          f"rounds={e.stats['rounds']} transforms={e.stats['transforms']} "
          f"placements={e.stats['placements']} wall={dt:.1f}s")
    e.close()

    # ---- chain kernel, 256 chains, same wall budget (iters calibrated:
    # ~5.8 it/s/chain on this cell when idle -> ~350 iters in 60 s)
    rng = np.random.default_rng(5)
    code = chain.random_code(q, n, M, rng)
    t0 = time.time()
    r = chain.run(q, n, R, code, nchains=256, iters=350, seed=5)
    dt = time.time() - t0
    agg = int(r["iters"].sum())
    print(f"chain-kernel: best_u_min={int(r['best_u'].min())} "
          f"best_u_median={int(np.median(r['best_u']))} "
          f"agg_moves={agg} wall={dt:.1f}s agg_moves_per_s={agg/dt:.0f}")

    print("cpu-reference (recorded): free search ~7.75M, symsearch ~6.44M "
          "uncovered at 55-60s; socket ~1200 moves/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
