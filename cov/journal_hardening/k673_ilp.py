#!/usr/bin/env python3
"""Task 5a: plain set-covering ILP baseline on K6(7,3) (referee M11).

The only Task-5 cell whose plain ILP fits in memory:
V = 6^7 = 279,936 binary vars, V covering rows, |B_3| = 4,936 nnz per row,
nnz = 1.382e9 (~17 GB of index+value data).

Modes (1h HiGHS each, 8 threads):
  min     : minimize sum x  (what can a plain ILP achieve in 1h?)
  feas245 : cardinality row sum x <= 245 (= Keri ub 246 - 1), objective 0
  feas227 : cardinality row sum x <= 227 (= our record), objective 0
No symmetry breaking, no warm start - deliberately plain.
"""
import argparse
import gc
import sys
import time

import numpy as np

Q, N, R = 6, 7, 3
V = Q ** N


def build_ball_targets(extra_row=False):
    """int32 array T of shape (B(+1), V): T[o, w] = row index of the o-th
    ball member of column w (rows = words to cover); optional extra row V
    (cardinality)."""
    from itertools import combinations, product
    dig = np.empty((V, N), dtype=np.int32)
    w = np.arange(V, dtype=np.int64)
    for i in range(N):
        dig[:, N - 1 - i] = (w // Q ** i) % Q
    pw = np.array([Q ** (N - 1 - i) for i in range(N)], dtype=np.int64)
    offs = [None]  # identity handled separately
    rows = [np.arange(V, dtype=np.int32)]
    for k in range(1, R + 1):
        for pos in combinations(range(N), k):
            for deltas in product(range(1, Q), repeat=k):
                t = w.copy()
                for p, d in zip(pos, deltas):
                    nd = (dig[:, p] + d) % Q
                    t += (nd.astype(np.int64) - dig[:, p]) * pw[p]
                rows.append(t.astype(np.int32))
    B = len(rows)
    if extra_row:
        rows.append(np.full(V, V, dtype=np.int32))
    T = np.vstack(rows)
    return T, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["min", "feas245", "feas227"])
    ap.add_argument("--tl", type=float, default=3600.0)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import highspy

    card = {"min": None, "feas245": 245, "feas227": 227}[a.mode]
    t0 = time.time()
    T, B = build_ball_targets(extra_row=card is not None)
    nrows_per_col = T.shape[0]
    print("[ilp] ball targets built: B=%d rows/col=%d nnz=%.3e (%.0fs)"
          % (B, nrows_per_col, nrows_per_col * float(V), time.time() - t0),
          flush=True)
    assert B == 4936
    a_index = np.ascontiguousarray(T.T).ravel()
    del T
    gc.collect()
    nnz = a_index.size
    a_start = np.arange(V + 1, dtype=np.int64) * nrows_per_col
    a_value = np.ones(nnz, dtype=np.float64)
    print("[ilp] CSC ready: nnz=%d (%.0fs)" % (nnz, time.time() - t0),
          flush=True)

    lp = highspy.HighsLp()
    nrows = V + (1 if card is not None else 0)
    lp.num_col_ = V
    lp.num_row_ = nrows
    lp.col_cost_ = np.ones(V) if card is None else np.zeros(V)
    lp.col_lower_ = np.zeros(V)
    lp.col_upper_ = np.ones(V)
    rl = np.ones(nrows)
    ru = np.full(nrows, highspy.kHighsInf)
    if card is not None:
        rl[V] = -highspy.kHighsInf
        ru[V] = float(card)
    lp.row_lower_ = rl
    lp.row_upper_ = ru
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = a_start
    lp.a_matrix_.index_ = a_index
    lp.a_matrix_.value_ = a_value
    lp.integrality_ = [highspy.HighsVarType.kInteger] * V
    h = highspy.Highs()
    h.setOptionValue("time_limit", a.tl)
    h.setOptionValue("threads", a.threads)
    st = h.passModel(lp)
    print("[ilp] passModel: %s (%.0fs)" % (st, time.time() - t0), flush=True)
    del a_index, a_value, lp
    gc.collect()
    t1 = time.time()
    h.run()
    info = h.getInfo()
    print("[ilp] [done] mode=%s status=%s solve=%.0fs primal=%s dual=%s"
          % (a.mode, h.modelStatusToString(h.getModelStatus()),
             time.time() - t1, info.objective_function_value,
             info.mip_dual_bound), flush=True)
    sol = np.array(h.getSolution().col_value[:V])
    chosen = np.flatnonzero(sol > 0.5)
    print("[ilp] [done] integral vars set: %d" % len(chosen), flush=True)


if __name__ == "__main__":
    sys.exit(main())
