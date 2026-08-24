#!/usr/bin/env python3
"""K8(4,2) exact: settle 22 vs 23 by ILP (HiGHS via highspy).

Minimum set cover over Z_8^4 (4096 words), balls of Hamming radius 2
(|B_2| = 1 + 4*7 + 6*49 = 323).

Symmetry breaking used: fix x_0 = 1 (the all-zero word is a codeword).
VALIDITY: Hamming distance on Z_q^n is invariant under translation by any
vector t (d(u+t, v+t) = d(u,v) coordinatewise since u_i+t_i == v_i+t_i iff
u_i == v_i).  Hence if C is a covering code with radius R and c in C, then
C - c is a covering code of the same size containing 0.  So the minimum size
is attained by a code containing 0, and fixing x_0 = 1 in the minimization
(or in the feasibility test at a given M) loses nothing.  This is checked
numerically at startup (translation of a random cover stays a cover).

Modes:
  opt   : minimize sum x, x_0 = 1, time limit --tl seconds.
  feas22: decision problem "is there a cover of size <= 22", x_0 = 1.
Both log HiGHS output to stdout; run under tee/redirect.
"""
import argparse
import sys
import time

import numpy as np

Q, N, R = 8, 4, 2
V = Q ** N  # 4096


def all_words():
    w = np.arange(V, dtype=np.int64)
    dig = np.empty((V, N), dtype=np.int8)
    for i in range(N):
        dig[:, N - 1 - i] = (w // Q ** i) % Q
    return dig


def ball_matrix():
    """Boolean (V,V) matrix A with A[v,w] = 1 iff d(v,w) <= R."""
    dig = all_words()
    # pairwise distances via chunks (4096x4096 x 4 fits easily)
    d = np.zeros((V, V), dtype=np.int8)
    for i in range(N):
        d += (dig[:, None, i] != dig[None, :, i])
    return d <= R


def check_translation_invariance(A):
    """Sanity: translating a random covering set keeps it covering."""
    rng = np.random.default_rng(0)
    # greedy cover
    uncov = np.ones(V, bool)
    C = []
    while uncov.any():
        gains = A[:, uncov].sum(axis=1)
        c = int(np.argmax(gains))
        C.append(c)
        uncov &= ~A[c]
    C = np.array(C)
    dig = all_words()
    t = rng.integers(0, Q, N)
    Cd = (dig[C] + t) % Q
    Cw = np.zeros(len(C), dtype=np.int64)
    for i in range(N):
        Cw = Cw * Q + Cd[:, i]
    ok = A[Cw].any(axis=0).all()
    print(f"[sanity] greedy cover size {len(C)}; translated by {t.tolist()}: "
          f"{'still a cover' if ok else 'NOT A COVER (BUG)'}", flush=True)
    assert ok
    return len(C)


def build_model(h, A, fix_zero=True, card=None):
    import highspy
    inf = highspy.kHighsInf
    lower = np.zeros(V)
    upper = np.ones(V)
    if fix_zero:
        lower[0] = 1.0
    obj = np.ones(V)
    h.addVars(V, lower, upper)
    h.changeColsCost(V, np.arange(V, dtype=np.int32), obj)
    for j in range(V):
        h.changeColIntegrality(j, highspy.HighsVarType.kInteger)
    # covering rows: for each word v, sum_{w in B(v)} x_w >= 1
    for v in range(V):
        idx = np.flatnonzero(A[v]).astype(np.int32)
        h.addRow(1.0, inf, len(idx), idx, np.ones(len(idx)))
    if card is not None:
        h.addRow(-inf, float(card), V, np.arange(V, dtype=np.int32),
                 np.ones(V))
    h.changeObjectiveSense(highspy.ObjSense.kMinimize)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["opt", "feas22"])
    ap.add_argument("--tl", type=float, default=7200.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None,
                    help="write code file here if a cover <= 22 is found")
    ap.add_argument("--warm", default=None,
                    help="code file (base-36 lines) used as MIP start, "
                         "translated so 0000 is a codeword")
    a = ap.parse_args()

    import highspy
    A = ball_matrix()
    print(f"[setup] ball size check: {A[0].sum()} (expect 323)", flush=True)
    assert A[0].sum() == 323
    check_translation_invariance(A)

    h = highspy.Highs()
    h.setOptionValue("time_limit", a.tl)
    h.setOptionValue("threads", a.threads)
    h.setOptionValue("output_flag", True)
    h.setOptionValue("log_to_console", True)
    if a.mode == "opt":
        build_model(h, A, fix_zero=True, card=None)
    else:
        build_model(h, A, fix_zero=True, card=22)
    if a.warm:
        words = []
        for line in open(a.warm):
            line = line.strip()
            if line and not line.startswith("#"):
                words.append([int(ch, 36) for ch in line])
        Cd = np.array(words, dtype=np.int64)
        Cd = (Cd - Cd[0]) % Q          # translate: first codeword -> 0000
        Cw = np.zeros(len(Cd), dtype=np.int64)
        for i in range(N):
            Cw = Cw * Q + Cd[:, i]
        cov_ok = A[Cw].any(axis=0).all()
        print(f"[warm] {a.warm}: {len(Cw)} words, translated, "
              f"cover check: {cov_ok}", flush=True)
        assert cov_ok
        x0 = np.zeros(V)
        x0[Cw] = 1.0
        sol = highspy.HighsSolution()
        sol.col_value = list(x0)
        st = h.setSolution(sol)
        print(f"[warm] setSolution status: {st}", flush=True)
    t0 = time.time()
    h.run()
    dt = time.time() - t0
    st = h.getModelStatus()
    info = h.getInfo()
    print(f"\n[done] mode={a.mode} status={h.modelStatusToString(st)} "
          f"time={dt:.0f}s", flush=True)
    print(f"[done] primal bound={info.objective_function_value} "
          f"dual bound={info.mip_dual_bound} gap={info.mip_gap}", flush=True)
    sol = h.getSolution()
    x = np.array(sol.col_value[:V])
    chosen = np.flatnonzero(x > 0.5)
    if len(chosen) and A[chosen].any(axis=0).all():
        print(f"[done] solution IS a verified cover of size {len(chosen)}",
              flush=True)
        if a.out and len(chosen) <= 22:
            dig = all_words()[chosen]
            with open(a.out, "w") as f:
                for row in dig:
                    f.write("".join(str(d) for d in row) + "\n")
            print(f"[done] wrote {a.out}", flush=True)
    else:
        print("[done] no integral cover in returned solution", flush=True)


if __name__ == "__main__":
    sys.exit(main())
