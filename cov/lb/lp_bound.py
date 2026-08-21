#!/usr/bin/env python3
"""
lp_bound.py -- step 1 of the lower-bound track: the fractional covering LP,
its symmetry reduction, and an exact rational certificate checker.

THE LP
------
K_q(n,R) is the optimum of the 0/1 covering program

    min sum_c x_c   s.t.   sum_{c in B_R(w)} x_c >= 1 for all w,  x in {0,1}.

Its LP relaxation is a lower bound.  By LP duality, *any* y : [q]^n -> R_{>=0}
with

    sum_{w in B_R(c)} y_w <= 1     for every potential codeword c        (*)

gives  K_q(n,R) >= sum_w y_w.  This module

  (a) reduces y to a function of Hamming weight (i.e. restricts to the
      subgroup Aut_0(q,n) fixing the zero word, whose orbits on [q]^n are the
      weight classes), so that (*) becomes n+1 inequalities in n+1 unknowns,
      checkable *exactly*;
  (b) solves that reduced LP numerically (scipy, optional);
  (c) verifies a rational y exactly (fractions only) and reports the
      certified integer bound ceil(sum_w y_w).

THE HONEST CONCLUSION
---------------------
Aut(q,n) = S_q wr S_n is transitive on [q]^n, and (*) is Aut(q,n)-invariant.
Averaging any feasible y over the group therefore produces a feasible
*constant* y with the same objective, and a constant y is feasible iff
y <= 1/V with V = |B_R(0)|.  Hence

    LP optimum  =  q^n / V   exactly,

i.e. the fractional covering LP is *exactly* the sphere covering bound and
cannot do better -- no choice of symmetry class refines it.  This is verified
numerically below for many cells (`--sweep`).  It is the reason the
lower-bound track has to go to the semidefinite relaxation (certify.py);
LP-based improvements in the literature (van Wee, Zhang, Habsieger-Plagne,
Haas et al.) all add *extra combinatorial inequalities* beyond (*), they are
not obtained from this LP.

Usage:
    python3 lp_bound.py --sweep                 # LP == sphere bound check
    python3 lp_bound.py --cell q n R            # certificate for one cell
    python3 lp_bound.py --check FILE.json       # verify a rational y
"""

import sys
import json
import argparse
from math import comb
from fractions import Fraction


def C(n, k):
    if n < 0 or k < 0 or k > n:
        return 0
    return comb(n, k)


def ball_size(q, n, R):
    return sum(C(n, i) * (q - 1) ** i for i in range(R + 1))


def sphere_covering_bound(q, n, R):
    V = ball_size(q, n, R)
    return -(-(q ** n) // V)


def weight_ball_counts(q, n, R):
    """A[k][j] = #{w : wt(w)=j, d(c,w) <= R} for any fixed c of weight k."""
    A = [[0] * (n + 1) for _ in range(n + 1)]
    for k in range(n + 1):
        for r in range(R + 1):
            for a in range(min(k, r) + 1):
                b = r - a
                if b > n - k:
                    continue
                for z in range(a + 1):
                    j = k - z + b
                    if j < 0 or j > n:
                        continue
                    cnt = (C(k, a) * C(a, z) * (q - 2) ** (a - z)
                           * C(n - k, b) * (q - 1) ** b)
                    if cnt:
                        A[k][j] += cnt
    return A


def weight_class_sizes(q, n):
    return [C(n, j) * (q - 1) ** j for j in range(n + 1)]


def verify_rational_y(q, n, R, y):
    """Exact check of (*) for a weight-indexed rational y.

    Returns (feasible, objective as Fraction, violated_k or None).
    """
    y = [Fraction(v) for v in y]
    if len(y) != n + 1:
        raise ValueError("y must have n+1 entries")
    for v in y:
        if v < 0:
            return False, Fraction(0), -1
    A = weight_ball_counts(q, n, R)
    for k in range(n + 1):
        s = sum(A[k][j] * y[j] for j in range(n + 1))
        if s > 1:
            return False, Fraction(0), k
    sizes = weight_class_sizes(q, n)
    obj = sum(sizes[j] * y[j] for j in range(n + 1))
    return True, obj, None


def certified_bound(q, n, R, y):
    ok, obj, bad = verify_rational_y(q, n, R, y)
    if not ok:
        return None, bad
    num, den = obj.numerator, obj.denominator
    return -(-num // den), obj


def uniform_certificate(q, n, R):
    """The optimal (constant) dual:  y_w = 1/V."""
    V = ball_size(q, n, R)
    return [Fraction(1, V)] * (n + 1)


def solve_reduced_lp(q, n, R):
    """Numerically maximise sum_w y_w over the weight-reduced dual (scipy)."""
    try:
        import numpy as np
        from scipy.optimize import linprog
    except Exception:                                            # noqa: BLE001
        return None
    A = weight_ball_counts(q, n, R)
    sizes = weight_class_sizes(q, n)
    c = -np.array([float(s) for s in sizes])
    Aub = np.array([[float(A[k][j]) for j in range(n + 1)]
                    for k in range(n + 1)])
    # normalise rows to keep the LP well scaled
    scale = Aub.max(axis=1)
    scale[scale == 0] = 1.0
    Aub = Aub / scale[:, None]
    bub = 1.0 / scale
    c = c / max(abs(c).max(), 1.0)
    res = linprog(c, A_ub=Aub, b_ub=bub, bounds=[(0, None)] * (n + 1),
                  method="highs")
    if not res.success:
        return None
    y = res.x
    return float(sum(sizes[j] * y[j] for j in range(n + 1)))


def sweep(cells):
    print("%-14s %12s %12s %12s %s" %
          ("cell", "q^n/V (exact)", "ceil", "LP (scipy)", "agree"))
    allok = True
    for (q, n, R) in cells:
        V = ball_size(q, n, R)
        exact = Fraction(q ** n, V)
        sph = sphere_covering_bound(q, n, R)
        lp = solve_reduced_lp(q, n, R)
        # exact certificate from the uniform dual
        b, obj = certified_bound(q, n, R, uniform_certificate(q, n, R))
        # theory: LP optimum == q^n/V exactly.  scipy's HiGHS answer is only
        # a numerical check, so allow 1e-3 relative slop on it.
        agree = (b == sph) and (lp is None
                                or abs(lp - float(exact)) <= 1e-3 * float(exact))
        allok = allok and agree
        print("%-14s %12.4f %12d %12s %s"
              % ("K%d(%d,%d)" % (q, n, R), float(exact), sph,
                 ("%.4f" % lp) if lp is not None else "n/a",
                 "yes" if agree else "NO"))
    print("\nall agree: %s" % allok)
    return 0 if allok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--cell", nargs=3, type=int, metavar=("Q", "N", "R"))
    ap.add_argument("--check", metavar="FILE")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.check:
        cert = json.load(open(a.check))
        q, n, R = cert["q"], cert["n"], cert["R"]
        y = [Fraction(s) for s in cert["y_by_weight"]]
        b, obj = certified_bound(q, n, R, y)
        if b is None:
            print("LP CERTIFICATE INVALID (violated weight class %s)" % obj)
            return 1
        print("LP CERTIFICATE VALID")
        print("  K_%d(%d,%d) >= %s = %d" % (q, n, R, obj, b))
        return 0

    if a.cell:
        q, n, R = a.cell
        y = uniform_certificate(q, n, R)
        b, obj = certified_bound(q, n, R, y)
        print("K_%d(%d,%d) >= %s -> %d   (V=%d, q^n=%d)"
              % (q, n, R, obj, b, ball_size(q, n, R), q ** n))
        if a.out:
            json.dump({"q": q, "n": n, "R": R,
                       "y_by_weight": [str(v) for v in y],
                       "objective": str(obj), "K_lower_bound": b,
                       "note": "uniform dual y_w = 1/|B_R(0)|; this is the "
                               "exact optimum of the fractional covering LP"},
                      open(a.out, "w"), indent=1)
            print("wrote %s" % a.out)
        return 0

    if a.sweep:
        cells = []
        for q in range(2, 8):
            for n in range(2, 11):
                for R in range(1, min(5, n)):
                    cells.append((q, n, R))
        return sweep(cells)

    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
