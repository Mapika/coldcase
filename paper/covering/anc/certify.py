#!/usr/bin/env python3
"""
certify.py -- exact rational certificate checker for lower bounds on q-ary
covering codes  K_q(n,R),  via the symmetry-reduced semidefinite program of

    D. Gijswijt, S. Polak, "Semidefinite lower bounds for covering codes",
    arXiv:2504.01932 (Theorem 4.18, non-binary case, q >= 3).

STANDALONE: Python 3 standard library only (math, fractions, json, sys).
No numpy, no scipy, no solver.  Everything below is exact integer / rational
arithmetic.

---------------------------------------------------------------------------
WHAT IS CERTIFIED
---------------------------------------------------------------------------
The SDP of Theorem 4.18 is a minimisation

    min  c^T x   s.t.   x >= 0,
                        L_k(x) := <l_k, x> + l_k^0 >= 0        (k = 1..m)
                        A_b(x) := C_b + sum_v x_v A_b^v  psd    (b = 1..B)

with the property that *every* code C in [q]^n of covering radius <= R yields
a feasible x with c^T x = |C|^3.  Hence  K_q(n,R)^3 >= min = OPT.

A certificate is a dual feasible point: rationals y_k >= 0 and symmetric
positive-semidefinite rational matrices Y_b such that, writing

    d_v := sum_k y_k * l_k[v] + sum_b <Y_b, A_b^v>          (v = 1..N)
    d_0 := sum_k y_k * l_k^0  + sum_b <Y_b, C_b>

we have  d_v <= c_v  for every variable v.  Then for every feasible x,

    c^T x - (-d_0) = sum_v (c_v - d_v) x_v + sum_k y_k L_k(x)
                                           + sum_b <Y_b, A_b(x)>  >= 0,

so  OPT >= -d_0  and therefore  K_q(n,R) >= ceil( (-d_0)^{1/3} ).

The certificate file stores integers, together with one common positive
integer denominator `den`, so that y_k = a_k/den and Y_b = B_b/den.  All
checks below are therefore pure integer arithmetic except the exact
positive-semidefiniteness test, which uses `fractions.Fraction`.

---------------------------------------------------------------------------
CERTIFICATE FILE FORMAT (JSON)
---------------------------------------------------------------------------
{
  "problem":  {"q":6, "n":7, "R":3, "lambda":[1,1,1,1,0,0,0,0], "beta":1},
  "den":      <positive int, as decimal string>,
  "dual_lin": [<int as string>, ...]              # length = #linear constraints
  "dual_psd": [ [[<int as string>,...],...], ...] # one symmetric matrix / block
  "claim":    {"K_lower_bound": <int>}            # optional, re-derived anyway
}

Run:   python3 certify.py CERT.json
       python3 certify.py --selftest
       python3 certify.py --model q n R      (print model size only)
"""

import sys
import json
from math import comb, factorial
from fractions import Fraction

# =========================================================================
# 0. small exact combinatorial helpers
# =========================================================================


def C(n, k):
    """Binomial coefficient, 0 outside the triangle (also for negative args)."""
    if n < 0 or k < 0 or k > n:
        return 0
    return comb(n, k)


def multinom(*parts):
    """Multinomial coefficient (sum parts)! / prod parts!.  Parts >= 0."""
    s = 0
    for p in parts:
        if p < 0:
            return 0
        s += p
    r = factorial(s)
    for p in parts:
        r //= factorial(p)
    return r


def sgn(e):
    """(-1)**e for possibly negative integer e, staying in int arithmetic."""
    return -1 if (e % 2) else 1


# =========================================================================
# 1. orbits of triples (0,u,v) in the q-ary Hamming space
# =========================================================================
#
# A triple (0,u,v) is classified per coordinate l into five types:
#
#   i1 : u_l = 0, v_l = 0                       ("000")
#   i2 : u_l = 0, v_l != 0                      ("001")
#   i3 : u_l != 0, v_l = 0                      ("010")
#   i4 : u_l = v_l != 0                         ("011")
#   i5 : u_l != v_l, both != 0                  ("012")
#
# and (i,j,t,p) = (d(0,u), d(0,v), #{u_l!=0 and v_l!=0}, #{u_l=v_l!=0}) is
#
#   i = i3+i4+i5,  j = i2+i4+i5,  t = i4+i5,  p = i4,  d(u,v) = i2+i3+i5.
#
# By Prop. 4.11(iii) of the paper the SDP variable x^{t,p}_{i,j} depends only
# on the multiset {d(0,u), d(0,v), d(u,v)} together with i5 = t-p.


def comps5(n):
    """All (i1,i2,i3,i4,i5) >= 0 with sum n."""
    for i2 in range(n + 1):
        for i3 in range(n - i2 + 1):
            for i4 in range(n - i2 - i3 + 1):
                for i5 in range(n - i2 - i3 - i4 + 1):
                    yield (n - i2 - i3 - i4 - i5, i2, i3, i4, i5)


def ijtp_of(c5):
    i1, i2, i3, i4, i5 = c5
    return (i3 + i4 + i5, i2 + i4 + i5, i4 + i5, i4)


def c5_of(n, i, j, t, p):
    """Inverse map; returns None if the tuple is not a valid orbit."""
    i1 = n - i - j + t
    i2 = j - t
    i3 = i - t
    i4 = p
    i5 = t - p
    if i1 < 0 or i2 < 0 or i3 < 0 or i4 < 0 or i5 < 0:
        return None
    return (i1, i2, i3, i4, i5)


def orbit_key(i, j, t, p):
    """Canonical S_3-invariant key of the orbit of the triple {0,u,v}."""
    d = i + j - t - p
    return (tuple(sorted((i, j, d))), t - p)


# =========================================================================
# 2. block diagonalisation coefficients  (alpha, beta of the Julia reference
#    implementation https://github.com/CoveringCodes/Julia)
# =========================================================================


def _beta_coef(m, t, i, j, k):
    s = 0
    for u in range(m + 1):
        c1 = C(u, t)
        if c1 == 0:
            continue
        c2 = C(m - 2 * k, m - k - u)
        if c2 == 0:
            continue
        c3 = C(m - k - u, i - u)
        if c3 == 0:
            continue
        c4 = C(m - k - u, j - u)
        if c4 == 0:
            continue
        s += sgn(t - u) * c1 * c2 * c3 * c4
    return s


def alpha(n, i, j, t, p, a, k, q):
    """Entry (i,j) of the block (a,k) image of the orbit matrix M^{t,p}_{i,j}.

    Integral normalisation, identical to the reference Julia code; it differs
    from the paper's alpha by the diagonal congruence diag((q-1)^{i/2}), which
    does not affect positive semidefiniteness.
    """
    b = _beta_coef(n - a, t - a, i - a, j - a, k - a)
    if b == 0:
        return 0
    s2 = 0
    for g in range(p + 1):
        if t - a >= p - g:
            s2 += sgn(a - g) * C(a, g) * C(t - a, p - g) * (q - 2) ** (t - a - p + g)
    if s2 == 0:
        return 0
    return b * (q - 1) ** (i + j - t) * s2


# =========================================================================
# 3. Lasserre shift:  sum_{d} lambda_d sum_{w in S_d(0)} x[orbit{w,u,v}]
# =========================================================================


def lasserre_triple(q, n, lam, c5):
    """dict orbit_key -> int coefficient, for a triple orbit c5=(i1..i5)."""
    i1, i2, i3, i4, i5 = c5
    R = len(lam) - 1
    out = {}
    q1, q2, q3 = q - 1, q - 2, q - 3
    for e1 in range(min(i1, R) + 1):
        f1 = C(i1, e1) * q1 ** e1
        r1 = R - e1
        for e2 in range(min(i2, r1) + 1):
            for e3 in range(min(i2 - e2, r1 - e2) + 1):
                f2 = f1 * multinom(e2, e3, i2 - e2 - e3) * q2 ** e3
                if f2 == 0:
                    continue
                r2 = r1 - e2 - e3
                for e4 in range(min(i3, r2) + 1):
                    for e5 in range(min(i3 - e4, r2 - e4) + 1):
                        f3 = f2 * multinom(e4, e5, i3 - e4 - e5) * q2 ** e5
                        if f3 == 0:
                            continue
                        r3 = r2 - e4 - e5
                        for e6 in range(min(i4, r3) + 1):
                            for e7 in range(min(i4 - e6, r3 - e6) + 1):
                                f4 = f3 * multinom(e6, e7, i4 - e6 - e7) * q2 ** e7
                                if f4 == 0:
                                    continue
                                r4 = r3 - e6 - e7
                                for e8 in range(min(i5, r4) + 1):
                                    for e9 in range(min(i5 - e8, r4 - e8) + 1):
                                        for e10 in range(
                                            min(i5 - e8 - e9, r4 - e8 - e9) + 1
                                        ):
                                            f5 = (
                                                f4
                                                * multinom(
                                                    e8, e9, e10, i5 - e8 - e9 - e10
                                                )
                                                * q3 ** e10
                                            )
                                            if f5 == 0:
                                                continue
                                            w = lam[e1 + e2 + e3 + e4 + e5 + e6
                                                    + e7 + e8 + e9 + e10]
                                            if w == 0:
                                                continue
                                            j1 = i1 - e1 + e6
                                            j2 = i2 - e2 - e3 + e4 + e8
                                            j3 = i3 - e4 - e5 + e2 + e9
                                            j4 = i4 - e6 + e1
                                            j5 = i5 - e8 - e9 + e3 + e5
                                            key = orbit_key(*ijtp_of(
                                                (j1, j2, j3, j4, j5)))
                                            out[key] = out.get(key, 0) + w * f5
    return out


def lasserre_pair(q, n, lam, i):
    """Same, for the pair orbit (0,v) with d(0,v)=i.  dict key -> int."""
    i1, i2 = n - i, i
    R = len(lam) - 1
    out = {}
    for e1 in range(min(i1, R) + 1):
        f1 = C(i1, e1) * (q - 1) ** e1
        for e2 in range(min(i2, R - e1) + 1):
            for e3 in range(min(i2 - e2, R - e1 - e2) + 1):
                f = f1 * multinom(e2, e3, i2 - e2 - e3) * (q - 2) ** e3
                if f == 0:
                    continue
                w = lam[e1 + e2 + e3]
                if w == 0:
                    continue
                m = i2 - e2 + e1          # new  d(0, v-w)
                key = orbit_key(0, m, 0, 0)
                out[key] = out.get(key, 0) + w * f
    return out


# =========================================================================
# 4. matrix-cut distribution numbers (Lemma 4.15 of the paper)
# =========================================================================


def makedistr(q, n, i, j, t, p, lam):
    """coef[(j2,t2,p2)] = sum_d lambda_d * #{w : dbar(u,w)=(i,j2,t2,p2),
                                              d(v,w)=d}."""
    ti, tj, tp, pp = i - t, j - t, t - p, p
    rest = n + t - i - j
    q1, q2, q3 = q - 1, q - 2, q - 3
    coef = {}
    for a1 in range(ti + 1):
        for a2 in range(ti - a1 + 1):
            fA = multinom(ti - a1 - a2, a1, a2) * q2 ** a2
            if fA == 0:
                continue
            for b1 in range(tj + 1):
                for b2 in range(tj - b1 + 1):
                    fB = fA * multinom(tj - b1 - b2, b1, b2) * q2 ** b2
                    if fB == 0:
                        continue
                    for c1 in range(pp + 1):
                        for c2 in range(pp - c1 + 1):
                            fC = fB * multinom(pp - c1 - c2, c1, c2) * q2 ** c2
                            if fC == 0:
                                continue
                            for d1 in range(tp + 1):
                                for d2 in range(tp - d1 + 1):
                                    for d3 in range(tp - d1 - d2 + 1):
                                        fD = (fC
                                              * multinom(tp - d1 - d2 - d3,
                                                         d1, d2, d3)
                                              * q3 ** d3)
                                        if fD == 0:
                                            continue
                                        base = a1 + a2 + j - b1 - c1 - d2
                                        j0 = (a1 + a2 + b1 + b2 + c1 + c2
                                              + d1 + d2 + d3)
                                        t2 = a1 + a2 + c1 + c2 + d1 + d2 + d3
                                        p2 = a1 + c1 + d1
                                        for e in range(rest + 1):
                                            dist = base + e
                                            if dist < 0 or dist >= len(lam):
                                                continue
                                            w = lam[dist]
                                            if w == 0:
                                                continue
                                            val = (w * fD * C(rest, e)
                                                   * q1 ** e)
                                            k = (j0 + e, t2, p2)
                                            coef[k] = coef.get(k, 0) + val
    return coef


# =========================================================================
# 5. the model
# =========================================================================


class Model(object):
    """Exact integer description of the Theorem 4.18 SDP.

    obj  : list of int, length nvars               (already multiplied by q^n)
    lin  : list of (dict var->int, int const)      meaning  <l,x> + const >= 0
    psd  : list of blocks; block = list of rows, row = list of
           (dict var->int, int const); the matrix is
           A_b(x) = C_b + sum_v x_v A_b^v  and must be psd.
    """

    __slots__ = ("q", "n", "R", "lam", "beta", "nvars", "keys", "index",
                 "obj", "lin", "psd", "lin_tags")


def _lf_add(lf, var, c):
    if c:
        lf[var] = lf.get(var, 0) + c


def build_model(q, n, R, lam=None, beta=1, verbose=False):
    """Build the symmetry-reduced SDP of Theorem 4.18 for K_q(n,R)."""
    if q < 3:
        raise ValueError("this module implements the non-binary case q >= 3 "
                         "(Theorem 4.18); q=2 uses Theorem 4.9")
    if not (0 <= R < n):
        raise ValueError("need 0 <= R < n")
    if lam is None:
        lam = [1] * (R + 1) + [0] * (n - R)          # sphere covering ineqs
    lam = list(lam)
    if len(lam) < n + 1:
        lam = lam + [0] * (n + 1 - len(lam))
    beta = int(beta)

    # ---- variables ------------------------------------------------------
    tuples = []                    # all (i,j,t,p) in I(q,n)
    keys = []
    index = {}
    for c5 in comps5(n):
        i, j, t, p = ijtp_of(c5)
        tuples.append((i, j, t, p))
        k = orbit_key(i, j, t, p)
        if k not in index:
            index[k] = len(keys)
            keys.append(k)
    nvars = len(keys)

    def vidx(i, j, t, p):
        return index[orbit_key(i, j, t, p)]

    def ball(m):
        """variable  x^{0,0}_{0,m}  =  M'_{0,u} for d(0,u)=m."""
        return index[orbit_key(0, m, 0, 0)]

    # ---- objective ------------------------------------------------------
    obj = [0] * nvars
    qn = q ** n
    for c5 in comps5(n):
        i1, i2, i3, i4, i5 = c5
        i, j, t, p = ijtp_of(c5)
        gamma = ((q - 1) ** (i + j - t) * (q - 2) ** (t - p)
                 * multinom(i1, i2, i3, i4, i5))
        obj[vidx(i, j, t, p)] += qn * gamma

    lin = []
    lin_tags = []

    # ---- Prop 4.11 (i),(ii): the basic order inequalities ---------------
    for (i, j, t, p) in tuples:
        v = vidx(i, j, t, p)
        d = i + j - t - p
        bi, bd, b0 = ball(i), ball(d), ball(0)
        f = {}
        _lf_add(f, bi, 1)
        _lf_add(f, v, -1)
        lin.append((f, 0))
        lin_tags.append(("p411_a", i, j, t, p))
        f = {}
        _lf_add(f, bd, 1)
        _lf_add(f, v, -1)
        lin.append((f, 0))
        lin_tags.append(("p411_b", i, j, t, p))
        f = {}
        _lf_add(f, v, 1)
        _lf_add(f, bi, -1)
        _lf_add(f, bd, -1)
        _lf_add(f, b0, 1)
        lin.append((f, 0))
        lin_tags.append(("p411_c", i, j, t, p))

    # ---- Prop 4.17: matrix cut inequalities -----------------------------
    b0 = ball(0)
    for (i, j, t, p) in tuples:
        coef = makedistr(q, n, i, j, t, p, lam)
        bi = ball(i)
        f1, f2, f3, f4 = {}, {}, {}, {}
        for (j2, t2, p2), cc in coef.items():
            if cc == 0:
                continue
            if c5_of(n, i, j2, t2, p2) is None:
                continue
            vv = vidx(i, j2, t2, p2)
            h = i + j2 - t2 - p2
            bj2, bh = ball(j2), ball(h)
            _lf_add(f1, vv, cc)
            _lf_add(f2, vv, -cc)
            _lf_add(f2, bj2, cc)
            _lf_add(f3, vv, -cc)
            _lf_add(f3, bh, cc)
            _lf_add(f4, vv, cc)
            _lf_add(f4, bh, -cc)
            _lf_add(f4, b0, cc)
            _lf_add(f4, bj2, -cc)
        _lf_add(f1, bi, -beta)
        _lf_add(f2, bi, beta)
        _lf_add(f2, b0, -beta)
        _lf_add(f3, bi, beta)
        _lf_add(f3, b0, -beta)
        _lf_add(f4, b0, 2 * beta)
        _lf_add(f4, bi, -beta)
        lin.append((f1, 0))
        lin_tags.append(("cut1", i, j, t, p))
        lin.append((f2, 0))
        lin_tags.append(("cut2", i, j, t, p))
        lin.append((f3, 0))
        lin_tags.append(("cut3", i, j, t, p))
        lin.append((f4, -beta))
        lin_tags.append(("cut4", i, j, t, p))

    # ---- semidefinite blocks -------------------------------------------
    # Precompute the Lasserre shift per orbit tuple.
    lass = {}
    for (i, j, t, p) in tuples:
        lass[(i, j, t, p)] = lasserre_triple(q, n, lam, c5_of(n, i, j, t, p))
    lass_pair = {m: lasserre_pair(q, n, lam, m) for m in range(n + 1)}

    blocks_ak = []
    for a in range(n + 1):
        for k in range(a, n + 1):
            if 2 * k <= n + a:
                blocks_ak.append((a, k, list(range(k, n + a - k + 1))))

    psd = []

    def new_block(sz):
        return [[({}, 0) for _ in range(sz)] for _ in range(sz)]

    # family 1: M'  (Prop 4.12, first matrices of (34) and (35))
    for (a, k, idxs) in blocks_ak:
        sz = len(idxs)
        X = new_block(sz)
        for r in range(sz):
            i = idxs[r]
            for c in range(sz):
                j = idxs[c]
                f = {}
                for t in range(n + 1):
                    for p in range(t + 1):
                        if c5_of(n, i, j, t, p) is None:
                            continue
                        al = alpha(n, i, j, t, p, a, k, q)
                        if al:
                            _lf_add(f, vidx(i, j, t, p), al)
                X[r][c] = (f, 0)
        psd.append(X)

    # family 2: M''  (Prop 4.12, second matrices of (34) and (35))
    for (a, k, idxs) in blocks_ak:
        sz = len(idxs)
        bord = 1 if k == 0 else 0
        X = new_block(sz + bord)
        for r in range(sz):
            i = idxs[r]
            for c in range(sz):
                j = idxs[c]
                f = {}
                for t in range(n + 1):
                    for p in range(t + 1):
                        if c5_of(n, i, j, t, p) is None:
                            continue
                        al = alpha(n, i, j, t, p, a, k, q)
                        if al:
                            _lf_add(f, ball(i + j - t - p), al)
                            _lf_add(f, vidx(i, j, t, p), -al)
                X[r + bord][c + bord] = (f, 0)
            if bord:
                sc = C(n, i) * (q - 1) ** i
                f = {}
                _lf_add(f, ball(0), sc)
                _lf_add(f, ball(i), -sc)
                X[0][r + 1] = (f, 0)
                X[r + 1][0] = (dict(f), 0)
        if bord:
            f = {}
            _lf_add(f, ball(0), -1)
            X[0][0] = (f, 1)
        psd.append(X)

    # family 3: N  (Prop 4.14, (39) and (40))
    for (a, k, idxs) in blocks_ak:
        sz = len(idxs)
        bord = 1 if k == 0 else 0
        X = new_block(sz + bord)
        for r in range(sz):
            i = idxs[r]
            for c in range(sz):
                j = idxs[c]
                f = {}
                for t in range(n + 1):
                    for p in range(t + 1):
                        if c5_of(n, i, j, t, p) is None:
                            continue
                        al = alpha(n, i, j, t, p, a, k, q)
                        if not al:
                            continue
                        for key, cc in lass[(i, j, t, p)].items():
                            _lf_add(f, index[key], al * cc)
                        _lf_add(f, ball(i + j - t - p), -al * beta)
                X[r + bord][c + bord] = (f, 0)
            if bord:
                sc = C(n, i) * (q - 1) ** i
                f = {}
                for key, cc in lass_pair[i].items():
                    _lf_add(f, index[key], sc * cc)
                _lf_add(f, ball(0), -sc * beta)
                X[0][r + 1] = (f, 0)
                X[r + 1][0] = (dict(f), 0)
        if bord:
            tot = sum(lam[d] * C(n, d) * (q - 1) ** d for d in range(n + 1))
            f = {}
            _lf_add(f, ball(0), tot)
            X[0][0] = (f, -beta)
        psd.append(X)

    # ---- deduplicate: drop identically-true rows and repeated rows --------
    # (repeated rows create an unnecessarily degenerate dual; removing them
    # is purely a conditioning measure and does not change the SDP value.)
    seen = {}
    lin2, tags2 = [], []
    for k, (f, c0) in enumerate(lin):
        ff = {v: c for v, c in f.items() if c}
        if not ff:
            if c0 < 0:
                raise AssertionError("infeasible constant row generated")
            continue                      # 0 + c0 >= 0 with c0 >= 0: vacuous
        key = (tuple(sorted(ff.items())), c0)
        if key in seen:
            continue
        seen[key] = True
        lin2.append((ff, c0))
        tags2.append(lin_tags[k])
    lin, lin_tags = lin2, tags2

    psd2 = []
    seenb = set()
    for X in psd:
        key = tuple(
            tuple((tuple(sorted(f.items())), c0) for (f, c0) in row)
            for row in X)
        if key in seenb:
            continue
        seenb.add(key)
        psd2.append(X)
    psd = psd2

    m = Model()
    m.q, m.n, m.R, m.lam, m.beta = q, n, R, lam, beta
    m.nvars, m.keys, m.index = nvars, keys, index
    m.obj, m.lin, m.psd, m.lin_tags = obj, lin, psd, lin_tags
    if verbose:
        sys.stderr.write(
            "model q=%d n=%d R=%d : %d vars, %d linear ineqs, %d psd blocks "
            "(sum size %d, sum size^2 %d)\n"
            % (q, n, R, nvars, len(lin), len(psd),
               sum(len(b) for b in psd), sum(len(b) ** 2 for b in psd)))
    return m


# =========================================================================
# 6. exact positive semidefiniteness test
# =========================================================================


def is_psd_exact(M):
    """Exact PSD test for a symmetric matrix of ints/Fractions.

    Symmetric-pivot LDL^T.  Returns (True, None) or (False, reason).
    """
    nn = len(M)
    A = [[Fraction(M[r][c]) for c in range(nn)] for r in range(nn)]
    for r in range(nn):
        for c in range(r + 1, nn):
            if A[r][c] != A[c][r]:
                return False, "not symmetric at (%d,%d)" % (r, c)
    live = list(range(nn))
    while live:
        # pick pivot = largest diagonal entry
        best = max(live, key=lambda r: A[r][r])
        if A[best][best] < 0:
            return False, "negative diagonal pivot"
        if A[best][best] == 0:
            # every remaining row through `best` must vanish
            for c in live:
                if A[best][c] != 0:
                    return False, "zero pivot with nonzero off-diagonal"
            live.remove(best)
            continue
        piv = A[best][best]
        live.remove(best)
        col = {r: A[r][best] for r in live}
        for r in live:
            cr = col[r]
            if cr == 0:
                continue
            f = cr / piv
            Ar = A[r]
            Ab = A[best]
            for c in live:
                Ar[c] -= f * Ab[c]
    return True, None


# =========================================================================
# 7. certificate evaluation
# =========================================================================


def integer_ceil_cuberoot(num, den):
    """Smallest integer m with m^3 >= num/den  (num,den > 0 integers)."""
    if num <= 0:
        return 0
    m = max(1, int(round((num / den) ** (1.0 / 3.0))) if den else 1)
    # integer search, robust against float error
    while m ** 3 * den >= num:
        m -= 1
    while m ** 3 * den < num:
        m += 1
    return m


def check_lambda_beta_valid(n, R, lam, beta):
    """Is (lambda, beta) a valid covering inequality for radius R?

    The relaxation is only sound if every code C of covering radius <= R
    satisfies  sum_d lambda_d A_d(u) >= beta  for all u, where
    A_d(u) = |C ∩ S_d(u)|.  The sphere covering inequalities give
    sum_{d<=R} A_d(u) >= 1 for every u.  A sufficient (and, for the
    non-negative lambda we use, natural) condition is therefore

        lambda_d >= 0 for all d,   beta > 0,   min_{d<=R} lambda_d >= beta,

    since then sum_d lambda_d A_d >= (min_{d<=R} lambda_d) * sum_{d<=R} A_d
    >= beta.  Returns (ok, reason).

    This check is what stops a certificate from smuggling in an unsound
    (lambda, beta) and getting arithmetically-correct nonsense verified.
    """
    if beta <= 0:
        return False, "beta must be positive"
    if len(lam) < R + 1:
        return False, "lambda too short"
    for d, v in enumerate(lam):
        if v < 0:
            return False, "lambda[%d] = %s < 0" % (d, v)
    m = min(lam[d] for d in range(R + 1))
    if m < beta:
        return False, ("min_{d<=R} lambda_d = %s < beta = %s, so the "
                       "inequality is not implied by sphere covering"
                       % (m, beta))
    return True, None


def evaluate_certificate(model, den, dual_lin, dual_psd):
    """Return dict with the exact verification result.

    den       : positive int
    dual_lin  : list of ints, y_k = dual_lin[k]/den
    dual_psd  : list of integer symmetric matrices, Y_b = dual_psd[b]/den
    """
    res = {"ok": False, "reasons": []}
    okl, whyl = check_lambda_beta_valid(model.n, model.R, model.lam, model.beta)
    if not okl:
        res["reasons"].append("invalid covering inequality (lambda,beta): %s"
                              % whyl)
        return res
    if den <= 0:
        res["reasons"].append("denominator must be positive")
        return res
    if len(dual_lin) != len(model.lin):
        res["reasons"].append("dual_lin has %d entries, model has %d linear "
                              "constraints" % (len(dual_lin), len(model.lin)))
        return res
    if len(dual_psd) != len(model.psd):
        res["reasons"].append("dual_psd has %d blocks, model has %d"
                              % (len(dual_psd), len(model.psd)))
        return res

    # (a) nonnegativity of the linear multipliers
    for k, y in enumerate(dual_lin):
        if y < 0:
            res["reasons"].append("dual_lin[%d] = %s < 0" % (k, y))
            return res

    # (b) psd-ness of the block multipliers
    for b, Y in enumerate(dual_psd):
        if len(Y) != len(model.psd[b]):
            res["reasons"].append("dual_psd[%d] has size %d, block size %d"
                                  % (b, len(Y), len(model.psd[b])))
            return res
        ok, why = is_psd_exact(Y)
        if not ok:
            res["reasons"].append("dual_psd[%d] not psd: %s" % (b, why))
            return res

    # (c) accumulate d_v (times den) and d_0 (times den)
    d = [0] * model.nvars
    d0 = 0
    for k, (f, c0) in enumerate(model.lin):
        y = dual_lin[k]
        if y == 0:
            continue
        for v, cc in f.items():
            d[v] += y * cc
        d0 += y * c0
    for b, X in enumerate(model.psd):
        Y = dual_psd[b]
        sz = len(X)
        for r in range(sz):
            Xr = X[r]
            Yr = Y[r]
            for c in range(sz):
                yv = Yr[c]
                if yv == 0:
                    continue
                f, c0 = Xr[c]
                for v, cc in f.items():
                    d[v] += yv * cc
                d0 += yv * c0

    # (d) d_v <= c_v  (times den)
    worst = None
    for v in range(model.nvars):
        slack = model.obj[v] * den - d[v]
        if slack < 0:
            res["reasons"].append(
                "dual coefficient exceeds objective on variable %d (key %s): "
                "slack*den = %s" % (v, model.keys[v], slack))
            return res
        if worst is None or slack < worst:
            worst = slack

    if d0 >= 0:
        res["reasons"].append("d_0 = %s/%s >= 0, bound is not positive"
                              % (d0, den))
        return res

    num = -d0                       # bound = num/den  >=  OPT  >= ... no:
    # OPT >= num/den, and K^3 >= OPT.
    K = integer_ceil_cuberoot(num, den)
    res["ok"] = True
    res["bound_num"] = num
    res["bound_den"] = den
    res["bound_float"] = num / den
    res["cube_root_float"] = (num / den) ** (1.0 / 3.0)
    res["K_lower_bound"] = K
    res["min_slack_times_den"] = worst
    return res


def check_certificate_file(path, verbose=True):
    with open(path) as fh:
        cert = json.load(fh)
    pr = cert["problem"]
    q, n, R = int(pr["q"]), int(pr["n"]), int(pr["R"])
    lam = [int(x) for x in pr.get("lambda", [1] * (R + 1) + [0] * (n - R))]
    beta = int(pr.get("beta", 1))
    if verbose:
        sys.stderr.write("rebuilding model for q=%d n=%d R=%d ...\n" % (q, n, R))
    model = build_model(q, n, R, lam=lam, beta=beta, verbose=verbose)
    den = int(cert["den"])
    dual_lin = [int(x) for x in cert["dual_lin"]]
    dual_psd = [[[int(x) for x in row] for row in Y] for Y in cert["dual_psd"]]
    res = evaluate_certificate(model, den, dual_lin, dual_psd)
    res["problem"] = {"q": q, "n": n, "R": R}
    claim = cert.get("claim", {}).get("K_lower_bound")
    if claim is not None and res.get("ok"):
        res["claim_matches"] = (int(claim) == res["K_lower_bound"])
    return res


# =========================================================================
# 8. primal validity self-test:  feed an actual covering code into the model
# =========================================================================


def code_primal_point(model, code):
    """Exact primal point x of the SDP induced by an explicit code.

    x^{t,p}_{i,j} = q^{-n} * gamma^{-1} * #{(u,v,w) in C^3 with the given
    orbit type}.  Returns a list of Fractions of length model.nvars.
    """
    q, n = model.q, model.n
    M = len(code)
    counts = {}
    for u in code:
        for v in code:
            for w in code:
                i1 = i2 = i3 = i4 = i5 = 0
                for l in range(n):
                    a, b, c = u[l], v[l], w[l]
                    if b == a and c == a:
                        i1 += 1
                    elif b == a:
                        i2 += 1
                    elif c == a:
                        i3 += 1
                    elif b == c:
                        i4 += 1
                    else:
                        i5 += 1
                key = orbit_key(*ijtp_of((i1, i2, i3, i4, i5)))
                counts[key] = counts.get(key, 0) + 1
    x = [Fraction(0)] * model.nvars
    qn = q ** n
    # gamma summed over the (i,j,t,p) tuples inside one orbit key
    gam = [0] * model.nvars
    for c5 in comps5(n):
        i1, i2, i3, i4, i5 = c5
        i, j, t, p = ijtp_of(c5)
        g = ((q - 1) ** (i + j - t) * (q - 2) ** (t - p)
             * multinom(i1, i2, i3, i4, i5))
        gam[model.index[orbit_key(i, j, t, p)]] += g
    for key, cnt in counts.items():
        v = model.index[key]
        x[v] = Fraction(cnt, qn * gam[v])
    return x, M


def check_primal(model, code, report=None):
    """Verify that the code's primal point satisfies every model constraint
    and that the objective equals |C|^3."""
    x, M = code_primal_point(model, code)
    problems = []
    val = sum(Fraction(model.obj[v]) * x[v] for v in range(model.nvars))
    if val != M ** 3:
        problems.append("objective %s != |C|^3 = %d" % (val, M ** 3))
    for v in range(model.nvars):
        if x[v] < 0:
            problems.append("x[%d] < 0" % v)
    for k, (f, c0) in enumerate(model.lin):
        s = Fraction(c0)
        for v, cc in f.items():
            s += cc * x[v]
        if s < 0:
            problems.append("linear constraint %d (%s) violated by %s"
                            % (k, model.lin_tags[k], s))
    for b, X in enumerate(model.psd):
        sz = len(X)
        A = [[Fraction(0)] * sz for _ in range(sz)]
        for r in range(sz):
            for c in range(sz):
                f, c0 = X[r][c]
                s = Fraction(c0)
                for v, cc in f.items():
                    s += cc * x[v]
                A[r][c] = s
        ok, why = is_psd_exact(A)
        if not ok:
            problems.append("psd block %d violated: %s" % (b, why))
    if report is not None:
        report["objective"] = val
        report["M3"] = M ** 3
    return problems


# =========================================================================
# 9. CLI
# =========================================================================


def _selftest():
    from itertools import product as iproduct

    def covering_radius(code, q, n):
        best = 0
        for w in iproduct(range(q), repeat=n):
            dmin = n
            for c in code:
                dd = sum(1 for l in range(n) if c[l] != w[l])
                if dd < dmin:
                    dmin = dd
            if dmin > best:
                best = dmin
        return best

    cases = []
    # whole space, covering radius 0
    cases.append((3, 3, 1, [tuple(w) for w in iproduct(range(3), repeat=3)]))
    cases.append((4, 2, 1, [tuple(w) for w in iproduct(range(4), repeat=2)]))
    # ternary Hamming [4,2,3] code: perfect, covering radius 1
    ham3 = []
    for a in range(3):
        for b in range(3):
            ham3.append((a, b, (a + b) % 3, (a + 2 * b) % 3))
    cases.append((3, 4, 1, ham3))
    cases.append((3, 4, 2, ham3))
    # a small q=5 code
    c5 = [(0, 0, 0), (1, 2, 3), (2, 4, 1), (3, 1, 4), (4, 3, 2)]
    cases.append((5, 3, 2, c5))
    # a small q=6 code
    c6 = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5)]
    cases.append((6, 3, 2, c6))

    # genuine randomly grown covering codes for q >= 6 (the regime we target).
    # Codewords must be DISTINCT: M' is defined for a set, and feeding a
    # multiset in makes the psd constraints fail (a good sanity check in
    # itself).
    import random
    rnd = random.Random(20260820)
    for (q, n, R) in [(6, 4, 1), (6, 4, 2), (6, 5, 3), (7, 4, 2), (8, 4, 2)]:
        code, seen = [], set()
        U = set(iproduct(range(q), repeat=n))
        while U:
            c = (rnd.choice(sorted(U)) if rnd.random() < 0.5
                 else tuple(rnd.randrange(q) for _ in range(n)))
            if c in seen:
                continue
            seen.add(c)
            code.append(c)
            U = {w for w in U
                 if sum(1 for l in range(n) if c[l] != w[l]) > R}
        cases.append((q, n, R, code))

    allok = True
    for (q, n, R, code) in cases:
        cr = covering_radius(code, q, n)
        if cr > R:
            print("  SKIP q=%d n=%d R=%d : code has covering radius %d"
                  % (q, n, R, cr))
            continue
        model = build_model(q, n, R)
        rep = {}
        probs = check_primal(model, code, rep)
        status = "OK " if not probs else "FAIL"
        print("  %s q=%d n=%d R=%d |C|=%d covrad=%d  vars=%d lin=%d psd=%d "
              "obj=%s (|C|^3=%s)"
              % (status, q, n, R, len(code), cr, model.nvars,
                 len(model.lin), len(model.psd), rep.get("objective"),
                 rep.get("M3")))
        for p in probs[:5]:
            print("      %s" % p)
            allok = False
        if probs:
            allok = False
    print("SELFTEST %s" % ("PASSED" if allok else "FAILED"))
    return 0 if allok else 1


def main(argv):
    if len(argv) >= 2 and argv[1] == "--selftest":
        return _selftest()
    if len(argv) >= 5 and argv[1] == "--model":
        q, n, R = int(argv[2]), int(argv[3]), int(argv[4])
        build_model(q, n, R, verbose=True)
        return 0
    if len(argv) < 2:
        print(__doc__)
        return 2
    res = check_certificate_file(argv[1])
    if not res.get("ok"):
        print("CERTIFICATE INVALID")
        for r in res["reasons"]:
            print("  " + r)
        return 1
    p = res["problem"]
    print("CERTIFICATE VALID")
    print("  problem            K_%d(%d,%d)" % (p["q"], p["n"], p["R"]))
    print("  certified SDP bnd  %s / %s" % (res["bound_num"], res["bound_den"]))
    print("                     ~ %.6f  (cube root %.6f)"
          % (res["bound_float"], res["cube_root_float"]))
    print("  => K_%d(%d,%d) >= %d" % (p["q"], p["n"], p["R"],
                                      res["K_lower_bound"]))
    if "claim_matches" in res:
        print("  claim in file      %s"
              % ("matches" if res["claim_matches"] else "DOES NOT MATCH"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
