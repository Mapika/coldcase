#!/usr/bin/env python3
"""
xcheck.py -- diff our exact Python model generator (certify.py) against the
ORIGINAL Gijswijt-Polak Julia code, entry by entry.

    julia gp_dump.jl Q N R dump.txt
    python3 xcheck.py Q N R dump.txt

Every coefficient family used by the SDP is compared:
  * the orbit partition of the triple configurations   (DetermineOrbitNumbersQary)
  * the objective coefficients gamma'                  (gammaprime)
  * the block-diagonalisation coefficients alpha       (alpha / beta)
  * the matrix-cut distribution numbers                (MakeDistrQary)
  * the Lasserre shift expansion                       (the eta loop in CovQary)
"""

import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import certify as ct                                             # noqa: E402


def main():
    q, n, R, path = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), \
        sys.argv[4]
    model = ct.build_model(q, n, R)
    lam = model.lam

    nvars_jl = None
    orb = {}
    gam = {}
    alp = {}
    dis = {}
    las = {}
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if not p:
                continue
            if p[0] == "NVARS":
                nvars_jl = int(p[1])
            elif p[0] == "ORB":
                orb[tuple(int(x) for x in p[1:6])] = int(p[6])
            elif p[0] == "GAM":
                gam[tuple(int(x) for x in p[1:5])] = int(p[5])
            elif p[0] == "ALP":
                alp[tuple(int(x) for x in p[1:7])] = int(p[7])
            elif p[0] == "DIS":
                dis[tuple(int(x) for x in p[1:8])] = int(p[8])
            elif p[0] == "LAS":
                k = (tuple(int(x) for x in p[1:6]),
                     tuple(int(x) for x in p[6:11]))
                las[k] = int(p[11])

    fails = []

    # --- 1. orbit partition -------------------------------------------------
    if nvars_jl != model.nvars:
        fails.append("nvars: julia %s, python %s" % (nvars_jl, model.nvars))
    fwd, back = {}, {}
    for c5, num in orb.items():
        mine = model.index[ct.orbit_key(*ct.ijtp_of(c5))]
        if num in fwd and fwd[num] != mine:
            fails.append("orbit %s: julia class %d maps to python %d and %d"
                         % (c5, num, fwd[num], mine))
        if mine in back and back[mine] != num:
            fails.append("orbit %s: python class %d maps to julia %d and %d"
                         % (c5, mine, back[mine], num))
        fwd[num] = mine
        back[mine] = num
    n_orb = len(orb)

    # --- 2. objective coefficients -----------------------------------------
    n_gam = 0
    for (i2, i3, i4, i5), v in gam.items():
        i1 = n - i2 - i3 - i4 - i5
        i, j, t, p = ct.ijtp_of((i1, i2, i3, i4, i5))
        mine = ((q - 1) ** (i + j - t) * (q - 2) ** (t - p)
                * ct.multinom(i1, i2, i3, i4, i5))
        n_gam += 1
        if mine != v:
            fails.append("gamma%s: julia %s python %s"
                         % ((i2, i3, i4, i5), v, mine))

    # --- 3. alpha -----------------------------------------------------------
    n_alp = 0
    for (i, j, t, p, a, k), v in alp.items():
        mine = ct.alpha(n, i, j, t, p, a, k, q)
        n_alp += 1
        if mine != v:
            fails.append("alpha(n=%d,i=%d,j=%d,t=%d,p=%d,a=%d,k=%d): "
                         "julia %s python %s" % (n, i, j, t, p, a, k, v, mine))

    # --- 4. matrix-cut distribution ----------------------------------------
    bytuple = {}
    for (i, j, t, p, j2, t2, p2), v in dis.items():
        bytuple.setdefault((i, j, t, p), {})[(j2, t2, p2)] = v
    n_dis = 0
    for key, want in bytuple.items():
        got = ct.makedistr(q, n, key[0], key[1], key[2], key[3], lam)
        got = {kk: vv for kk, vv in got.items() if vv}
        want = {kk: vv for kk, vv in want.items() if vv}
        n_dis += len(want)
        if got != want:
            only_j = {kk: want[kk] for kk in want if got.get(kk) != want[kk]}
            fails.append("makedistr%s mismatch on %d entries, e.g. %s"
                         % (key, len(only_j), list(only_j.items())[:3]))

    # --- 5. Lasserre shift --------------------------------------------------
    by5 = {}
    for (src, dst), v in las.items():
        by5.setdefault(src, {})
        kk = ct.orbit_key(*ct.ijtp_of(dst))
        by5[src][kk] = by5[src].get(kk, 0) + v
    n_las = 0
    for src, want in by5.items():
        got = ct.lasserre_triple(q, n, lam, src)
        got = {kk: vv for kk, vv in got.items() if vv}
        want = {kk: vv for kk, vv in want.items() if vv}
        n_las += len(want)
        if got != want:
            fails.append("lasserre%s mismatch: julia %d keys, python %d keys"
                         % (src, len(want), len(got)))

    print("cross-check q=%d n=%d R=%d against CoveringCodes/Julia" % (q, n, R))
    print("  orbit classes compared : %d   (nvars julia %s, python %d)"
          % (n_orb, nvars_jl, model.nvars))
    print("  gamma' values compared : %d" % n_gam)
    print("  alpha values compared  : %d" % n_alp)
    print("  MakeDistr entries      : %d" % n_dis)
    print("  Lasserre shift entries : %d" % n_las)
    if fails:
        print("  MISMATCHES: %d" % len(fails))
        for f in fails[:20]:
            print("    " + f)
        return 1
    print("  ALL MATCH")
    return 0


if __name__ == "__main__":
    sys.exit(main())
