#!/usr/bin/env python3
"""
solve_ipm.py -- interior-point (cvxopt) dual point for the Gijswijt-Polak
covering-code SDP, rounded to an exact rational certificate.

Same contract as solve.py: floating point is used only to *find* a dual point;
the emitted certificate is verified with exact integer arithmetic by
certify.py.

Usage:
    python3 solve_ipm.py Q N R [--out cert.json] [--bits 46]
"""

import sys
import os
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as ct                                              # noqa: E402
from solve import (compute_scalings, pow2, make_certificate)      # noqa: E402


def build_cvxopt(model, svar, srow, sblk, gobj=0):
    """cvxopt.solvers.sdp form:

        minimise   c'x
        s.t.       Gl x + sl = hl,        sl >= 0
                   Gs_k x + ss_k = hs_k,  ss_k psd   (column-stacked)

    Our constraints:  <l_k,x> + c0_k >= 0    ->  -l_k' x <= c0_k
                      C_b + sum x_v A_b^v psd -> sum x_v (-A_b^v) <= C_b
    """
    from cvxopt import matrix, spmatrix
    nv = model.nvars
    c = matrix([float(model.obj[v]) * pow2(svar[v] + gobj)
                for v in range(nv)])

    rows, cols, vals, hl = [], [], [], []
    nrow = 0
    for v in range(nv):                      # x_v >= 0
        rows.append(nrow); cols.append(v); vals.append(-1.0)
        hl.append(0.0); nrow += 1
    for k, (f, c0) in enumerate(model.lin):
        s = pow2(srow[k])
        for v, cc in f.items():
            rows.append(nrow); cols.append(v)
            vals.append(-float(cc) * pow2(svar[v]) * s)
        hl.append(float(c0) * s); nrow += 1
    Gl = spmatrix(vals, rows, cols, (nrow, nv))
    hlm = matrix(hl)

    Gs, hs, blk_sizes = [], [], []
    for b, X in enumerate(model.psd):
        sz = len(X)
        blk_sizes.append(sz)
        D = [pow2(e) for e in sblk[b]]
        G = np.zeros((sz * sz, nv))
        H = np.zeros((sz, sz))
        for r in range(sz):
            for cc_ in range(sz):
                f, c0 = X[r][cc_]
                sc = D[r] * D[cc_]
                H[r, cc_] = float(c0) * sc
                for v, coef in f.items():
                    G[cc_ * sz + r, v] = -float(coef) * pow2(svar[v]) * sc
        Gs.append(matrix(G))
        hs.append(matrix(H))
    return c, Gl, hlm, Gs, hs, blk_sizes


TOL_LADDER = [(1e-10, 1e-8), (1e-9, 1e-7), (1e-8, 1e-6), (1e-7, 1e-5),
              (1e-6, 1e-4), (1e-5, 1e-3), (1e-4, 1e-2)]


def run(q, n, R, out=None, bits=46, verbose=True, lam=None, beta=1,
        maxiters=400, tols=None, tag="", model=None, scaling="both"):
    from cvxopt import solvers
    t0 = time.time()
    if model is None:
        model = ct.build_model(q, n, R, lam=lam, beta=beta, verbose=verbose)
    t_model = time.time() - t0
    if verbose:
        sys.stderr.write("  model built in %.1f s\n" % t_model)

    solvers.options["show_progress"] = False
    solvers.options["maxiters"] = maxiters
    solvers.options["abstol"] = 1e-30

    modes = ([scaling] if scaling != "both"
              else ["geo", "rho2", "rho", "obj"])
    best = None
    t_solve = 0.0
    for mode in modes:
      svar, srow, sblk, gobj = compute_scalings(model, mode=mode)
      c, Gl, hl, Gs, hs, blk_sizes = build_cvxopt(model, svar, srow, sblk, gobj)
      for (reltol, feastol) in (tols or TOL_LADDER):
          solvers.options["reltol"] = reltol
          solvers.options["feastol"] = feastol
          t1 = time.time()
          try:
              sol = solvers.sdp(c, Gl=Gl, hl=hl, Gs=Gs, hs=hs)
          except Exception as exc:                              # noqa: BLE001
              sys.stderr.write("  cvxopt failed at reltol=%g: %s\n"
                               % (reltol, exc))
              continue
          t_solve += time.time() - t1
          if verbose:
              sys.stderr.write(
                  "  reltol=%.0e feastol=%.0e -> status=%s pobj=%s dobj=%s\n"
                  % (reltol, feastol, sol["status"],
                     sol["primal objective"], sol["dual objective"]))
          zl = np.array(sol["zl"]).ravel()
          nvv = model.nvars
          y_lin = [float(zl[nvv + k]) * pow2(srow[k] - gobj)
                   for k in range(len(model.lin))]
          Y_blocks = []
          for b, sz in enumerate(blk_sizes):
              Z = np.array(sol["zs"][b]).reshape((sz, sz), order="F")
              Dg = np.array([pow2(e) for e in sblk[b]])
              Y_blocks.append(pow2(-gobj) * Dg[:, None]
                              * (0.5 * (Z + Z.T)) * Dg[None, :])
          made = make_certificate(model, y_lin, Y_blocks, bits=bits,
                                  verbose=False)
          if made is None:
              continue
          D, a, Bs, res = made
          if not res.get("ok"):
              continue
          if verbose:
              sys.stderr.write("     certified %.8g  (K >= %d)\n"
                               % (res["bound_float"], res["K_lower_bound"]))
          if best is None or res["bound_num"] * best[3]["bound_den"] > \
                  best[3]["bound_num"] * res["bound_den"]:
              best = (D, a, Bs, res, sol["status"], sol["dual objective"])
    if best is None:
        sys.stderr.write("  no valid certificate produced\n")
        return None
    D, a, Bs, res, status, dobj = best
    sol = {"status": status, "dual objective": dobj}

    cert = {
        "problem": {"q": q, "n": n, "R": R, "lambda": model.lam,
                    "beta": model.beta},
        "den": str(D),
        "dual_lin": [str(x) for x in a],
        "dual_psd": [[[str(x) for x in row] for row in B] for B in Bs],
        "claim": {"K_lower_bound": res["K_lower_bound"],
                  "sdp_bound_num": str(res["bound_num"]),
                  "sdp_bound_den": str(res["bound_den"])},
        "provenance": {
            "generator": "cov/lb/solve_ipm.py (cvxopt IPM dual, rounded"
                         " + theta-scaled)",
            "cvxopt_status": sol["status"],
            "cvxopt_dobj_scaled": (float(sol["dual objective"])
                                   if sol["dual objective"] is not None
                                   else None),
            "seconds_model": t_model, "seconds_solve": t_solve, "tag": tag,
        },
    }
    if out:
        with open(out, "w") as fh:
            json.dump(cert, fh)
        if verbose:
            sys.stderr.write("  wrote %s (%.2f MB)\n"
                             % (out, os.path.getsize(out) / 1e6))
    print("q=%d n=%d R=%d  certified SDP value %.6f  cube root %.6f  "
          "=> K >= %d" % (q, n, R, res["bound_float"],
                          res["cube_root_float"], res["K_lower_bound"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("q", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("R", type=int)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bits", type=int, default=46)
    ap.add_argument("--maxiters", type=int, default=400)
    ap.add_argument("--scaling", default="both")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    r = run(a.q, a.n, a.R, out=a.out, bits=a.bits, maxiters=a.maxiters,
            scaling=a.scaling, verbose=not a.quiet)
    return 0 if r else 1


if __name__ == "__main__":
    sys.exit(main())
