#!/usr/bin/env python3
"""
solve.py -- find a good dual point for the Gijswijt-Polak covering-code SDP
(built exactly by certify.build_model), round it to rationals, repair it to
exact dual feasibility, and emit a certificate that certify.py can verify.

The floating point solver is used ONLY to find a good dual point.  Nothing it
produces is trusted: the emitted certificate is checked with exact integer
arithmetic by certify.py.

Usage:
    python3 solve.py Q N R [--out cert.json] [--eps 1e-12] [--iters 400000]
                          [--bits 40] [--backend scs|admm]
"""

import sys
import os
import json
import time
import argparse
from fractions import Fraction

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify as ct                                              # noqa: E402
from certify import C                                             # noqa: E402


# ---------------------------------------------------------------------------
# power-of-two scalings (exact in binary floating point and in rationals)
# ---------------------------------------------------------------------------

def pow2(e):
    return float(2.0 ** e)


def exp2_round(x):
    """Nearest power of two exponent of |x| (0 for x == 0)."""
    if x == 0:
        return 0
    return int(round(np.log2(abs(float(x)))))


def rho_exponents(model):
    """Expected order of magnitude of each variable.

    M'_{u,v} ~ rho^3 for a generic pair, rho^2 when two of {0,u,v} coincide,
    and rho for u=v=0, where rho = |C|/q^n.  Using rho = (sphere covering
    bound)/q^n gives an excellent a-priori diagonal scaling.
    """
    q, n, R = model.q, model.n, model.R
    V = sum(C(n, i) * (q - 1) ** i for i in range(R + 1))
    rho = float(-(-(q ** n) // V)) / float(q ** n)
    out = []
    for key in model.keys:
        (d1, d2, d3), _i5 = key
        zeros = (d1 == 0) + (d2 == 0) + (d3 == 0)
        e = 3 if zeros == 0 else (2 if zeros == 1 else 1)
        out.append(rho ** e)
    return out


def compute_scalings(model, mode="rho"):
    """Return (svar_exp, srow_exp, sblk_exp, gobj_exp).

    x_v          = 2^svar_exp[v]  * z_v
    scaled row k = 2^srow_exp[k]  * row k
    scaled block = D A D,  D = diag(2^sblk_exp[b][r])
    objective    = 2^gobj_exp * (true objective)

    `gobj_exp` matters: the SDP optimum is |C|^3, which for the interesting
    cells is 1e8..1e13, while every constraint constant is O(1).  A homogeneous
    self-dual interior point method needs c and b of comparable size, otherwise
    the dual iterate has to carry the whole 1e13 and the method breaks down
    (cvxopt then reports a spurious "primal infeasible").  Normalising the
    objective by (sphere covering bound)^3 fixes exactly that.
    """
    nv = model.nvars
    gobj = 0
    q, n, R = model.q, model.n, model.R
    Vb = sum(C(n, i) * (q - 1) ** i for i in range(R + 1))
    Mest = float(-(-(q ** n) // Vb))
    if mode == "geo":
        # geometric-mean equilibration: make the scaled objective vector and
        # the scaled primal variables have the SAME magnitude profile
        # sqrt(contribution_v)/M^{3/2}.  Without this the objective
        # coefficients span ~1e10 and the interior point solver's dual
        # residual, which is measured relative to ||c||, is meaningless for
        # the variables with small c_v -- which is exactly what destroyed the
        # rounded certificate (theta ~ 0.45) on the larger cells.
        mag = rho_exponents(model)
        svar = []
        for v in range(nv):
            ov = float(model.obj[v]) if model.obj[v] else 1.0
            svar.append(exp2_round((mag[v] / ov) ** 0.5 * Mest ** 1.5))
        gobj = -exp2_round(Mest ** 3)
    elif mode in ("rho", "rho2"):
        mag = rho_exponents(model)
        svar = [exp2_round(mag[v]) for v in range(nv)]
        if mode == "rho2":
            gobj = -exp2_round(Mest ** 3)
    else:
        # make objective coefficients ~ 1
        svar = [-exp2_round(model.obj[v]) if model.obj[v] else 0
                for v in range(nv)]

    # linear rows: make the largest scaled entry ~ 1
    srow = []
    for (f, c0) in model.lin:
        m = 0.0
        for v, cc in f.items():
            m = max(m, abs(float(cc)) * pow2(svar[v]))
        if c0:
            m = max(m, abs(float(c0)))
        srow.append(-exp2_round(m) if m > 0 else 0)

    # psd blocks: symmetric diagonal equilibration on the magnitude matrix
    sblk = []
    for X in model.psd:
        sz = len(X)
        G = np.zeros((sz, sz))
        for r in range(sz):
            for c in range(sz):
                f, c0 = X[r][c]
                m = abs(float(c0))
                for v, cc in f.items():
                    m = max(m, abs(float(cc)) * pow2(svar[v]))
                G[r, c] = m
        # want f_r + f_c + log2 G_rc ~ 0; damped Jacobi on the symmetric system
        L = np.where(G > 0, np.log2(np.maximum(G, 1e-300)), np.nan)
        fvec = np.zeros(sz)
        for _ in range(60):
            newf = np.zeros(sz)
            for r in range(sz):
                row = L[r] + fvec
                row = row[~np.isnan(row)]
                newf[r] = -np.mean(row) if len(row) else 0.0
            fvec = 0.5 * fvec + 0.5 * newf
        sblk.append([int(round(v)) for v in fvec])
    return svar, srow, sblk, gobj


# ---------------------------------------------------------------------------
# conic form for SCS
# ---------------------------------------------------------------------------

SQ2 = np.sqrt(2.0)


def psd_vec_len(k):
    return k * (k + 1) // 2


def build_conic(model, svar, srow, sblk, gobj=0):
    """Return (A, b, c, cone, meta) for SCS in the *scaled* variables z."""
    nv = model.nvars
    c = np.array([float(model.obj[v]) * pow2(svar[v] + gobj)
                  for v in range(nv)])

    rows, cols, vals = [], [], []
    bvals = []
    nrow = 0

    # --- z >= 0 -----------------------------------------------------------
    for v in range(nv):
        rows.append(nrow)
        cols.append(v)
        vals.append(-1.0)
        bvals.append(0.0)
        nrow += 1
    n_nonneg_var = nv

    # --- linear model constraints ----------------------------------------
    for k, (f, c0) in enumerate(model.lin):
        s = pow2(srow[k])
        for v, cc in f.items():
            rows.append(nrow)
            cols.append(v)
            vals.append(-float(cc) * pow2(svar[v]) * s)
        bvals.append(float(c0) * s)
        nrow += 1
    n_lin = len(model.lin)

    # --- psd blocks -------------------------------------------------------
    blk_sizes = []
    for b, X in enumerate(model.psd):
        sz = len(X)
        blk_sizes.append(sz)
        D = [pow2(e) for e in sblk[b]]
        # SCS lower-triangular column-major scaled vectorisation
        for cc_ in range(sz):
            for rr in range(cc_, sz):
                mult = 1.0 if rr == cc_ else SQ2
                f, c0 = X[rr][cc_]
                sc = D[rr] * D[cc_] * mult
                for v, coef in f.items():
                    rows.append(nrow)
                    cols.append(v)
                    vals.append(-float(coef) * pow2(svar[v]) * sc)
                bvals.append(float(c0) * sc)
                nrow += 1

    A = sp.csc_matrix((vals, (rows, cols)), shape=(nrow, nv))
    b = np.array(bvals)
    cone = {"z": 0, "l": n_nonneg_var + n_lin, "s": blk_sizes}
    meta = {"n_nonneg_var": n_nonneg_var, "n_lin": n_lin,
            "blk_sizes": blk_sizes}
    return A, b, c, cone, meta


def unvec_psd(vecs, sz):
    """Inverse of the SCS scaled lower-triangular vectorisation."""
    Y = np.zeros((sz, sz))
    idx = 0
    for c in range(sz):
        for r in range(c, sz):
            v = vecs[idx]
            idx += 1
            if r == c:
                Y[r, r] = v
            else:
                Y[r, c] = v / SQ2
                Y[c, r] = v / SQ2
    return Y


# ---------------------------------------------------------------------------
# rounding to an exact certificate
# ---------------------------------------------------------------------------

def make_certificate(model, y_lin, Y_blocks, bits=40, eps_list=None,
                     verbose=True):
    """y_lin: list of floats >= 0 (already in *unscaled* model coordinates)
       Y_blocks: list of numpy symmetric matrices (unscaled coordinates)

    Returns (den, dual_lin_int, dual_psd_int, result_dict) or None.
    """
    if eps_list is None:
        eps_list = [1e-9, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2]
    D = 1 << bits

    a = [max(0, int(round(v * D))) for v in y_lin]

    Bs = None
    for eps in eps_list:
        Bs = []
        ok = True
        for Y in Y_blocks:
            sz = Y.shape[0]
            W = 0.5 * (Y + Y.T)
            w, V = np.linalg.eigh(W)
            scale = max(np.max(np.abs(w)), 1e-300)
            w = np.clip(w, 0.0, None) + eps * scale
            W = (V * w) @ V.T
            W = 0.5 * (W + W.T)
            B = np.rint(W * D).astype(object)
            B = [[int(B[r][c]) for c in range(sz)] for r in range(sz)]
            for r in range(sz):
                for c in range(r):
                    B[r][c] = B[c][r]
            good, _ = ct.is_psd_exact(B)
            if not good:
                ok = False
                break
            Bs.append(B)
        if ok:
            if verbose:
                sys.stderr.write("  psd rounding succeeded with eps=%g\n" % eps)
            break
        Bs = None
    if Bs is None:
        sys.stderr.write("  FAILED to round psd duals to exact psd matrices\n")
        return None

    # exact residual computation
    d, d0 = accumulate_dual(model, a, Bs)

    # theta-scaling to restore d_v <= c_v * D exactly
    theta = Fraction(1)
    for v in range(model.nvars):
        cv = model.obj[v] * D
        if d[v] > cv:
            theta = min(theta, Fraction(cv, d[v]))
    if theta < 1:
        # dyadic under-approximation, keeps everything integral
        SH = 1 << 40
        num = (theta.numerator * SH) // theta.denominator
        if num <= 0:
            sys.stderr.write("  theta collapsed to 0; dual point unusable\n")
            return None
        a = [x * num for x in a]
        Bs = [[[x * num for x in row] for row in B] for B in Bs]
        D = D * SH
        d = [x * num for x in d]
        d0 = d0 * num
        if verbose:
            sys.stderr.write("  theta = %.15f\n" % float(Fraction(num, SH)))

    res = ct.evaluate_certificate(model, D, a, Bs)
    return D, a, Bs, res


def accumulate_dual(model, a, Bs):
    d = [0] * model.nvars
    d0 = 0
    for k, (f, c0) in enumerate(model.lin):
        y = a[k]
        if y == 0:
            continue
        for v, cc in f.items():
            d[v] += y * cc
        d0 += y * c0
    for b, X in enumerate(model.psd):
        B = Bs[b]
        sz = len(X)
        for r in range(sz):
            Xr = X[r]
            Br = B[r]
            for c in range(sz):
                yv = Br[c]
                if yv == 0:
                    continue
                f, c0 = Xr[c]
                for v, cc in f.items():
                    d[v] += yv * cc
                d0 += yv * c0
    return d, d0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(q, n, R, out=None, eps=1e-11, iters=200000, bits=40, verbose=True,
        lam=None, beta=1, tag="", scaling="both", model=None):
    t0 = time.time()
    if model is None:
        model = ct.build_model(q, n, R, lam=lam, beta=beta, verbose=verbose)
    t_model = time.time() - t0
    if verbose:
        sys.stderr.write("  model built in %.1f s\n" % t_model)

    import scs
    modes = ([scaling] if scaling != "both"
             else ["geo", "rho2", "obj", "rho"])
    best = None
    t_solve = 0.0
    info = {"status": "n/a", "dobj": None}
    for mode in modes:
        svar, srow, sblk, gobj = compute_scalings(model, mode=mode)
        A, b, c, cone, meta = build_conic(model, svar, srow, sblk, gobj)
        t1 = time.time()
        solver = scs.SCS({"A": A, "b": b, "c": c}, cone,
                         eps_abs=eps, eps_rel=eps, max_iters=iters,
                         verbose=False, acceleration_lookback=10)
        sol = solver.solve()
        t_solve += time.time() - t1
        inf2 = sol["info"]
        if verbose:
            sys.stderr.write("  scs[%s] status=%s pobj=%.10g dobj=%.10g "
                             "(%.1f s)\n" % (mode, inf2["status"],
                                             inf2["pobj"], inf2["dobj"],
                                             time.time() - t1))
        y = sol["y"]
        nvv = meta["n_nonneg_var"]
        nlin = meta["n_lin"]
        y_lin_scaled = y[nvv:nvv + nlin]
        off = nvv + nlin
        Y_blocks = []
        for bi, sz in enumerate(meta["blk_sizes"]):
            L = psd_vec_len(sz)
            Yb = unvec_psd(y[off:off + L], sz)
            off += L
            Dg = np.array([pow2(e) for e in sblk[bi]])
            Y_blocks.append(pow2(-gobj) * Dg[:, None] * Yb * Dg[None, :])
        y_lin = [float(y_lin_scaled[k]) * pow2(srow[k] - gobj)
                 for k in range(nlin)]
        made = make_certificate(model, y_lin, Y_blocks, bits=bits,
                                verbose=False)
        if made is None:
            continue
        D, a, Bs, res = made
        if not res.get("ok"):
            continue
        if verbose:
            sys.stderr.write("     certified %.8g (K >= %d)\n"
                             % (res["bound_float"], res["K_lower_bound"]))
        if best is None or (res["bound_num"] * best[3]["bound_den"]
                            > best[3]["bound_num"] * res["bound_den"]):
            best = (D, a, Bs, res)
            info = inf2
    if best is None:
        sys.stderr.write("  no valid certificate produced\n")
        return None
    D, a, Bs, res = best

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
            "generator": "cov/lb/solve.py (SCS dual, rounded + theta-scaled)",
            "scs_status": info["status"],
            "scs_dobj_scaled": (float(info["dobj"])
                                if info["dobj"] is not None else None),
            "seconds_model": t_model, "seconds_solve": t_solve,
            "tag": tag,
        },
    }
    if out:
        with open(out, "w") as fh:
            json.dump(cert, fh)
        if verbose:
            sys.stderr.write("  wrote %s (%.1f MB)\n"
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
    ap.add_argument("--eps", type=float, default=1e-11)
    ap.add_argument("--iters", type=int, default=200000)
    ap.add_argument("--bits", type=int, default=40)
    ap.add_argument("--scaling", default="both")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    r = run(a.q, a.n, a.R, out=a.out, eps=a.eps, iters=a.iters, bits=a.bits,
            scaling=a.scaling, verbose=not a.quiet)
    return 0 if r else 1


if __name__ == "__main__":
    sys.exit(main())
