#!/usr/bin/env python3
"""
solve_hp.py -- HIGH-PRECISION dual points for the Gijswijt-Polak covering-code
SDP via a native aarch64 build of SDPA-GMP, rounded to exact rational
certificates that the frozen checker cov/lb/certify.py verifies.

Same contract as solve.py / solve_ipm.py: the multiprecision solver is used
ONLY to find a dual point.  Nothing it produces is trusted; the emitted
certificate is checked with exact integer arithmetic by certify.py.

Pipeline per cell:
  1. build the exact integer model (certify.build_model);
  2. apply the power-of-two geometric-mean equilibration of solve.py Section
     3.5 (exact both in floats and rationals);
  3. write the scaled SDP in SDPA sparse format with EXACT decimal strings
     (every coefficient is an integer times a power of two, which has a
     finite decimal expansion), so the solver sees the model exactly;
  4. run sdpa_gmp (GMP arithmetic, `precision` bits, 40-digit output);
  5. parse the dual matrix yMat back into exact Fractions, undo the scalings
     (powers of two, exact), round over den = 2^bits with a tiny multiple of
     the identity added to each block to survive rounding,
  6. exact feasibility repair (theta-shrink) and evaluation via
     certify.evaluate_certificate, then write the certificate JSON.

Usage:
    python3 solve_hp.py Q N R [--out cert.json] [--bits 64] [--prec 250]
                             [--maxit 300] [--threads 4] [--eps 1e-30]
                             [--workdir DIR] [--keep]
"""

import sys
import os
import re
import json
import time
import argparse
import subprocess
import pickle
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import certify as ct                                              # noqa: E402
from solve import compute_scalings, accumulate_dual               # noqa: E402

_SDPA_CANDIDATES = [
    os.path.join(HERE, "tools", "sdpa-gmp-aarch64", "sdpa_gmp"),
    "/tmp/claude-1000/-lambda-nfs-new-fs-longshots/"
    "3ae6d111-07fa-4880-8ce4-cca841595d21/scratchpad/sdpa-gmp/sdpa_gmp",
]
SDPA_GMP = os.environ.get(
    "SDPA_GMP",
    next((p for p in _SDPA_CANDIDATES if os.path.exists(p)),
         _SDPA_CANDIDATES[0]))

DEFAULT_WORKDIR = os.path.join(
    "/tmp/claude-1000/-lambda-nfs-new-fs-longshots/"
    "3ae6d111-07fa-4880-8ce4-cca841595d21/scratchpad", "hp_runs")

MODEL_CACHE = os.path.join(
    "/tmp/claude-1000/-lambda-nfs-new-fs-longshots/"
    "3ae6d111-07fa-4880-8ce4-cca841595d21/scratchpad", "model_cache")


# ---------------------------------------------------------------------------
# exact decimal strings for integer * 2^e
# ---------------------------------------------------------------------------

def dec(cint, e):
    """Exact decimal string of cint * 2**e."""
    if cint == 0:
        return "0"
    if e >= 0:
        return str(cint << e)
    m = -e
    num = cint * 5 ** m
    sgn = "-" if num < 0 else ""
    a = str(abs(num)).rjust(m + 1, "0")
    return sgn + a[:-m] + "." + a[-m:]


# ---------------------------------------------------------------------------
# model cache (build_model is deterministic; n=10 builds take minutes)
# ---------------------------------------------------------------------------

def get_model(q, n, R, lam=None, beta=1, verbose=True):
    os.makedirs(MODEL_CACHE, exist_ok=True)
    key = "m_q%d_n%d_R%d_%s_%s.pkl" % (
        q, n, R, "-".join(map(str, lam)) if lam else "def", beta)
    path = os.path.join(MODEL_CACHE, key)
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    model = ct.build_model(q, n, R, lam=lam, beta=beta, verbose=verbose)
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "wb") as fh:
        pickle.dump(model, fh, protocol=4)
    os.replace(tmp, path)
    return model


# ---------------------------------------------------------------------------
# SDPA sparse format writer
# ---------------------------------------------------------------------------

def write_dats(model, svar, srow, sblk, gobj, path):
    """Layout: block 1 = diag(nvars) for x >= 0, block 2 = diag(nlin) for the
    linear rows, blocks 3.. = the psd blocks.  Variables are the scaled z_v
    (x_v = 2^svar[v] z_v)."""
    nv = model.nvars
    nlin = len(model.lin)
    out = []
    out.append("%d = mDIM" % nv)
    out.append("%d = nBLOCK" % (2 + len(model.psd)))
    bstruct = ["-%d" % nv, "-%d" % nlin] + [str(len(X)) for X in model.psd]
    out.append(" ".join(bstruct))
    out.append(" ".join(dec(model.obj[v], svar[v] + gobj) for v in range(nv)))

    # F_v block 1: identity entry (v,v) -- z_v >= 0
    for v in range(nv):
        out.append("%d 1 %d %d 1" % (v + 1, v + 1, v + 1))
    # block 2: linear rows
    for k, (f, c0) in enumerate(model.lin):
        if c0:
            out.append("0 2 %d %d %s" % (k + 1, k + 1, dec(-c0, srow[k])))
        for v, cc in sorted(f.items()):
            out.append("%d 2 %d %d %s"
                       % (v + 1, k + 1, k + 1, dec(cc, svar[v] + srow[k])))
    # psd blocks
    for b, X in enumerate(model.psd):
        sz = len(X)
        sb = sblk[b]
        bn = b + 3
        for r in range(sz):
            for c in range(r, sz):
                f, c0 = X[r][c]
                e = sb[r] + sb[c]
                if c0:
                    out.append("0 %d %d %d %s"
                               % (bn, r + 1, c + 1, dec(-c0, e)))
                for v, cc in sorted(f.items()):
                    out.append("%d %d %d %d %s"
                               % (v + 1, bn, r + 1, c + 1, dec(cc, svar[v] + e)))
    with open(path, "w") as fh:
        fh.write("\n".join(out))
        fh.write("\n")


def write_param(path, prec=250, maxit=300, eps="1.0E-30", lstar="1.0E4",
                ostar="2.0"):
    body = """%d	unsigned int maxIteration;
%s	double 0.0 < epsilonStar;
%s   double 0.0 < lambdaStar;
%s   	double 1.0 < omegaStar;
-1.0E7  double lowerBound;
1.0E7   double upperBound;
0.1     double 0.0 <= betaStar <  1.0;
0.3     double 0.0 <= betaBar  <  1.0, betaStar <= betaBar;
0.9     double 0.0 < gammaStar  <  1.0;
%s	double 0.0 < epsilonDash;
%d     precision
NOPRINT     char*  xPrint   (default %%+8.3e,   NOPRINT skips printout)
NOPRINT     char*  XPrint   (default %%+8.3e,   NOPRINT skips printout)
%%+50.40Fe     char*  YPrint   (default %%+8.3e,   NOPRINT skips printout)
%%+50.40Fe     char*  infPrint (default %%+10.16e, NOPRINT skips printout)
""" % (maxit, eps, lstar, ostar, eps, prec)
    with open(path, "w") as fh:
        fh.write(body)


# ---------------------------------------------------------------------------
# result parsing
# ---------------------------------------------------------------------------

_NUM = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def frac_of_decimal(s):
    """Exact Fraction of a decimal string like +1.234e-05."""
    return Fraction(s)


def parse_result(path, model):
    """Return (phase, objd_float, y_lin, Y_blocks) with y/Y as exact Fractions
    in UNSCALED model coordinates -- scalings must be undone by the caller."""
    with open(path) as fh:
        txt = fh.read()
    phase = None
    m = re.search(r"phase\.value\s*=\s*(\S+)", txt)
    if m:
        phase = m.group(1)
    objd = None
    m = re.search(r"objValDual\s*=\s*(\S+)", txt)
    if m:
        objd = float(m.group(1))
    i = txt.find("yMat =")
    if i < 0:
        return phase, objd, None, None
    j = txt.find("main loop time", i)
    seg = txt[i + 6: j if j > 0 else len(txt)]
    vals = _NUM.findall(seg)
    nv = model.nvars
    nlin = len(model.lin)
    need = nv + nlin + sum(len(X) ** 2 for X in model.psd)
    if len(vals) != need:
        raise ValueError("yMat parse: got %d values, expected %d"
                         % (len(vals), need))
    it = iter(vals)
    u = [frac_of_decimal(next(it)) for _ in range(nv)]      # noqa: F841
    y_lin = [frac_of_decimal(next(it)) for _ in range(nlin)]
    Y_blocks = []
    for X in model.psd:
        sz = len(X)
        B = [[frac_of_decimal(next(it)) for _ in range(sz)]
             for _ in range(sz)]
        Y_blocks.append(B)
    return phase, objd, y_lin, Y_blocks


# ---------------------------------------------------------------------------
# exact rounding of a high-precision dual to a certificate
# ---------------------------------------------------------------------------

def eps_ladder(bits):
    """Diagonal-shift ladder for psd rounding.  The shift must dominate the
    2^-bits rounding noise but stay far below it in absolute effect on d_v:
    start a few bits above the rounding grain and only escalate on failure."""
    exps = [bits - 10, bits - 20, bits - 34, bits - 50, 58, 50, 42, 34, 26,
            18, 12]
    out, seen = [], set()
    for e in exps:
        if e >= 8 and e not in seen:
            seen.add(e)
            out.append(Fraction(1, 1 << e))
    return sorted(out)


def make_certificate_hp(model, y_sc, Y_sc, srow, sblk, gobj, bits=64,
                        verbose=True):
    """Round the SCALED dual (y_sc, Y_sc: Fractions, equilibrated
    coordinates where every block has O(1)-ish dynamic range) over 2^bits,
    then unscale EXACTLY by absorbing the power-of-two exponents into a
    common denominator.  Rounding noise is then congruence-aligned with the
    block equilibration, so psd-ness survives.

    y_k  = y_sc[k] * 2^(srow[k]-gobj)
    Y_b  = D_b Ysc_b D_b * 2^(-gobj),  D_b = diag(2^sblk[b][r])

    Returns (den, a, Bs, res) or None."""
    nlin = len(model.lin)
    # global power-of-two shift so that every scaled integer, once shifted
    # by its exponent, is an integer over den = 2^(bits+S)
    exps_lin = [srow[k] - gobj for k in range(nlin)]
    exps_blk = [[[sblk[b][r] + sblk[b][c] - gobj
                  for c in range(len(Y))] for r in range(len(Y))]
                for b, Y in enumerate(Y_sc)]
    mine = 0
    if exps_lin:
        mine = min(mine, min(exps_lin))
    for eb in exps_blk:
        for row in eb:
            mine = min(mine, min(row))
    S = -mine
    D = 1 << (bits + S)
    Dl = 1 << bits

    a = [max(0, int(y_sc[k] * Dl)) << (exps_lin[k] + S) for k in range(nlin)]

    Bs = None
    for eps in eps_ladder(bits):
        Bs = []
        ok = True
        for b, Y in enumerate(Y_sc):
            sz = len(Y)
            dmax = max(max(Y[r][r] for r in range(sz)), Fraction(0))
            shift = eps * dmax
            B = [[0] * sz for _ in range(sz)]
            for r in range(sz):
                for c in range(r, sz):
                    w = (Y[r][c] + Y[c][r]) / 2
                    if r == c:
                        w += shift
                    x = int(w * Dl) << (exps_blk[b][r][c] + S)
                    B[r][c] = x
                    B[c][r] = x
            good, _ = ct.is_psd_exact(B)
            if not good:
                ok = False
                break
            Bs.append(B)
        if ok:
            if verbose:
                sys.stderr.write("  psd rounding ok with eps=2^-%d (S=%d)\n"
                                 % (eps.denominator.bit_length() - 1, S))
            break
        Bs = None
    if Bs is None:
        sys.stderr.write("  FAILED to round psd duals exactly\n")
        return None

    d, d0 = accumulate_dual(model, a, Bs)

    # dyadic theta-shrink to restore d_v <= c_v * D exactly
    theta = Fraction(1)
    for v in range(model.nvars):
        cv = model.obj[v] * D
        if d[v] > cv:
            theta = min(theta, Fraction(cv, d[v]))
    theta_f = float(theta)

    cand = []
    if theta < Fraction(999999, 1000000):
        # the global shrink would lose real value: try the smarter repair --
        # keep the rounded psd blocks (up to one nonneg scale per block) and
        # re-optimize every linear multiplier by LP
        rep = lp_repair(model, a, Bs, D, verbose=verbose)
        if rep is not None:
            cand.append(rep)

    SH = 1 << 48
    num = (theta.numerator * SH) // theta.denominator
    if num > 0:
        a1 = [x * num for x in a]
        Bs1 = [[[x * num for x in row] for row in B] for B in Bs]
        cand.append((a1, Bs1, D * SH, theta_f))
    if not cand:
        sys.stderr.write("  theta collapsed to 0 and no LP repair\n")
        return None

    best = None
    for (a2, Bs2, D2, tf) in cand:
        _, d02 = accumulate_dual(model, a2, Bs2)
        val = Fraction(-d02, D2)
        if best is None or val > best[0]:
            best = (val, a2, Bs2, D2, tf)
    _, a, Bs, D, theta_f = best
    if verbose:
        sys.stderr.write("  theta = %.15f  (repair candidates: %d)\n"
                         % (theta_f, len(cand)))

    res = ct.evaluate_certificate(model, D, a, Bs)
    if res.get("ok"):
        res["theta"] = theta_f
    return D, a, Bs, res


def lp_repair(model, a, Bs, D, verbose=True):
    """NOTES.md section 7 'smarter dual repair': FIX the rounded psd blocks
    Y_b = Bs[b]/D up to one nonnegative scale s_b per block, and re-optimize
    all linear multipliers y_k >= 0, maximizing the certified bound subject
    to d_v <= c_v.  This is an LP; solve it in floats (HiGHS), then round
    the solution DOWN dyadically and let the caller's exact arithmetic have
    the final word.  Returns (a2, Bs2, D2, theta_marker) or None."""
    try:
        import numpy as np
        import scipy.sparse as sp
        from scipy.optimize import linprog
    except ImportError:
        return None
    nv, nb, nlin = model.nvars, len(model.psd), len(model.lin)

    # exact per-block dual columns g[b][v] and constants g0[b] (over den D)
    g = [dict() for _ in range(nb)]
    g0 = [0] * nb
    for b, X in enumerate(model.psd):
        B = Bs[b]
        gb = g[b]
        for r in range(len(X)):
            Xr, Br = X[r], B[r]
            for c in range(len(X)):
                yv = Br[c]
                if yv == 0:
                    continue
                f, c0 = Xr[c]
                for v, cc in f.items():
                    gb[v] = gb.get(v, 0) + yv * cc
                g0[b] += yv * c0

    # row scaling by the objective (c_v > 0 for every v when q >= 3)
    re_ = [model.obj[v].bit_length() - 1 for v in range(nv)]

    def p2f(e):
        return float(2.0 ** e)

    # column scales (powers of two)
    cs_blk = []
    for b in range(nb):
        m = 0.0
        for v, gv in g[b].items():
            m = max(m, abs(gv) / D * p2f(-re_[v]))
        cs_blk.append(-(int(m).bit_length() - 1) if m >= 1
                      else (int(1 / m).bit_length() - 1) if m > 0 else 0)
    cs_lin = []
    for k, (f, c0) in enumerate(model.lin):
        m = 0.0
        for v, cc in f.items():
            m = max(m, abs(cc) * p2f(-re_[v]))
        cs_lin.append(-(int(m).bit_length() - 1) if m >= 1
                      else (int(1 / m).bit_length() - 1) if m > 0 else 0)

    rows, cols, vals = [], [], []
    for b in range(nb):
        sc = p2f(cs_blk[b])
        for v, gv in g[b].items():
            rows.append(v)
            cols.append(b)
            vals.append(gv / D * p2f(-re_[v]) * sc)
    for k, (f, c0) in enumerate(model.lin):
        sc = p2f(cs_lin[k])
        for v, cc in f.items():
            rows.append(v)
            cols.append(nb + k)
            vals.append(float(cc) * p2f(-re_[v]) * sc)
    A = sp.csc_matrix((vals, (rows, cols)), shape=(nv, nb + nlin))
    bub = np.array([float(model.obj[v]) * p2f(-re_[v]) for v in range(nv)])

    obj = np.zeros(nb + nlin)
    for b in range(nb):
        obj[b] = -g0[b] / D * p2f(cs_blk[b])
    for k, (f, c0) in enumerate(model.lin):
        obj[nb + k] = -float(c0) * p2f(cs_lin[k])
    onorm = np.max(np.abs(obj))
    if onorm == 0:
        return None
    oscale = p2f(-(int(onorm).bit_length() - 1) if onorm >= 1 else 0)

    try:
        sol = linprog(-obj * oscale, A_ub=A, b_ub=bub, bounds=(0, None),
                      method="highs")
    except Exception as exc:                                     # noqa: BLE001
        sys.stderr.write("  lp_repair: linprog failed: %s\n" % exc)
        return None
    if not sol.success:
        sys.stderr.write("  lp_repair: LP not solved (%s)\n" % sol.message)
        return None
    x = sol.x

    SH = 1 << 48
    # block scales s_b (in true units: s_b = x_b * 2^cs_blk[b])
    Bs2 = []
    for b in range(nb):
        sb = max(0.0, float(x[b]))
        # exact dyadic floor of sb * 2^cs_blk over 2^48
        num = max(0, int(Fraction(sb) * (1 << 48) * Fraction(2) ** cs_blk[b]))
        Bs2.append([[e * num for e in row] for row in Bs[b]])
    a2 = []
    for k in range(nlin):
        yk = max(0.0, float(x[nb + k]))
        num = int(Fraction(yk) * (1 << 48) * Fraction(2) ** cs_lin[k] * D)
        a2.append(max(0, num))
    D2 = D * SH

    # exact residual; mop-up theta for the float crumbs
    d, d0 = accumulate_dual(model, a2, Bs2)
    theta = Fraction(1)
    for v in range(nv):
        cv = model.obj[v] * D2
        if d[v] > cv:
            theta = min(theta, Fraction(cv, d[v]))
    if theta < 1:
        num = (theta.numerator * SH) // theta.denominator
        if num <= 0:
            return None
        a2 = [t * num for t in a2]
        Bs2 = [[[t * num for t in row] for row in B] for B in Bs2]
        D2 = D2 * SH
    if verbose:
        sys.stderr.write("  lp_repair: LP ok, mop-up theta = %.12f\n"
                         % float(theta))
    return a2, Bs2, D2, -2.0


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(q, n, R, out=None, bits=128, prec=250, maxit=300, threads=4,
        eps="1.0E-30", lstar="1.0E4", ostar="2.0", lam=None, beta=1,
        workdir=None, keep=False, reuse=False, verbose=True, scaling="geo",
        tag=""):
    t0 = time.time()
    model = get_model(q, n, R, lam=lam, beta=beta, verbose=verbose)
    t_model = time.time() - t0
    svar, srow, sblk, gobj = compute_scalings(model, mode=scaling)

    wd = workdir or DEFAULT_WORKDIR
    os.makedirs(wd, exist_ok=True)
    base = os.path.join(wd, "q%d_n%d_R%d_p%d%s" % (
        q, n, R, prec, "" if lstar == "1.0E4" else "_l" + lstar))
    dats, parf, resf = base + ".dat-s", base + ".param", base + ".result"

    t1 = time.time()
    write_dats(model, svar, srow, sblk, gobj, dats)
    write_param(parf, prec=prec, maxit=maxit, eps=eps, lstar=lstar,
                ostar=ostar)
    t_write = time.time() - t1
    if verbose:
        sys.stderr.write("  model %.1fs, dat-s %.1fs (%.1f MB)\n"
                         % (t_model, t_write, os.path.getsize(dats) / 1e6))

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    t2 = time.time()
    if reuse and os.path.exists(resf) and os.path.getsize(resf) > 1000:
        proc = None
    else:
        proc = subprocess.run(
            ["nice", "-n", "10", SDPA_GMP, "-ds", dats, "-o", resf,
             "-p", parf],
            capture_output=True, text=True, env=env)
    t_solve = time.time() - t2
    if not os.path.exists(resf):
        sys.stderr.write("  sdpa_gmp produced no result file\n%s\n"
                         % (proc.stdout[-2000:] if proc else ""))
        return None

    phase, objd, y_sc, Y_sc = parse_result(resf, model)
    if verbose:
        sys.stderr.write("  sdpa_gmp: phase=%s objd=%s (%.1f s)\n"
                         % (phase, objd, t_solve))
    if y_sc is None:
        sys.stderr.write("  no yMat in result\n")
        return None

    made = make_certificate_hp(model, y_sc, Y_sc, srow, sblk, gobj,
                               bits=bits, verbose=verbose)
    if not keep:
        try:
            os.remove(dats)          # result file is kept for re-rounding
        except OSError:
            pass
    if made is None:
        return None
    D, a, Bs, res = made
    if not res.get("ok"):
        sys.stderr.write("  certificate evaluation failed: %s\n"
                         % res.get("reasons"))
        return None

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
            "generator": "cov/lb/solve_hp.py (SDPA-GMP %d-bit dual, exact "
                         "rounding + theta)" % prec,
            "sdpa_phase": phase,
            "sdpa_objd_scaled": objd,
            "theta": res.get("theta"),
            "seconds_model": t_model, "seconds_solve": t_solve,
            "scaling": scaling, "bits": bits, "tag": tag,
        },
    }
    if out:
        with open(out, "w") as fh:
            json.dump(cert, fh)
        if verbose:
            sys.stderr.write("  wrote %s (%.2f MB)\n"
                             % (out, os.path.getsize(out) / 1e6))
    print("q=%d n=%d R=%d  certified SDP value %.6f  cube root %.6f  "
          "=> K >= %d   (phase=%s theta=%s)"
          % (q, n, R, res["bound_float"], res["cube_root_float"],
             res["K_lower_bound"], phase, res.get("theta")))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("q", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("R", type=int)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bits", type=int, default=128)
    ap.add_argument("--prec", type=int, default=250)
    ap.add_argument("--maxit", type=int, default=300)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eps", default="1.0E-30")
    ap.add_argument("--lstar", default="1.0E4")
    ap.add_argument("--ostar", default="2.0")
    ap.add_argument("--scaling", default="geo")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="re-round an existing .result without re-solving")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    r = run(a.q, a.n, a.R, out=a.out, bits=a.bits, prec=a.prec,
            maxit=a.maxit, threads=a.threads, eps=a.eps, lstar=a.lstar,
            ostar=a.ostar, scaling=a.scaling,
            workdir=a.workdir, keep=a.keep, reuse=a.reuse, tag=a.tag)
    return 0 if r else 1


if __name__ == "__main__":
    sys.exit(main())
