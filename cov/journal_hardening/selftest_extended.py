#!/usr/bin/env python3
"""Extended primal self-test battery for the LB SDP transcription (referee M10).

cov/lb/certify.py --selftest validates the model by feeding explicit covering
codes into check_primal, but only at n <= 5.  This battery extends the check to
n = 6..10 using this project's own verified record codes (all in cov/results/,
each gated by verify_cov.py + verify_independent.py):

    K8(6,4)  = 19    K12(6,4) = 39     (n = 6)
    K6(7,3)  = 227                     (n = 7)
    K8(8,5)  = 79    K6(8,4)  = 166    (n = 8)
    K6(9,5)  = 119                     (n = 9)
    K8(10,6) = 335                     (n = 10)

For each case the model is built for that exact (q, n, R) and the code must be
accepted as a feasible primal point with objective exactly |C|^3 (all
arithmetic exact Fractions inside certify.check_primal).

Usage:
    python3 selftest_extended.py            # run all cases in order
    python3 selftest_extended.py K6_9_5     # run one case by prefix

Cost note: code_primal_point is Theta(|C|^3 * n) (triple loop over the code);
model build + exact PSD checks grow with the orbit count (O(n^4) variables).
Each case prints build/check wall times so the coverage table is honest.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
COV = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(COV, "lb"))

import certify  # noqa: E402

CASES = [
    # (results file,          q,  n,  R)
    ("K8_6_4_M19.txt",        8,  6,  4),
    ("K12_6_4_M39.txt",      12,  6,  4),
    ("K6_7_3_M227.txt",       6,  7,  3),
    ("K8_8_5_M79.txt",        8,  8,  5),
    ("K6_8_4_M166.txt",       6,  8,  4),
    ("K6_9_5_M119.txt",       6,  9,  5),
    ("K8_10_6_M335.txt",      8, 10,  6),
]


def read_code(path, q, n):
    code = []
    for raw in open(path):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        if any(ch in line for ch in " ,\t"):
            w = tuple(int(t) for t in line.replace(",", " ").split())
        else:
            w = tuple(int(ch, 36) for ch in line)
        assert len(w) == n and all(0 <= d < q for d in w), (path, line)
        code.append(w)
    assert len(set(code)) == len(code), "duplicate codewords in " + path
    return code


def main():
    sel = sys.argv[1] if len(sys.argv) > 1 else None
    allok = True
    ran = 0
    for fname, q, n, R in CASES:
        if sel and not fname.startswith(sel):
            continue
        ran += 1
        path = os.path.join(COV, "results", fname)
        code = read_code(path, q, n)
        M = len(code)
        t0 = time.time()
        model = certify.build_model(q, n, R)
        t_build = time.time() - t0
        t0 = time.time()
        rep = {}
        probs = certify.check_primal(model, code, rep)
        t_check = time.time() - t0
        status = "OK " if not probs else "FAIL"
        print("  %s q=%-2d n=%-2d R=%d |C|=%-4d vars=%d lin=%d psd=%d "
              "obj=%s |C|^3=%s  build=%.1fs check=%.1fs"
              % (status, q, n, R, M, model.nvars, len(model.lin),
                 len(model.psd), rep.get("objective"), rep.get("M3"),
                 t_build, t_check), flush=True)
        for p in probs[:5]:
            print("      %s" % p, flush=True)
        if probs:
            allok = False
    if ran == 0:
        print("no case matches %r" % sel)
        return 2
    print("EXTENDED SELFTEST %s" % ("PASSED" if allok else "FAILED"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
