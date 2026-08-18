#!/usr/bin/env python3
"""Two-phase campaign driver.

Phase 1 (scan)  : short sweep over a whole range of orders; dd_search --verbose
                  reports the best f = N - |B_D| reached for each N.
Phase 2 (focus) : spend a long budget on the promising orders only, largest N
                  first, stopping at the first verified hit.

Rationale: a flat sweep gives every N the same time, but the reachable orders are
sparse and structured (for (14,3) every hit had n = 3).  Phase 1 costs a few percent
of the budget and tells us where to spend the rest.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "src", "dd_search2")
TABLE = os.path.join(HERE, "table_current.json")
RESULTS = os.path.join(HERE, "results")

SCAN_RE = re.compile(r"N=(\d+)\s+best_f=(\d+)\s+\(m=(\d+) n=(\d+) a=(-?\d+)\)")


def run(args, secs, out=None, extra=()):
    cmd = [BIN] + list(args) + ["--time", str(secs)] + list(extra)
    if out:
        cmd += ["--out", out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stderr, p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=int, required=True)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--lo", type=int, default=None)
    ap.add_argument("--hi", type=int, default=None)
    ap.add_argument("--span", type=int, default=250)
    ap.add_argument("--scan-secs", type=float, default=120)
    ap.add_argument("--focus-secs", type=float, default=180)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--minn", type=int, default=3)
    ap.add_argument("--maxa", type=int, default=80)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--models", type=int, default=3, help="1=metacyclic 2=affine2 3=both")
    a = ap.parse_args()

    rec = json.load(open(TABLE))["cells"]["%d,%d" % (a.delta, a.D)]["N"]
    lo = a.lo or rec + 1
    hi = a.hi or (lo + a.span - 1)
    tag = a.tag or "f%d_%d" % (a.delta, a.D)
    os.makedirs(os.path.join(RESULTS, "raw"), exist_ok=True)
    outf = os.path.join(RESULTS, "raw", tag + ".jsonl")

    base = ["--delta", str(a.delta), "--diam", str(a.D), "--threads", str(a.threads),
            "--nonab", "--faithful", "--maxa", str(a.maxa), "--minn", str(a.minn),
            "--iters", "200000", "--nostop", "--verbose", "--models", str(a.models)]

    print("cell (%d,%d) record=%d   scanning N=%d..%d for %.0fs"
          % (a.delta, a.D, rec, lo, hi, a.scan_secs), flush=True)
    err, _ = run(base + ["--Nmin", str(lo), "--Nmax", str(hi)], a.scan_secs, outf)
    scan = [(int(m.group(1)), int(m.group(2))) for m in SCAN_RE.finditer(err)]
    if not scan:
        print("scan produced nothing:\n" + err[-2000:])
        return 1
    hits = [N for N, f in scan if f == 0]
    print("  scan: %d orders probed, %d already solved: %s"
          % (len(scan), len(hits), sorted(hits)[-6:]), flush=True)

    # promising = unsolved, small f, prefer large N
    cand = sorted([(N, f) for N, f in scan if f > 0], key=lambda x: (x[1], -x[0]))
    cand = cand[:a.topk]
    cand.sort(key=lambda x: -x[0])
    print("  focusing on: %s" % [(N, f) for N, f in cand], flush=True)

    best = max(hits) if hits else 0
    for N, f0 in cand:
        if N <= best:
            continue
        err, _ = run(base + ["--Nmin", str(N), "--Nmax", str(N)], a.focus_secs, outf)
        m = re.search(r"best f=(\d+)", err)
        f = int(m.group(1)) if m else -1
        print("    N=%-8d f: %d -> %d %s" % (N, f0, f, "HIT" if f == 0 else ""), flush=True)
        if f == 0:
            best = max(best, N)

    print("best order reached: %d (record %d)%s"
          % (best, rec, "  *** IMPROVEMENT ***" if best > rec else ""))
    print("raw hits in %s" % outf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
