#!/usr/bin/env python3
"""Mine the campaign logs for near-misses and pound on them.

The scan phase of a campaign leaves a trail of "(cell, N) reached f = k" lines.
Orders with a small residual f are the ones worth a long, concentrated run: they
are cases where a Cayley graph of that order very nearly has the target diameter,
and closing the last few vertices is exactly what a big compute budget buys.

    python3 nearmiss.py --logs /tmp/campaign3.log /tmp/campaign_gpu.log --list
    python3 nearmiss.py --logs ... --run --secs 900 --engine gpu
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "table_current.json")
RESULTS = os.path.join(HERE, "results")

CELL = re.compile(r"cell \((\d+),(\d+)\) record=(\d+)")
FOCUS = re.compile(r"^\s+N=(\d+)\s+f: (\d+) -> (\d+)")
SCANLINE = re.compile(r"^\s+N=(\d+)\s+best_f=(\d+)\s+\(m=(\d+) n=(\d+) a=(-?\d+)\)")
GPUHDR = re.compile(r"##### GPU \((\d+),(\d+)\)")
GPUFOCUS = re.compile(r"##### GPU (?:FOCUS|push) \((\d+),(\d+)\)")


def mine(paths):
    """-> {(delta,D): {N: best_f}}"""
    out = {}
    cur = None
    for p in paths:
        if not os.path.exists(p):
            continue
        for line in open(p, errors="replace"):
            m = CELL.search(line)
            if m:
                cur = (int(m.group(1)), int(m.group(2)))
                continue
            m = GPUHDR.search(line) or GPUFOCUS.search(line)
            if m:
                cur = (int(m.group(1)), int(m.group(2)))
                continue
            if cur is None:
                continue
            m = FOCUS.match(line)
            if m:
                N, f = int(m.group(1)), int(m.group(3))
                d = out.setdefault(cur, {})
                d[N] = min(d.get(N, 1 << 30), f)
                continue
            m = SCANLINE.match(line)
            if m:
                N, f = int(m.group(1)), int(m.group(2))
                d = out.setdefault(cur, {})
                d[N] = min(d.get(N, 1 << 30), f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="*", default=["/tmp/campaign3.log", "/tmp/campaign_gpu.log"])
    ap.add_argument("--maxf", type=int, default=4, help="only orders with residual f <= this")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--secs", type=float, default=900)
    ap.add_argument("--engine", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--threads", type=int, default=20, help="CPU threads (share the box)")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    table = json.load(open(TABLE))["cells"]
    # orders already solved (a hit exists) are not worth re-running
    solved = set()
    for f in glob.glob(os.path.join(RESULTS, "raw", "*.jsonl")):
        for line in open(f):
            line = line.strip()
            if line:
                h = json.loads(line)
                solved.add((h["delta"], h["D"], h["N"]))
    data = mine(a.logs)
    cand = []
    for (delta, D), d in data.items():
        rec = table.get("%d,%d" % (delta, D), {}).get("N")
        if rec is None:
            continue
        for N, f in d.items():
            if N > rec and 0 < f <= a.maxf and (delta, D, N) not in solved:
                cand.append((f, -(N - rec), delta, D, N, rec))
    cand.sort()
    cand = cand[:a.top]

    print("%-8s %-9s %-9s %-7s %s" % ("cell", "N", "record", "margin", "residual f"))
    for f, negm, delta, D, N, rec in cand:
        print("(%d,%d)%s %-9d %-9d +%-6d %d" %
              (delta, D, " " * max(0, 8 - len("(%d,%d)" % (delta, D))), N, rec, N - rec, f))
    if not a.run:
        return 0

    os.makedirs(os.path.join(RESULTS, "raw"), exist_ok=True)
    for f, negm, delta, D, N, rec in cand:
        tag = "nm%d_%d" % (delta, D)
        outf = os.path.join(RESULTS, "raw", tag + ".jsonl")
        if a.engine == "gpu":
            cmd = [os.path.join(HERE, "src", "dd_gpu"), "--delta", str(delta), "--diam", str(D),
                   "--Nmin", str(N), "--Nmax", str(N), "--time", str(a.secs),
                   "--blocks", "1536", "--threads", "256", "--iters", "100000",
                   "--stall", "800", "--kicks", "200", "--maxa", "200", "--minn", "3",
                   "--out", outf]
        else:
            cmd = [os.path.join(HERE, "src", "dd_search2"), "--delta", str(delta), "--diam", str(D),
                   "--Nmin", str(N), "--Nmax", str(N), "--time", str(a.secs),
                   "--threads", str(a.threads), "--nonab", "--faithful", "--maxa", "200", "--minn", "3",
                   "--iters", "500000", "--stall", "800", "--kicks", "200", "--nostop",
                   "--out", outf]
        print("\n>>> (%d,%d) N=%d (record %d, was f=%d)  %.0fs on %s"
              % (delta, D, N, rec, f, a.secs, a.engine), flush=True)
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        last = [l for l in p.stderr.strip().splitlines() if "best f=" in l]
        print("    %s  (%.0fs)" % (last[-1] if last else "(no output)", time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
