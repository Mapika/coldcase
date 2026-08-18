#!/usr/bin/env python3
"""Keep pushing above whatever we currently hold.

For every cell where results/ already contains a verified graph larger than the
published record, sweep the window just above our own best order.  Each pass that
succeeds raises the floor, so the next pass starts higher -- a ratchet.

    python3 ratchet.py --secs 700 --width 400 --engine gpu --rounds 6
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TABLE = os.path.join(HERE, "table_current.json")


def current_best():
    """cell -> (our best N, published record)"""
    cells = json.load(open(TABLE))["cells"]
    best = {}
    for jf in glob.glob(os.path.join(RESULTS, "*.json")):
        try:
            m = json.load(open(jf))
        except Exception:
            continue
        if "delta" not in m or "S" not in m:
            continue
        key = (m["delta"], m["D"])
        rec = cells.get("%d,%d" % key, {}).get("N")
        if rec is None:
            continue
        if m["N"] > best.get(key, (0,))[0]:
            best[key] = (m["N"], rec)
    return {k: v for k, v in best.items() if v[0] > v[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=700)
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--engine", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=1)
    a = ap.parse_args()

    for rnd in range(a.rounds):
        held = current_best()
        if not held:
            print("nothing held yet")
            return 0
        print("\n########## ratchet round %d -- %s ##########"
              % (rnd + 1, time.strftime("%H:%M:%S", time.gmtime())), flush=True)
        for (delta, D), (ours, rec) in sorted(held.items(), key=lambda kv: -kv[1][0]):
            lo, hi = ours + 1, ours + a.width
            tag = "rt%d_%d" % (delta, D)
            outf = os.path.join(RESULTS, "raw", tag + ".jsonl")
            if a.engine == "gpu":
                cmd = [os.path.join(HERE, "src", "dd_gpu"),
                       "--delta", str(delta), "--diam", str(D),
                       "--Nmin", str(lo), "--Nmax", str(hi), "--time", str(a.secs),
                       "--blocks", "1536", "--threads", "256", "--iters", "40000",
                       "--maxa", "40", "--minn", "3", "--out", outf]
            else:
                cmd = [os.path.join(HERE, "src", "dd_search2"),
                       "--delta", str(delta), "--diam", str(D),
                       "--Nmin", str(lo), "--Nmax", str(hi), "--time", str(a.secs),
                       "--threads", str(a.threads), "--nonab", "--faithful",
                       "--maxa", "40", "--minn", "3", "--iters", "300000",
                       "--nostop", "--verbose", "--out", outf]
            print("\n>>> ratchet (%d,%d): ours %d (record %d) -> sweeping %d..%d"
                  % (delta, D, ours, rec, lo, hi), flush=True)
            p = subprocess.run(cmd, capture_output=True, text=True)
            for line in p.stderr.strip().splitlines():
                if "best f=" in line or "best_f=" in line:
                    print("    " + line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
