#!/usr/bin/env python3
"""Spray phase: brief seeded-tabu probes across many A(n,d,w) cells.

Reads cells from data/tables/quick_targets.json (or a filter), gives each a
short GPU budget via hunt_cwc.hunt_cell, and logs outcomes to
results/spray_log.jsonl.  Cells that improve are recorded for the deepening
phase.  All improvements are PROVISIONAL until checked against the live table.
"""
import json, os, sys, time, argparse
sys.stdout.reconfigure(line_buffering=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from hunt_cwc import hunt_cell, find_incumbent

LOG = os.path.join(ROOT, "results", "spray_log.jsonl")


def already_done(n, d, w):
    if not os.path.exists(LOG):
        return False
    with open(LOG) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r["n"], r["d"], r["w"]) == (n, d, w):
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, nargs="*", default=None)
    ap.add_argument("--nmin", type=int, default=6)
    ap.add_argument("--nmax", type=int, default=64)
    ap.add_argument("--lbmax", type=int, default=2000)
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--chains", type=int, default=1536)
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--limit", type=int, default=10**9)
    args = ap.parse_args()

    cells = json.load(open(os.path.join(ROOT, "data", "tables", "andw_targets.json")))["targets"]
    cells = [c for c in cells
             if (args.d is None or c["d"] in args.d)
             and args.nmin <= c["n"] <= args.nmax
             and c["lb"] + 1 <= 2048 and c["lb"] <= args.lbmax
             and c.get("fits_gpu_word", True)
             and not c.get("lb_lost", False)
             and c["ub"] - c["lb"] >= 2]
    # prioritize by the table's gap ranking (already sorted by gap_ratio)
    cells.sort(key=lambda c: c["rank"])
    print(f"{len(cells)} cells queued")

    done = 0
    for c in cells:
        n, d, w, lb = c["n"], c["d"], c["w"], c["lb"]
        if already_done(n, d, w):
            continue
        t0 = time.time()
        res = hunt_cell(n, d, w, lb, max_rounds=args.rounds, nchains=args.chains,
                        iters=(args.iters or None), time_budget_s=args.budget)
        rec = {"n": n, "d": d, "w": w, "lb": lb, "achieved": res,
               "improved": bool(res and res > lb), "secs": round(time.time() - t0, 1),
               "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if rec["improved"]:
            print(f"IMPROVED (provisional): A({n},{d},{w}) {lb} -> {res}")
        done += 1
        if done >= args.limit:
            break


if __name__ == "__main__":
    main()
