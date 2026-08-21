#!/usr/bin/env python3
"""Frozen judge for the solver arena. Usage:
   arena_judge.py entryA [entryB ...] [--cells public|heldout] [--seeds 15] [--time 120]
Runs each entry's run_entry.sh on the benchmark cells, verifies every output
with the independent verifier, and reports score tables."""
import argparse, os, subprocess, sys, tempfile, time, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verify"))

PUBLIC = [(6, 6, 3, 41), (6, 8, 4, 169), (8, 9, 4, 940), (3, 11, 4, 81)]
# held-out set kept in a separate file the judge reads at judging time
HELDOUT_FILE = os.path.join(ROOT, "cov", "arena", ".heldout.json")


def coverage_deficit(q, n, R, path):
    """Return (valid, M_found, uncovered) using numpy dilation (independent)."""
    import numpy as np
    try:
        words = [[int(c) for c in l.strip()] for l in open(path) if l.strip()]
        if not words:
            return False, 0, q ** n
        for wd in words:
            if len(wd) != n or any(c < 0 or c >= q for c in wd):
                return False, 0, q ** n
        arr = np.zeros((q,) * n, dtype=bool)
        for wd in words:
            arr[tuple(wd)] = True
        M = len(set(map(tuple, words)))
        for _ in range(R):
            base = arr
            new = base.copy()
            for ax in range(n):
                new |= base.any(axis=ax, keepdims=True)
            arr = new
        return True, M, int(q ** n - arr.sum())
    except Exception:
        return False, 0, q ** n


def run_one(entry_dir, q, n, R, M, seed, time_s):
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
    t0 = time.time()
    try:
        subprocess.run(["bash", os.path.join(entry_dir, "run_entry.sh"),
                        str(q), str(n), str(R), str(M), str(seed), str(time_s), out],
                       timeout=time_s + 60, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        pass
    wall = time.time() - t0
    valid, mf, uncov = coverage_deficit(q, n, R, out)
    os.unlink(out)
    if not valid:
        return -2 * 10**6, wall
    if mf >= M and uncov == 0:
        return 1000, wall
    # partial: normalized uncovered fraction in [-1e6, 0] so a valid partial
    # always beats invalid output regardless of q^n (rules amendment 1)
    return -int(uncov * 10**6 / (q ** n)) - 1, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entries", nargs="+")
    ap.add_argument("--cells", default="public")
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--time", type=int, default=120)
    args = ap.parse_args()
    cells = PUBLIC if args.cells == "public" else json.load(open(HELDOUT_FILE))
    results = {}
    for e in args.entries:
        ed = os.path.join(ROOT, "cov", "arena", e)
        rows = []
        for (q, n, R, M) in [tuple(c) for c in cells]:
            scores, walls = [], []
            for s in range(args.seeds):
                sc, w = run_one(ed, q, n, R, M, 1000 + s, args.time)
                scores.append(sc)
                walls.append(w)
            solved = sum(1 for s in scores if s == 1000)
            med = sorted(scores)[len(scores) // 2]
            rows.append({"cell": f"K{q}({n},{R})@{M}", "solved": solved,
                         "of": args.seeds, "median_score": med,
                         "median_wall": round(sorted(walls)[len(walls) // 2], 1)})
            print(f"{e:12} K{q}({n},{R})@{M}: solved {solved}/{args.seeds} "
                  f"median_score {med} median_wall {rows[-1]['median_wall']}s")
        results[e] = {"rows": rows,
                      "total_solved": sum(r["solved"] for r in rows),
                      "sum_median": sum(r["median_score"] for r in rows)}
    print(json.dumps({e: {k: v for k, v in r.items() if k != "rows"}
                      for e, r in results.items()}, indent=1))


if __name__ == "__main__":
    main()
