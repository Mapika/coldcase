#!/usr/bin/env python3
"""Score cov/search/covsearch.c (built here as ./baseline) under exactly the
arena judge's protocol, so the entry can be compared against it honestly.

Uses arena_judge.coverage_deficit -- the same verifier -- and the same
kill-at-TIME_S+60 rule.  The baseline is run with --threads 6 (the thread cap in
RULES.md); note that on the two small cells --threads 1 is measurably better for
it, see NOTES.md section 2.

    ./cmp_baseline.py [--seeds 5] [--time 60]
"""
import argparse, os, subprocess, sys, tempfile, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from arena_judge import coverage_deficit, PUBLIC          # noqa: E402


def run_one(q, n, R, M, seed, time_s, threads):
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
    t0 = time.time()
    try:
        subprocess.run(["nice", "-n", "15", os.path.join(HERE, "baseline"),
                        "-q", str(q), "-n", str(n), "-R", str(R), "-M", str(M),
                        "-t", str(time_s), "-s", str(seed),
                        "--threads", str(threads), "--quiet", "--out", out],
                       timeout=time_s + 60, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        pass
    wall = time.time() - t0
    valid, mf, uncov = coverage_deficit(q, n, R, out)
    os.unlink(out)
    if not valid:
        return -2 * 10 ** 6, wall
    if mf >= M and uncov == 0:
        return 1000, wall
    # rules amendment 1: normalized uncovered fraction
    return -int(uncov * 10 ** 6 / (q ** n)) - 1, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--time", type=int, default=60)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    rows = []
    for (q, n, R, M) in PUBLIC:
        scores, walls = [], []
        for s in range(args.seeds):
            sc, w = run_one(q, n, R, M, 1000 + s, args.time, args.threads)
            scores.append(sc); walls.append(w)
        solved = sum(1 for s in scores if s == 1000)
        med = sorted(scores)[len(scores) // 2]
        rows.append({"cell": f"K{q}({n},{R})@{M}", "solved": solved,
                     "of": args.seeds, "median_score": med,
                     "median_wall": round(sorted(walls)[len(walls) // 2], 1),
                     "scores": scores})
        print(f"baseline     K{q}({n},{R})@{M}: solved {solved}/{args.seeds} "
              f"median_score {med} median_wall {rows[-1]['median_wall']}s "
              f"scores {scores}", flush=True)
    print(json.dumps({"baseline": {"total_solved": sum(r["solved"] for r in rows),
                                   "sum_median": sum(r["median_score"] for r in rows)}},
                     indent=1))


if __name__ == "__main__":
    main()
