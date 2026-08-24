#!/usr/bin/env python3
"""Task 5 (referee M11): external baselines on the small improved cells.

For each cell and each target M in {Keri_ub - 1, our record}, run PAIRED
1h budgets of:
  tabu   : covsearch2 --preset base  (== covsearch_base, the plain tabu/greedy
           local search; every covsearch2 improvement is behind a flag and
           defaults OFF, so this is the honest "plain tabu" baseline)
  engine : covsearch2 --preset p5b   (the production configuration used by
           the paper's campaigns)

Pairing: both configs run simultaneously with the same worker count (8), the
same per-worker seconds (3600), and the same seed set, so they see identical
machine conditions.  A solved target (uncovered=0) is re-verified with
cov/verify_cov.py (numpy) on the output file re-read from disk before being
reported as solved.  Nothing here is recorded to cov/results (these are
baseline measurements, not record claims).

Results append to t5_results.jsonl next to this script.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
COV = os.path.dirname(HERE)
ROOT = os.path.dirname(COV)
BIN = os.path.join(COV, "opt", "covsearch2")
SCRATCH = os.path.join(HERE, "t5_scratch")
RESULTS = os.path.join(HERE, "t5_results.jsonl")

WORKERS = 8
TSEC = 3600
SEED0 = 71000

# cell -> (keri_ub, our_record)
CELLS = [
    ((8, 6, 4), 20, 19),
    ((12, 6, 4), 41, 39),
    ((13, 6, 4), 46, 45),
    ((14, 6, 4), 52, 50),
    ((15, 6, 4), 59, 57),
    ((6, 7, 3), 246, 227),
]


def launch(q, n, R, M, preset, tag):
    procs = []
    for w in range(WORKERS):
        out = os.path.join(SCRATCH, "%s_w%02d.txt" % (tag, w))
        cmd = ["nice", "-n", "11", BIN, "-q", str(q), "-n", str(n),
               "-R", str(R), "-M", str(M), "-t", str(TSEC),
               "-s", str(SEED0 + w), "--threads", "1",
               "--out", out, "--quiet", "--preset", preset]
        procs.append((subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True),
                      out))
    return procs


def collect(procs):
    best, bestpath = None, None
    for p, out in procs:
        stdout, _ = p.communicate()
        unc = None
        for line in stdout.splitlines():
            if line.startswith("RESULT "):
                kv = dict(t.split("=", 1) for t in line.split()[1:] if "=" in t)
                unc = int(kv["uncovered"])
        if unc is not None and (best is None or unc < best):
            best, bestpath = unc, out
    return best, bestpath


def verify(path, q, n, R):
    r = subprocess.run(
        [sys.executable, os.path.join(COV, "verify_cov.py"), path,
         "-q", str(q), "-n", str(n), "-R", str(R), "--method", "numpy"],
        capture_output=True, text=True)
    return r.returncode == 0


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None  # e.g. "8_6_4"
    for (q, n, R), keri, ours in CELLS:
        cell = "%d_%d_%d" % (q, n, R)
        if only and cell != only:
            continue
        targets = sorted(set([keri - 1, ours]), reverse=True)
        for M in targets:
            t0 = time.time()
            print("[t5] K%d(%d,%d) M=%d : paired base vs p5b, %dx%ds"
                  % (q, n, R, M, WORKERS, TSEC), flush=True)
            pb = launch(q, n, R, M, "base", "K%s_M%d_base" % (cell, M))
            pe = launch(q, n, R, M, "p5b", "K%s_M%d_p5b" % (cell, M))
            res = {}
            for name, procs in (("base", pb), ("p5b", pe)):
                unc, path = collect(procs)
                solved = unc == 0
                ver = verify(path, q, n, R) if solved else None
                res[name] = {"best_uncovered": unc, "solved": solved,
                             "verified": ver, "path": path}
                print("[t5]   %-4s best_uncovered=%s solved=%s verified=%s"
                      % (name, unc, solved, ver), flush=True)
            rec = {"cell": [q, n, R], "M": M, "keri_ub": keri, "ours": ours,
                   "workers": WORKERS, "t_per_worker_s": TSEC,
                   "seed0": SEED0, "wall_s": round(time.time() - t0),
                   "base": res["base"], "p5b": res["p5b"],
                   "ts": time.strftime("%Y-%m-%d %H:%M")}
            with open(RESULTS, "a") as f:
                f.write(json.dumps(rec) + "\n")
    print("[t5] DONE", flush=True)


if __name__ == "__main__":
    main()
