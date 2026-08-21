#!/usr/bin/env python3
"""End-to-end time-to-target benchmark (Table 2) + run-time distributions.

Protocol
--------
* Three cells, each started from a FIXED code file (`seeds_*.txt`) so that
  every run of every variant begins from the identical state and the seed
  only drives the stochastic search.
* 20 seeds per (cell, variant), a fixed per-run CPU budget, and a fixed
  target uncovered count.  The solver records the CPU time and the iteration
  at which `best_uncovered` first reached the target.
* Runs that never reach the target inside the budget are censored; the
  censoring rate is reported and medians are only quoted when the target was
  reached by more than half the runs.
* The metric is the solver's own CLOCK_PROCESS_CPUTIME_ID, because the host
  is shared with a production sweep that saturates all 64 cores; wall clock
  would measure the sweep, not the solver.  Run order is shuffled.
"""
import csv, os, random, re, statistics, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
CPUSET = os.environ.get("COVBENCH_CPUSET", "")   # "" = no pinning; see micro.py
POOL = int(os.environ.get("COVBENCH_POOL", "8"))
NSEEDS = int(os.environ.get("COVBENCH_SEEDS", "20"))

CELLS = [
    dict(name="K6(6,3)", args="-q 6 -n 6 -R 3 -M 41   --in seeds_K6_6_3_M41.txt",
         target=30, cpu=45, init=1411),
    dict(name="K6(8,4)", args="-q 6 -n 8 -R 4 -M 169  --in seeds_K6_8_4_M169.txt",
         target=20, cpu=90, init=854),
    dict(name="K8(9,4)", args="-q 8 -n 9 -R 4 -M 2944 --in seeds_K8_9_4_M2944.txt",
         target=10800, cpu=100, init=11640),
]

VARIANTS = [
    dict(name="base",        args="--preset base"),
    dict(name="p5",          args="--preset p5"),
    dict(name="p5+wide",     args="--preset p5 --wide"),
    dict(name="p5b",         args="--preset p5b"),
    dict(name="p5b+wide",    args="--preset p5b --wide"),
    # fix0 quotients out the q^n translations by freezing one codeword; the
    # effect must be largest when that is 1 of 41 codewords and negligible when
    # it is 1 of 2944, so it is measured on the two smaller cells.
    dict(name="p5+fix0",     args="--preset p5 --fix0", skip=["K8(9,4)"]),
    # NOTE: --early only applies to the per-candidate evaluator, which --preset
    # p5 replaces; it is measured directly as a pruning rate with the -DPROF
    # build instead (see METHODS.md H.3).
    dict(name="p5+ucache",   args="--preset p5 --ucache", skip=["K8(9,4)"]),
    dict(name="p5+upick",    args="--preset p5 --upick", skip=["K8(9,4)"]),
]

RESULT_RE = re.compile(r"RESULT (.*)")


def run(job):
    cell, var, seed = job
    pin = ["taskset", "-c", CPUSET] if CPUSET else []
    cmd = (pin + ["nice", "-n", "15",
            os.path.join(HERE, "covsearch2")] + cell["args"].split()
           + var["args"].split()
           + ["--target", str(cell["target"]), "--cpu", str(cell["cpu"]),
              "-t", "999999", "-s", str(seed), "--threads", "1", "--quiet"])
    env = dict(os.environ); env["OMP_NUM_THREADS"] = "1"
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=HERE)
    m = RESULT_RE.search(p.stdout)
    if not m:
        sys.stderr.write("FAILED %s\n%s\n" % (" ".join(cmd), p.stderr[-400:]))
        return None
    d = dict(kv.split("=", 1) for kv in m.group(1).split())
    return dict(cell=cell["name"], variant=var["name"], seed=seed,
                target=cell["target"], budget=cell["cpu"],
                ttt_cpu=float(d["ttt_cpu"]), ttt_it=int(d["ttt_it"]),
                cpu=float(d["cpu"]), iters=int(d["iters"]),
                best=int(d["uncovered"]), wall=float(d["time"]))


def main():
    jobs = []
    for c in CELLS:
        for v in VARIANTS:
            if c["name"] in v.get("skip", []):
                continue
            for s in range(1, NSEEDS + 1):
                jobs.append((c, v, s))
    random.Random(23).shuffle(jobs)
    print(f"{len(jobs)} runs", flush=True)
    rows, done, t0 = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=POOL) as ex:
        for r in ex.map(run, jobs):
            done += 1
            if r:
                rows.append(r)
            if done % 15 == 0:
                print(f"  {done}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    out = os.path.join(HERE, "results_ttt.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
