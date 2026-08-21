#!/usr/bin/env python3
"""Fixed-iteration micro-benchmark (Table 1).

Every variant here reproduces the baseline search trajectory bit-for-bit
(see selftest.sh), so at a fixed --iters budget all of them perform exactly
the same search and differ only in how fast they perform it.  CPU time at
fixed iterations is therefore a fully controlled comparison: no
search-quality confound and no variance from the stochastic search.

Residual variance comes from the co-running production sweep, handled by
(a) measuring CPU time rather than wall clock, (b) shuffling the run order,
(c) medians over repetitions.
"""
import csv, os, random, re, statistics, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
# Empty CPUSET = no taskset pinning.  The 8-core budget is enforced by running
# at most POOL single-threaded processes (OMP_NUM_THREADS=1), which is at most
# 8 cores of work.  Pinning to a fixed 8 cores turned out to be actively
# harmful here: the co-running sweep puts ~4 spinning OpenMP threads per
# worker on those same cores, so a pinned nice-15 process got only ~5% of a
# core, while an unpinned one gets its fair share spread over all 64.
CPUSET = os.environ.get("COVBENCH_CPUSET", "")
POOL = int(os.environ.get("COVBENCH_POOL", "8"))

CELLS = [
    ("K6(6,3)", "-q 6 -n 6 -R 3 -M 41   --in seeds_K6_6_3_M41.txt",   20000, 3, 3),
    ("K6(8,4)", "-q 6 -n 8 -R 4 -M 169  --in seeds_K6_8_4_M169.txt",    600, 3, 3),
    ("K8(9,4)", "-q 8 -n 9 -R 4 -M 2944 --in seeds_K8_9_4_M2944.txt",    50, 3, 2),
]

VARIANTS = [
    ("base",            "--preset base"),
    ("p1 walk+hoist",   "--preset p1"),
    ("p2 +uint8 state", "--preset p2"),
    ("p2b 2-bit state", "--preset p2b"),
    ("p3 +shared sphere", "--preset p3"),
    ("p4 +uncov. list", "--preset p4"),
    ("p5 +hugepages",   "--preset p5"),
    ("p5b 2-bit+shared", "--preset p5b"),
    ("p5+prefetch",     "--preset p5 --pf"),
]

RESULT_RE = re.compile(r"RESULT (.*)")


def run(job):
    name, extra, cell, cargs, iters, seed, rep = job
    pin = ["taskset", "-c", CPUSET] if CPUSET else []
    cmd = (pin + ["nice", "-n", "15",
            os.path.join(HERE, "covsearch2")] + cargs.split() + extra.split()
           + ["--iters", str(iters), "-t", "999999", "-s", str(seed),
              "--threads", "1", "--quiet"])
    env = dict(os.environ); env["OMP_NUM_THREADS"] = "1"
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=HERE)
    m = RESULT_RE.search(p.stdout)
    if not m:
        sys.stderr.write("FAILED: %s\n%s\n" % (" ".join(cmd), p.stderr[-400:]))
        return None
    d = dict(kv.split("=", 1) for kv in m.group(1).split())
    return dict(variant=name, cell=cell, seed=seed, rep=rep,
                cpu=float(d["cpu"]), iters=int(d["iters"]),
                uncovered=int(d["uncovered"]), wall=float(d["time"]))


def main():
    jobs = []
    for cell, cargs, iters, nseeds, reps in CELLS:
        for name, extra in VARIANTS:
            for seed in range(1, nseeds + 1):
                for rep in range(reps):
                    jobs.append((name, extra, cell, cargs, iters, seed, rep))
    random.Random(11).shuffle(jobs)
    print(f"{len(jobs)} runs", flush=True)
    rows, done, t0 = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=POOL) as ex:
        for r in ex.map(run, jobs):
            done += 1
            if r:
                rows.append(r)
            if done % 20 == 0:
                print(f"  {done}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    out = os.path.join(HERE, "results_micro.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("wrote", out)
    summarize(rows)


def summarize(rows):
    for cell, cargs, iters, nseeds, reps in CELLS:
        print(f"\n== {cell}, fixed {iters} iterations ==")
        base = None
        for name, _ in VARIANTS:
            v = sorted(r["cpu"] for r in rows if r["cell"] == cell and r["variant"] == name)
            if not v:
                continue
            med = statistics.median(v)
            if name == "base":
                base = med
            uu = sorted({r["uncovered"] for r in rows
                         if r["cell"] == cell and r["variant"] == name})
            lo = v[max(0, int(0.25 * len(v)) - 0)] if len(v) > 3 else v[0]
            hi = v[min(len(v) - 1, int(0.75 * len(v)))] if len(v) > 3 else v[-1]
            print(f"  {name:20s} med {med:8.3f}s  [{lo:.3f},{hi:.3f}]  "
                  f"x{base/med if base else 0:5.2f}  n={len(v)}  unc={uu}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        rows = [dict(r, cpu=float(r["cpu"]), uncovered=int(r["uncovered"]))
                for r in csv.DictReader(open(os.path.join(HERE, "results_micro.csv")))]
        summarize(rows)
    else:
        main()
