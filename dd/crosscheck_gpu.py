#!/usr/bin/env python3
"""Cross-check the CUDA ball kernel against the CPU engine and pure Python.

Three independent implementations must agree on |B_k| for random
(m, n, a, S, D):  dd_gpu (CUDA), dd_search (C++/OpenMP), and a plain Python BFS.
"""
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GPU = os.environ.get("DD_GPU", os.path.join(HERE, "src", "dd_gpu"))
CPU = os.path.join(HERE, "src", "dd_search")

sys.path.insert(0, HERE)
from crosscheck import apowers, inv, py_balls, sample_group, sample_S  # noqa: E402


def run(binary, spec, D):
    r = subprocess.run([binary, "--eval", spec, "--D", str(D)],
                       capture_output=True, text=True)
    return [int(x.split("=")[1]) for x in r.stdout.split() if x.startswith("|B")]


def main():
    rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    bad = done = 0
    while done < trials:
        m, n, a = sample_group(rng)
        N = m * n
        if N > 4000 or N < 8:
            continue
        hi = min(16, N - 1)
        if hi < 2:
            continue
        deg = rng.randint(2, hi)
        S = sample_S(m, n, a, rng, deg)
        if not S:
            continue
        D = rng.randint(1, 6)
        spec = "%d,%d,%d:%s" % (m, n, a, ",".join(map(str, S)))
        py = py_balls(m, n, a, S, D)
        cpu = run(CPU, spec, D)
        gpu = run(GPU, spec, D)
        if not (py == cpu == gpu):
            bad += 1
            print("MISMATCH m=%d n=%d a=%d deg=%d D=%d\n  py =%s\n  cpu=%s\n  gpu=%s"
                  % (m, n, a, deg, D, py, cpu, gpu))
        done += 1
    print("crosscheck_gpu: %d trials, %d mismatches (py == cpu == gpu)" % (done, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
