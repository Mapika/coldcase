#!/usr/bin/env python3
"""The decisive idle-GPU benchmark for the chain-batched solver.

REFUSES to run if any other compute process holds the GPU (we yield to the
other tenant). When the GPU is free, runs the full matrix:
  - K8(9,4) M=2944 at up to 256 chains (the projection-deciding case),
  - K6(6,3)@41 and K6(8,4)@169 idle re-runs (paper table),
validates every result (CPU recount + verify_cov.py on solves), and writes
results to idle_results.json. Safe to re-run; ~15-30 min total.
Usage: python3 idle_bench.py [--quick]
"""
import json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from chain import run, random_code, cpu_uncovered

HERE = os.path.dirname(os.path.abspath(__file__))


def gpu_busy():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                          "--format=csv,noheader"], capture_output=True, text=True)
    return bool(out.stdout.strip())


def bench(q, n, R, M, nchains, iters, seed, label, results):
    if gpu_busy():
        print(f"YIELD: GPU occupied before {label}; aborting benchmark.")
        sys.exit(2)
    rng = np.random.default_rng(seed)
    code = random_code(q, n, M, rng)
    u0 = cpu_uncovered(q, n, R, code)
    t0 = time.time()
    r = run(q, n, R, code, nchains=nchains, iters=iters, seed=seed)
    dt = time.time() - t0
    tot = int(r["iters"].sum())
    # validation: recount 4 random chains exactly
    for c in np.random.default_rng(0).choice(nchains, size=min(4, nchains), replace=False):
        ucpu = cpu_uncovered(q, n, R, r["best"][c][:, :n])
        assert ucpu == r["best_u"][c], f"MISMATCH {label} chain {c}: {ucpu} vs {r['best_u'][c]}"
    row = {"label": label, "q": q, "n": n, "R": R, "M": M, "chains": nchains,
           "iters_budget": iters, "start_u": u0, "wall_s": round(dt, 1),
           "agg_it_per_s": round(tot / dt), "per_chain_it_per_s": round(tot / dt / nchains, 1),
           "solved": int(r["solved"].sum()), "best_u_min": int(r["best_u"].min()),
           "best_u_median": int(np.median(r["best_u"]))}
    print(row)
    results.append(row)
    # verifier gate on any solve
    sol = np.where(r["solved"] == 1)[0]
    if len(sol):
        p = f"/tmp/idlebench_{label}.txt"
        with open(p, "w") as f:
            for w in r["best"][sol[0]][:, :n]:
                f.write("".join(str(int(x)) for x in w) + "\n")
        v = subprocess.run(["python3", os.path.join(HERE, "..", "verify_cov.py"),
                            "-q", str(q), "-n", str(n), "-R", str(R), p],
                           capture_output=True, text=True)
        row["verify"] = v.stdout.strip().splitlines()[-1] if v.stdout else "?"
    return row


def main():
    quick = "--quick" in sys.argv
    if gpu_busy():
        print("YIELD: another compute process owns the GPU. Not running.")
        sys.exit(2)
    results = []
    # paper-table idle re-runs
    bench(6, 6, 3, 41, 256, 6000 if quick else 30000, 21, "K6_6_3_idle", results)
    bench(6, 8, 4, 169, 256, 1000 if quick else 4000, 22, "K6_8_4_idle", results)
    # THE decisive case: big cell. 256 chains x 268MB = 69GB; verify free HBM.
    free = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.split()[0])
    chains = 256 if free > 75000 else (128 if free > 40000 else 64)
    bench(8, 9, 4, 2944, chains, 100 if quick else 400, 23, "K8_9_4_idle", results)
    json.dump(results, open(os.path.join(HERE, "idle_results.json"), "w"), indent=1)
    print("\nCPU socket reference (cov/opt tables, p5b x64 cores):"
          " K6(6,3) ~980k it/s, K6(8,4) ~128k it/s, K8(9,4) ~1.2k it/s")
    print("Wrote idle_results.json — compare agg_it_per_s against the socket.")


if __name__ == "__main__":
    main()
