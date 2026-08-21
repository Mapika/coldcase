#!/usr/bin/env python3
"""
reproduce.py -- reproduction gate against Gijswijt-Polak arXiv:2504.01932.

Runs our own exact model + IPM dual + exact certificate on the cells the paper
tabulates (Tables 7/8/9, non-binary q=3,4,5) and compares the *certified*
cube-root value with their published value.

Usage:
    python3 reproduce.py [--max-n 8] [--jobs 8] [--out results/repro.json]
"""

import os
import sys
import json
import time
import argparse
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Published cube-root SDP values, Tables 7 (q=3), 8 (q=4), 9 (q=5) of
# arXiv:2504.01932v2.  "*" in the paper = improves on Keri's table.
PUBLISHED = {
    3: {(6, 1): 60.8568, (6, 2): 13.1228,
        (7, 1): 150.9556, (7, 2): 26.3830, (7, 3): 8.5250,
        (8, 1): 402.9463, (8, 2): 57.4972, (8, 3): 15.5959,
        (9, 1): 1063.9751, (9, 2): 131.8916, (9, 3): 30.1035, (9, 4): 10.2323,
        (10, 1): 2811.8571, (10, 2): 313.0763, (10, 3): 60.4226,
        (10, 4): 17.9976, (10, 5): 7.3464,
        (11, 2): 728.9999, (11, 3): 128.2649, (11, 4): 33.2215,
        (11, 5): 11.9665},
    4: {(6, 1): 226.59, (6, 2): 32.91, (6, 3): 8.76, (6, 4): 3.35,
        (7, 1): 775.07, (7, 2): 87.63, (7, 3): 18.78, (7, 4): 6.40,
        (8, 1): 2694.38, (8, 2): 250.87, (8, 3): 45.02, (8, 4): 12.56,
        (9, 1): 9362.28, (9, 2): 774.46, (9, 3): 115.28, (9, 4): 26.66,
        (9, 5): 9.03,
        (10, 2): 2459.70, (10, 3): 310.69, (10, 4): 61.09, (10, 5): 17.54,
        (10, 6): 6.87},
    5: {(5, 1): 161.03, (5, 2): 21.66,
        (6, 1): 624.99, (6, 2): 68.86, (6, 3): 13.81, (6, 4): 4.37,
        (7, 1): 2764.89, (7, 2): 235.35, (7, 3): 37.40, (7, 4): 9.70,
        (8, 1): 12133.70, (8, 2): 860.13, (8, 3): 110.24, (8, 4): 23.04,
        (8, 5): 7.27,
        (9, 2): 3279.51, (9, 3): 353.32, (9, 4): 61.18, (9, 5): 15.67,
        (9, 6): 5.79},
}


def _one(args):
    q, n, R, outdir, bits = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    sys.path.insert(0, HERE)
    import solve_ipm
    t0 = time.time()
    out = None
    if outdir:
        out = os.path.join(outdir, "cert_q%d_n%d_R%d.json" % (q, n, R))
    try:
        res = solve_ipm.run(q, n, R, out=out, bits=bits, verbose=False,
                            tag="reproduce")
    except Exception as exc:                                     # noqa: BLE001
        return {"q": q, "n": n, "R": R, "error": repr(exc),
                "seconds": time.time() - t0}
    if res is None:
        return {"q": q, "n": n, "R": R, "error": "no certificate",
                "seconds": time.time() - t0}
    return {"q": q, "n": n, "R": R,
            "certified_cube_root": res["cube_root_float"],
            "certified_value_num": str(res["bound_num"]),
            "certified_value_den": str(res["bound_den"]),
            "K_lower_bound": res["K_lower_bound"],
            "cert_file": out,
            "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=8)
    ap.add_argument("--min-n", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--bits", type=int, default=46)
    ap.add_argument("--qs", default="3,4,5")
    ap.add_argument("--outdir", default=os.path.join(HERE, "certs"))
    ap.add_argument("--out", default=os.path.join(HERE, "results",
                                                  "reproduce.json"))
    a = ap.parse_args()
    qs = [int(x) for x in a.qs.split(",")]
    jobs = []
    for q in qs:
        for (n, R) in sorted(PUBLISHED.get(q, {})):
            if a.min_n <= n <= a.max_n:
                jobs.append((q, n, R, a.outdir, a.bits))
    if a.outdir:
        os.makedirs(a.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    print("running %d cells on %d workers" % (len(jobs), a.jobs))
    with mp.Pool(a.jobs) as pool:
        rows = pool.map(_one, jobs)

    print("\n%-12s %14s %14s %10s %8s" %
          ("cell", "published", "ours(certified)", "rel.diff", "sec"))
    ok = bad = 0
    for r in rows:
        q, n, R = r["q"], r["n"], r["R"]
        pub = PUBLISHED[q][(n, R)]
        if "error" in r:
            print("%-12s %14.4f %14s %10s %8.1f"
                  % ("K%d(%d,%d)" % (q, n, R), pub, r["error"], "-",
                     r["seconds"]))
            bad += 1
            r["published"] = pub
            continue
        ours = r["certified_cube_root"]
        rel = (ours - pub) / pub
        r["published"] = pub
        r["rel_diff"] = rel
        # published values are rounded to 4 (q=3) or 2 (q=4,5) decimals
        tolabs = 5e-5 * max(1.0, pub) if q == 3 else 5e-3 * max(1.0, pub)
        tolabs = max(tolabs, 1e-4 * pub)
        r["match"] = bool(abs(ours - pub) <= max(tolabs, 6e-5 * pub)
                          or (ours <= pub and abs(rel) < 2e-4))
        print("%-12s %14.4f %14.4f %10.2e %8.1f"
              % ("K%d(%d,%d)" % (q, n, R), pub, ours, rel, r["seconds"]))
        if r["match"]:
            ok += 1
        else:
            bad += 1
    print("\nmatched %d / %d" % (ok, ok + bad))
    with open(a.out, "w") as fh:
        json.dump({"published_source": "arXiv:2504.01932v2 Tables 7,8,9",
                   "rows": rows}, fh, indent=1)
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
