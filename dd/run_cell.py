#!/usr/bin/env python3
"""Orchestrate a dd_search campaign on one (Delta, D) cell.

Two modes:
  --mode probe   descend from Nstart until the engine finds ANY valid graph.
                 Used to CALIBRATE the machinery against the published record.
  --mode attack  spend the whole budget on N > record, largest N first.

Every hit is expanded to an edge list and run through the independent verifier.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "src", "dd_search")
TABLE = os.path.join(HERE, "table_current.json")
RESULTS = os.path.join(HERE, "results")
LOG = os.path.join(HERE, "results", "campaign_log.jsonl")


def record_for(delta, D):
    t = json.load(open(TABLE))
    c = t["cells"].get("%d,%d" % (delta, D))
    return c["N"] if c else None


def run(delta, D, N, secs, threads, seed, extra):
    out = os.path.join("/tmp", "ddhit_%d_%d_%d_%d.json" % (delta, D, N, seed))
    cmd = [BIN, "--delta", str(delta), "--diam", str(D), "--Nmin", str(N),
           "--time", str(secs), "--threads", str(threads), "--seed", str(seed),
           "--out", out] + extra
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    stats = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ""
    hits = []
    if os.path.exists(out):
        hits = [json.loads(l) for l in open(out) if l.strip()]
        os.unlink(out)
    return hits, stats, dt


def verify(hit, tag):
    hf = "/tmp/_hit.json"
    open(hf, "w").write(json.dumps(hit) + "\n")
    p = subprocess.run([sys.executable, os.path.join(HERE, "emit_graph.py"), hf,
                        "--outdir", RESULTS, "--tag", tag],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return False, "emit failed: " + p.stderr
    path = p.stdout.strip().splitlines()[-1]
    v = subprocess.run([sys.executable, os.path.join(HERE, "verify_dd.py"), path],
                       capture_output=True, text=True)
    return v.returncode == 0, v.stdout + v.stderr


def log(rec):
    os.makedirs(RESULTS, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=int, required=True)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--mode", choices=["probe", "attack"], default="probe")
    ap.add_argument("--secs", type=float, default=20, help="seconds per N")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start", type=int, default=None, help="override starting N")
    ap.add_argument("--steps", type=int, default=40, help="how many N values to try")
    ap.add_argument("--stepsize", type=int, default=1)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()

    rec = record_for(a.delta, a.D)
    print("cell (%d,%d): published record N = %s" % (a.delta, a.D, rec))
    if a.mode == "probe":
        N = a.start if a.start else rec
        direction = -a.stepsize
    else:
        N = a.start if a.start else rec + 1
        direction = -a.stepsize   # attack largest-first, descending toward rec+1

    best = None
    for step in range(a.steps):
        if N <= 0:
            break
        if a.mode == "attack" and N <= rec:
            print("reached the record; stopping attack")
            break
        hits, stats, dt = run(a.delta, a.D, N, a.secs, a.threads, a.seed + step, a.extra)
        status = "HIT" if hits else "miss"
        print("  N=%-8d %-5s  %s" % (N, status, stats))
        log({"cell": [a.delta, a.D], "N": N, "record": rec, "mode": a.mode,
             "hits": len(hits), "stats": stats, "secs": dt, "t": time.time()})
        if hits:
            h = hits[0]
            ok, txt = verify(h, "%s" % a.mode)
            print("  verifier: %s" % ("PASS" if ok else "FAIL"))
            for line in txt.strip().splitlines():
                print("    " + line)
            log({"cell": [a.delta, a.D], "N": N, "record": rec, "verified": ok, "hit": h})
            best = N
            if a.mode == "probe":
                break
            if a.mode == "attack":
                print("*** RECORD CANDIDATE: (%d,%d) N=%d > %d ***" % (a.delta, a.D, N, rec))
                break
        N += direction

    if a.mode == "probe":
        if best:
            print("PROBE RESULT: engine reaches N=%d vs record %d  (%.1f%% of record)"
                  % (best, rec, 100.0 * best / rec))
        else:
            print("PROBE RESULT: no graph found in the scanned range")
    return 0


if __name__ == "__main__":
    sys.exit(main())
