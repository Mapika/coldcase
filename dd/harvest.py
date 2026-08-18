#!/usr/bin/env python3
"""Harvest campaign output: for every raw hit, keep only the largest N per cell,
re-derive the graph, run the standalone verifier, and print a claim table.

    python3 harvest.py [results/raw/*.jsonl]

A hit is only reported if it (a) exceeds the published record for its cell and
(b) passes verify_dd.py.
"""
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "table_current.json")
RESULTS = os.path.join(HERE, "results")


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(RESULTS, "raw", "*.jsonl")))
    table = json.load(open(TABLE))["cells"]
    best = {}
    for fp in files:
        if not os.path.exists(fp):
            continue
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            h = json.loads(line)
            key = (h["delta"], h["D"])
            if key not in best or h["N"] > best[key]["N"]:
                best[key] = h

    rows = []
    for (delta, D), h in sorted(best.items()):
        rec = table.get("%d,%d" % (delta, D), {}).get("N")
        status = "improves" if (rec and h["N"] > rec) else "below record"
        verified = None
        if rec and h["N"] > rec:
            hf = "/tmp/_harvest_hit.json"
            open(hf, "w").write(json.dumps(h) + "\n")
            p = subprocess.run([sys.executable, os.path.join(HERE, "emit_graph.py"),
                                hf, "--outdir", RESULTS, "--tag", "best"],
                               capture_output=True, text=True)
            if p.returncode != 0:
                verified = "EMIT-FAIL: " + p.stderr.strip().splitlines()[-1]
            else:
                path = p.stdout.strip().splitlines()[-1]
                v = subprocess.run([sys.executable, os.path.join(HERE, "verify_dd.py"), path],
                                   capture_output=True, text=True)
                verified = "PASS" if v.returncode == 0 else "FAIL\n" + v.stdout + v.stderr
                if v.returncode == 0:
                    print(v.stdout)
        grp = ("(Z_%d)^2 x| Z_%d" % (h["p"], h["n"])) if h.get("model") == "affine2" \
            else ("Z_%d x|_%d Z_%d" % (h["m"], h["a"], h["n"]))
        rows.append((delta, D, h["N"], rec, status, verified, grp))

    print("%-8s %-9s %-9s %-18s %-13s %s" % ("cell", "found N", "record", "group", "vs rec", "verified"))
    for delta, D, N, rec, status, ver, grp in rows:
        print("(%d,%d)%s %-9d %-9s %-18s %-13s %s"
              % (delta, D, " " * max(0, 8 - len("(%d,%d)" % (delta, D))), N,
                 rec, grp, status, (ver or "-").splitlines()[0]))


if __name__ == "__main__":
    main()
