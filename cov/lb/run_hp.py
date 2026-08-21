#!/usr/bin/env python3
"""
run_hp.py -- queue runner for solve_hp.py over a list of cells.

Usage: python3 run_hp.py cells.txt [--jobs 6] [--threads 2]
where cells.txt lines are:  q n R [prec] [maxit]
Writes per-cell logs results/hiprec/log_qQ_nN_RR.txt, appends one-line
outcomes to results/hiprec/sweep.log, certificates to certs_hp/.
"""

import sys
import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "hiprec")


def run_cell(args, jobargs):
    q, n, R = jobargs[0], jobargs[1], jobargs[2]
    prec = jobargs[3] if len(jobargs) > 3 else 200
    maxit = jobargs[4] if len(jobargs) > 4 else 200
    out = os.path.join(HERE, "certs_hp", "cert_q%d_n%d_R%d.json" % (q, n, R))
    log = os.path.join(RES, "log_q%d_n%d_R%d.txt" % (q, n, R))
    lstars = [args.lstar] if args.lstar else ["1.0E4", "1.0E2", "1.0E1"]
    for ls in lstars:
        cmd = ["nice", "-n", "10", sys.executable,
               os.path.join(HERE, "solve_hp.py"), str(q), str(n), str(R),
               "--out", out, "--prec", str(prec), "--maxit", str(maxit),
               "--threads", str(args.threads), "--reuse",
               "--lstar", ls]
        with open(log, "a") as fh:
            p = subprocess.run(cmd, stdout=fh, stderr=fh)
        if p.returncode == 0:
            with open(log) as fh:
                txt = fh.read()
            if "phase=pdOPT" in txt.split("sdpa_gmp:")[-1]:
                break
    tail = ""
    try:
        with open(log) as fh:
            lines = [l.strip() for l in fh if l.strip()]
        for l in reversed(lines):
            if ("certified SDP value" in l or "FAILED" in l
                    or "failed" in l or "no yMat" in l):
                tail = l
                break
    except OSError:
        pass
    with open(os.path.join(RES, "sweep.log"), "a") as fh:
        fh.write("DONE q=%d n=%d R=%d rc=%d prec=%d :: %s\n"
                 % (q, n, R, p.returncode, prec, tail))
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--lstar", default=None)
    args = ap.parse_args()
    cells = []
    for line in open(args.cells):
        line = line.split("#")[0].strip()
        if not line:
            continue
        cells.append([int(x) for x in line.split()])
    os.makedirs(os.path.join(HERE, "certs_hp"), exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        rcs = list(ex.map(lambda c: run_cell(args, c), cells))
    print("all done:", rcs)


if __name__ == "__main__":
    main()
