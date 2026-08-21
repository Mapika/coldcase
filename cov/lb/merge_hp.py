#!/usr/bin/env python3
"""
merge_hp.py -- copy each frozen-checker-VERIFIED certificate from certs_hp/
into certs_all/ iff it is strictly better (higher exact SDP bound) than the
certificate already there (or none is there).  Reads verification verdicts
from results/hiprec/status.json (produced by hp_check.py), so run hp_check.py
first; only certificates with ok=true are considered.

Usage: python3 merge_hp.py [--dry]
"""

import os
import json
import glob
import shutil
import argparse
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))


def bound_of(path):
    c = json.load(open(path))
    cl = c.get("claim", {})
    if "sdp_bound_num" in cl:
        return Fraction(int(cl["sdp_bound_num"]), int(cl["sdp_bound_den"]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    status = json.load(open(os.path.join(HERE, "results", "hiprec",
                                         "status.json")))
    verified = {os.path.basename(r["path"]) for r in status if r.get("ok")}
    n_copy = n_skip = 0
    for p in sorted(glob.glob(os.path.join(HERE, "certs_hp", "*.json"))):
        base = os.path.basename(p)
        if base not in verified:
            print("SKIP (not verified):", base)
            n_skip += 1
            continue
        newb = bound_of(p)
        dest = os.path.join(HERE, "certs_all", base)
        oldb = bound_of(dest) if os.path.exists(dest) else None
        if oldb is None or newb > oldb:
            print("COPY %s  (%.4f -> %.4f cube root)"
                  % (base,
                     float(oldb) ** (1 / 3.0) if oldb else 0.0,
                     float(newb) ** (1 / 3.0)))
            if not a.dry:
                shutil.copy2(p, dest)
            n_copy += 1
        else:
            print("KEEP existing %s (not weaker)" % base)
            n_skip += 1
    print("copied %d, skipped %d" % (n_copy, n_skip))


if __name__ == "__main__":
    main()
