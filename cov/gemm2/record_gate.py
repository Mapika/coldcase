#!/usr/bin/env python3
"""Record a candidate cover through the project's ONE verification gate.

Thin wrapper: imports record() from cov/campaign.py unchanged, so every file
that lands in cov/results/ passes cov/verify_cov.py re-read from disk exactly
as every prior record did.  Run niced; used synchronously or as a background
subprocess by siege.py.

Usage: record_gate.py Q N R M PATH [METHOD]
Exit 0 iff verified and stored.
"""
import os
import sys

os.nice(10)
HERE = os.path.dirname(os.path.abspath(__file__))
COV = os.path.dirname(HERE)
sys.path.insert(0, COV)

import campaign as CP  # noqa: E402


def keri_ub(q, n, R):
    try:
        import constructions as C
        e = C.load_bounds().get((q, n, R))
        return e["ub"] if e else None
    except Exception:
        return None


def main():
    q, n, R, M = (int(x) for x in sys.argv[1:5])
    path = sys.argv[5]
    method = sys.argv[6] if len(sys.argv) > 6 else \
        "gpuchain transform-engine LNS (exact distance-count transforms, GH200)"
    ub = keri_ub(q, n, R)
    if ub is not None and M >= ub:
        print(f"SKIPPED: M={M} does not beat the Keri upper bound {ub}; "
              f"cover kept at {path}, NOT stored in cov/results/")
        return 3
    dest = CP.record(q, n, R, M, path, method)
    return 0 if dest else 1


if __name__ == "__main__":
    sys.exit(main())
