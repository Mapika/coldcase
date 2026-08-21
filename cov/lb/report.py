#!/usr/bin/env python3
"""
report.py -- re-verify every certificate in cov/lb/certs/ with the standalone
exact checker and cross-check the result against cov/bounds.json.

Hard invariants (a failure here means our code is wrong, not that we found
something):
  * the certified bound must never exceed the best known UPPER bound;
  * the certified bound must be a valid lower bound, i.e. the certificate must
    verify exactly.

Usage:  python3 report.py [--certs DIR] [--out results/summary.json]
                          [--md results/summary.md] [--jobs 8]
"""

import os
import sys
import json
import glob
import math
import argparse
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BOUNDS = os.path.join(HERE, os.pardir, "bounds.json")


def _verify(path):
    sys.path.insert(0, HERE)
    import certify as ct
    try:
        res = ct.check_certificate_file(path, verbose=False)
    except Exception as exc:                                     # noqa: BLE001
        return {"cert": path, "ok": False, "reasons": [repr(exc)]}
    res["cert"] = path
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--certs", default=os.path.join(HERE, "certs"))
    ap.add_argument("--out", default=os.path.join(HERE, "results",
                                                  "summary.json"))
    ap.add_argument("--md", default=os.path.join(HERE, "results",
                                                 "summary.md"))
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    with open(BOUNDS) as fh:
        bj = json.load(fh)
    E = {(e["q"], e["n"], e["R"]): e for e in bj["entries"]}
    GP = {(e["q"], e["n"], e["R"]): e for e in bj["lower_bound_updates_2025"]}

    files = sorted(glob.glob(os.path.join(a.certs, "cert_*.json")))
    print("verifying %d certificates on %d workers" % (len(files), a.jobs))
    with mp.Pool(a.jobs) as pool:
        rows = pool.map(_verify, files)

    out = []
    errors = []
    for r in rows:
        if not r.get("ok"):
            errors.append((r["cert"], r.get("reasons")))
            out.append({"cert": os.path.basename(r["cert"]), "valid": False,
                        "reasons": r.get("reasons")})
            continue
        p = r["problem"]
        key = (p["q"], p["n"], p["R"])
        e = E.get(key, {})
        keri_lb = e.get("lb")
        keri_ub = e.get("ub")
        gp_lb = GP[key]["new_lb"] if key in GP else None
        best_known_lb = max([x for x in (keri_lb, gp_lb, e.get("lb_updated"))
                             if x is not None], default=None)
        K = r["K_lower_bound"]
        # the SDP is provably at least as tight as the sphere covering bound,
        # so a certified value below it means the floating point solver failed
        # on this cell -- report it as such rather than as a weak bound.
        V = sum(math.comb(p["n"], i) * (p["q"] - 1) ** i
                for i in range(p["R"] + 1))
        sphere = -(-(p["q"] ** p["n"]) // V)
        row = {
            "cert": os.path.basename(r["cert"]),
            "q": p["q"], "n": p["n"], "R": p["R"],
            "valid": True,
            "sdp_value_num": str(r["bound_num"]),
            "sdp_value_den": str(r["bound_den"]),
            "sdp_value": r["bound_float"],
            "cube_root": r["cube_root_float"],
            "K_lower_bound": K,
            "keri_lb": keri_lb, "keri_lb_key": e.get("lb_key"),
            "keri_ub": keri_ub,
            "gp2025_lb": gp_lb,
            "best_known_lb": best_known_lb,
            "improves_best_known": (best_known_lb is not None
                                    and K > best_known_lb),
            "matches_best_known": (best_known_lb is not None
                                   and K == best_known_lb),
            "exceeds_known_ub": (keri_ub is not None and K > keri_ub),
            "sphere_covering": sphere,
            "solver_failed": K < sphere,
        }
        if row["exceeds_known_ub"]:
            errors.append((r["cert"], "certified LB %d exceeds known UB %d"
                           % (K, keri_ub)))
        out.append(row)

    out.sort(key=lambda r: (r.get("q", 0), r.get("n", 0), r.get("R", 0)))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)

    hdr = ("| cell | our certified SDP value | our LB | best known LB "
           "(source) | known UB | verdict |")
    lines = [hdr, "|---|---|---|---|---|---|"]
    nimp = nmatch = 0
    for r in out:
        if not r.get("valid"):
            lines.append("| %s | INVALID | | | | %s |"
                         % (r["cert"], r.get("reasons")))
            continue
        src = []
        if r["gp2025_lb"] is not None:
            src.append("GP2025")
        if r["keri_lb_key"]:
            src.append("Keri key %s" % r["keri_lb_key"])
        verdict = ("solver failed" if r.get("solver_failed")
                   else ("**IMPROVES**" if r["improves_best_known"]
                         else ("matches" if r["matches_best_known"]
                               else "below")))
        nimp += bool(r["improves_best_known"]
                     and not r.get("solver_failed"))
        nmatch += bool(r["matches_best_known"])
        lines.append("| K_%d(%d,%d) | %.4f (cube root %.4f) | %d | %s (%s) "
                     "| %s | %s |"
                     % (r["q"], r["n"], r["R"], r["sdp_value"],
                        r["cube_root"], r["K_lower_bound"],
                        r["best_known_lb"], ", ".join(src) or "-",
                        r["keri_ub"], verdict))
    md = "\n".join(lines)
    os.makedirs(os.path.dirname(a.md), exist_ok=True)
    with open(a.md, "w") as fh:
        fh.write("# Certified SDP lower bounds (all re-verified by "
                 "certify.py)\n\n" + md + "\n")
    print(md)
    nfail = sum(1 for r in out if r.get("solver_failed"))
    print("\nimproves best known: %d   matches: %d   solver failed: %d   "
          "total valid: %d"
          % (nimp, nmatch, nfail, sum(1 for r in out if r.get("valid"))))
    if errors:
        print("\nERRORS (%d):" % len(errors))
        for e in errors[:20]:
            print("  %s" % (e,))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
