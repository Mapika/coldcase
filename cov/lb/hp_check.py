#!/usr/bin/env python3
"""
hp_check.py -- verify every certificate in certs_hp/ with the FROZEN checker
(cov/lb/certify.py, run as a subprocess) and compare the certified bound with
the best known lower bound (Keri 2011 + the 2025 update list in
cov/bounds.json).  Writes results/hiprec/status.md and status.json.

Usage:  python3 hp_check.py [cert.json ...]     (default: all of certs_hp/)
"""

import sys
import os
import re
import json
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def incumbents():
    b = json.load(open(os.path.join(HERE, "..", "bounds.json")))
    inc = {}
    for e in b["entries"]:
        if e.get("lb") is not None:
            inc[(e["q"], e["n"], e["R"])] = (e["lb"], e.get("lb_key"),
                                             e.get("ub"))
    for e in b.get("lower_bound_updates_2025", []):
        k = (e["q"], e["n"], e["R"])
        old = inc.get(k, (0, None, None))
        if e["new_lb"] > old[0]:
            inc[k] = (e["new_lb"], "GP2025", old[2])
    return inc


def check(path):
    out = subprocess.run(
        ["nice", "-n", "10", sys.executable,
         os.path.join(HERE, "certify.py"), path],
        capture_output=True, text=True)
    txt = out.stdout
    ok = "CERTIFICATE VALID" in txt
    m = re.search(r"=> K_(\d+)\((\d+),(\d+)\) >= (\d+)", txt)
    v = re.search(r"~ ([\d.]+)\s+\(cube root ([\d.]+)\)", txt)
    if not ok or not m:
        return {"path": path, "ok": False, "raw": txt[-500:] +
                out.stderr[-300:]}
    return {"path": path, "ok": True,
            "q": int(m.group(1)), "n": int(m.group(2)), "R": int(m.group(3)),
            "K_lb": int(m.group(4)),
            "sdp_value": float(v.group(1)) if v else None,
            "cube_root": float(v.group(2)) if v else None}


def main(argv):
    paths = argv[1:] or sorted(glob.glob(os.path.join(HERE, "certs_hp",
                                                      "*.json")))
    inc = incumbents()
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(check, paths))
    rows = []
    for r in results:
        if r["ok"]:
            key = (r["q"], r["n"], r["R"])
            lb, srckey, ub = inc.get(key, (None, None, None))
            r["incumbent"] = lb
            r["incumbent_key"] = srckey
            r["known_ub"] = ub
            if ub is not None and r["K_lb"] > ub:
                r["verdict"] = "ERROR_EXCEEDS_UB"
            elif lb is None:
                r["verdict"] = "no incumbent"
            elif r["K_lb"] > lb:
                r["verdict"] = "IMPROVES (+%d)" % (r["K_lb"] - lb)
            elif r["K_lb"] == lb:
                r["verdict"] = "matches"
            else:
                r["verdict"] = "below (-%d)" % (lb - r["K_lb"])
        rows.append(r)
        print("%-40s %s" % (os.path.basename(r["path"]),
                            r.get("verdict", "INVALID") if r["ok"]
                            else "INVALID: " + r.get("raw", "")[:200]))
    resdir = os.path.join(HERE, "results", "hiprec")
    os.makedirs(resdir, exist_ok=True)
    with open(os.path.join(resdir, "status.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    lines = ["# certs_hp status (frozen-checker verified)", "",
             "| cell | cube root | our LB | incumbent (key) | verdict |",
             "|---|---|---|---|---|"]
    for r in rows:
        if not r["ok"]:
            lines.append("| %s | INVALID | | | |" % os.path.basename(r["path"]))
            continue
        lines.append("| K_%d(%d,%d) | %.4f | %d | %s (%s) | %s |"
                     % (r["q"], r["n"], r["R"], r["cube_root"] or 0,
                        r["K_lb"], r["incumbent"], r["incumbent_key"],
                        r["verdict"]))
    with open(os.path.join(resdir, "status.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
