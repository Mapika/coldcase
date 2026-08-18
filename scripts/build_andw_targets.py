#!/usr/bin/env python3
"""Rank the open A(n,d,w) instances by attractiveness for a record hunt.

Input : data/tables/andw_bounds.json   (scripts/parse_andw.py)
Output: data/tables/andw_targets.json

An instance is *open* when the parsed table does not mark it exact.

gap_ratio = ub / lb.  The upper bound in Brouwer's table is authoritative and is
used whenever it is present; only when the table quotes no upper bound do we
compute a Johnson bound ourselves:

    J(n,d,w) = floor( n/w * J(n-1,d,w-1) )                    (first Johnson bound)
    J(n,d,w) = floor( n/(n-w) * J(n-1,d,w) )                  (its second part)

recursing down to base cases
    J(n,d,w) = 1                  if 2w < d  (or w = 0)
    J(n,d,w) = table value        whenever the table has an upper bound / exact
                                  value for (n,d,w)  -- a strictly stronger base
                                  case than the pure recursion
and using the complement symmetry A(n,d,w) = A(n,d,n-w).

`johnson_first` is the first-Johnson-only value, `johnson_min` additionally
takes the second part into account (Brouwer's page notes it is sometimes
stronger).  gap_ratio uses `johnson_first`, i.e. the form specified for this
database; `johnson_min` is carried along for reference.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "data", "tables", "andw_bounds.json")
OUT = os.path.join(ROOT, "data", "tables", "andw_targets.json")
CWC = os.path.join(ROOT, "data", "raw", "cwc")

sys.setrecursionlimit(10000)


def build():
    doc = json.load(open(IN, encoding="utf-8"))
    entries = doc["entries"]

    # ---- lookup tables, normalised by the complement symmetry --------------
    def key(n, d, w):
        return (n, d, min(w, n - w))

    ub_tab, lb_tab = {}, {}
    for e in entries:
        k = key(e["n"], e["d"], e["w"])
        if e["ub"] is not None:
            ub_tab[k] = min(e["ub"], ub_tab.get(k, e["ub"]))
        if e["lb"] is not None:
            lb_tab[k] = max(e["lb"], lb_tab.get(k, e["lb"]))

    memo = {}

    def J(n, d, w, second):
        if w < 0 or w > n:
            return 0
        w = min(w, n - w)
        if w == 0 or 2 * w < d:
            return 1
        k = (n, d, w, second)
        if k in memo:
            return memo[k]
        memo[k] = None                       # cycle guard
        tab = ub_tab.get((n, d, w))
        if tab is not None:
            memo[k] = tab
            return tab
        v = (n * J(n - 1, d, w - 1, second)) // w
        if second and n - w > 0:
            v = min(v, (n * J(n - 1, d, w, second)) // (n - w))
        memo[k] = v
        return v

    # ---- targets -----------------------------------------------------------
    targets = []
    for e in entries:
        if e["exact"] or e["lb"] is None:
            continue
        n, d, w, lb = e["n"], e["d"], e["w"], e["lb"]
        if e["ub"] is not None:
            ub, ub_src = e["ub"], "table"
            jf = jm = None
        else:
            jf = J(n, d, w, second=False)
            jm = J(n, d, w, second=True)
            ub, ub_src = jf, "johnson"
        code_file = e.get("code_file")
        local = None
        if code_file:
            p = os.path.join(ROOT, "data", "raw", code_file)
            if os.path.isfile(p):
                local = os.path.relpath(p, ROOT)
        targets.append({
            "n": n, "d": d, "w": w,
            "lb": lb, "ub": ub, "ub_source": ub_src,
            "gap_ratio": round(ub / lb, 6),
            "gap_abs": ub - lb,
            "johnson_first": jf,
            "johnson_min": jm,
            "fits_gpu_word": n <= 64,
            "has_local_code": local is not None,
            "code_file": code_file,
            "local_code_path": local,
            "lb_label": e["lb_label"],
            "ub_label": e["ub_label"],
            "lb_new_on_page": bool(e.get("nw")),
            "lb_lost": bool(e.get("lost")),
        })

    targets.sort(key=lambda t: (-t["gap_ratio"], t["n"], t["d"], t["w"]))
    for i, t in enumerate(targets, 1):
        t["rank"] = i

    # ---- self-check: reproduce the page's own unlabelled Johnson bounds ----
    agree = disagree = 0
    examples = []
    for e in entries:
        if (e["n"] > 28 and e["ub"] is not None and e["ub_label"] is None
                and not e["exact"]):
            n, d, w = e["n"], e["d"], e["w"]
            k = key(n, d, w)
            saved = ub_tab.pop(k, None)
            memo.clear()
            got = min(J(n, d, w, False), J(n, d, w, True))
            if saved is not None:
                ub_tab[k] = saved
            memo.clear()
            if got == e["ub"]:
                agree += 1
            else:
                disagree += 1
                if len(examples) < 15:
                    examples.append({"n": n, "d": d, "w": w,
                                     "table_ub": e["ub"], "our_johnson": got})

    doc_out = {
        "source": "derived from data/tables/andw_bounds.json",
        "generated_by": "scripts/build_andw_targets.py",
        "definition": {
            "open": "the table does not mark the cell exact",
            "gap_ratio": "ub / lb, ub from the table when present, otherwise "
                         "the first Johnson bound computed here",
            "fits_gpu_word": "n <= 64: a codeword fits in one 64-bit word",
            "has_local_code": "the incumbent construction is mirrored under "
                              "data/raw/cwc/",
            "ranking": "descending gap_ratio, ties broken by n, d, w",
        },
        "johnson_selfcheck": {
            "description": "cells with n > 28 whose upper bound the page gives "
                           "without a label are, per the page, Johnson bounds; "
                           "we recompute them with the table removed as a base case",
            "agree": agree, "disagree": disagree, "examples": examples,
        },
        "counts": {
            "open_instances": len(targets),
            "ub_from_table": sum(1 for t in targets if t["ub_source"] == "table"),
            "ub_from_johnson": sum(1 for t in targets if t["ub_source"] == "johnson"),
            "with_local_code": sum(1 for t in targets if t["has_local_code"]),
            "fits_gpu_word": sum(1 for t in targets if t["fits_gpu_word"]),
        },
        "targets": targets,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc_out, f, indent=1, ensure_ascii=False)
    print("wrote %s: %d open instances (%d table ub, %d johnson ub)"
          % (OUT, len(targets), doc_out["counts"]["ub_from_table"],
             doc_out["counts"]["ub_from_johnson"]))
    print("johnson self-check vs page: agree=%d disagree=%d" % (agree, disagree))
    for x in examples:
        print("   ", x)


if __name__ == "__main__":
    build()
