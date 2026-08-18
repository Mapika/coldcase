#!/usr/bin/env python3
"""Classify each table cell by the construction family of its standing record.

The source site has a description page per diameter (desc_g2.html ... desc_g5.html)
naming the construction behind every entry.  Knowing which records are Cayley
graphs of semidirect products tells us exactly which cells this engine can compete
in: our family provably reaches a record that is itself in our family, so a
systematic sweep there should beat a casual one.  Cells whose records are compound
graphs or generalized-quadrangle polarity quotients with vertex additions are
structurally out of reach for a pure Cayley search.

Writes dd/data/cell_families.json.
"""
import html
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BASE = "https://web.mat.upc.edu/francesc.comellas/delta-d/desc_g/desc_g%d.html"

HDR = re.compile(
    r'(?:Delta\s*=\s*(\d+)\s*,\s*Diam\s*=\s*(\d+)\s*;\s*N\s*=\s*([\d ]+)'
    r'|Degree\s*=\s*(\d+)\s*,\s*Diameter\s*=\s*(\d+)\s*;\s*Order\s*=\s*([\d ]+))')

KEYWORDS = [
    ("semidirect", "metacyclic-Cayley"),
    ("cayley", "Cayley"),
    ("circulant", "circulant"),
    ("voltage", "voltage-lift"),
    ("lift", "lift"),
    ("compound", "compound"),
    ("polarity", "GQ/PG-polarity"),
    ("generalized quadrangle", "GQ"),
    ("added", "vertex-addition"),
    ("addition of vertices", "vertex-addition"),
    ("regulariz", "vertex-addition"),
    ("ai tools", "AI-assisted"),
    ("llm", "LLM-assisted"),
]


def fetch(D, refresh):
    path = os.path.join(DATA, "desc_g%d.html" % D)
    if refresh or not os.path.exists(path):
        with urllib.request.urlopen(BASE % D, timeout=60) as r:
            open(path, "wb").write(r.read())
    return open(path, encoding="utf-8", errors="replace").read()


def main():
    refresh = "--refresh" in sys.argv
    table = json.load(open(os.path.join(HERE, "table_current.json")))["cells"]
    out = {}
    for D in range(2, 11):
        try:
            h = fetch(D, refresh)
        except Exception as e:
            print("skip D=%d (%s)" % (D, e))
            continue
        t = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", h)))
        for m in HDR.finditer(t):
            g = m.groups()
            if g[0]:
                delta, dd, N = int(g[0]), int(g[1]), int(g[2].replace(" ", ""))
            else:
                delta, dd, N = int(g[3]), int(g[4]), int(g[5].replace(" ", ""))
            cell = table.get("%d,%d" % (delta, dd))
            if not cell or cell["N"] != N:
                continue          # this is a superseded entry, not the record
            body = t[m.end():m.end() + 500]
            b = body.lower()
            fams = []
            for kw, lbl in KEYWORDS:
                if kw in b and lbl not in fams:
                    fams.append(lbl)
            if "cayley ? false" in b and "Cayley" in fams:
                fams.remove("Cayley")
            out["%d,%d" % (delta, dd)] = {
                "delta": delta, "D": dd, "N": N,
                "families": fams,
                "reachable_by_this_engine": "metacyclic-Cayley" in fams or "Cayley" in fams,
                "note": body[:300].strip(),
            }
    p = os.path.join(DATA, "cell_families.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote %s (%d cells classified)" % (p, len(out)))
    print("\ncells whose record is a Cayley graph (this engine's family):")
    for k, v in sorted(out.items(), key=lambda kv: (kv[1]["D"], kv[1]["delta"])):
        if v["reachable_by_this_engine"]:
            print("  (%d,%d) = %-8d %s" % (v["delta"], v["D"], v["N"], ",".join(v["families"])))


if __name__ == "__main__":
    main()
