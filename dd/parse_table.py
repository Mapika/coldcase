#!/usr/bin/env python3
"""Parse the Comellas degree-diameter table into JSON with provenance.

Source: https://web.mat.upc.edu/francesc.comellas/delta-d/table_degree_diameter.html
Reads dd/data/table_raw.html, writes dd/table_current.json.

Provenance is recovered from the cell background colour, cross-referenced with the
legend list at the bottom of the page (colour -> author/date/cells).
"""
import json
import os
import re
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "table_raw.html")
OUT = os.path.join(HERE, "table_current.json")
URL = "https://web.mat.upc.edu/francesc.comellas/delta-d/table_degree_diameter.html"


def strip_tags(s):
    s = re.sub(r"<[^>]*>", " ", s)
    s = s.replace("&nbsp;", " ").replace("&eacute;", "e").replace("&oacute;", "o")
    s = s.replace("&Scaron;", "S").replace("&aacute;", "a").replace("&ncaron;", "n")
    s = s.replace("&Agrave;", "A").replace("&iacute;", "i").replace("&egrave;", "e")
    return re.sub(r"\s+", " ", s).strip()


def moore_bound(delta, D):
    if delta <= 2:
        return 2 * D + 1
    return 1 + delta * ((delta - 1) ** D - 1) // (delta - 2)


def parse_legend(html):
    """colour(lowercased hex) -> list of legend entries."""
    tail = html[html.lower().find("</table>"):]
    legend = {}
    for m in re.finditer(
        r'background-color:\s*(#[0-9A-Fa-f]{6})\s*;.*?</span>(.*?)(?=<LI|</UL>)',
        tail, re.S | re.I
    ):
        colour = m.group(1).lower()
        text = strip_tags(m.group(2))
        text = re.sub(r"\s*See\s*\[.*", "", text).strip()
        cells = [(int(a), int(b)) for a, b in re.findall(r"\((\d+)\s*[,.]\s*(\d+)\)", text)]
        author = text.split(",")[0].strip()
        dates = re.findall(
            r"((?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December)[a-z\- ]*\d{0,2},?\s*\d{4}|\b(?:19|20)\d{2}\b)", text)
        legend.setdefault(colour, []).append(
            {"author": author, "text": text, "cells": cells, "dates": dates})
    return legend


def parse_cells(html):
    lo = html.lower().find("<table")
    hi = html.lower().find("</table>")
    tbl = html[lo:hi]
    rows = re.split(r"<tr\b", tbl, flags=re.I)[1:]
    cells = {}
    diameters = None
    for row in rows:
        ths = re.findall(r"<th\b[^>]*>(.*?)</th>", row, re.S | re.I)
        tds = re.findall(r"(<td\b[^>]*>.*?</td>)", row, re.S | re.I)
        if not tds:
            # header row: extract diameter columns
            vals = [strip_tags(t) for t in ths]
            nums = [int(v) for v in vals if v.isdigit()]
            if nums:
                diameters = nums
            continue
        delta = int(strip_tags(ths[0]))
        for k, td in enumerate(tds):
            D = diameters[k]
            txt = strip_tags(re.sub(r"<a\b[^>]*>|</a>", "", td, flags=re.I))
            num = re.sub(r"[^\d]", "", txt)
            if not num:
                continue
            cm = re.search(r'bgcolor\s*=\s*"?\s*(#[0-9A-Fa-f]{6})', td, re.I)
            colour = cm.group(1).lower() if cm else None
            href = re.search(r'href="([^"]+)"', td, re.I)
            cells[(delta, D)] = {
                "N": int(num),
                "colour": colour,
                "href": href.group(1) if href else None,
            }
    return cells


def main():
    html = open(RAW, encoding="utf-8", errors="replace").read()
    legend = parse_legend(html)
    cells = parse_cells(html)

    out = {
        "source_url": URL,
        "fetched_utc": datetime.fromtimestamp(os.path.getmtime(RAW), timezone.utc).isoformat(),
        "parsed_utc": datetime.now(timezone.utc).isoformat(),
        "note": "N = largest known order of a graph with max degree Delta and diameter D "
                "(lower bound on the true optimum). Provenance inferred from cell "
                "background colour matched against the page legend.",
        "legend": {c: v for c, v in legend.items()},
        "cells": {},
    }
    for (delta, D), info in sorted(cells.items()):
        key = f"{delta},{D}"
        prov = []
        for e in legend.get(info["colour"], []):
            # prefer the legend entry that explicitly lists this cell
            explicit = (delta, D) in e["cells"]
            prov.append({"author": e["author"], "dates": e["dates"],
                         "explicit_cell": explicit, "text": e["text"]})
        prov.sort(key=lambda p: not p["explicit_cell"])
        mb = moore_bound(delta, D)
        out["cells"][key] = {
            "delta": delta, "D": D, "N": info["N"],
            "moore_bound": mb,
            "ratio_to_moore": round(info["N"] / mb, 4),
            "colour": info["colour"],
            "href": info["href"],
            "provenance": prov,
        }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}: {len(out['cells'])} cells, {len(legend)} legend colours")

    hot = []
    for k, v in out["cells"].items():
        p = v["provenance"][0] if v["provenance"] else None
        if p and p["explicit_cell"] and any("2026" in d for d in p["dates"]):
            hot.append((k, v["N"], p["author"], p["dates"]))
    print("\ncells updated in 2026 (soft targets):")
    for k, n, a, d in sorted(hot, key=lambda x: (int(x[0].split(",")[0]))):
        print(f"  ({k}) = {n:>9}   {a}  {d}")


if __name__ == "__main__":
    sys.exit(main())
