#!/usr/bin/env python3
"""Parse Brouwer's unrestricted binary code table A(n,d) into JSON.

Source: data/raw/binary-1.html (mirror of
https://www.win.tue.nl/~aeb/codes/binary-1.html, table frozen 2019).

Output: data/tables/and_bounds.json  --  entries {"n", "d", "lb", "ub", "exact"}.

Cell forms handled:
    "16"              exact value
    "2816-3276"       lower - upper bound
    "2<SUP>19</SUP>"  power of two (lower bound)
    "2<SUP>19</SUP>-599184"
Anything else lands in "unparsed".
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "raw", "binary-1.html")
OUT = os.path.join(ROOT, "data", "tables", "and_bounds.json")

NUM = r"(?:\d+(?:<SUP>\d+</SUP>)?)"
CELL_RE = re.compile(r"^(%s)(?:\s*[-–]\s*(%s))?$" % (NUM, NUM), re.I)


def value(tok):
    m = re.fullmatch(r"(\d+)<SUP>(\d+)</SUP>", tok, re.I)
    if m:
        return int(m.group(1)) ** int(m.group(2))
    return int(tok)


def main():
    raw = open(SRC, encoding="utf-8").read()
    src = re.sub(r"<!--.*?-->", "", raw, flags=re.S)

    tables = re.findall(r"<TABLE[^>]*>.*?</TABLE>", src, flags=re.S | re.I)
    if len(tables) != 1:
        sys.exit("expected exactly one table in %s, found %d" % (SRC, len(tables)))
    t = tables[0]

    rows = []
    for tr in re.split(r"</TR>", t, flags=re.I):
        cells = re.findall(r"<TD[^>]*>(.*?)(?=<TD|</TR>|$)", tr, flags=re.S | re.I)
        cells = [re.sub(r"\s+", " ", c).replace("</TD>", "").strip() for c in cells]
        if any(cells):
            rows.append(cells)

    header = rows[0]
    ds = []
    for c in header:
        m = re.fullmatch(r"d=(\d+)", c)
        if m:
            ds.append(int(m.group(1)))
    assert ds == [4, 6, 8, 10, 12, 14, 16], ds

    entries, unparsed = [], []
    prev_n = None
    for row in rows[1:]:
        n = None
        try:
            n = int(row[0])
        except (ValueError, IndexError):
            continue
        assert prev_n is None or n == prev_n + 1, (prev_n, n)
        prev_n = n
        body = [c for c in row[1:] if c != ""]
        if len(body) != len(ds):
            unparsed.append({"n": n, "raw_html": str(row),
                             "reason": "row has %d value cells, expected %d"
                                       % (len(body), len(ds))})
            continue
        for d, c in zip(ds, body):
            m = CELL_RE.match(c)
            if not m:
                unparsed.append({"n": n, "d": d, "raw_html": c,
                                 "reason": "cell does not match grammar"})
                continue
            lb = value(m.group(1))
            ub = value(m.group(2)) if m.group(2) else lb
            if lb > ub:
                unparsed.append({"n": n, "d": d, "raw_html": c,
                                 "reason": "lb > ub"})
                continue
            entries.append({"n": n, "d": d, "lb": lb, "ub": ub,
                            "exact": lb == ub, "cell_raw": c})

    doc = {
        "source": "data/raw/binary-1.html (A. E. Brouwer, "
                  "https://www.win.tue.nl/~aeb/codes/binary-1.html), table frozen 2019",
        "generated_by": "scripts/parse_and.py",
        "conventions": {
            "range": "only even d and 6 <= n <= 28 are tabulated; "
                     "A(n-1,2e-1) = A(n,2e), A(n,1)=2^n, A(n,2)=2^(n-1), "
                     "A(n,d)=1 for d>n and 2 for 3d/2>n",
            "exact": "lb == ub",
            "powers": "cells written 2^k are expanded to their integer value",
        },
        "counts": {"entries": len(entries), "unparsed": len(unparsed)},
        "unparsed": unparsed,
        "entries": entries,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    print("wrote %s: %d entries, %d unparsed" % (OUT, len(entries), len(unparsed)))
    for u in unparsed:
        print("  UNPARSED", u)


if __name__ == "__main__":
    main()
