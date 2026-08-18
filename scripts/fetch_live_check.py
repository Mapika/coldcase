#!/usr/bin/env python3
"""Pre-claim verification: re-fetch Brouwer's live tables and report the CURRENT
bound for one cell.

    python3 scripts/fetch_live_check.py 18 6 6      # A(n,d,w), constant weight
    python3 scripts/fetch_live_check.py 21 10       # A(n,d),   unrestricted binary

Options:
    --json          machine-readable output
    --no-compare    do not diff against data/tables/*.json

Never claim a record without running this first: the local JSON databases are
snapshots, the live page moves.

Deliberately dependency-free (urllib + re), deliberately structure-driven rather
than line-number driven, so that it keeps working when the page is edited.
Rate limited to one request per second, User-Agent "longshots-research".
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parse_andw import parse_cell, tokenize_cell, rows_of, is_sep, plain_int  # noqa: E402

ANDW_URL = "https://aeb.win.tue.nl/codes/Andw.html"
AND_URL = "https://aeb.win.tue.nl/codes/binary-1.html"
UA = "longshots-research"
MIN_INTERVAL = 1.0

_last = [0.0]
_cache = {}


def fetch(url):
    if url in _cache:
        return _cache[url]
    wait = MIN_INTERVAL - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    _last[0] = time.time()
    try:
        html = data.decode("utf-8")
    except UnicodeDecodeError:
        html = data.decode("latin-1")
    _cache[url] = html
    return html


def strip_comments(s):
    return re.sub(r"<!--.*?-->", "", s, flags=re.S)


def tables_with_headings(src):
    """-> list of (heading_text, table_html) in document order."""
    out = []
    pos = 0
    heading = ""
    for m in re.finditer(r"(<h[1-6][^>]*>.*?</h[1-6]>)|(<table[^>]*>.*?</table>)",
                         src, flags=re.S | re.I):
        if m.group(1):
            heading = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m.group(1))).strip()
        else:
            out.append((heading, m.group(2)))
        pos = m.end()
    return out


def table_mode(heading, rows):
    """(lb_only, all_exact) for one table, inferred from its own contents.

    A table that quotes upper bounds always shows at least one "lb-ub" cell; if
    a table has no such cell it either lists lower bounds only ("Bounds on
    A(n,4,w)", "Lower bounds for A(n,8,8)") or exclusively known values
    ("Values of A(n,6,4)").  This reproduces the hand-built table registry of
    scripts/parse_andw.py on the frozen 2026-08-18 snapshot.
    """
    for row in rows:
        for tag, _attrs, inner in row:
            if tag == "td" and re.search(r"[-–]", tokenize_cell(inner)[0]):
                return False, False
    if heading.startswith("Values of"):
        return False, True
    return True, False


# ---------------------------------------------------------------------------
def lookup_cw(html, n, d, w):
    """Return a list of hits {table, cell_raw, lb, ub, exact, ...}."""
    src = strip_comments(html)
    hits = []
    for heading, thtml in tables_with_headings(src):
        rows = rows_of(thtml)
        if not rows:
            continue
        head0 = re.sub(r"<[^>]+>", "", rows[0][0][2]).strip()
        lb_only, all_exact = table_mode(heading, rows)

        # --- main n x w grid for one d --------------------------------------
        if head0 in ("n\\w", "n/w"):
            mh = re.search(r"A\(n,\s*(\d+),\s*w\)", heading)
            if not mh or int(mh.group(1)) != d:
                continue
            weights = [plain_int(c[2]) for c in rows[0][1:]]
            if any(x is None for x in weights):
                continue
            for row in rows[1:]:
                if all(t == "th" for t, _, _ in row):
                    continue
                if plain_int(row[0][2]) != n:
                    continue
                for ww, (_t, attrs, inner) in zip(weights, row[1:]):
                    if ww != w:
                        continue
                    p = parse_cell(inner, lb_only, all_exact)
                    if p and "error" not in p:
                        hits.append(dict(p, table=heading,
                                         cell_raw=re.sub(r"\s+", " ", inner).strip()))
            continue

        # --- per-(d,w) strip -------------------------------------------------
        mh = re.findall(r"A\(n,\s*(\d+),\s*(\d+)\)", heading)
        pairs = {(int(a), int(b)) for a, b in mh}
        if not pairs or d not in {a for a, _ in pairs}:
            continue
        wrows = len({b for a, b in pairs if a == d}) > 1
        ns = None
        for row in rows:
            if is_sep(row):
                ns = None
                continue
            if ns is None:
                cells = row[1:] if row[0][0] == "th" else row
                ns = [plain_int(c[2]) for c in cells]
                if any(x is None for x in ns):
                    ns = None
                continue
            lab = re.sub(r"<[^>]+>", "", row[0][2]).strip()
            mw = re.fullmatch(r"w=(\d+)", lab)
            if mw:
                ww, body = int(mw.group(1)), row[1:]
            elif wrows:
                continue
            else:
                ww = sorted(b for a, b in pairs if a == d)[0]
                body = row
            if ww != w or len(body) != len(ns):
                continue
            for nn, (_t, attrs, inner) in zip(ns, body):
                if nn != n:
                    continue
                p = parse_cell(inner, lb_only, all_exact)
                if p and "error" not in p:
                    hits.append(dict(p, table=heading,
                                     cell_raw=re.sub(r"\s+", " ", inner).strip()))
    return hits


def lookup_binary(html, n, d):
    src = strip_comments(html)
    m = re.search(r"<TABLE[^>]*>.*?</TABLE>", src, flags=re.S | re.I)
    if not m:
        return []
    rows = []
    for tr in re.split(r"</TR>", m.group(0), flags=re.I):
        cells = re.findall(r"<TD[^>]*>(.*?)(?=<TD|</TR>|$)", tr, flags=re.S | re.I)
        cells = [re.sub(r"\s+", " ", c).replace("</TD>", "").strip() for c in cells]
        if any(cells):
            rows.append(cells)
    ds = [int(x) for c in rows[0] for x in re.findall(r"^d=(\d+)$", c)]
    hits = []
    for row in rows[1:]:
        try:
            if int(row[0]) != n:
                continue
        except (ValueError, IndexError):
            continue
        body = [c for c in row[1:] if c != ""]
        if len(body) != len(ds):
            continue
        for dd, c in zip(ds, body):
            if dd != d:
                continue
            mm = re.fullmatch(r"(\d+(?:<SUP>\d+</SUP>)?)"
                              r"(?:\s*[-–]\s*(\d+(?:<SUP>\d+</SUP>)?))?", c, re.I)
            if not mm:
                continue

            def val(t):
                p = re.fullmatch(r"(\d+)<SUP>(\d+)</SUP>", t, re.I)
                return int(p.group(1)) ** int(p.group(2)) if p else int(t)

            lb = val(mm.group(1))
            ub = val(mm.group(2)) if mm.group(2) else lb
            hits.append({"lb": lb, "ub": ub, "exact": lb == ub,
                         "cell_raw": c, "table": "A(n,d) general binary table"})
    return hits


def local_cw(n, d, w):
    p = os.path.join(ROOT, "data", "tables", "andw_bounds.json")
    if not os.path.isfile(p):
        return None
    for e in json.load(open(p, encoding="utf-8"))["entries"]:
        if (e["n"], e["d"], e["w"]) == (n, d, w):
            return e
    return None


def local_binary(n, d):
    p = os.path.join(ROOT, "data", "tables", "and_bounds.json")
    if not os.path.isfile(p):
        return None
    for e in json.load(open(p, encoding="utf-8"))["entries"]:
        if (e["n"], e["d"]) == (n, d):
            return e
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("params", nargs="+", type=int, metavar="N",
                    help="n d [w]  --  three values query A(n,d,w), two query A(n,d)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-compare", action="store_true")
    args = ap.parse_args()

    if len(args.params) == 3:
        n, d, w = args.params
        url = ANDW_URL
        hits = lookup_cw(fetch(url), n, d, w)
        if not hits and w > n - w:                 # complement symmetry
            hits = [dict(h, note="via A(n,d,w)=A(n,d,n-w)")
                    for h in lookup_cw(fetch(url), n, d, n - w)]
        label, local = "A(%d,%d,%d)" % (n, d, w), None if args.no_compare else local_cw(n, d, w)
    elif len(args.params) == 2:
        n, d = args.params
        url = AND_URL
        hits = lookup_binary(fetch(url), n, d)
        label, local = "A(%d,%d)" % (n, d), None if args.no_compare else local_binary(n, d)
    else:
        ap.error("give 2 (n d) or 3 (n d w) integers")

    result = {"query": label, "url": url,
              "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "hits": hits, "local": local}

    if hits:
        lbs = [h["lb"] for h in hits if h["lb"] is not None]
        ubs = [h["ub"] for h in hits if h["ub"] is not None]
        result["live_lb"] = max(lbs) if lbs else None
        result["live_ub"] = min(ubs) if ubs else None
        result["live_exact"] = any(h["exact"] for h in hits)
        if local:
            result["matches_local"] = (local["lb"] == result["live_lb"]
                                       and local["ub"] == result["live_ub"])

    if args.json:
        print(json.dumps(result, indent=1, ensure_ascii=False))
        return 0 if hits else 1

    print("%s  --  live table %s" % (label, url))
    if not hits:
        print("  NOT FOUND on the live page (check the parameters, or the page "
              "layout changed)")
        return 1
    for h in hits:
        print("  [%s]" % h["table"])
        print("    cell   : %s" % h["cell_raw"])
        print("    lb=%s  ub=%s  exact=%s  labels=%s/%s%s"
              % (h["lb"], h["ub"], h["exact"], h.get("lb_label"), h.get("ub_label"),
                 "  " + h["note"] if h.get("note") else ""))
        if h.get("code_file"):
            print("    code   : %s" % h["code_file"])
    print("  ==> LIVE  lb=%s  ub=%s  exact=%s"
          % (result["live_lb"], result["live_ub"], result["live_exact"]))
    if local:
        print("  ==> LOCAL lb=%s  ub=%s  exact=%s   %s"
              % (local["lb"], local["ub"], local["exact"],
                 "MATCH" if result.get("matches_local") else "*** DIFFERS ***"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
