#!/usr/bin/env python3
"""Parse Brouwer's "Bounds for constant weight codes" page into JSON.

Source: data/raw/Andw.html  (mirror of https://www.win.tue.nl/~aeb/codes/Andw.html,
fetched 2026-08-18).

Output: data/tables/andw_bounds.json

Notation honoured (see README of the page itself):
  * main grids  (<h1> "Bounds on A(n,d,w)", anchors #d4 .. #d18): rows = n, cols = w.
  * extension tables (<h3> "Values of A(n,d,w)" / "Bounds on A(n,d,w)"): a row of n
    values followed by one (or, for A(n,4,3)/A(n,4,4), two) rows of cells; blocks are
    separated by <tr><td colspan=.. class="sep"></td></tr>.
  * a cell "LB-UB" gives lower and upper bound; "LB" alone in a table that carries
    upper bounds means the value is determined (lb == ub); a trailing "." always means
    the value is exact; "LB-..." means no upper bound is quoted.
  * <sup>..</sup> after a number is a source label (before the dash -> lower bound
    source, after the dash -> upper bound source).
  * <span class="nw"> (yellow background) marks a bound first given on the page.
  * <td class="be"> marks a d=4 lower bound from [Brouwer & Etzion 2011].
  * <a href="cwc/..."> links an explicit code file.

Anything that does not match the grammar is emitted into the "unparsed" list rather
than guessed at.
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "raw", "Andw.html")
OUT = os.path.join(ROOT, "data", "tables", "andw_bounds.json")

# ---------------------------------------------------------------------------
# Table registry.  Keyed by the 1-based line number of the "<table" tag in the
# frozen source file.  Every table in the file is listed; the ones that are not
# bound tables carry kind="skip" with a reason, so that a future re-fetch that
# adds/removes a table trips the assertion below instead of silently changing
# the output.
#
#   kind "main" : n x w grid for one d
#   kind "ext"  : n-indexed strip for one (d, w)  (or several w, see "wrows")
#   flags:
#     lb_only   : table quotes lower bounds only -> a bare number is NOT exact
#     all_exact : every entry in the table is a known exact value
# ---------------------------------------------------------------------------
TABLES = {
    144: dict(kind="skip", why="'Lost codes' summary table, handled separately"),
    419: dict(kind="main", d=4, lb_only=True, heading="Bounds on A(n,4,w)"),
    649: dict(kind="ext", d=4, wrows=True, all_exact=True, lb_only=True,
              heading="Values of A(n,4,3) and A(n,4,4)"),
    714: dict(kind="ext", d=4, w=5, lb_only=True, heading="Bounds on A(n,4,5)"),
    832: dict(kind="main", d=6, heading="Bounds on A(n,6,w)"),
    1098: dict(kind="ext", d=6, w=4, all_exact=True, heading="Values of A(n,6,4)"),
    1135: dict(kind="ext", d=6, w=5, heading="Values of A(n,6,5)"),
    1223: dict(kind="ext", d=6, w=6, heading="Values of A(n,6,6)"),
    1344: dict(kind="main", d=8, heading="Bounds on A(n,8,w)"),
    1552: dict(kind="ext", d=8, w=5, heading="Values of A(n,8,5)"),
    1675: dict(kind="ext", d=8, w=6, heading="Values of A(n,8,6)"),
    1768: dict(kind="ext", d=8, w=7, heading="Values of A(n,8,7)"),
    1865: dict(kind="ext", d=8, w=8, lb_only=True,
               heading="Lower bounds for A(n,8,8) and kissing numbers"),
    1919: dict(kind="skip", why="kissing numbers tau_n, not A(n,d,w)"),
    1946: dict(kind="skip", why="lexmin vs reverse-lexmin comparison, not the table"),
    2040: dict(kind="main", d=10, heading="Bounds on A(n,10,w)"),
    2267: dict(kind="ext", d=10, w=6, heading="Values of A(n,10,6)"),
    2363: dict(kind="ext", d=10, w=7, heading="Values of A(n,10,7)"),
    2447: dict(kind="ext", d=10, w=8, heading="Values of A(n,10,8)"),
    2525: dict(kind="main", d=12, heading="Bounds on A(n,12,w)"),
    2733: dict(kind="ext", d=12, w=7, heading="Values of A(n,12,7)"),
    2854: dict(kind="ext", d=12, w=8, heading="Values of A(n,12,8)"),
    2928: dict(kind="ext", d=12, w=12, lb_only=True,
               heading="Lower bounds for A(n,12,12)"),
    2949: dict(kind="main", d=14, heading="Bounds on A(n,14,w)"),
    3100: dict(kind="ext", d=14, w=8, heading="Values of A(n,14,8)"),
    3203: dict(kind="ext", d=14, w=14, lb_only=True,
               heading="Lower bounds for A(n,14,14)"),
    3233: dict(kind="main", d=16, heading="Bounds on A(n,16,w)"),
    3347: dict(kind="ext", d=16, w=9, heading="Values of A(n,16,9)"),
    3430: dict(kind="main", d=18, heading="Bounds on A(n,18,w)"),
    3557: dict(kind="ext", d=18, w=10, heading="Values of A(n,18,10)"),
}


# ---------------------------------------------------------------------------
# cell grammar
# ---------------------------------------------------------------------------
SUP = "\x01"  # placeholder marking where a <sup> label sat


def tokenize_cell(html):
    """Return (text, labels, code_file, nw, lost).

    `text` is the cell with anchors/spans removed and every <sup>..</sup>
    replaced by \x01; `labels` is the list of label strings in document order.
    """
    s = html
    lost = 'class="lost"' in s
    nw = 'class="nw"' in s
    m = re.search(r'<a\s+href="(cwc/[^"]+)"', s)
    code_file = m.group(1) if m else None

    labels = []

    def grab(mo):
        lab = re.sub(r"<[^>]+>", "", mo.group(1)).strip()
        labels.append(lab)
        return SUP

    s = re.sub(r"<sup>(.*?)</sup>", grab, s, flags=re.S)
    s = re.sub(r"</?(?:a|span|tt|b|i)\b[^>]*>", "", s)
    s = s.replace("&nbsp;", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s, labels, code_file, nw, lost


# number, then any mix of "." and label markers; optionally  - <same> or - "..."
_SIDE = r"(\d+)((?:\s*(?:\.|%s))*)" % SUP
CELL_RE = re.compile(
    r"^(?:%s)?\s*(?:[-–]\s*(?:%s|(\.\.\.)))?$" % (_SIDE, _SIDE)
)


def parse_cell(html, lb_only=False, all_exact=False):
    """-> dict of parsed fields, or None for an empty cell, or {'error': ...}."""
    text, labels, code_file, nw, lost = tokenize_cell(html)
    if text in ("", "-", "–"):
        return None

    m = CELL_RE.match(text)
    if not m:
        return {"error": "cell does not match grammar", "text": text}

    lb_s, lb_tail, ub_s, ub_tail, dots = m.groups()

    if lb_s is None and ub_s is None and dots is None:
        return {"error": "no numbers in cell", "text": text}

    lb = int(lb_s) if lb_s is not None else None
    ub = int(ub_s) if ub_s is not None else None
    has_dash = ("-" in text) or ("–" in text)
    lb_dot = "." in (lb_tail or "")
    ub_dot = "." in (ub_tail or "")

    # split the label list: labels appearing in lb_tail belong to the lower
    # bound, the rest to the upper bound.
    n_lb_labels = (lb_tail or "").count(SUP)
    lb_labels = labels[:n_lb_labels]
    ub_labels = labels[n_lb_labels:]

    if lb is not None and ub is None and dots is None and not has_dash:
        # bare "LB" (possibly dotted)
        if all_exact or lb_dot:
            exact, ub = True, lb
        elif lb_only:
            exact, ub = False, None
        else:
            # table carries upper bounds and none is quoted -> value determined
            exact, ub = True, lb
    elif lb is not None and ub is not None:
        exact = (lb == ub) or lb_dot or ub_dot
        if lb > ub:
            return {"error": "lb > ub", "text": text}
    elif lb is not None and dots is not None:
        exact, ub = False, None          # "LB-..."  upper bound not quoted
    elif lb is None and ub is not None:
        exact = ub_dot                   # "-UB"     lower bound not quoted
        if exact:
            return {"error": "'-UB' cell marked exact", "text": text}
    else:
        return {"error": "unhandled cell", "text": text}

    return {
        "lb": lb,
        "ub": ub,
        "exact": exact,
        "lb_label": " ".join(lb_labels) or None,
        "ub_label": " ".join(ub_labels) or None,
        "code_file": code_file,
        "nw": nw,
        "lost": lost,
    }


# ---------------------------------------------------------------------------
# html helpers
# ---------------------------------------------------------------------------
CELL_TAG_RE = re.compile(r"<(t[hd])([^>]*)>(.*?)</\1>", re.S | re.I)


def rows_of(table_html):
    out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)(?=<tr[^>]*>|</table>|$)", table_html,
                         flags=re.S | re.I):
        cells = [(tag.lower(), attrs, inner)
                 for tag, attrs, inner in CELL_TAG_RE.findall(tr)]
        if cells:
            out.append(cells)
    return out


def is_sep(row):
    return len(row) == 1 and 'class="sep"' in row[0][1]


def plain_int(inner):
    t = re.sub(r"<[^>]+>", "", inner).replace("&nbsp;", " ").strip()
    return int(t) if re.fullmatch(r"\d+", t) else None


# ---------------------------------------------------------------------------
def main():
    raw = open(SRC, encoding="utf-8").read()
    src = re.sub(r"<!--.*?-->", "", raw, flags=re.S)   # drop HTML comments

    # locate tables in the *original* text so line numbers match the registry
    starts = [m.start() for m in re.finditer(r"<table", raw, flags=re.I)]
    line_of = {p: raw.count("\n", 0, p) + 1 for p in starts}
    lines = sorted(line_of[p] for p in starts)
    if sorted(TABLES) != lines:
        sys.exit("table registry out of date: file has tables at lines %s" % lines)

    # re-find the (comment-stripped) tables in document order
    tables = re.findall(r"<table[^>]*>.*?</table>", src, flags=re.S | re.I)
    assert len(tables) == len(lines), (len(tables), len(lines))

    entries = []
    unparsed = []

    def emit(n, d, w, cell_html, parsed, where):
        if parsed is None:
            return
        if "error" in parsed:
            unparsed.append({"n": n, "d": d, "w": w, "raw_html": cell_html.strip(),
                             "location": where, "reason": parsed["error"]})
            return
        e = {"n": n, "d": d, "w": w,
             "lb": parsed["lb"], "ub": parsed["ub"], "exact": parsed["exact"],
             "lb_label": parsed["lb_label"], "ub_label": parsed["ub_label"],
             "cell_raw": re.sub(r"\s+", " ", cell_html).strip(),
             "source_table": where}
        if parsed["code_file"]:
            e["code_file"] = parsed["code_file"]
        if parsed["nw"]:
            e["nw"] = True
        if parsed["lost"]:
            e["lost"] = True
        entries.append(e)

    for line, thtml in zip(lines, tables):
        spec = TABLES[line]
        if spec["kind"] == "skip":
            continue
        d = spec["d"]
        lb_only = spec.get("lb_only", False)
        all_exact = spec.get("all_exact", False)
        where = "%s @line %d" % (spec["heading"], line)
        rows = rows_of(thtml)

        if spec["kind"] == "main":
            header = rows[0]
            assert header[0][2].strip() in ("n\\w", "n/w"), header[0]
            weights = [plain_int(c[2]) for c in header[1:]]
            assert all(x is not None for x in weights), weights
            prev_n = None
            for row in rows[1:]:
                if all(t == "th" for t, _, _ in row):
                    continue                       # repeated header at the foot
                # the row header is normally <th>n</th>; a few rows of the d=18
                # grid use <td>n</td> instead, so key on "plain integer" instead
                # of on the tag.
                n = plain_int(row[0][2])
                assert n is not None, (where, row[0])
                assert prev_n is None or n == prev_n + 1, (where, prev_n, n)
                prev_n = n
                body = row[1:]
                if len(body) > len(weights):
                    unparsed.append({"n": n, "d": d, "w": None,
                                     "raw_html": str(row), "location": where,
                                     "reason": "more cells than columns"})
                    continue
                for w, (_tag, attrs, inner) in zip(weights, body):
                    p = parse_cell(inner, lb_only, all_exact)
                    if p and "error" not in p and 'class="be"' in attrs:
                        p["be"] = True
                    e_before = len(entries)
                    emit(n, d, w, inner, p, where)
                    if 'class="be"' in attrs and len(entries) > e_before:
                        entries[-1]["be"] = True
        else:
            ns = None
            block_row = 0
            for row in rows:
                if is_sep(row):
                    ns, block_row = None, 0
                    continue
                if ns is None:
                    cells = row[1:] if row[0][0] == "th" else row
                    ns = [plain_int(c[2]) for c in cells]
                    assert all(x is not None for x in ns), (where, row)
                    continue
                if spec.get("wrows"):
                    lab = re.sub(r"<[^>]+>", "", row[0][2]).strip()
                    m = re.fullmatch(r"w=(\d+)", lab)
                    assert m, (where, lab)
                    w = int(m.group(1))
                    body = row[1:]
                else:
                    w = spec["w"]
                    body = row
                assert len(body) == len(ns), (where, len(body), len(ns))
                for n, (_tag, attrs, inner) in zip(ns, body):
                    emit(n, d, w, inner, parse_cell(inner, lb_only, all_exact), where)
                block_row += 1

    # ---- the "lost codes" section: best *explicitly available* code ---------
    lost_codes = []
    m = re.search(r'<h3><a name="lost">.*?</table>', src, flags=re.S)
    if m:
        rows = rows_of(m.group(0))
        params = [re.sub(r"<[^>]+>", "", c[2]).strip() for c in rows[0][1:]]
        bsss = [re.sub(r"<[^>]+>", "", c[2]).strip() for c in rows[1][1:]]
        code = rows[2][1:]
        for p, b, (_t, _a, inner) in zip(params, bsss, code):
            mm = re.fullmatch(r"A\((\d+),(\d+),(\d+)\)", p)
            assert mm, p
            n, d, w = map(int, mm.groups())
            tok = tokenize_cell(inner)
            lost_codes.append({
                "n": n, "d": d, "w": w,
                "bsss_lb": int(b),
                "best_explicit_lb": int(re.match(r"\d+", tok[0]).group(0)),
                "best_explicit_label": " ".join(tok[1]) or None,
                "code_file": tok[2],
                "note": "BSSS lower bound has no surviving explicit construction; "
                        "best_explicit_lb is the largest code actually available.",
            })

    # ---- consistency: same (n,d,w) appearing in two tables -----------------
    seen = {}
    conflicts = []
    for e in entries:
        k = (e["n"], e["d"], e["w"])
        if k in seen:
            a = seen[k]
            if (a["lb"], a["ub"], a["exact"]) != (e["lb"], e["ub"], e["exact"]):
                conflicts.append({"key": list(k), "a": a, "b": e})
        else:
            seen[k] = e

    # de-duplicate by merging: keep the strongest bound of each kind.  (The main
    # d=... grids and the per-(d,w) strips overlap around n=29..36 and the page is
    # not always kept in sync; taking max(lb)/min(ub) is safe in both directions.)
    merged = {}
    order = []
    for e in entries:
        k = (e["n"], e["d"], e["w"])
        if k not in merged:
            merged[k] = dict(e)
            order.append(k)
            continue
        m = merged[k]
        if e["lb"] is not None and (m["lb"] is None or e["lb"] > m["lb"]):
            for f in ("lb", "lb_label", "nw", "be", "lost"):
                m.pop(f, None)
            m["lb"] = e["lb"]
            m["lb_label"] = e["lb_label"]
            for f in ("nw", "be", "lost"):
                if e.get(f):
                    m[f] = True
            if e.get("code_file"):
                m["code_file"] = e["code_file"]
        if e["ub"] is not None and (m["ub"] is None or e["ub"] < m["ub"]):
            m["ub"] = e["ub"]
            m["ub_label"] = e["ub_label"]
        if e["exact"]:
            m["exact"] = True
        if m.get("code_file") is None and e.get("code_file"):
            m["code_file"] = e["code_file"]
        if m["lb"] is not None and m["lb"] == m["ub"]:
            m["exact"] = True
        m["cell_raw"] = m["cell_raw"] + " | " + e["cell_raw"]
        m["source_table"] = m["source_table"] + " | " + e["source_table"]
    deduped = [merged[k] for k in order]
    deduped.sort(key=lambda e: (e["d"], e["n"], e["w"]))

    doc = {
        "source": "data/raw/Andw.html (A. E. Brouwer, "
                  "https://www.win.tue.nl/~aeb/codes/Andw.html), fetched 2026-08-18",
        "generated_by": "scripts/parse_andw.py",
        "conventions": {
            "exact": "lb == ub is proven; ub is then filled in with lb",
            "ub_null": "no upper bound is quoted on the page for this cell "
                       "(d=4 tables and the A(n,8,8)/A(n,12,12)/A(n,14,14) lower-bound "
                       "strips quote lower bounds only; a few cells read 'LB-...')",
            "lb_null": "cells of the form '-UB' quote an upper bound only",
            "labels": "for a cell without a dash both bounds share the label list, "
                      "which is reported in lb_label",
            "nw": "true when the page shows the bound on a yellow background "
                  "(first given on the page itself)",
            "be": "true for d=4 cells with class 'be' (Brouwer & Etzion 2011)",
            "symmetry": "entries are as printed; A(n,d,w) = A(n,d,n-w) is NOT "
                        "materialised here",
        },
        "counts": {
            "entries": len(deduped),
            "duplicates_dropped": len(entries) - len(deduped),
            "conflicts": len(conflicts),
            "unparsed": len(unparsed),
        },
        "lost_codes": lost_codes,
        "conflicts": conflicts,
        "unparsed": unparsed,
        "entries": deduped,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    print("wrote %s: %d entries, %d unparsed, %d conflicts, %d duplicates"
          % (OUT, len(deduped), len(unparsed), len(conflicts),
             len(entries) - len(deduped)))
    for c in conflicts:
        print("  CONFLICT", c["key"], c["a"]["cell_raw"], "vs", c["b"]["cell_raw"])
    for u in unparsed:
        print("  UNPARSED", u)


if __name__ == "__main__":
    main()
