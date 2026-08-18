#!/usr/bin/env python3
"""pdfaudit.py -- print a Keri PDF page as baseline-only text.

Strips the 8pt glyphs (subscript q in the titles, superscript classification
counts in exact cells) so the table can be eyeballed against bounds.json.
Usage: python3 pdfaudit.py FILE.pdf [PAGE]
"""
import sys
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar


def page_chars(path, pageno):
    pg = list(extract_pages(path))[pageno]
    out = []

    def walk(o):
        if isinstance(o, LTChar):
            out.append(o)
        elif hasattr(o, '__iter__'):
            for c in o:
                walk(c)
    walk(pg)
    return out


def dump(path, pageno, minsize=9.0):
    chars = [c for c in page_chars(path, pageno) if c.size >= minsize and c.get_text().strip()]
    rows = {}
    for c in chars:
        rows.setdefault(round(c.y0), []).append(c)
    # merge near-equal y
    keys = sorted(rows, reverse=True)
    merged = []
    for k in keys:
        if merged and abs(merged[-1][0] - k) <= 2:
            merged[-1][1].extend(rows[k])
        else:
            merged.append([k, list(rows[k])])
    for y, cs in merged:
        cs.sort(key=lambda c: c.x0)
        line = []
        prev = None
        for c in cs:
            if prev is not None and c.x0 - prev > 1.2:
                line.append(' ')
            line.append(c.get_text())
            prev = c.x1
        print('y=%5d | %s' % (y, ''.join(line)))


if __name__ == '__main__':
    p = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    dump(p, n)
