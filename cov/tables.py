#!/usr/bin/env python3
"""
tables.py -- extract the Keri (2011) bounds tables on K_q(n,R) from the frozen
PDFs at http://old.sztaki.hu/~keri/codes/ into cov/bounds.json.

K_q(n,R) = minimum size of a q-ary code of length n whose Hamming balls of
radius R cover Z_q^n.

PDF layout (discovered by character-level inspection with pdfminer.six):

  * Every table cell is rendered on a text baseline as
        [lower-bound key letter] VALUE [upper-bound key letter]
    where VALUE is either a single integer (bounds coincide, i.e. the value is
    known exactly) or "lb-ub" with an en-dash (U+2013) or, occasionally, an
    ASCII hyphen.
  * Exactly-known cells may additionally carry a RAISED, SMALLER (8pt vs 10.9pt)
    integer immediately after the value.  That is Keri's classification count:
    the number of inequivalent optimal covering codes.  It is NOT part of the
    bound.  Naive text extraction glues it onto the value (e.g. K_3(6,3)=6 with
    28 inequivalent optima extracts as "628"), which is why this parser works
    from glyph coordinates and font sizes rather than from extracted text.
  * The table title "Bounds on K_q(n,R)" carries q as a LOWERED 8pt subscript.
  * Key letters are explained on a trailing "Key to the tables" page, separately
    for lower and upper bounds; the same letter can mean different things for
    the two, and across table families.

Cells are assigned to (n,R) by x-coordinate against the column centres taken
from the "n  R = 1  R = 2 ..." header line, so the lower/upper key-letter
ambiguity that plagues pure token parsing ("g 8 g 4 1") never arises.

Post-2011 lower-bound updates from Gijswijt & Polak, "Semidefinite lower bounds
for covering codes", arXiv:2504.01932, are merged in as a separate provenance
layer.  They only ever raise lower bounds, so they do not move the upper-bound
targets we hunt; they are recorded for gap reporting.

Usage:
    python3 tables.py            # rebuild bounds.json from the PDFs
    python3 tables.py --check    # rebuild + run the cross-check suite
"""

import json
import os
import re
import sys
from collections import defaultdict

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw')
RAW = os.path.normpath(RAW)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bounds.json')

PDFS = [
    'keri_2_tables.pdf',
    'keri_3_tables.pdf',
    'keri_4-5_tables.pdf',
    'keri_6-21_tables.pdf',
]

# Latin ordinal words Keri uses in the sub-title of each table, for cross-check
# of the subscript q read off the title glyphs.
ORDINALS = {
    'binary': 2, 'ternary': 3, 'quaternary': 4, 'quinary': 5, 'senary': 6,
    'septenary': 7, 'octonary': 8, 'novenary': 9, 'denary': 10,
    'undenary': 11, 'duodenary': 12, 'tredenary': 13, 'quattuordenary': 14,
    'quindenary': 15, 'sexdenary': 16, 'septemdenary': 17, 'octodenary': 18,
    'undevicenary': 19, 'vicenary': 20, 'viginti-unary': 21,
}

BASE_SIZE_MIN = 9.0          # glyphs at >= this size sit on the text baseline
DASHES = '–—-'     # en dash, em dash, hyphen


# --------------------------------------------------------------------------
# glyph extraction
# --------------------------------------------------------------------------

def page_glyphs(path):
    """Yield, per page, a list of (x0, x1, y0, size, char) for non-blank glyphs."""
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTChar

    for page in extract_pages(path):
        out = []

        def walk(obj):
            if isinstance(obj, LTChar):
                t = obj.get_text()
                if t.strip():
                    out.append((obj.x0, obj.x1, obj.y0, obj.size, t))
            elif hasattr(obj, '__iter__'):
                for c in obj:
                    walk(c)
        walk(page)
        yield out


def group_lines(glyphs, tol=2.0):
    """Group baseline glyphs into text lines by y0.  Returns [(y, [glyph...])]."""
    buckets = defaultdict(list)
    for g in glyphs:
        buckets[round(g[2])].append(g)
    lines = []
    for y in sorted(buckets, reverse=True):
        if lines and abs(lines[-1][0] - y) <= tol:
            lines[-1][1].extend(buckets[y])
        else:
            lines.append((y, list(buckets[y])))
    for _, gs in lines:
        gs.sort(key=lambda g: g[0])
    return lines


def line_text_map(gs, gap=1.2):
    """Render a glyph run to text, inserting a space at every visual gap.

    Returns (text, imap) where imap[i] is the index into gs of the glyph that
    produced text[i], or None for an inserted space.
    """
    parts, imap, prev = [], [], None
    for k, g in enumerate(gs):
        if prev is not None and g[0] - prev > gap:
            parts.append(' ')
            imap.append(None)
        parts.append(g[4])
        imap.append(k)
        prev = g[1]
    return ''.join(parts), imap


def line_text(gs, gap=1.2):
    return line_text_map(gs, gap)[0]


# --------------------------------------------------------------------------
# table parsing
# --------------------------------------------------------------------------

VALUE_RE = re.compile(r'^(\d+)(?:[' + DASHES + r'](\d+))?$')

CELL_GAP = 9.0   # pt; splits table cells, never splits "key value key"


def split_cells(gs, gap):
    """Split a sorted glyph run into groups separated by gaps > `gap` points."""
    cells, cur, prev = [], [], None
    for g in gs:
        if prev is not None and g[0] - prev > gap:
            cells.append(cur)
            cur = []
        cur.append(g)
        prev = g[1]
    if cur:
        cells.append(cur)
    # A cell never consists of a key letter alone.  Such a group appears when an
    # exact value carries a wide classification superscript ("d 6^28 u"), which
    # pushes the trailing key letter more than `gap` away from the value.  Merge
    # it back into the horizontally nearer neighbouring cell (never into the
    # row-label cell at index 0).
    i = 1
    while i < len(cells):
        c = cells[i]
        if len(c) == 1 and re.fullmatch(r'[a-z]', c[0][4]):
            dl = c[0][0] - cells[i - 1][-1][1] if i - 1 >= 1 else float('inf')
            dr = cells[i + 1][0][0] - c[0][1] if i + 1 < len(cells) else float('inf')
            if dl == dr == float('inf'):
                i += 1
                continue
            j = i - 1 if dl <= dr else i + 1
            cells[j] = sorted(cells[j] + c, key=lambda g: g[0])
            del cells[i]
            continue
        i += 1
    return cells


def parse_cell(gs, base_y):
    """Parse one cell's glyphs into (lb, ub, lb_key, ub_key, n_optimal)."""
    base = [g for g in gs if g[3] >= BASE_SIZE_MIN]
    sup = [g for g in gs if g[3] < BASE_SIZE_MIN and g[2] > base_y + 1.0]
    if not base:
        return None
    txt = line_text(base)
    toks = txt.split()
    lb_key = ub_key = None
    vals = [t for t in toks if VALUE_RE.match(t)]
    if len(vals) != 1:
        raise ValueError('cell %r: expected exactly one value token, got %r' % (txt, vals))
    vi = toks.index(vals[0])
    pre, post = toks[:vi], toks[vi + 1:]
    if len(pre) > 1 or len(post) > 1:
        raise ValueError('cell %r: unexpected key tokens' % txt)
    if pre:
        if not re.fullmatch(r'[a-z]', pre[0]):
            raise ValueError('cell %r: bad lower key %r' % (txt, pre[0]))
        lb_key = pre[0]
    if post:
        if not re.fullmatch(r'[a-z]', post[0]):
            raise ValueError('cell %r: bad upper key %r' % (txt, post[0]))
        ub_key = post[0]
    m = VALUE_RE.match(vals[0])
    lb = int(m.group(1))
    ub = int(m.group(2)) if m.group(2) else lb
    n_opt = None
    if sup:
        s = line_text(sup).replace(' ', '')
        if s.isdigit():
            n_opt = int(s)
    return lb, ub, lb_key, ub_key, n_opt


def parse_page(glyphs, src, pageno):
    """Parse every table on one page.  Returns (entries, legend_or_None)."""
    base = [g for g in glyphs if g[3] >= BASE_SIZE_MIN]
    small = [g for g in glyphs if g[3] < BASE_SIZE_MIN]
    lines = group_lines(base)

    entries = []
    legend = None
    q = None
    cols = None            # [(R, xlo, xhi)]
    n_xhi = None

    for y, gs in lines:
        txt = line_text(gs)
        flat = txt.replace(' ', '')

        if flat.startswith('KeytothetablesforK'):
            legend = parse_legend(lines, src, pageno)
            return entries, legend

        if flat.startswith('BoundsonK'):
            # q is the subscript: small glyphs just below this baseline,
            # positioned right after the 'K'.
            kx = next(g[1] for g in gs if g[4] == 'K')
            sub = [g for g in small if base_y_near(g[2], y, -6.0, 0.0) and kx - 1 <= g[0] <= kx + 14]
            sub.sort(key=lambda g: g[0])
            s = ''.join(g[4] for g in sub)
            q = int(s) if s.isdigit() else None
            cols = None
            continue

        # sub-title, e.g. "(lower and upper bounds on the size of senary ...)"
        if flat.startswith('(lowerandupper'):
            # longest word first: "denary" is a suffix of "undenary", "ternary"
            # of "quaternary", etc.
            for word in sorted(ORDINALS, key=len, reverse=True):
                if re.search(r'(?<![a-z-])' + re.escape(word) + r'(?![a-z])', txt):
                    qq = ORDINALS[word]
                    if q is not None and q != qq:
                        raise ValueError('%s p%d: subscript q=%s but sub-title says %s'
                                         % (src, pageno + 1, q, word))
                    q = qq
                    break
            continue

        # column header:  n   R = 1   R = 2   ...
        if re.match(r'^n\b', txt) and re.search(r'R\s*=\s*\d', txt):
            _, imap = line_text_map(gs)
            centres = []
            for m in re.finditer(r'R\s*=\s*(\d+)', txt):
                gi = [imap[i] for i in range(m.start(), m.end()) if imap[i] is not None]
                centres.append((int(m.group(1)),
                                (gs[gi[0]][0] + gs[gi[-1]][1]) / 2.0))
            if not centres:
                continue
            nx = next(g[1] for g in gs if g[4] == 'n')
            cols = centres           # [(R, x_centre)]
            n_centre = nx - 3.0
            continue

        if cols is None or q is None:
            continue

        # Split the row into cells at large horizontal gaps.  Inside a cell the
        # widest gap is the word space between the value and a key letter (~4pt);
        # between cells the gap is >= ~15pt in every Keri table.
        cells = split_cells(gs, CELL_GAP)
        if not cells:
            continue
        label = line_text(cells[0]).strip()
        if not label.isdigit():
            continue
        n = int(label)

        # assign each remaining cell to the nearest column centre
        used = {}
        for cg in cells[1:]:
            mid = (cg[0][0] + cg[-1][1]) / 2.0
            R = min(cols, key=lambda rc: abs(rc[1] - mid))[0]
            if R in used:
                print('WARN: %s p%d n=%d: two cells map to R=%d (%r, %r)'
                      % (src, pageno + 1, n, R, line_text(used[R]), line_text(cg)),
                      file=sys.stderr)
            used[R] = cg

        for R, cg in sorted(used.items()):
            lo, hi = cg[0][0] - 2.0, cg[-1][1] + 2.0
            cellsmall = [g for g in small
                         if lo <= g[0] <= hi and base_y_near(g[2], y, 1.0, 9.0)]
            try:
                got = parse_cell(cg + cellsmall, y)
            except ValueError as ex:
                print('WARN: %s p%d K_%s(%d,%d): %s' % (src, pageno + 1, q, n, R, ex),
                      file=sys.stderr)
                continue
            if got is None:
                continue
            lb, ub, lk, uk, nopt = got
            entries.append(dict(q=q, n=n, R=R, lb=lb, ub=ub,
                                lb_key=lk, ub_key=uk, n_optimal=nopt,
                                src=src, page=pageno + 1))
    return entries, legend


def base_y_near(y, base, lo, hi):
    return base + lo <= y <= base + hi


def parse_legend(lines, src, pageno):
    """Parse a 'Key to the tables' page into {'lower': {...}, 'upper': {...}}."""
    out = {'lower': {}, 'upper': {}, 'src': src, 'page': pageno + 1, 'applies_to': None}
    which = None
    pending = []
    for y, gs in lines:
        txt = line_text(gs).strip()
        flat = txt.replace(' ', '')
        if flat.startswith('KeytothetablesforK'):
            out['applies_to'] = txt
            continue
        if flat == 'Lowerbounds':
            which = 'lower'
            continue
        if flat == 'Upperbounds':
            which = 'upper'
            continue
        if which is None:
            continue
        if txt in ('¨', '`', '˚', '~', '^'):
            pending.append(txt)          # stray combining accent on its own line
            continue
        m = re.match(r'^([a-z]|unmarked)\s+(.*)$', txt)
        if m:
            out[which][m.group(1)] = m.group(2).strip()
        pending = []
    return out


# --------------------------------------------------------------------------
# post-2011 lower-bound updates
# --------------------------------------------------------------------------

# Gijswijt & Polak, "Semidefinite lower bounds for covering codes",
# arXiv:2504.01932 (v1 2025-04-02; revised 2026-06-19).  Tables 1-3: every cell
# where their symmetry-reduced SDP beats the previously known lower bound.
# Fields: q, n, R, old lower bound, new lower bound, upper bound as printed in
# their table.  Only q in {2,3,4,5} appears anywhere in the paper: the q >= 6
# corner is untouched, which is exactly the corner we search.
GP2025 = [
    # (q, n, R, old_lb, new_lb, ub)
    (2, 13, 1, 598, 607, 704), (2, 14, 1, 1172, 1185, 1408),
    (2, 17, 1, 7419, 7426, 8192), (2, 21, 1, 96125, 96477, 122880),
    (2, 22, 1, 190651, 191501, 245760), (2, 25, 1, 1298238, 1301089, 1556480),
    (2, 26, 1, 2581111, 2589179, 3112960), (2, 29, 1, 17997161, 18000844, 23068672),
    (2, 33, 1, 253523901, 253764801, 268435456),
    (2, 13, 2, 97, 101, 128), (2, 14, 2, 159, 170, 248), (2, 17, 2, 859, 889, 1536),
    (2, 23, 2, 30686, 30828, 32768), (2, 26, 2, 191229, 192747, 262144),
    (2, 29, 2, 1231554, 1239885, 2097152), (2, 32, 2, 8170308, 8173960, 16776960),
    (2, 12, 3, 18, 19, 28), (2, 27, 3, 40683, 41012, 65536),
    (2, 15, 4, 22, 23, 32), (2, 16, 4, 33, 34, 64),
    (2, 16, 5, 13, 14, 28), (2, 17, 5, 19, 20, 32), (2, 18, 5, 27, 28, 64),
    (2, 18, 6, 12, 13, 28), (2, 19, 6, 16, 17, 32), (2, 21, 6, 33, 34, 64),
    (2, 21, 7, 14, 15, 32), (2, 22, 7, 20, 21, 64),
    (2, 23, 8, 13, 14, 32), (2, 24, 8, 18, 19, 64),
    (2, 25, 9, 12, 13, 32), (2, 26, 9, 16, 17, 56),
    (2, 27, 10, 11, 12, 32), (2, 29, 10, 19, 20, 64),
    (3, 8, 1, 402, 403, 486), (3, 9, 1, 1060, 1064, 1269),
    (3, 7, 2, 26, 27, 34), (3, 8, 2, 54, 58, 81), (3, 9, 2, 130, 132, 219),
    (3, 14, 2, 12204, 12323, 19683),
    (3, 8, 3, 14, 16, 27), (3, 9, 3, 27, 31, 54), (3, 10, 3, 57, 61, 105),
    (3, 11, 3, 117, 129, 243), (3, 13, 3, 612, 640, 1215),
    (3, 10, 4, 17, 18, 36), (3, 11, 4, 30, 34, 81), (3, 12, 4, 62, 65, 175),
    (3, 13, 4, 123, 130, 335), (3, 14, 4, 255, 273, 729),
    (3, 11, 5, 11, 12, 27), (3, 12, 5, 18, 21, 54), (3, 13, 5, 33, 37, 108),
    (3, 14, 5, 59, 69, 243),
    (3, 13, 6, 13, 14, 36), (3, 14, 6, 21, 24, 81),
    (4, 7, 1, 762, 776, 992), (4, 11, 1, 123846, 124941, 131072),
    (4, 6, 2, 32, 33, 52), (4, 7, 2, 84, 88, 128), (4, 8, 2, 240, 251, 352),
    (4, 9, 2, 751, 775, 1024), (4, 10, 2, 2412, 2460, 4096),
    (4, 11, 2, 7974, 8072, 15872),
    (4, 8, 3, 44, 46, 96), (4, 9, 3, 110, 116, 256), (4, 11, 3, 849, 885, 2048),
    (4, 9, 4, 26, 27, 64), (4, 10, 4, 59, 62, 208), (4, 11, 4, 148, 150, 512),
    (4, 11, 5, 36, 37, 128),
    (5, 5, 1, 160, 162, 184), (5, 7, 1, 2722, 2765, 3125),
    (5, 8, 1, 11945, 12134, 15625), (5, 9, 1, 53138, 53896, 78125),
    (5, 10, 1, 238993, 241122, 390625),
    (5, 7, 2, 225, 236, 525), (5, 8, 2, 821, 861, 1625),
    (5, 11, 2, 52842, 53309, 78125),
    (5, 8, 3, 109, 111, 325), (5, 9, 3, 330, 354, 1275),
    (5, 10, 3, 1163, 1215, 3125), (5, 11, 3, 4255, 4366, 15625),
    (5, 10, 4, 162, 177, 875), (5, 11, 4, 535, 546, 3125),
]

GP_REF = {
    'id': 'arXiv:2504.01932',
    'title': 'Semidefinite lower bounds for covering codes',
    'authors': 'Dion Gijswijt, Sven Polak',
    'submitted': '2025-04-02',
    'revised': '2026-06-19',
    'note': ('Improvements are LOWER bounds only; they never move an upper bound. '
             'The paper treats q in {2,3,4,5} exclusively -- q >= 6 is untouched. '
             'Their SDP also gives K_5(11,5) >= 100, which does not beat Keri.'),
    'code': 'https://github.com/CoveringCodes',
}


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build():
    entries = []
    legends = []
    for fn in PDFS:
        path = os.path.join(RAW, fn)
        if not os.path.exists(path):
            print('WARN: missing %s' % path, file=sys.stderr)
            continue
        for pageno, glyphs in enumerate(page_glyphs(path)):
            try:
                ents, leg = parse_page(glyphs, fn, pageno)
            except Exception as e:
                print('WARN: %s p%d: %s' % (fn, pageno + 1, e), file=sys.stderr)
                continue
            entries.extend(ents)
            if leg:
                legends.append(leg)

    # sanity: no duplicate (q,n,R) with conflicting values
    seen = {}
    for e in entries:
        k = (e['q'], e['n'], e['R'])
        if k in seen and (seen[k]['lb'], seen[k]['ub']) != (e['lb'], e['ub']):
            print('WARN: conflicting duplicate for %s: %s vs %s'
                  % (k, seen[k], e), file=sys.stderr)
        seen[k] = e

    # merge GP2025 lower bounds
    gp = []
    for q, n, R, old, new, ub in GP2025:
        rec = dict(q=q, n=n, R=R, old_lb=old, new_lb=new, ub_in_paper=ub)
        e = seen.get((q, n, R))
        if e is not None:
            rec['keri_lb'] = e['lb']
            rec['keri_ub'] = e['ub']
            rec['keri_lb_matches_paper_old'] = (e['lb'] == old)
            rec['keri_ub_matches_paper'] = (e['ub'] == ub)
            e['lb_updated'] = new
            e['lb_updated_src'] = GP_REF['id']
        gp.append(rec)

    db = {
        'meta': {
            'description': 'Best known bounds on K_q(n,R), the minimum size of a '
                           'q-ary covering code of length n and covering radius R.',
            'primary_source': {
                'author': 'Gerzson Keri',
                'title': 'Tables for bounds on covering codes',
                'url': 'http://old.sztaki.hu/~keri/codes/',
                'frozen': '2011',
                'files': PDFS,
            },
            'lower_bound_updates': GP_REF,
            'fields': {
                'lb': 'Keri 2011 lower bound',
                'ub': 'Keri 2011 upper bound (this is what we try to beat)',
                'lb_key': 'Keri key letter for the lower bound (see keys[])',
                'ub_key': 'Keri key letter for the upper bound (see keys[])',
                'n_optimal': 'number of inequivalent optimal codes (Keri superscript), '
                             'only present when lb == ub',
                'lb_updated': 'post-2011 improved lower bound, if any',
            },
            'generated_by': 'cov/tables.py',
        },
        'keys': legends,
        'entries': sorted(entries, key=lambda e: (e['q'], e['R'], e['n'])),
        'lower_bound_updates_2025': gp,
    }
    return db


# --------------------------------------------------------------------------
# cross-checks
# --------------------------------------------------------------------------

# (q, n, R, expected_lb, expected_ub, source_of_expectation)
CHECKS = [
    (10, 10, 5, 632, 7106, 'task brief'),
    (7, 10, 5, 160, 1225, 'task brief'),
    (6, 10, 4, 417, 2952, 'task brief'),
    (5, 11, 5, 103, 625, 'task brief'),
    (3, 11, 4, 30, 81, 'Keri (brief quotes lb 34 = Gijswijt-Polak 2025)'),
    (3, 14, 5, 59, 243, 'Keri (brief quotes lb 69 = Gijswijt-Polak 2025)'),
    (3, 12, 5, 18, 54, 'Keri (brief quotes lb 21 = Gijswijt-Polak 2025)'),
    (9, 10, 4, 3872, 19683, 'Keri; brief quoted 481-3969 which is K_9(10,5)'),
    (8, 9, 4, 409, 2944, 'Keri; brief quoted 287-2461 which is K_8(10,5)'),
    (9, 10, 5, 481, 3969, 'value the brief attributed to K_9(10,4)'),
    (8, 10, 5, 287, 2461, 'value the brief attributed to K_8(9,4)'),
    # textbook / structural sanity
    (3, 4, 1, 9, 9, 'perfect ternary Hamming code [4,2] -> 9'),
    (2, 3, 1, 2, 2, 'perfect binary repetition code'),
    (5, 6, 1, 625, 625, 'perfect 5-ary Hamming code [6,4] -> 625'),
    (7, 8, 1, 117649, 117649, 'perfect 7-ary Hamming code [8,6] -> 7^6'),
    (9, 10, 1, 43046721, 43046721, 'perfect 9-ary Hamming code [10,8] -> 9^8'),
    (3, 13, 1, 59049, 59049, 'perfect ternary Hamming code [13,10] -> 3^10'),
]


THIRD_PARTY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'data', 'keri_third_party.csv')


def diff_third_party(db, path=THIRD_PARTY):
    """Diff our parse against an independent transcription of the same tables.

    data/keri_third_party.csv is the Keri snapshot shipped with
    github.com/florath/covering-codes-lean (reference-data/keri/
    non_mixed_covering_codes.csv), transcribed by someone else, from the same
    PDFs, for a Lean formalisation project.  Agreement on every cell is far
    stronger evidence than any internal consistency check we can run on
    ourselves.
    """
    import csv
    from collections import defaultdict
    if not os.path.exists(path):
        print('  (no third-party table at %s)' % path)
        return
    theirs = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        theirs[(int(r['q']), int(r['n']), int(r['r']))][r['bound_type']] = \
            int(r['bound_value'])
    mine = {(e['q'], e['n'], e['R']): (e['lb'], e['ub']) for e in db['entries']}
    shared = set(mine) & set(theirs)
    bad = 0
    for k in sorted(shared):
        a = mine[k]
        b = (theirs[k].get('lower'), theirs[k].get('upper'))
        if a != b:
            bad += 1
            if bad <= 20:
                print('  MISMATCH K_%d(%d,%d): ours %s, theirs %s'
                      % (k[0], k[1], k[2], a, b))
    print('  ours %d cells, theirs %d cells, %d only ours, %d only theirs'
          % (len(mine), len(theirs), len(set(mine) - set(theirs)),
             len(set(theirs) - set(mine))))
    print('  => %d mismatches out of %d shared cells' % (bad, len(shared)))


def audit_internal(db):
    """Is a tabulated upper bound ever beaten by Keri's own rules on other cells?

    Applies, to every cell, the three derivation rules the table itself uses --
    K_q(n,R) <= K_q(n-1,R-1) (key c), K_q(n,R) <= q K_q(n-1,R) (key e), and the
    direct sum over every split (key f) -- using only tabulated values, and
    reports any cell where the derived value is smaller than the printed one.
    """
    idx = {(e['q'], e['n'], e['R']): e for e in db['entries']}

    def ub(q, n, R):
        if R >= n:
            return 1
        if R == 0:
            return q ** n
        e = idx.get((q, n, R))
        return e['ub'] if e else None

    hits = 0
    for (q, n, R), e in sorted(idx.items()):
        cands = []
        a = ub(q, n - 1, R - 1) if n >= 2 and R >= 1 else None
        if a:
            cands.append((a, 'K_%d(%d,%d) = %d  [key c]' % (q, n - 1, R - 1, a)))
        a = ub(q, n - 1, R) if n >= 2 else None
        if a:
            cands.append((q * a, '%d * K_%d(%d,%d) = %d * %d  [key e]'
                          % (q, q, n - 1, R, q, a)))
        for n1 in range(1, n):
            for R1 in range(0, min(R, n1) + 1):
                R2 = R - R1
                if R2 > n - n1:
                    continue
                x, y = ub(q, n1, R1), ub(q, n - n1, R2)
                if x and y:
                    cands.append((x * y, 'K_%d(%d,%d) * K_%d(%d,%d) = %d * %d  [key f]'
                                  % (q, n1, R1, q, n - n1, R2, x, y)))
        if cands:
            m = min(cands)
            if m[0] < e['ub']:
                hits += 1
                print('  K_%d(%d,%d): table says %d (key %s), but %s'
                      % (q, n, R, e['ub'], e['ub_key'], m[1]))
    print('  => %d cell(s) where the table is beaten by its own rules' % hits)


def check(db):
    idx = {(e['q'], e['n'], e['R']): e for e in db['entries']}
    ok = bad = 0
    print('--- cross-checks against independently quoted values ---')
    for q, n, R, lb, ub, why in CHECKS:
        e = idx.get((q, n, R))
        if e is None:
            print('  MISSING  K_%d(%d,%d)   [%s]' % (q, n, R, why))
            bad += 1
            continue
        good = (e['lb'], e['ub']) == (lb, ub)
        print('  %s K_%-2d(%2d,%d) parsed %s-%s  expected %s-%s   %s'
              % ('OK  ' if good else 'FAIL', q, n, R, e['lb'], e['ub'], lb, ub, why))
        ok += good
        bad += (not good)
    print('  => %d ok, %d failed' % (ok, bad))

    print()
    print('--- structural sanity over all parsed entries ---')
    prob = 0
    for e in db['entries']:
        q, n, R = e['q'], e['n'], e['R']
        if e['lb'] > e['ub']:
            print('  lb>ub at K_%d(%d,%d): %s' % (q, n, R, e))
            prob += 1
        if R >= n and (e['lb'], e['ub']) != (1, 1):
            print('  K_%d(%d,%d) should be 1: %s' % (q, n, R, e))
            prob += 1
        if e['ub'] > q ** n:
            print('  ub exceeds q^n at K_%d(%d,%d)' % (q, n, R))
            prob += 1
        # sphere-covering bound
        vol = ball_volume(q, n, R)
        if e['lb'] * vol < q ** n:
            print('  lb violates sphere covering at K_%d(%d,%d): %d*%d < %d'
                  % (q, n, R, e['lb'], vol, q ** n))
            prob += 1
        if e.get('lb_updated') and e['lb_updated'] > e['ub']:
            print('  updated lb > ub at K_%d(%d,%d)' % (q, n, R))
            prob += 1
    # monotonicity in n for fixed q,R: K_q(n+1,R) >= K_q(n,R)
    idx2 = {(e['q'], e['n'], e['R']): e for e in db['entries']}
    for (q, n, R), e in sorted(idx2.items()):
        nxt = idx2.get((q, n + 1, R))
        if nxt and nxt['ub'] < e['lb']:
            print('  monotonicity: K_%d(%d,%d)ub=%d < K_%d(%d,%d)lb=%d'
                  % (q, n + 1, R, nxt['ub'], q, n, R, e['lb']))
            prob += 1
        # K_q(n+1,R+1) <= K_q(n,R)
        dia = idx2.get((q, n + 1, R + 1))
        if dia and dia['lb'] > e['ub']:
            print('  diagonal: K_%d(%d,%d)lb=%d > K_%d(%d,%d)ub=%d'
                  % (q, n + 1, R + 1, dia['lb'], q, n, R, e['ub']))
            prob += 1
    print('  => %d structural problems' % prob)

    print()
    print('--- Gijswijt-Polak 2025 consistency with parsed Keri table ---')
    m = mm = miss = 0
    for r in db['lower_bound_updates_2025']:
        if 'keri_lb' not in r:
            miss += 1
            continue
        if r['keri_lb_matches_paper_old'] and r['keri_ub_matches_paper']:
            m += 1
        else:
            mm += 1
            print('  mismatch q=%d n=%d R=%d: paper old_lb=%d ub=%d, Keri lb=%d ub=%d'
                  % (r['q'], r['n'], r['R'], r['old_lb'], r['ub_in_paper'],
                     r['keri_lb'], r['keri_ub']))
    print('  => %d agree, %d disagree, %d cells not in parsed table' % (m, mm, miss))
    return bad == 0


def ball_volume(q, n, R):
    """|B_R| = sum_{i<=R} C(n,i) (q-1)^i."""
    from math import comb
    return sum(comb(n, i) * (q - 1) ** i for i in range(min(R, n) + 1))


def main():
    db = build()
    with open(OUT, 'w') as f:
        json.dump(db, f, indent=1, sort_keys=False)
    print('wrote %s: %d entries, %d legend blocks, %d lower-bound updates'
          % (OUT, len(db['entries']), len(db['keys']),
             len(db['lower_bound_updates_2025'])))
    per_q = defaultdict(int)
    for e in db['entries']:
        per_q[e['q']] += 1
    print('entries per q: ' + ', '.join('q=%d:%d' % (q, per_q[q]) for q in sorted(per_q)))
    if '--check' in sys.argv:
        print()
        check(db)
        print()
        print('--- diff against an independent third-party transcription ---')
        diff_third_party(db)
        print()
        print("--- audit: is the table beaten by its own derivation rules? ---")
        audit_internal(db)


if __name__ == '__main__':
    main()
