#!/usr/bin/env python3
"""
verify_cov.py -- exact, standalone verifier for q-ary covering codes.

Given a code file (one codeword per line) together with n, q and R, decide
whether EVERY word of Z_q^n lies within Hamming distance R of some codeword.
This gates every record claim in the project, so it is written to be boring and
obviously correct rather than clever.

Two independent methods are implemented and they can be run against each other:

  pure   Dependency-free.  Allocates a bytearray of q^n flags and marks, for
         every codeword, every word of its Hamming ball by direct enumeration
         over (subset of changed positions, new digit values).  Cost M*V where
         V = sum_{i<=R} C(n,i)(q-1)^i.  No numpy, no cleverness.

  numpy  Meet in the middle.  Split the coordinates n = n1 + n2 and index a word
         as w = x*q^n2 + y.  Then d(c,w) = d1(c1,x) + d2(c2,y), so w is covered
         iff some codeword i has d2(c2_i, y) <= R - d1(c1_i, x).  Precompute, for
         each codeword i and each radius r <= R, the bitset over y of
         {y : d2(c2_i,y) <= r}; then one pass over the q^n1 values of x ORs the
         relevant bitsets together and popcounts the result.  Never materialises
         q^n bytes, so it scales past what `pure` can hold.

Both methods are exhaustive.  Neither samples.

Code file format
----------------
One codeword per line.  Blank lines and lines starting with '#' are ignored.
A codeword is either
  * a run of base-36 digits, one character per coordinate ("01221010101"), or
  * whitespace- or comma-separated integers ("0 1 2 2 1"), which is what you
    need once q > 36.
Every digit must be in [0,q).  Duplicate codewords are reported and de-duplicated
(they never help coverage, and silently keeping them would inflate M).

Usage
-----
  python3 verify_cov.py CODEFILE -q Q -n N -R R [--method auto|pure|numpy|both]
  python3 verify_cov.py CODEFILE --json META.json        # take q,n,R from sidecar
  python3 verify_cov.py CODEFILE -q 3 -n 11 -R 4 --radius   # also report the
                                                            # true covering radius

Exit status is 0 iff the code is a verified covering code of radius <= R.
"""

import argparse
import json
import os
import sys
from math import comb

DIGITS = '0123456789abcdefghijklmnopqrstuvwxyz'


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def parse_code(path, q, n):
    """Read a code file.  Returns (codewords, n_duplicates)."""
    words = []
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.split('#')[0].strip()
            if not line:
                continue
            if any(ch in line for ch in ' ,\t'):
                toks = [t for t in line.replace(',', ' ').split() if t]
                try:
                    w = [int(t) for t in toks]
                except ValueError:
                    raise SystemExit('%s:%d: cannot parse %r' % (path, lineno, line))
            else:
                w = []
                for ch in line:
                    k = DIGITS.find(ch.lower())
                    if k < 0:
                        raise SystemExit('%s:%d: bad digit %r' % (path, lineno, ch))
                    w.append(k)
            if len(w) != n:
                raise SystemExit('%s:%d: length %d, expected n=%d'
                                 % (path, lineno, len(w), n))
            for v in w:
                if not 0 <= v < q:
                    raise SystemExit('%s:%d: digit %d out of range for q=%d'
                                     % (path, lineno, v, q))
            words.append(tuple(w))
    if not words:
        raise SystemExit('%s: no codewords' % path)
    uniq = list(dict.fromkeys(words))
    return uniq, len(words) - len(uniq)


def fmt_word(w):
    if max(w) < 36:
        return ''.join(DIGITS[v] for v in w)
    return ' '.join(str(v) for v in w)


def ball_volume(q, n, R):
    return sum(comb(n, i) * (q - 1) ** i for i in range(min(R, n) + 1))


# --------------------------------------------------------------------------
# method 1: pure python ball marking
# --------------------------------------------------------------------------

def verify_pure(code, n, q, R, want_witness=True, max_witness=5):
    """Exhaustive check with a q^n bytearray.  Returns (n_uncovered, witnesses)."""
    total = q ** n
    if total > 3 * 10 ** 8:
        raise MemoryError('pure method needs q^n = %d bytes' % total)
    pw = [q ** (n - 1 - p) for p in range(n)]     # positional weights
    covered = bytearray(total)

    for c in code:
        idx0 = 0
        for p in range(n):
            idx0 += c[p] * pw[p]
        covered[idx0] = 1
        # frontier holds (index, first position that may still be changed)
        frontier = [(idx0, 0)]
        for _ in range(R):
            nxt = []
            for ix, p0 in frontier:
                for p in range(p0, n):
                    w = pw[p]
                    base = ix - c[p] * w
                    for v in range(q):
                        if v == c[p]:
                            continue
                        j = base + v * w
                        covered[j] = 1
                        nxt.append((j, p + 1))
            frontier = nxt
            if not frontier:
                break

    n_unc = total - sum(covered)
    wit = []
    if n_unc and want_witness:
        for idx in range(total):
            if not covered[idx]:
                w, t = [], idx
                for p in range(n):
                    w.append((t // pw[p]) % q)
                    t -= w[-1] * pw[p]
                wit.append(tuple(w))
                if len(wit) >= max_witness:
                    break
    return n_unc, wit


# --------------------------------------------------------------------------
# method 2: numpy meet-in-the-middle
# --------------------------------------------------------------------------

def _digit_table(np, q, m):
    """(q^m, m) uint8 array: row k holds the base-q digits of k, MSB first."""
    size = q ** m
    out = np.empty((size, m), dtype=np.uint8)
    idx = np.arange(size, dtype=np.int64)
    for p in range(m):
        out[:, p] = (idx // (q ** (m - 1 - p))) % q
    return out


def verify_numpy(code, n, q, R, want_witness=True, max_witness=5, n1=None):
    """Exhaustive meet-in-the-middle check.  Returns (n_uncovered, witnesses)."""
    import numpy as np

    M = len(code)
    if n1 is None:
        n1 = n // 2
    n2 = n - n1
    X, Y = q ** n1, q ** n2
    W8 = (Y + 7) // 8

    C = np.array(code, dtype=np.uint8)
    C1, C2 = C[:, :n1], C[:, n1:]

    H1 = _digit_table(np, q, n1)
    H2 = _digit_table(np, q, n2)

    # D1[i,x] = Hamming distance between codeword i's prefix and prefix-word x
    D1 = np.zeros((M, X), dtype=np.int16)
    for p in range(n1):
        D1 += (H1[:, p][None, :] != C1[:, p][:, None])

    # bits[i,r] = bitset over y of {y : d2(c2_i, y) <= r}
    bits = np.zeros((M, R + 1, W8), dtype=np.uint8)
    d2 = np.zeros(Y, dtype=np.int16)
    for i in range(M):
        d2[:] = 0
        for p in range(n2):
            d2 += (H2[:, p] != C2[i, p])
        for r in range(R + 1):
            bits[i, r] = np.packbits(d2 <= r, bitorder='little')
    bits = bits.reshape(M * (R + 1), W8)

    # mask off the padding bits of the last byte
    tail = np.zeros(W8, dtype=np.uint8)
    tail[:] = 0xFF
    if Y % 8:
        tail[-1] = (1 << (Y % 8)) - 1

    popc = np.array([bin(i).count('1') for i in range(256)], dtype=np.int64)

    n_unc = 0
    wit = []
    base = np.arange(M, dtype=np.int64) * (R + 1)
    for x in range(X):
        t = R - D1[:, x]
        ok = np.nonzero(t >= 0)[0]
        if ok.size == 0:
            row = np.zeros(W8, dtype=np.uint8)
        else:
            sel = bits[base[ok] + t[ok]]
            row = np.bitwise_or.reduce(sel, axis=0)
        row &= tail
        cnt = int(popc[row].sum())
        if cnt != Y:
            n_unc += Y - cnt
            if want_witness and len(wit) < max_witness:
                miss = np.nonzero(~np.unpackbits(row, count=Y, bitorder='little')
                                  .astype(bool))[0]
                for y in miss[:max_witness - len(wit)]:
                    wit.append(tuple(H1[x].tolist()) + tuple(H2[int(y)].tolist()))
    return n_unc, wit


# --------------------------------------------------------------------------
# covering radius
# --------------------------------------------------------------------------

def covering_radius(code, n, q, method, rmax=None):
    """Smallest r such that the code covers Z_q^n.  Monotone -> binary search."""
    lo, hi = 0, rmax if rmax is not None else n
    # make sure hi covers (r = n always does, for a non-empty code)
    while lo < hi:
        mid = (lo + hi) // 2
        unc, _ = method(code, n, q, mid, want_witness=False)
        if unc == 0:
            hi = mid
        else:
            lo = mid + 1
    return lo


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def choose_method(n, q, R, M, forced):
    if forced != 'auto':
        return forced
    total = q ** n
    if total <= 3 * 10 ** 8 and M * ball_volume(q, n, R) <= 5 * 10 ** 7:
        return 'pure'
    try:
        import numpy  # noqa: F401
    except ImportError:
        return 'pure'
    return 'numpy'


def run(method, code, n, q, R, **kw):
    if method == 'pure':
        return verify_pure(code, n, q, R, **kw)
    return verify_numpy(code, n, q, R, **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('codefile')
    ap.add_argument('-q', type=int, help='alphabet size')
    ap.add_argument('-n', type=int, help='code length')
    ap.add_argument('-R', type=int, help='claimed covering radius')
    ap.add_argument('--json', help='sidecar JSON with q, n, R (and optionally M)')
    ap.add_argument('--method', default='auto',
                    choices=['auto', 'pure', 'numpy', 'both'])
    ap.add_argument('--radius', action='store_true',
                    help='also compute the exact covering radius')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args(argv)

    q, n, R, claimed_M = a.q, a.n, a.R, None
    if a.json:
        meta = json.load(open(a.json))
        q = q if q is not None else meta['q']
        n = n if n is not None else meta['n']
        R = R if R is not None else meta['R']
        claimed_M = meta.get('M')
    if None in (q, n, R):
        ap.error('need q, n and R (directly or via --json)')
    if q < 2 or n < 1 or R < 0:
        ap.error('bad parameters')

    code, ndup = parse_code(a.codefile, q, n)
    M = len(code)
    V = ball_volume(q, n, R)

    p = (lambda *s: None) if a.quiet else (lambda *s: print(*s))
    p('code      : %s' % os.path.abspath(a.codefile))
    p('parameters: q=%d n=%d R=%d   q^n=%d   ball volume=%d' % (q, n, R, q ** n, V))
    p('codewords : M=%d%s' % (M, '  (%d duplicate lines removed)' % ndup if ndup else ''))
    if claimed_M is not None and claimed_M != M:
        p('WARNING   : sidecar claims M=%d but file has %d distinct codewords'
          % (claimed_M, M))
    if M * V < q ** n:
        p('IMPOSSIBLE: M*V = %d < q^n = %d (sphere-covering bound)' % (M * V, q ** n))

    methods = ['pure', 'numpy'] if a.method == 'both' else [choose_method(n, q, R, M, a.method)]
    results = {}
    import time
    for m in methods:
        t0 = time.time()
        try:
            unc, wit = run(m, code, n, q, R)
        except MemoryError as e:
            p('method %-5s: SKIPPED (%s)' % (m, e))
            continue
        results[m] = unc
        p('method %-5s: uncovered=%d   (%.2fs)' % (m, unc, time.time() - t0))
        if wit:
            p('  first uncovered words: %s'
              % ', '.join(fmt_word(w) for w in wit))
    if not results:
        p('RESULT    : NO METHOD COULD RUN')
        return 2
    vals = set(results.values())
    if len(vals) > 1:
        p('RESULT    : METHODS DISAGREE %s  -- DO NOT TRUST' % results)
        return 3
    unc = vals.pop()

    if a.radius:
        meth = verify_pure if 'pure' in results else verify_numpy
        cr = covering_radius(code, n, q, meth, rmax=n)
        p('covering radius (exact): %d' % cr)

    if unc == 0:
        p('RESULT    : VERIFIED  -- K_%d(%d,%d) <= %d' % (q, n, R, M))
        return 0
    p('RESULT    : NOT A COVERING CODE (%d of %d words uncovered)' % (unc, q ** n))
    return 1


if __name__ == '__main__':
    sys.exit(main())
