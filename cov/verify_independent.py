#!/usr/bin/env python3
"""
verify_independent.py -- a SECOND, deliberately different verifier.

`verify_cov.py` proves coverage by marking Hamming balls around codewords (or,
for large instances, by meet-in-the-middle bitsets). Both of those work outward
from the codewords, and both are mine. If a record claim rests on them alone, a
single shared misconception about what "covered" means would sink it.

This file works the other way round and shares no code with `verify_cov.py`.
It offers two methods, both exhaustive, and runs both when it can afford to:

  mindist  Enumerate every word of Z_q^n in chunks, compute that word's Hamming
           distance to every codeword directly, take the minimum.  The code
           covers at radius R iff the maximum over all words of that minimum is
           at most R.  No balls, no bitsets, no incremental counters -- just
           distances.  Reports the exact covering radius and the full
           distribution of minimum distances, which makes an off-by-one in R
           impossible to miss.  Cost q^n * M * n.

  dilate   Never computes a Hamming distance at all.  A Hamming ball of radius R
           is the R-fold dilation of a point in the Hamming graph, so: start
           from the indicator vector of the code over Z_q^n and apply, R times,
           the operator

               (D f)(w) = OR over p of ( OR over v of f(w with w_p := v) )

           which is exactly dilation by radius 1.  Each application is n passes
           of a reduction along one axis of the q^n array reshaped as
           (q^p, q, q^(n-1-p)).  The code covers iff every entry is set after R
           rounds.  Cost R * n * q^n, independent of M, which is what makes the
           big cells checkable at all.

Both are slower per call than `verify_cov.py`.  That is the point: these are the
checks you run once on a claim, not the ones you run in a loop.

Usage:
  python3 verify_independent.py CODEFILE -q 6 -n 8 -R 4
  python3 verify_independent.py results/K6_8_4_M181.txt --json results/K6_8_4_M181.json
  python3 verify_independent.py CODE --method dilate      # for large q^n
"""

import argparse
import json
import sys

import numpy as np

DIGITS = '0123456789abcdefghijklmnopqrstuvwxyz'


def read_code(path, q, n):
    """Deliberately re-implemented, not imported."""
    out = []
    for lineno, raw in enumerate(open(path), 1):
        line = raw.split('#')[0].strip()
        if not line:
            continue
        if any(ch in line for ch in ' ,\t'):
            w = [int(t) for t in line.replace(',', ' ').split()]
        else:
            w = [DIGITS.index(ch.lower()) for ch in line]
        if len(w) != n:
            sys.exit('%s:%d: length %d != n=%d' % (path, lineno, len(w), n))
        if min(w) < 0 or max(w) >= q:
            sys.exit('%s:%d: digit outside [0,%d)' % (path, lineno, q))
        out.append(w)
    return np.array(out, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('codefile')
    ap.add_argument('-q', type=int)
    ap.add_argument('-n', type=int)
    ap.add_argument('-R', type=int)
    ap.add_argument('--json', help='read q, n, R from this sidecar')
    ap.add_argument('--chunk', type=int, default=1 << 20)
    ap.add_argument('--method', default='auto',
                    choices=['auto', 'mindist', 'dilate', 'both'])
    a = ap.parse_args()

    q, n, R = a.q, a.n, a.R
    if a.json:
        m = json.load(open(a.json))
        q = q if q is not None else m['q']
        n = n if n is not None else m['n']
        R = R if R is not None else m['R']
    if None in (q, n, R):
        ap.error('need q, n, R (directly or via --json)')

    C = read_code(a.codefile, q, n)
    M = C.shape[0]
    uniq = len(set(map(tuple, C.tolist())))
    total = q ** n
    print('independent verifier')
    print('  code       : %s' % a.codefile)
    print('  parameters : q=%d n=%d R=%d   q^n=%d' % (q, n, R, total))
    print('  codewords  : %d in file, %d distinct' % (M, uniq))
    if uniq != M:
        print('  WARNING    : file contains duplicates; M should be %d' % uniq)

    methods = a.method
    if methods == 'auto':
        methods = 'both' if total * M * n <= 4e10 else 'dilate'

    ok = {}
    if methods in ('dilate', 'both'):
        ok['dilate'] = run_dilate(C, q, n, R, total)
    if methods in ('mindist', 'both'):
        ok['mindist'] = run_mindist(C, q, n, R, total, M, a.chunk)

    print('  methods run: %s' % ', '.join('%s=%s' % (k, 'covers' if v else 'DOES NOT COVER')
                                          for k, v in ok.items()))
    if len(set(ok.values())) > 1:
        print('RESULT: METHODS DISAGREE -- DO NOT TRUST')
        return 3
    if not all(ok.values()):
        print('RESULT: NOT a covering code of radius %d' % R)
        return 1
    print('RESULT: VERIFIED  -- every one of the %d words of Z_%d^%d is within '
          'distance %d of the code' % (total, q, n, R))
    print('        K_%d(%d,%d) <= %d' % (q, n, R, uniq))
    return 0


def run_dilate(C, q, n, R, total):
    """R-fold dilation of the code's indicator vector in the Hamming graph."""
    cov = np.zeros(total, dtype=bool)
    w = np.array([q ** (n - 1 - p) for p in range(n)], dtype=np.int64)
    cov[(C.astype(np.int64) * w).sum(axis=1)] = True
    print('  dilate     : %d codeword cells set' % cov.sum())
    for r in range(R):
        nxt = np.zeros(total, dtype=bool)
        for p in range(n):
            v = cov.reshape(q ** p, q, q ** (n - 1 - p))
            nxt.reshape(q ** p, q, q ** (n - 1 - p))[:] |= v.any(axis=1, keepdims=True)
        cov = nxt
        print('  dilate     : after radius %d, covered %d / %d' % (r + 1, cov.sum(), total))
    return bool(cov.all())


def run_mindist(C, q, n, R, total, M, chunk):
    weights = np.array([q ** (n - 1 - p) for p in range(n)], dtype=np.int64)
    hist = np.zeros(n + 2, dtype=np.int64)
    worst = 0
    worst_word = None
    done = 0
    while done < total:
        m = min(chunk, total - done)
        idx = np.arange(done, done + m, dtype=np.int64)
        # digits of every word in the chunk, MSB first
        dig = np.empty((m, n), dtype=np.uint8)
        for p in range(n):
            dig[:, p] = ((idx // weights[p]) % q).astype(np.uint8)
        # min distance to the code
        best = np.full(m, n + 1, dtype=np.int16)
        for i in range(M):
            d = np.zeros(m, dtype=np.int16)
            ci = C[i]
            for p in range(n):
                d += (dig[:, p] != ci[p])
            np.minimum(best, d, out=best)
        np.add.at(hist, best, 1)
        mx = int(best.max())
        if mx > worst:
            worst = mx
            j = int(np.argmax(best))
            worst_word = dig[j].tolist()
        done += m
        pct = 100.0 * done / total
        print('\r  scanned    : %6.2f%%  worst distance so far %d' % (pct, worst),
              end='', file=sys.stderr)
    print('', file=sys.stderr)

    print('  mindist    : distance distribution (min distance to the code):')
    for d in range(n + 2):
        if hist[d]:
            print('      d=%-2d : %d' % (d, hist[d]))
    print('  mindist    : covering radius (exact) = %d' % worst)
    if worst > R:
        print('  mindist    : farthest word %s'
              % ''.join(DIGITS[v] for v in worst_word))
    return worst <= R


if __name__ == '__main__':
    sys.exit(main())
