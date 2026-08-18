#!/usr/bin/env python3
"""
constructions.py -- explicit covering-code constructions, used to seed search
and to reproduce the incumbent upper bounds in Keri's tables.

What is here
------------
trivial          K_q(n,0) = q^n (the whole space) and K_q(n,R) <= 1 for R >= n.
hamming_code     Perfect radius-1 code for any prime power q and any r >= 2:
                 length n = (q^r - 1)/(q - 1), size q^(n-r), covering radius 1.
                 This is the source of every "h" entry in Keri's tables.
direct_sum       K_q(n1+n2, R1+R2) <= K_q(n1,R1) * K_q(n2,R2), by concatenation.
                 This is Keri's upper-bound key "f", and it is how essentially
                 every q >= 6 upper bound in the table is obtained -- which is
                 why those cells have never been searched directly.
shorten          K_q(n+1, R+1) <= K_q(n,R)  (Keri's key "c"): append a constant
                 coordinate; the extra coordinate can absorb one more error.
blow_up          K_q(n+1, R) <= q * K_q(n,R)  (Keri's key "e").
seed_for         Recursively build the smallest code the above constructions can
                 reach for a cell, using bounds.json as the source of exactly
                 known small components.

Everything returns a plain list of tuples of ints, so the output feeds straight
into verify_cov.py.

Run `python3 constructions.py -q Q -n N -R R` to see how the incumbent upper
bound for a cell decomposes, and `--out FILE` to write the seed code it can
build.
"""

import argparse
import json
import os
import sys
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
DIGITS = '0123456789abcdefghijklmnopqrstuvwxyz'


# --------------------------------------------------------------------------
# finite fields (needed for Hamming codes over prime-power alphabets)
# --------------------------------------------------------------------------

# irreducible polynomials over GF(p), listed low-degree-coefficient first,
# monic, for the prime powers we care about
IRRED = {
    4: (2, (1, 1, 1)),          # GF(4)  = GF(2)[x]/(x^2+x+1)
    8: (2, (1, 1, 0, 1)),       # GF(8)  = GF(2)[x]/(x^3+x+1)
    9: (3, (1, 0, 1)),          # GF(9)  = GF(3)[x]/(x^2+1)
    16: (2, (1, 1, 0, 0, 1)),   # GF(16) = GF(2)[x]/(x^4+x+1)
    25: (5, (2, 0, 1)),         # GF(25) = GF(5)[x]/(x^2+2)
    27: (3, (1, 2, 0, 1)),      # GF(27) = GF(3)[x]/(x^3+2x+1)
}


def is_prime(k):
    if k < 2:
        return False
    i = 2
    while i * i <= k:
        if k % i == 0:
            return False
        i += 1
    return True


class GF(object):
    """Field of order q, elements labelled 0..q-1.

    For prime q the label is the residue.  For a prime power q = p^m the label
    is the integer whose base-p digits are the polynomial coefficients (constant
    term least significant), which keeps 0 and 1 at labels 0 and 1.
    """

    def __init__(self, q):
        self.q = q
        if is_prime(q):
            self.p, self.m = q, 1
            self.add = lambda a, b: (a + b) % q
            self.neg = lambda a: (-a) % q
            self.mul = lambda a, b: (a * b) % q
        elif q in IRRED:
            p, poly = IRRED[q]
            m = len(poly) - 1
            self.p, self.m = p, m
            self._poly = poly
            self._mul_tab = [[self._pmul(a, b) for b in range(q)] for a in range(q)]
            self.add = lambda a, b: self._padd(a, b)
            self.neg = lambda a: self._padd(0, a) if p == 2 else self._pneg(a)
            self.mul = lambda a, b: self._mul_tab[a][b]
        else:
            raise ValueError('GF(%d) not supported' % q)

    # polynomial helpers for prime-power fields
    def _digits(self, a):
        d = []
        for _ in range(self.m):
            d.append(a % self.p)
            a //= self.p
        return d

    def _undigits(self, d):
        v = 0
        for x in reversed(d):
            v = v * self.p + x
        return v

    def _padd(self, a, b):
        da, db = self._digits(a), self._digits(b)
        return self._undigits([(x + y) % self.p for x, y in zip(da, db)])

    def _pneg(self, a):
        return self._undigits([(-x) % self.p for x in self._digits(a)])

    def _pmul(self, a, b):
        da, db = self._digits(a), self._digits(b)
        prod = [0] * (2 * self.m)
        for i, x in enumerate(da):
            if x:
                for j, y in enumerate(db):
                    prod[i + j] = (prod[i + j] + x * y) % self.p
        poly = self._poly            # monic, degree m
        for k in range(2 * self.m - 1, self.m - 1, -1):
            c = prod[k]
            if c:
                prod[k] = 0
                for i in range(self.m):
                    prod[k - self.m + i] = (prod[k - self.m + i] - c * poly[i]) % self.p
        return self._undigits(prod[:self.m])

    def elements(self):
        return range(self.q)


# --------------------------------------------------------------------------
# constructions
# --------------------------------------------------------------------------

def full_space(q, n):
    """K_q(n,0) = q^n."""
    return [tuple(w) for w in product(range(q), repeat=n)]


def single(n):
    """K_q(n,R) <= 1 for R >= n."""
    return [tuple([0] * n)]


def direct_sum(A, B):
    """K_q(n1+n2, R1+R2) <= |A| |B| for A of radius R1 and B of radius R2.

    If d(a,x) <= R1 and d(b,y) <= R2 then d((a,b),(x,y)) <= R1+R2, and for any
    (x,y) such a pair exists because A and B each cover their own factor.
    """
    return [a + b for a in A for b in B]


def shorten(A):
    """K_q(n+1,R+1) <= K_q(n,R): append a constant coordinate.

    A word (x, t) is within R+1 of (a, 0) whenever d(a,x) <= R, because the last
    coordinate contributes at most 1.
    """
    return [a + (0,) for a in A]


def blow_up(A, q):
    """K_q(n+1,R) <= q K_q(n,R): append every possible last symbol."""
    return [a + (v,) for a in A for v in range(q)]


def hamming_code(q, r):
    """Perfect [n, n-r] Hamming code over GF(q), n = (q^r - 1)/(q - 1).

    Covering radius 1, size q^(n-r), and q^(n-r) * (1 + n(q-1)) = q^n exactly.
    Built as the null space of the r x n parity-check matrix whose columns are
    the projective points of PG(r-1, q) (one representative per 1-dim subspace,
    normalised so the leading nonzero entry is 1).
    """
    F = GF(q)
    # projective points: vectors whose first nonzero coordinate is 1
    cols = []
    for v in product(range(q), repeat=r):
        nz = next((i for i, x in enumerate(v) if x), None)
        if nz is not None and v[nz] == 1:
            cols.append(v)
    n = (q ** r - 1) // (q - 1)
    assert len(cols) == n, (len(cols), n)

    # H is r x n with columns `cols`; solve H c^T = 0 by putting H in the form
    # [I_r | A] (the identity columns are the unit vectors, which are present)
    unit = []
    for i in range(r):
        e = tuple(1 if j == i else 0 for j in range(r))
        unit.append(cols.index(e))
    rest = [j for j in range(n) if j not in unit]
    # information symbols sit on `rest`, parity symbols on `unit`
    code = []
    for info in product(range(q), repeat=len(rest)):
        c = [0] * n
        for j, val in zip(rest, info):
            c[j] = val
        # parity: sum over all columns must be zero, so the unit column i
        # carries minus the i-th coordinate of sum_{j in rest} info_j * col_j
        acc = [0] * r
        for j, val in zip(rest, info):
            col = cols[j]
            for i in range(r):
                acc[i] = F.add(acc[i], F.mul(val, col[i]))
        for i in range(r):
            c[unit[i]] = F.neg(acc[i])
        code.append(tuple(c))
    return code


# --------------------------------------------------------------------------
# seeding from the bounds table
# --------------------------------------------------------------------------

def load_bounds(path=None):
    path = path or os.path.join(HERE, 'bounds.json')
    db = json.load(open(path))
    return {(e['q'], e['n'], e['R']): e for e in db['entries']}


def ub(idx, q, n, R):
    """Best known upper bound for K_q(n,R), including the trivial cases."""
    if R >= n:
        return 1
    if R == 0:
        return q ** n
    e = idx.get((q, n, R))
    return e['ub'] if e else None


def decompose(idx, q, n, R, use_table=False, _memo=None):
    """Cheapest way to reach cell (q,n,R).

    Returns (size, description, recipe) where recipe is a nested tuple that
    build_seed() can execute (only when use_table is False).

    use_table=False: leaves are only constructions we can actually instantiate
        (whole space, single codeword, perfect Hamming code).  The result is
        therefore >= Keri's upper bound whenever Keri's entry comes from
        something we do not implement (direct search, adjoint codes, linear
        codes, ...), and it is exactly the seed we can build for local search.

    use_table=True: leaves may additionally be Keri's tabulated upper bound for
        a smaller cell.  This shows how a table entry decomposes -- in
        particular it identifies the cells whose upper bound is nothing but a
        direct sum of smaller ones, which are the never-searched targets.
    """
    if _memo is None:
        _memo = {}
    key = (q, n, R, use_table)
    if key in _memo:
        return _memo[key]
    _memo[key] = None                      # cycle guard

    if R >= n:
        res = (1, 'single codeword (R >= n)', ('single', n))
        _memo[key] = res
        return res
    if R == 0:
        res = (q ** n, 'whole space q^%d' % n, ('full', q, n))
        _memo[key] = res
        return res

    best = None

    def better(cand):
        return cand if (best is None or cand[0] < best[0]) else best

    # perfect Hamming code
    if R == 1:
        r = 2
        while (q ** r - 1) // (q - 1) <= n:
            if (q ** r - 1) // (q - 1) == n:
                try:
                    GF(q)
                    best = better((q ** (n - r),
                                   'perfect Hamming code [%d,%d]_%d' % (n, n - r, q),
                                   ('hamming', q, r)))
                except ValueError:
                    pass
            r += 1

    # a tabulated value for this very cell counts as a leaf in table mode
    if use_table:
        e = idx.get((q, n, R))
        if e is not None:
            best = better((e['ub'], 'Keri table (key %s)' % e['ub_key'],
                           ('table', q, n, R)))

    # direct sum over all splits
    for n1 in range(1, n):
        n2 = n - n1
        for R1 in range(0, min(R, n1) + 1):
            R2 = R - R1
            if R2 > n2:
                continue
            a = decompose(idx, q, n1, R1, use_table, _memo)
            b = decompose(idx, q, n2, R2, use_table, _memo)
            if a is None or b is None:
                continue
            best = better((a[0] * b[0],
                           'direct sum (%d,%d)x(%d,%d) = %d*%d'
                           % (n1, R1, n2, R2, a[0], b[0]),
                           ('sum', a[2], b[2])))

    # shortening: K_q(n,R) <= K_q(n-1,R-1)
    if n >= 2 and R >= 1:
        a = decompose(idx, q, n - 1, R - 1, use_table, _memo)
        if a is not None:
            best = better((a[0], 'shorten from (%d,%d)' % (n - 1, R - 1),
                           ('shorten', a[2])))

    _memo[key] = best
    return best


def explain_ub(idx, q, n, R):
    """How does the tabulated upper bound for (q,n,R) arise from smaller cells?

    Considers direct sums of tabulated smaller cells, K_q(n,R) <= K_q(n-1,R-1)
    (Keri key "c"), K_q(n,R) <= q K_q(n-1,R) (key "e"), and the trivial cells.
    Returns the best (size, description) reachable that way, or None.
    """
    memo = {}

    def val(nn, RR):
        if RR >= nn:
            return 1
        if RR == 0:
            return q ** nn
        e = idx.get((q, nn, RR))
        return e['ub'] if e else None

    best = None
    for n1 in range(1, n):
        n2 = n - n1
        for R1 in range(0, min(R, n1) + 1):
            R2 = R - R1
            if R2 > n2:
                continue
            a, b = val(n1, R1), val(n2, R2)
            if a is None or b is None:
                continue
            cand = (a * b, 'direct sum (%d,%d)x(%d,%d) = %d*%d'
                    % (n1, R1, n2, R2, a, b))
            if best is None or cand[0] < best[0]:
                best = cand
    a = val(n - 1, R - 1) if n >= 2 and R >= 1 else None
    if a is not None and (best is None or a < best[0]):
        best = (a, 'K_q(n,R) <= K_q(%d,%d) = %d  [key c]' % (n - 1, R - 1, a))
    a = val(n - 1, R) if n >= 2 else None
    if a is not None and (best is None or q * a < best[0]):
        best = (q * a, 'K_q(n,R) <= q*K_q(%d,%d) = %d*%d  [key e]' % (n - 1, R, q, a))
    del memo
    return best


def build_seed(recipe):
    kind = recipe[0]
    if kind == 'single':
        return single(recipe[1])
    if kind == 'full':
        return full_space(recipe[1], recipe[2])
    if kind == 'hamming':
        return hamming_code(recipe[1], recipe[2])
    if kind == 'sum':
        return direct_sum(build_seed(recipe[1]), build_seed(recipe[2]))
    if kind == 'shorten':
        return shorten(build_seed(recipe[1]))
    raise ValueError(kind)


def write_code(path, code):
    q_needed = max(max(c) for c in code) + 1
    with open(path, 'w') as f:
        for c in code:
            if q_needed <= 36:
                f.write(''.join(DIGITS[v] for v in c) + '\n')
            else:
                f.write(' '.join(str(v) for v in c) + '\n')


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-q', type=int, required=True)
    ap.add_argument('-n', type=int, required=True)
    ap.add_argument('-R', type=int, required=True)
    ap.add_argument('--out', help='write the constructed seed code here')
    ap.add_argument('--bounds', help='path to bounds.json')
    a = ap.parse_args()

    idx = load_bounds(a.bounds)
    keri = idx.get((a.q, a.n, a.R))
    dec = decompose(idx, a.q, a.n, a.R)
    tab = explain_ub(idx, a.q, a.n, a.R)
    print('cell K_%d(%d,%d)' % (a.q, a.n, a.R))
    if keri:
        print('  Keri 2011      : %d - %d   (lower key %s, upper key %s)'
              % (keri['lb'], keri['ub'], keri['lb_key'], keri['ub_key']))
        if keri.get('lb_updated'):
            print('  lower bound now: %d  [%s]' % (keri['lb_updated'], keri['lb_updated_src']))
    if tab:
        mark = ''
        if keri:
            mark = ('  <-- the table UB is exactly this' if tab[0] == keri['ub']
                    else '  (table UB is %d, better than this)' % keri['ub'])
        print('  best UB from smaller cells: %d via %s%s' % (tab[0], tab[1], mark))
    if dec is None:
        print('  no constructive seed available')
        return 1
    print('  our constructions reach M = %d via %s' % (dec[0], dec[1]))
    if keri:
        if dec[0] == keri['ub']:
            print('  -> reproduces the incumbent upper bound exactly')
        elif dec[0] < keri['ub']:
            print('  -> BELOW the incumbent upper bound; check this!')
        else:
            print('  -> %d above the incumbent (%s came from something we do not '
                  'implement)' % (dec[0] - keri['ub'], keri['ub_key']))
    if a.out:
        code = build_seed(dec[2])
        write_code(a.out, code)
        print('  wrote %d codewords to %s' % (len(code), a.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
