#!/usr/bin/env python3
"""
test_verify.py -- test suite for verify_cov.py.

Three independent implementations must agree on every case:

  brute   the dumbest thing that could possibly work: enumerate every word of
          Z_q^n with itertools.product, and for each one scan the whole code
          computing Hamming distances.  O(q^n * M * n).  Only used on tiny
          instances, but it depends on nothing that the other two share.
  pure    verify_cov.verify_pure   (bytearray ball marking)
  numpy   verify_cov.verify_numpy  (meet-in-the-middle bitsets)

Cases: random codes over a range of (q,n,R,M), hand-built structural codes with
known answers (perfect Hamming codes, the whole space, single codewords, direct
sums), plus the file-parsing edge cases.

Run:  python3 test_verify.py
"""

import itertools
import os
import random
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_cov as V
import constructions as C


def brute(code, n, q, R):
    """Reference: uncovered count by exhaustive scan.  No shared code with V."""
    unc = 0
    for w in itertools.product(range(q), repeat=n):
        for c in code:
            d = 0
            for a, b in zip(c, w):
                if a != b:
                    d += 1
                    if d > R:
                        break
            if d <= R:
                break
        else:
            unc += 1
    return unc


FAIL = []


def expect(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg)
    if not cond:
        FAIL.append(msg)


def agree(code, n, q, R, label, use_brute=True):
    a = V.verify_pure(code, n, q, R, want_witness=False)[0]
    b = V.verify_numpy(code, n, q, R, want_witness=False)[0]
    vals = {'pure': a, 'numpy': b}
    if use_brute:
        vals['brute'] = brute(code, n, q, R)
    ok = len(set(vals.values())) == 1
    expect(ok, '%s -> %s' % (label, vals))
    return a


def main():
    random.seed(20260818)

    print('1. random codes: brute vs pure vs numpy')
    for (q, n) in [(2, 6), (3, 5), (3, 6), (4, 5), (5, 4), (6, 4), (7, 3), (2, 9)]:
        for R in range(0, min(n, 4) + 1):
            M = random.randint(1, 12)
            code = [tuple(random.randrange(q) for _ in range(n)) for _ in range(M)]
            code = list(dict.fromkeys(code))
            agree(code, n, q, R, 'q=%d n=%d R=%d M=%d' % (q, n, R, len(code)))

    print('2. odd splits of the meet-in-the-middle (n1 = 1 .. n-1)')
    q, n, R = 3, 6, 2
    code = [tuple(random.randrange(q) for _ in range(n)) for _ in range(9)]
    ref = brute(code, n, q, R)
    for n1 in range(1, n):
        got = V.verify_numpy(code, n, q, R, want_witness=False, n1=n1)[0]
        expect(got == ref, 'n1=%d -> %d (ref %d)' % (n1, got, ref))

    print('3. structural cases with known answers')
    full = C.full_space(2, 5)
    expect(V.verify_pure(full, 5, 2, 0, want_witness=False)[0] == 0,
           'whole space of Z_2^5 covers at R=0')
    expect(V.verify_pure(full[:-1], 5, 2, 0, want_witness=False)[0] == 1,
           'whole space minus one word leaves exactly 1 uncovered at R=0')
    one = [tuple([0] * 5)]
    expect(V.verify_pure(one, 5, 3, 5, want_witness=False)[0] == 0,
           'K_3(5,5) = 1: a single codeword covers at R=n')
    expect(V.verify_pure(one, 5, 3, 4, want_witness=False)[0] == 2 ** 5,
           'a single codeword at R=n-1 misses exactly (q-1)^n words')

    for q, r in [(2, 3), (3, 2), (4, 2), (5, 2)]:
        H = C.hamming_code(q, r)
        nn = (q ** r - 1) // (q - 1)
        exp = q ** (nn - r)
        expect(len(H) == exp, 'Hamming q=%d r=%d has %d words (expect %d)'
               % (q, r, len(H), exp))
        u = V.verify_pure(H, nn, q, 1, want_witness=False)[0]
        expect(u == 0, 'Hamming q=%d r=%d is a perfect radius-1 covering code' % (q, r))
        if exp > 1:
            u2 = V.verify_pure(H[:-1], nn, q, 1, want_witness=False)[0]
            expect(u2 > 0, 'Hamming q=%d r=%d minus one word no longer covers' % (q, r))

    print('4. direct sum K_q(n1+n2, R1+R2) <= K_q(n1,R1) K_q(n2,R2)')
    A = C.hamming_code(3, 2)                    # K_3(4,1) <= 9
    B = C.full_space(3, 2)                      # K_3(2,0) <= 9
    D = C.direct_sum(A, B)
    expect(len(D) == 81, 'direct sum size %d' % len(D))
    agree(D, 6, 3, 1, 'direct sum (4,1)+(2,0) -> (6,1)', use_brute=False)
    A2 = C.hamming_code(3, 2)
    B2 = [tuple([0, 0])]                        # K_3(2,2) <= 1
    D2 = C.direct_sum(A2, B2)
    agree(D2, 6, 3, 3, 'direct sum (4,1)+(2,2) -> (6,3)', use_brute=False)

    print('5. file parsing')
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'c.txt')
        with open(p, 'w') as f:
            f.write('# comment\n0000\n0111\n\n0222\n0111\n')
        code, ndup = V.parse_code(p, 3, 4)
        expect(len(code) == 3 and ndup == 1, 'duplicate detection: M=%d dup=%d'
               % (len(code), ndup))
        with open(p, 'w') as f:
            f.write('0 1 2 2\n1,1,1,1\n')
        code, _ = V.parse_code(p, 3, 4)
        expect(code == [(0, 1, 2, 2), (1, 1, 1, 1)], 'separated digits parse')
        # base-36 for q > 10
        with open(p, 'w') as f:
            f.write('0a9\n')
        code, _ = V.parse_code(p, 12, 3)
        expect(code == [(0, 10, 9)], 'base-36 digits parse')

    print('6. CLI end-to-end exit codes')
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'h.txt')
        H = C.hamming_code(3, 2)
        C.write_code(p, H)
        here = os.path.dirname(os.path.abspath(__file__))
        r = subprocess.run([sys.executable, os.path.join(here, 'verify_cov.py'),
                            p, '-q', '3', '-n', '4', '-R', '1', '--quiet'])
        expect(r.returncode == 0, 'CLI exit 0 on a valid covering code')
        C.write_code(p, H[:-1])
        r = subprocess.run([sys.executable, os.path.join(here, 'verify_cov.py'),
                            p, '-q', '3', '-n', '4', '-R', '1', '--quiet'])
        expect(r.returncode == 1, 'CLI exit 1 on a non-covering code')

    print()
    if FAIL:
        print('%d FAILURES' % len(FAIL))
        for m in FAIL:
            print('  ' + m)
        return 1
    print('all tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
