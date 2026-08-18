#!/usr/bin/env python3
"""
test_solver.py -- check that covsearch's incrementally maintained uncovered
count is the truth.

The solver never recomputes coverage from scratch: it maintains `cnt[w]` and the
running number of uncovered words through millions of sphere updates. If that
bookkeeping drifts, the solver will happily announce "uncovered = 0" on a code
that does not cover, and the only thing standing between that and a false record
claim is the verifier. This test closes the loop directly: run the solver with a
fixed budget, take the code it emits, and check that `verify_cov.py` agrees with
the number the solver printed -- not just when it is zero.

It also exercises the paths that are easy to get wrong: R = 0, R = n - 1, seeding
from a file, and the remove-and-repair descent that drops codewords from an
oversized seed.

Run:  python3 test_solver.py     (needs search/covsearch built)
"""

import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, 'search', 'covsearch')
sys.path.insert(0, HERE)
import verify_cov as V          # noqa: E402
import constructions as C       # noqa: E402

FAIL = []


def expect(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg)
    if not cond:
        FAIL.append(msg)


def run(args, out):
    cmd = [BIN] + [str(a) for a in args] + ['--out', out, '--quiet']
    p = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r'uncovered=(\d+)', p.stdout)
    if not m:
        raise AssertionError('no RESULT line:\n%s\n%s' % (p.stdout, p.stderr))
    return int(m.group(1)), p.stdout


def main():
    if not os.path.exists(BIN):
        sys.exit('build search/covsearch first (make -C search)')
    td = tempfile.mkdtemp(prefix='covtest_')

    print('1. solver-reported uncovered count matches the verifier')
    cases = [
        # q, n, R, M, iters
        (3, 4, 1, 5, 200),        # too few codewords: must end non-zero
        (3, 4, 1, 9, 5000),       # exactly the optimum
        (3, 5, 2, 6, 2000),
        (4, 4, 2, 5, 2000),
        (5, 4, 1, 40, 3000),
        (6, 4, 2, 12, 3000),
        (2, 7, 2, 8, 3000),
        (3, 3, 0, 20, 500),       # R = 0
        (3, 4, 3, 2, 200),        # R = n - 1
        (7, 3, 1, 20, 2000),
    ]
    for q, n, R, M, iters in cases:
        out = os.path.join(td, 'c.txt')
        unc, _ = run(['-q', q, '-n', n, '-R', R, '-M', M,
                      '--iters', iters, '-t', 30, '-s', 7], out)
        code, ndup = V.parse_code(out, q, n)
        # the solver writes its BEST code, so verify that one
        real = V.verify_pure(code, n, q, R, want_witness=False)[0]
        expect(unc == real,
               'q=%d n=%d R=%d M=%d: solver said %d, verifier says %d%s'
               % (q, n, R, M, unc, real,
                  ' (%d duplicate codewords)' % ndup if ndup else ''))

    print('2. the two verifier methods agree with the solver too')
    out = os.path.join(td, 'd.txt')
    unc, _ = run(['-q', 6, '-n', 5, '-R', 2, '-M', 40, '--iters', 4000,
                  '-t', 60, '-s', 3], out)
    code, _ = V.parse_code(out, 6, 5)
    a = V.verify_pure(code, 5, 6, 2, want_witness=False)[0]
    b = V.verify_numpy(code, 5, 6, 2, want_witness=False)[0]
    expect(unc == a == b, 'q=6 n=5 R=2 M=40: solver=%d pure=%d numpy=%d' % (unc, a, b))

    print('3. seeding from a file preserves the seed')
    seed = os.path.join(td, 'seed.txt')
    H = C.hamming_code(3, 2)                       # K_3(4,1) = 9
    C.write_code(seed, H)
    out = os.path.join(td, 'e.txt')
    unc, _ = run(['-q', 3, '-n', 4, '-R', 1, '-M', 9, '--in', seed,
                  '--iters', 0, '-t', 5, '-s', 1], out)
    got, _ = V.parse_code(out, 3, 4)
    expect(unc == 0, 'seeded with a perfect code and 0 iterations: uncovered=%d' % unc)
    expect(set(got) == set(H), 'the emitted code is the seed')

    print('4. remove-and-repair drops from an oversized seed')
    # 12 codewords covering Z_3^4 at R=1 (the perfect code plus 3 spares);
    # asking for M=9 must drop exactly 3 and still be able to cover.
    big = H + [(2, 2, 2, 2), (1, 0, 1, 0), (0, 2, 0, 2)]
    big = list(dict.fromkeys(big))
    bigf = os.path.join(td, 'big.txt')
    C.write_code(bigf, big)
    out = os.path.join(td, 'f.txt')
    unc, stdout = run(['-q', 3, '-n', 4, '-R', 1, '-M', 9, '--in', bigf,
                       '--iters', 20000, '-t', 30, '-s', 5], out)
    code, _ = V.parse_code(out, 3, 4)
    expect(len(code) == 9, 'descended to exactly 9 codewords (got %d)' % len(code))
    real = V.verify_pure(code, 4, 3, 1, want_witness=False)[0]
    expect(unc == real, 'after descent: solver=%d verifier=%d' % (unc, real))
    expect(unc == 0, 'descent from 12 to 9 still covers (uncovered=%d)' % unc)

    print('5. an impossible size never reports success')
    out = os.path.join(td, 'g.txt')
    unc, _ = run(['-q', 3, '-n', 4, '-R', 1, '-M', 8, '--iters', 200000,
                  '-t', 30, '-s', 11], out)
    expect(unc > 0, 'K_3(4,1)=9 is exact, so M=8 must stay uncovered (got %d)' % unc)
    code, _ = V.parse_code(out, 3, 4)
    real = V.verify_pure(code, 4, 3, 1, want_witness=False)[0]
    expect(unc == real, 'M=8: solver=%d verifier=%d' % (unc, real))

    print()
    if FAIL:
        print('%d FAILURES' % len(FAIL))
        for m in FAIL:
            print('  ' + m)
        return 1
    print('all solver tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
