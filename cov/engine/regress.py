#!/usr/bin/env python3
"""
regress.py -- re-derive existing records from scratch with the merged engine.

For each cell it starts at Kéri's tabulated upper bound with NO access to
cov/results/ at all (COVENGINE_NO_RESULTS=1: no incumbent seed, no direct-sum
factor read out of a code we already own) and descends one word at a time,
each step seeded only by the code the previous step produced.  Every step is
checked by cov/verify_cov.py before it is allowed to seed the next one, exactly
as the production gate does.

Nothing is written to cov/results/: the point is to find out whether the engine
still reaches what the old pipeline reached, not to re-record it.  Codes land
in cov/engine/regress/.

  regress.py --cells 6,7,3,246,232 6,8,4,216,167 7,8,4,343,329 -t 120 --cores 12
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
COV = os.path.dirname(HERE)
sys.path.insert(0, COV)
import verify_cov as V          # noqa: E402


def verify(path, q, n, R):
    """The project gate: re-read from disk, two exhaustive methods, and the
    distinct-codeword count.  Returns (M_distinct, uncovered) or None."""
    code, ndup = V.parse_code(path, q, n)
    unc_np = V.verify_numpy(code, n, q, R, want_witness=False)[0]
    unc_pure = None
    if q ** n <= 3 * 10 ** 8:
        unc_pure = V.verify_pure(code, n, q, R, want_witness=False)[0]
    if unc_pure is not None and unc_pure != unc_np:
        print('   VERIFIER DISAGREEMENT pure=%s numpy=%s' % (unc_pure, unc_np))
        return None
    return (len(code), unc_np)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', nargs='+', required=True,
                    help='q,n,R,start,target tuples')
    ap.add_argument('-t', type=float, default=120.0)
    ap.add_argument('--cores', type=int, default=12)
    ap.add_argument('--outdir', default=os.path.join(HERE, 'regress'))
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    os.makedirs(a.outdir, exist_ok=True)

    env = dict(os.environ)
    env['COVENGINE_NO_RESULTS'] = '1'
    env['COVENGINE_CORES'] = str(a.cores)
    env['COVENGINE_NICE'] = '12'

    summary = []
    for spec in a.cells:
        q, n, R, start, target = (int(x) for x in spec.split(','))
        print('\n=== K_%d(%d,%d): from scratch at M=%d, descending towards the '
              'recorded record M=%d ===' % (q, n, R, start, target))
        seed = None
        reached = None
        t_cell = time.time()
        M = start
        while M >= target:
            out = os.path.join(a.outdir, 'K%d_%d_%d_M%d.txt' % (q, n, R, M))
            cmd = [sys.executable, os.path.join(HERE, 'covengine'),
                   '-q', str(q), '-n', str(n), '-R', str(R), '-M', str(M),
                   '-t', str(a.t), '-s', str(M), '--cores', str(a.cores),
                   '--out', out, '--quiet']
            if seed:
                cmd += ['--in', seed]
            t0 = time.time()
            r = subprocess.run(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, env=env)
            unc = -1
            for line in (r.stdout or '').splitlines():
                if line.startswith('RESULT '):
                    kv = dict(t.split('=', 1) for t in line.split()[1:] if '=' in t)
                    unc = int(kv.get('uncovered', -1))
            if unc != 0:
                print('  M=%-5d engine best uncovered=%-8s (%.0fs)  -- stop'
                      % (M, unc, time.time() - t0))
                break
            v = verify(out, q, n, R)
            if v is None or v[1] != 0 or v[0] != M:
                print('  M=%-5d REJECTED BY VERIFIER: %s' % (M, v))
                break
            print('  M=%-5d VERIFIED covering (%.0fs)' % (M, time.time() - t0))
            reached = M
            seed = out
            M -= 1
        ok = reached is not None and reached <= target
        summary.append(dict(cell='K_%d(%d,%d)' % (q, n, R), start=start,
                            target=target, reached=reached, ok=ok,
                            seconds=round(time.time() - t_cell)))
        print('  -> reached M=%s (record is %d): %s'
              % (reached, target, 'RECORD RE-DERIVED' if ok else 'short of the record'))

    print('\n=== regression summary ===')
    for s in summary:
        print('  %-12s start %-5d record %-5d reached %-5s  %s  (%ds)'
              % (s['cell'], s['start'], s['target'], s['reached'],
                 'PASS' if s['ok'] else 'FAIL', s['seconds']))
    with open(os.path.join(a.outdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    return 0 if all(s['ok'] for s in summary) else 1


if __name__ == '__main__':
    sys.exit(main())
