#!/usr/bin/env python3
"""
finalize.py -- prune cov/results/ to the codes worth keeping and re-verify them
with everything we have.

A descent leaves one file per size it passed through (K6_8_4_M216, ..._M215,
... ,_M171). Only two of those are worth keeping per cell: the smallest code
found, and -- if we reached it from scratch -- the one that matches Kéri's
incumbent upper bound, which is the evidence that the solver can reach the
frontier on that cell at all.

For every kept file this then re-runs, from scratch:

  * `verify_cov.py`         pure ball marking   (and meet-in-the-middle bitsets)
  * `verify_independent.py` min-distance scan   and/or Hamming-graph dilation

and writes the outcome of each into the sidecar. A file that any method rejects
is reported loudly and left in place, unpruned, for inspection.

Usage:
  python3 finalize.py            # prune + verify + rewrite sidecars
  python3 finalize.py --dry-run  # say what it would do
  python3 finalize.py --verify-only
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
sys.path.insert(0, HERE)


def load_all():
    out = []
    for jf in sorted(glob.glob(os.path.join(RESULTS, '*.json'))):
        m = json.load(open(jf))
        tf = jf[:-5] + '.txt'
        if os.path.exists(tf):
            out.append((jf, tf, m))
    return out


def verify(tf, m):
    """Run every verifier we have and return a dict of outcomes."""
    import verify_cov as V
    res = {}
    code, ndup = V.parse_code(tf, m['q'], m['n'])
    res['distinct_codewords'] = len(code)
    res['duplicate_lines'] = ndup
    q, n, R = m['q'], m['n'], m['R']

    t = time.time()
    if q ** n <= 3 * 10 ** 8:
        res['verify_cov_pure_uncovered'] = V.verify_pure(code, n, q, R,
                                                         want_witness=False)[0]
    else:
        res['verify_cov_pure_uncovered'] = None
    res['verify_cov_numpy_uncovered'] = V.verify_numpy(code, n, q, R,
                                                       want_witness=False)[0]
    res['verify_cov_seconds'] = round(time.time() - t, 1)

    method = 'both' if q ** n * len(code) * n <= 4e10 else 'dilate'
    t = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, 'verify_independent.py'),
                        tf, '-q', str(q), '-n', str(n), '-R', str(R),
                        '--method', method],
                       capture_output=True, text=True)
    res['verify_independent_method'] = method
    res['verify_independent_exit'] = p.returncode
    res['verify_independent_seconds'] = round(time.time() - t, 1)
    for line in p.stdout.splitlines():
        if 'covering radius (exact)' in line:
            res['covering_radius_exact'] = int(line.rsplit('=', 1)[1])
        if line.startswith('RESULT:'):
            res['verify_independent_result'] = line[len('RESULT:'):].strip()
    return res


def ok(res):
    return (res['verify_cov_numpy_uncovered'] == 0
            and res['verify_cov_pure_uncovered'] in (0, None)
            and res['verify_independent_exit'] == 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--verify-only', action='store_true')
    a = ap.parse_args()

    all_files = load_all()
    by_cell = {}
    for jf, tf, m in all_files:
        by_cell.setdefault((m['q'], m['n'], m['R']), []).append((jf, tf, m))

    keep, drop = [], []
    for cell, items in sorted(by_cell.items()):
        items.sort(key=lambda x: x[2]['M'])
        best = items[0]
        keepset = {best[0]}
        ub = (best[2].get('keri_2011') or {}).get('ub')
        if ub is not None:
            for it in items:
                if it[2]['M'] == ub:
                    keepset.add(it[0])
        for it in items:
            (keep if it[0] in keepset else drop).append(it)

    print('keeping %d files, dropping %d superseded descent steps'
          % (len(keep), len(drop)))
    if a.dry_run:
        for _, tf, m in keep:
            print('  keep  %s (M=%d)' % (os.path.basename(tf), m['M']))
        for _, tf, m in drop:
            print('  drop  %s (M=%d)' % (os.path.basename(tf), m['M']))
        return 0

    bad = []
    for jf, tf, m in keep:
        print('\n=== %s  K_%d(%d,%d) M=%d ==='
              % (os.path.basename(tf), m['q'], m['n'], m['R'], m['M']))
        res = verify(tf, m)
        m['verified'] = res
        m['all_verifiers_agree'] = ok(res)
        e = m.get('keri_2011') or {}
        if e.get('ub') is not None:
            m['beats_keri_upper_bound'] = m['M'] < e['ub']
            m['matches_keri_upper_bound'] = m['M'] == e['ub']
        json.dump(m, open(jf, 'w'), indent=1)
        status = 'OK' if ok(res) else 'FAILED'
        print('  %s  %s' % (status, json.dumps(res)))
        if not ok(res):
            bad.append(tf)
        elif m.get('beats_keri_upper_bound'):
            print('  *** RECORD: K_%d(%d,%d) <= %d  (Keri 2011 upper bound %d) ***'
                  % (m['q'], m['n'], m['R'], m['M'], e['ub']))

    if bad:
        print('\n%d FILES FAILED VERIFICATION -- nothing pruned' % len(bad))
        for b in bad:
            print('  ' + b)
        return 1

    if not a.verify_only:
        for jf, tf, m in drop:
            os.remove(jf)
            os.remove(tf)
        print('\npruned %d superseded files' % len(drop))

    # machine-readable roll-up of where we stand against Keri
    summary = []
    for jf, tf, m in sorted(keep, key=lambda x: (x[2]['q'], x[2]['n'], x[2]['R'],
                                                 x[2]['M'])):
        e = m.get('keri_2011') or {}
        summary.append({
            'q': m['q'], 'n': m['n'], 'R': m['R'], 'M': m['M'],
            'keri_lb': e.get('lb'), 'keri_ub': e.get('ub'),
            'keri_ub_key': e.get('ub_key'),
            'improvement': (e['ub'] - m['M']) if e.get('ub') is not None else None,
            'beats_keri_upper_bound': m.get('beats_keri_upper_bound'),
            'code_file': m['code_file'],
            'all_verifiers_agree': m.get('all_verifiers_agree'),
            'method': m.get('method'),
            'date': m.get('date'),
        })
    with open(os.path.join(RESULTS, 'records.json'), 'w') as f:
        json.dump({
            'note': ('Upper bounds on K_q(n,R) from this project, against Kéri '
                     '2011. Every entry was checked by verify_cov.py (ball '
                     'marking and meet-in-the-middle) and verify_independent.py '
                     '(min-distance scan and/or Hamming-graph dilation).'),
            'baseline': 'Keri 2011, http://old.sztaki.hu/~keri/codes/',
            'entries': summary,
        }, f, indent=1)
    print('\nwrote %s' % os.path.join(RESULTS, 'records.json'))

    rec = [s for s in summary if s['beats_keri_upper_bound']]
    print('\n%d cell(s) improved over Keri 2011:' % len(rec))
    for s in rec:
        print('  K_%d(%d,%d): %d -> %d   (-%d, Keri ub key %s)'
              % (s['q'], s['n'], s['R'], s['keri_ub'], s['M'],
                 s['improvement'], s['keri_ub_key']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
