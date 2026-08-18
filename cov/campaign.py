#!/usr/bin/env python3
"""
campaign.py -- run covsearch over many seeds/parameter sets, verify anything it
finds, and record it.

For a cell (q,n,R) and a target size M it launches `--workers` independent
covsearch processes with different seeds, keeps the best code any of them
produced, and -- crucially -- runs verify_cov.py on any code that claims zero
uncovered words before letting it be called a result.  A claim that does not
pass the independent verifier is discarded loudly.

Successful codes land in cov/results/K<q>_<n>_<R>_M<M>.txt with a JSON sidecar
recording q, n, R, M, the date, the method and the verifier output, plus the
Keri incumbent that the size should be compared against.

Usage
-----
  # one cell, 16 seeds, 120 s each
  python3 campaign.py -q 3 -n 11 -R 4 -M 80 --workers 16 -t 120

  # descend: start at M0 and, every time a size is solved, retry at M-1
  python3 campaign.py -q 6 -n 6 -R 3 --descend-from 41 --workers 16 -t 120

  # seed from a previously found code
  python3 campaign.py -q 3 -n 11 -R 4 -M 79 --seed-code results/K3_11_4_M80.txt
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, 'search', 'covsearch')
RESULTS = os.path.join(HERE, 'results')

sys.path.insert(0, HERE)
import verify_cov as V           # noqa: E402
import constructions as C        # noqa: E402


def keri_entry(q, n, R):
    try:
        idx = C.load_bounds()
    except Exception:
        return None
    return idx.get((q, n, R))


def run_batch(q, n, R, M, workers, tlimit, threads, seed0, extra, seed_code, tmpd,
              verbose=True):
    """Launch `workers` covsearch processes; return (best_uncovered, best_path)."""
    procs = []
    for w in range(workers):
        out = os.path.join(tmpd, 'w%03d.txt' % w)
        cmd = [BIN, '-q', str(q), '-n', str(n), '-R', str(R), '-M', str(M),
               '-t', str(tlimit), '-s', str(seed0 + w), '--threads', str(threads),
               '--out', out, '--quiet']
        if seed_code:
            cmd += ['--in', seed_code]
        cmd += extra
        procs.append((subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True), out))
    best, bestpath, rates = None, None, []
    for p, out in procs:
        stdout, _ = p.communicate()
        unc = None
        for line in stdout.splitlines():
            if line.startswith('RESULT '):
                kv = dict(tok.split('=', 1) for tok in line.split()[1:] if '=' in tok)
                unc = int(kv['uncovered'])
                rates.append(float(kv.get('rate', 0)))
        if unc is None:
            print('  worker produced no RESULT line:\n%s' % stdout[-500:])
            continue
        if best is None or unc < best:
            best, bestpath = unc, out
    if verbose and rates:
        print('  workers: best uncovered=%s   throughput %.0f-%.0f moves/s each'
              % (best, min(rates), max(rates)))
    return best, bestpath


def record(q, n, R, M, path, method, extra_meta=None):
    """Verify independently, then store in cov/results/ with a sidecar."""
    os.makedirs(RESULTS, exist_ok=True)
    code, ndup = V.parse_code(path, q, n)
    if len(code) > M:
        print('  REJECTED: file has %d distinct codewords, more than the claimed M=%d'
              % (len(code), M))
        return None
    if len(code) < M:
        # Duplicates in the solver output.  The distinct set covers exactly the
        # same words, so the honest claim is the smaller one.
        print('  note: %d duplicate codewords; recording at M=%d instead of %d'
              % (M - len(code), len(code), M))
        M = len(code)
    unc_pure = None
    if q ** n <= 3 * 10 ** 8:
        unc_pure = V.verify_pure(code, n, q, R, want_witness=False)[0]
    unc_np = V.verify_numpy(code, n, q, R, want_witness=False)[0]
    if unc_np != 0 or (unc_pure is not None and unc_pure != 0):
        print('  REJECTED BY VERIFIER: pure=%s numpy=%s' % (unc_pure, unc_np))
        return None

    dest = os.path.join(RESULTS, 'K%d_%d_%d_M%d.txt' % (q, n, R, M))
    shutil.copy(path, dest)
    e = keri_entry(q, n, R)
    meta = {
        'q': q, 'n': n, 'R': R, 'M': M,
        'date': datetime.date.today().isoformat(),
        'method': method,
        'code_file': os.path.basename(dest),
        'verified': {
            'verify_cov_pure_uncovered': unc_pure,
            'verify_cov_numpy_uncovered': unc_np,
            'duplicates_in_file': ndup,
        },
        'keri_2011': None if e is None else {
            'lb': e['lb'], 'ub': e['ub'],
            'lb_key': e['lb_key'], 'ub_key': e['ub_key'],
        },
    }
    if e is not None:
        meta['beats_keri_upper_bound'] = M < e['ub']
        meta['matches_keri_upper_bound'] = M == e['ub']
        if e.get('lb_updated'):
            meta['lower_bound_2025'] = e['lb_updated']
    if extra_meta:
        meta.update(extra_meta)
    with open(dest.replace('.txt', '.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('  VERIFIED and stored: %s  (M=%d)' % (dest, M))
    if e is not None:
        if M < e['ub']:
            print('  *** THIS BEATS THE KERI 2011 UPPER BOUND %d -> %d ***' % (e['ub'], M))
        elif M == e['ub']:
            print('  (reproduces the incumbent upper bound %d)' % e['ub'])
        else:
            print('  (incumbent upper bound is %d, this is weaker)' % e['ub'])
    return dest


def main():
    # campaign logs are usually redirected to a file and watched with tail, so
    # keep them line-buffered instead of 8 kB-buffered
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-q', type=int, required=True)
    ap.add_argument('-n', type=int, required=True)
    ap.add_argument('-R', type=int, required=True)
    ap.add_argument('-M', type=int)
    ap.add_argument('--descend-from', type=int,
                    help='start at this M and keep decrementing while it succeeds')
    ap.add_argument('--floor', type=int, default=1, help='stop descending at this M')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--threads', type=int, default=4, help='OpenMP threads per worker')
    ap.add_argument('-t', type=float, default=60.0, help='seconds per worker')
    ap.add_argument('--seed0', type=int, default=1)
    ap.add_argument('--seed-code', help='seed every worker from this code file')
    ap.add_argument('--rounds', type=int, default=1,
                    help='repeat the batch this many times at each M')
    ap.add_argument('--extra', nargs=argparse.REMAINDER, default=[],
                    help='remaining args are passed to covsearch')
    a = ap.parse_args()

    if not os.path.exists(BIN):
        sys.exit('covsearch not built: run make in cov/search')

    e = keri_entry(a.q, a.n, a.R)
    print('cell K_%d(%d,%d)  q^n=%d  ball=%d' %
          (a.q, a.n, a.R, a.q ** a.n, V.ball_volume(a.q, a.n, a.R)))
    if e:
        print('  Keri 2011: %d - %d  (upper key %s)%s'
              % (e['lb'], e['ub'], e['ub_key'],
                 '   lower bound now %d [GP2025]' % e['lb_updated']
                 if e.get('lb_updated') else ''))

    Ms = []
    if a.descend_from is not None:
        Ms = list(range(a.descend_from, a.floor - 1, -1))
    elif a.M is not None:
        Ms = [a.M]
    else:
        ap.error('need -M or --descend-from')

    seed_code = a.seed_code
    seed0 = a.seed0
    tmpd = tempfile.mkdtemp(prefix='cov_campaign_')
    try:
        for M in Ms:
            print('\n--- M = %d ---' % M)
            solved = False
            for rnd in range(a.rounds):
                best, path = run_batch(a.q, a.n, a.R, M, a.workers, a.t, a.threads,
                                       seed0, a.extra, seed_code, tmpd)
                seed0 += a.workers
                if best == 0:
                    dest = record(a.q, a.n, a.R, M, path,
                                  'covsearch tabu/greedy local search, %ds x %d workers'
                                  % (int(a.t), a.workers))
                    if dest:
                        seed_code = dest
                        solved = True
                        break
                else:
                    print('  round %d: best uncovered = %s' % (rnd + 1, best))
            if not solved:
                print('  M=%d not solved; stopping descent' % M)
                break
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
