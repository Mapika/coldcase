#!/usr/bin/env python3
"""
campaign2.py -- cov/campaign.py, driving cov/engine/covengine instead of a bare
covsearch worker pool.

This is a thin fork.  The part that decides whether something is a result is
not forked at all: `record()` and `keri_entry()` are IMPORTED from
cov/campaign.py and used unchanged, so every code that lands in cov/results/
goes through exactly the same gate it always did --

    cov/verify_cov.py, both exhaustive methods, on the file re-read from disk,
    with the distinct-codeword count checked against the claimed M.

A code that fails is discarded loudly.  The solver's own "uncovered = 0" is
never the evidence, and nothing in this file changes that.

What is different from campaign.py:

  * one covengine invocation per (M, round) instead of `--workers` covsearch
    processes.  covengine IS the worker pool: it runs the algebraic phase, then
    a portfolio of independent chains over the whole core budget, and stops the
    world the moment one of them reports a cover (cov/engine/NOTES.md).
  * the descent seeds itself.  covengine looks in cov/results/ for a code of
    this cell at M or just above and hands it to a chain for remove-and-repair,
    so `--descend-from` does not have to be told where to start.
  * `--attack` runs a list of cells from a JSON file, so an overnight run is one
    command and one log.

Usage
-----
  # descend K_6(9,5) from 123 downwards, 30 minutes a notch, 40 cores
  python3 campaign2.py -q 6 -n 9 -R 5 --descend-from 122 --floor 118 \\
          -t 1800 --cores 40

  # a single size
  python3 campaign2.py -q 6 -n 6 -R 3 -M 40 -t 1800 --cores 40
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
COV = os.path.dirname(HERE)
ENGINE = os.path.join(HERE, 'covengine')

sys.path.insert(0, COV)
import verify_cov as V          # noqa: E402
import campaign as CP           # noqa: E402  (record/keri_entry, used unchanged)


def run_engine(q, n, R, M, tlimit, cores, seed, out, seed_code=None,
               no_results=False, nice=12, log=print):
    """One covengine run.  Returns the uncovered count it reports."""
    env = dict(os.environ)
    env['COVENGINE_CORES'] = str(cores)
    env['COVENGINE_NICE'] = str(nice)
    if no_results:
        env['COVENGINE_NO_RESULTS'] = '1'
    cmd = [sys.executable, ENGINE, '-q', str(q), '-n', str(n), '-R', str(R),
           '-M', str(M), '-t', str(tlimit), '-s', str(seed),
           '--cores', str(cores), '--out', out]
    if seed_code:
        cmd += ['--in', seed_code]
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env)
    unc = None
    for line in p.stdout:
        line = line.rstrip()
        if line.startswith('RESULT '):
            kv = dict(tok.split('=', 1) for tok in line.split()[1:] if '=' in tok)
            unc = int(kv.get('uncovered', -1))
        elif line:
            log('    | ' + line)
    p.wait()
    log('    engine finished in %.0fs, uncovered=%s' % (time.time() - t0, unc))
    return unc


def attack(q, n, R, Ms, args, log):
    e = CP.keri_entry(q, n, R)
    log('cell K_%d(%d,%d)  q^n=%d  ball=%d' % (q, n, R, q ** n, V.ball_volume(q, n, R)))
    if e:
        log('  Keri 2011: %d - %d  (upper key %s)%s'
            % (e['lb'], e['ub'], e['ub_key'],
               '   lower bound now %d [GP2025]' % e['lb_updated']
               if e.get('lb_updated') else ''))
    sphere_lb = (q ** n + V.ball_volume(q, n, R) - 1) // V.ball_volume(q, n, R)
    log('  sphere-covering bound: M >= %d' % sphere_lb)

    tmpd = tempfile.mkdtemp(prefix='campaign2_')
    seed_code = args.seed_code
    found = []
    try:
        for M in Ms:
            if M < sphere_lb:
                log('\n--- M = %d is below the sphere-covering bound %d; stopping'
                    % (M, sphere_lb))
                break
            log('\n--- M = %d  (%s) ---' % (M, datetime.datetime.now().strftime('%H:%M:%S')))
            solved = False
            for rnd in range(args.rounds):
                out = os.path.join(tmpd, 'K%d_%d_%d_M%d_r%d.txt' % (q, n, R, M, rnd))
                unc = run_engine(q, n, R, M, args.t, args.cores,
                                 args.seed0 + 1000 * rnd, out, seed_code,
                                 no_results=args.no_results, nice=args.nice,
                                 log=log)
                if unc == 0:
                    # THE GATE.  campaign.record re-reads the file from disk and
                    # runs verify_cov.py (pure + numpy) before anything is
                    # written to cov/results/.
                    dest = CP.record(q, n, R, M, out,
                                     'covengine portfolio (lincov/symsearch + '
                                     'covsearch2e p5b + covfast), %ds x %d cores'
                                     % (int(args.t), args.cores),
                                     extra_meta={'engine': 'cov/engine/covengine',
                                                 'engine_rounds': rnd + 1})
                    if dest:
                        found.append((M, dest))
                        seed_code = dest
                        solved = True
                        break
                    log('  VERIFIER REJECTED a claimed cover at M=%d -- not recorded' % M)
                else:
                    log('  round %d: best uncovered = %s' % (rnd + 1, unc))
            if not solved:
                log('  M=%d not solved' % M)
                if not args.keep_going:
                    break
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)

    if found:
        log('\n=== recorded ===')
        for (M, dest) in found:
            flag = ''
            if e:
                if M < e['ub']:
                    flag = '   *** BEATS KERI %d ***' % e['ub']
                elif M == e['ub']:
                    flag = '   (= Keri)'
            log('  K_%d(%d,%d) <= %d   %s%s' % (q, n, R, M, os.path.basename(dest), flag))
    return found


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-q', type=int)
    ap.add_argument('-n', type=int)
    ap.add_argument('-R', type=int)
    ap.add_argument('-M', type=int)
    ap.add_argument('--descend-from', type=int,
                    help='start at this M and keep decrementing while it succeeds')
    ap.add_argument('--floor', type=int, default=1, help='stop descending at this M')
    ap.add_argument('--cores', type=int, default=40)
    ap.add_argument('-t', type=float, default=1800.0, help='seconds per M per round')
    ap.add_argument('--rounds', type=int, default=1)
    ap.add_argument('--seed0', type=int, default=1)
    ap.add_argument('--seed-code', help='seed the first M from this code file')
    ap.add_argument('--nice', type=int, default=12)
    ap.add_argument('--keep-going', action='store_true',
                    help='keep trying smaller M even after one fails')
    ap.add_argument('--no-results', action='store_true',
                    help='do not seed from cov/results/ (regression / from-scratch runs)')
    ap.add_argument('--attack', help='JSON list of {"q","n","R","M"|"descend_from"} cells')
    a = ap.parse_args()

    if not os.path.exists(ENGINE):
        sys.exit('cov/engine/covengine missing')

    jobs = []
    if a.attack:
        for c in json.load(open(a.attack)):
            Ms = ([c['M']] if 'M' in c else
                  list(range(c['descend_from'], c.get('floor', 1) - 1, -1)))
            jobs.append((c['q'], c['n'], c['R'], Ms))
    else:
        if a.q is None or a.n is None or a.R is None:
            ap.error('need -q -n -R (or --attack)')
        if a.descend_from is not None:
            Ms = list(range(a.descend_from, a.floor - 1, -1))
        elif a.M is not None:
            Ms = [a.M]
        else:
            ap.error('need -M or --descend-from')
        jobs.append((a.q, a.n, a.R, Ms))

    allfound = []
    for (q, n, R, Ms) in jobs:
        print('\n' + '=' * 70)
        allfound += attack(q, n, R, Ms, a, print)
    print('\n' + '=' * 70)
    print('campaign2 done: %d code(s) verified and recorded' % len(allfound))
    return 0


if __name__ == '__main__':
    sys.exit(main())
