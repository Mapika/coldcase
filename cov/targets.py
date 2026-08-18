#!/usr/bin/env python3
"""
targets.py -- rank K_q(n,R) cells by how attackable their UPPER bound is.

An upper bound is beatable by search when three things line up:

  1. Nobody searched it.  Keri's upper-bound key letter says where the value
     came from.  Keys f (direct sum), c (K_q(n+1,R+1) <= K_q(n,R)), e
     (K_q(n+1,R) <= q K_q(n,R)), k (bound for smaller q) and the unmarked
     trivial entries are pure bookkeeping -- no code was ever searched for that
     cell.  Keys m/n/o/q/t/x/y/z are search or construction results by people
     who did look.

  2. The instance fits.  The counter array is q^n cells; the cost of one move
     is C(n-1,R)(q-1)^R.

  3. There is slack.  redundancy = M * |B_R| / q^n is how many times over the
     balls cover the space at the incumbent size M.  A perfect code has
     redundancy 1 and is unbeatable; the empirically solved instances in this
     project sat around 2.5-3, and the untouched direct-sum corner runs at 4-10,
     which is where the fat is.

The "gap" column (ub/lb) is the headline number but it is a poor predictor on
its own, because a big gap can mean a weak lower bound rather than a bad code.

Usage:
  python3 targets.py                 # top cells overall
  python3 targets.py --qmin 6        # only the untouched q >= 6 corner
  python3 targets.py --max-space 1e8 --max-M 3000
"""

import argparse
import json
import os
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))

# Keri upper-bound keys that mean "derived from another cell, never searched".
# See the key pages of the PDFs (reproduced in bounds.json under "keys").
BOOKKEEPING = {
    None: 'trivial',
    'c': 'K_q(n+1,R+1) <= K_q(n,R)',
    'e': 'K_q(n+1,R) <= q K_q(n,R)',
    'f': 'direct sum',
    'j': 'K_q(n,1) <= t K_q(n,1) / sigma bound',
    'k': 'bound for the same n,R and smaller q',
}


def ball_volume(q, n, R):
    return sum(comb(n, i) * (q - 1) ** i for i in range(min(R, n) + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bounds', default=os.path.join(HERE, 'bounds.json'))
    ap.add_argument('--qmin', type=int, default=3)
    ap.add_argument('--qmax', type=int, default=21)
    ap.add_argument('--max-space', type=float, default=1e11,
                    help='largest q^n to consider')
    ap.add_argument('--max-M', type=int, default=100000)
    ap.add_argument('--max-move', type=float, default=1e7,
                    help='largest C(n-1,R)(q-1)^R (cost of one move)')
    ap.add_argument('--only-unsearched', action='store_true',
                    help='keep only cells whose upper bound is bookkeeping')
    ap.add_argument('--top', type=int, default=40)
    ap.add_argument('--sort', default='redundancy',
                    choices=['redundancy', 'gap', 'space', 'M'])
    a = ap.parse_args()

    db = json.load(open(a.bounds))
    rows = []
    for e in db['entries']:
        q, n, R, M = e['q'], e['n'], e['R'], e['ub']
        if not (a.qmin <= q <= a.qmax):
            continue
        if e['lb'] == e['ub']:
            continue                       # solved exactly, nothing to beat
        if R < 1 or R >= n:
            continue
        space = q ** n
        if space > a.max_space or M > a.max_M:
            continue
        move = comb(n - 1, R) * (q - 1) ** R
        if move > a.max_move:
            continue
        unsearched = e['ub_key'] in BOOKKEEPING
        if a.only_unsearched and not unsearched:
            continue
        V = ball_volume(q, n, R)
        lb = e.get('lb_updated') or e['lb']
        rows.append(dict(q=q, n=n, R=R, lb=lb, ub=M, gap=M / lb,
                         space=space, V=V, move=move,
                         redundancy=M * V / space,
                         key=e['ub_key'],
                         how=BOOKKEEPING.get(e['ub_key'], 'searched/constructed'),
                         unsearched=unsearched))

    keyf = {'redundancy': lambda r: -r['redundancy'],
            'gap': lambda r: -r['gap'],
            'space': lambda r: r['space'],
            'M': lambda r: r['M'] if 'M' in r else r['ub']}[a.sort]
    rows.sort(key=keyf)

    print('%-14s %8s %8s %6s %11s %10s %10s %6s  %s'
          % ('cell', 'lb', 'ub', 'gap', 'q^n', 'ball', 'move', 'redun', 'ub provenance'))
    print('-' * 105)
    for r in rows[:a.top]:
        print('K_%-2d(%2d,%d)%s %8d %8d %6.1f %11.3g %10d %10d %6.2f  %s (key %s)'
              % (r['q'], r['n'], r['R'], ' ' if r['unsearched'] else '*',
                 r['lb'], r['ub'], r['gap'], r['space'], r['V'], r['move'],
                 r['redundancy'], r['how'], r['key']))
    print()
    print('* = the incumbent upper bound is a search/construction result, so it '
          'was actually looked at.')
    print('redun = ub * |B_R| / q^n; 1.0 is a perfect code, higher means more '
          'slack to remove.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
