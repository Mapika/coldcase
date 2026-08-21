#!/usr/bin/env python3
"""
tally.py -- regenerate cov/results/final_records.json from what is on disk.

Walks every code file in cov/results/, keeps the smallest M per cell, compares
it against Kéri's tabulated upper bound, and records the provenance read out of
each code's own sidecar -- so a cell produced by the concurrent cov_sweep is
never silently attributed to this engine.

Run it after any campaign2.py run; the descents keep going, so the file is a
snapshot, not a conclusion.
"""
import glob
import json
import os
import re
import sys

COV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COV)
import constructions as C          # noqa: E402

# what the record hunt held before the cov/engine session of 2026-08-20/21
PREVIOUS = {(6, 7, 3): 232, (6, 8, 3): 1045, (6, 8, 4): 167, (6, 9, 4): 703,
            (6, 9, 5): 123, (6, 10, 4): 2951, (6, 10, 5): 610,
            (7, 8, 4): 329, (7, 9, 4): 1743}


def main():
    best = {}
    for f in glob.glob(os.path.join(COV, 'results', 'K*_*_*_M*.txt')):
        m = re.match(r'K(\d+)_(\d+)_(\d+)_M(\d+)\.txt$', os.path.basename(f))
        if not m:
            continue
        q, n, R, M = (int(x) for x in m.groups())
        k = (q, n, R)
        if k not in best or M < best[k][0]:
            best[k] = (M, f)

    bounds = C.load_bounds()
    rows = []
    for k in sorted(best):
        q, n, R = k
        M, f = best[k]
        e = bounds.get(k)
        if not (e and M < e['ub']):
            continue
        side = f[:-4] + '.json'
        produced = 'unknown'
        if os.path.exists(side):
            meta = json.load(open(side))
            produced = ('cov/engine/covengine' if meta.get('engine')
                        else meta.get('method', '')[:60])
        rows.append(dict(q=q, n=n, R=R, keri=e['ub'], ours=M,
                         previous_ours=PREVIOUS.get(k),
                         code_file=os.path.basename(f), produced_by=produced))

    out = os.path.join(COV, 'results', 'final_records.json')
    with open(out, 'w') as fh:
        json.dump(rows, fh, indent=1)
    print('%-12s %7s %7s %7s %8s  %s' % ('cell', 'Keri', 'was', 'now', 'vs Keri', 'by'))
    for r in rows:
        print('K_%d(%d,%d)%s %7d %7s %7d %7.1f%%  %s'
              % (r['q'], r['n'], r['R'], ' ' * (5 - len(str(r['n'])) - len(str(r['R']))),
                 r['keri'], str(r['previous_ours'] or 'new'), r['ours'],
                 100.0 * (r['keri'] - r['ours']) / r['keri'],
                 'covengine' if 'covengine' in r['produced_by'] else 'cov_sweep'))
    print('%d cells below Kéri; written to %s' % (len(rows), out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
