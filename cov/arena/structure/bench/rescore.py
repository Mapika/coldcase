#!/usr/bin/env python3
"""Apply the RULES.md amendment-1 score to the raw baseline log.

The scoring change (partial -> -floor(uncovered*10^6/q^n)-1, invalid -> -2e6)
is a monotone remap of the uncovered counts already in bench/base60.log, so the
baseline does not need re-running to be compared under the new rule.
"""
import re, sys, statistics
cells = {}
for line in open(sys.argv[1] if len(sys.argv) > 1 else "bench/base60.log"):
    m = re.match(r"baseline K_(\d+)\((\d+),(\d+)\)@(\d+) seed=(\d+) wall=([\d.]+)s uncovered=(\S+)", line)
    if not m:
        continue
    q, n, R, M = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    u, wall = m.group(7), float(m.group(6))
    qn = q ** n
    sc = -2 * 10**6 if u == "INVALID" else (1000 if int(u) == 0
                                            else -int(int(u) * 10**6 // qn) - 1)
    cells.setdefault(f"K{q}({n},{R})@{M}", []).append((sc, wall))
tot_solved, tot_med = 0, 0
for k, v in cells.items():
    sc = sorted(s for s, _ in v)
    solved = sum(1 for s in sc if s == 1000)
    med = sc[len(sc) // 2]
    mw = sorted(w for _, w in v)[len(v) // 2]
    tot_solved += solved
    tot_med += med
    print(f"baseline     {k}: solved {solved}/{len(v)} median_score {med} median_wall {mw}s")
print(f'{{"baseline": {{"total_solved": {tot_solved}, "sum_median": {tot_med}}}}}')
