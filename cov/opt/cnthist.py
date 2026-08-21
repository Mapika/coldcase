#!/usr/bin/env python3
"""Coverage-multiplicity histogram of a code file.

Used to justify (or refute) two ideas in METHODS.md:
  * the shared-sphere leaving count is cheap because cnt(w)==1 is rare;
  * "inverting" the leaving count (iterating a maintained list of the
    cnt(w)==1 words instead of walking the sphere) is only profitable if
    |U_1| * n < C(n,R)(q-1)^R, which this script checks.

usage: cnthist.py CODEFILE -q Q -n N -R R
"""
import argparse, itertools, sys
from math import comb

ap = argparse.ArgumentParser()
ap.add_argument("codefile"); ap.add_argument("-q", type=int, required=True)
ap.add_argument("-n", type=int, required=True); ap.add_argument("-R", type=int, required=True)
a = ap.parse_args()
q, n, R = a.q, a.n, a.R
DIG = "0123456789abcdefghijklmnopqrstuvwxyz"

code = []
for line in open(a.codefile):
    line = line.split("#")[0].strip()
    if not line:
        continue
    if " " in line or "," in line:
        w = [int(t) for t in line.replace(",", " ").split()]
    else:
        w = [DIG.index(ch) for ch in line]
    assert len(w) == n, line
    code.append(w)
M = len(code)
NT = q ** n
pw = [q ** (n - 1 - j) for j in range(n)]
cnt = bytearray(NT)          # saturating at 255, enough for the histogram tail


def ball(c):
    base = sum(c[j] * pw[j] for j in range(n))
    yield base
    for r in range(1, R + 1):
        for pos in itertools.combinations(range(n), r):
            for vals in itertools.product(*[[v for v in range(q) if v != c[j]] for j in pos]):
                yield base + sum((vals[i] - c[pos[i]]) * pw[pos[i]] for i in range(r))


for c in code:
    for w in ball(c):
        if cnt[w] < 255:
            cnt[w] += 1

hist = {}
for v in cnt:
    hist[v] = hist.get(v, 0) + 1
tot = NT
print(f"{a.codefile}: q={q} n={n} R={R} M={M} q^n={NT} |B_R|={sum(comb(n,i)*(q-1)**i for i in range(R+1))}")
print(f"mean multiplicity = {sum(k*v for k,v in hist.items())/tot:.3f}")
for k in sorted(hist)[:10]:
    print(f"  cnt={k:3d}: {hist[k]:10d}  ({100*hist[k]/tot:6.3f}%)")
u0, u1 = hist.get(0, 0), hist.get(1, 0)
thresh_pos = comb(n, R) * (q - 1) ** R
thresh_neg = (R + 1) * comb(n - 1, R) * (q - 1) ** R
print(f"|U|  = {u0:9d}   list-based ENTERING side profitable iff |U|*n < {thresh_neg}"
      f"  -> n|U| = {u0*n} : {'YES' if u0*n < thresh_neg else 'no'}")
print(f"|U1| = {u1:9d}   list-based LEAVING  side profitable iff |U1|*n < {thresh_pos}"
      f"  -> n|U1| = {u1*n} : {'YES' if u1*n < thresh_pos else 'no'}")
