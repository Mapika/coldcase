#!/usr/bin/env python3
"""Rank candidate orders N > record for a (Delta, D) cell by metacyclic richness.

For each N we list the factorisations N = m*n admitting a faithful-ish action, i.e.
an a with a^n == 1 (mod m) and a != 1.  Orders with a large non-trivial n are the
interesting ones -- a purely abelian group (a == 1 everywhere) is a circulant/abelian
Cayley graph, and those obey a much stronger L1-ball bound than the Moore bound.

Also prints the abelian (L1 lattice-ball) bound so it is clear which cells can NOT
be reached with abelian groups at all.
"""
import argparse
import json
import math
import os
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "table_current.json")


def divisors(N):
    ds = []
    i = 1
    while i * i <= N:
        if N % i == 0:
            ds.append(i)
            if i != N // i:
                ds.append(N // i)
        i += 1
    return sorted(ds)


def actions(m, n):
    """all a in Z_m^* with a^n == 1 (mod m); returns the list."""
    if m == 1:
        return [0]
    return [a for a in range(1, m) if math.gcd(a, m) == 1 and pow(a, n, m) == 1 % m]


def abelian_bound(delta, D):
    """Max order of an abelian Cayley graph of degree <= delta and diameter <= D.

    Upper bound: an abelian group is a quotient of Z^k (k = number of generator
    pairs), so the ball of radius D has at most as many elements as the L1 ball of
    radius D in Z^k.  Best k maximises that count subject to 2k <= delta
    (involutions only shrink it).  Count = sum_i 2^i C(k,i) C(D,i).
    """
    best = 0
    for k in range(0, delta // 2 + 1):
        tot = sum((2 ** i) * comb(k, i) * comb(D, i) for i in range(0, min(k, D) + 1))
        best = max(best, tot)
    return best


def moore(delta, D):
    if delta <= 2:
        return 2 * D + 1
    return 1 + delta * ((delta - 1) ** D - 1) // (delta - 2)


def rank(N, min_n=2, max_specs=99):
    """Return (score, list of (m,n,#a,nonabelian_a_count))."""
    out = []
    for n in divisors(N):
        if n < min_n:
            continue
        m = N // n
        if m == 1:
            continue
        acts = actions(m, n)
        nonab = [a for a in acts if a != 1 % m]
        # only actions whose order is exactly n give a "new" group vs smaller n
        full = [a for a in nonab if all(pow(a, n // p, m) != 1 % m
                                        for p in set(prime_factors(n)))]
        if nonab:
            out.append((m, n, len(acts), len(nonab), len(full)))
    score = max([n * (1 if full else 0) for (_, n, _, _, full) in out] or [0])
    return score, out


def prime_factors(x):
    f = []
    d = 2
    while d * d <= x:
        while x % d == 0:
            f.append(d)
            x //= d
        d += 1
    if x > 1:
        f.append(x)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=int, required=True)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--span", type=int, default=200, help="how far above the record to scan")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--start", type=int, default=None)
    a = ap.parse_args()

    t = json.load(open(TABLE))
    rec = t["cells"]["%d,%d" % (a.delta, a.D)]["N"]
    mb = moore(a.delta, a.D)
    ab = abelian_bound(a.delta, a.D)
    print("cell (%d,%d): record %d, Moore %d (%.3f), abelian/circulant ceiling %d %s"
          % (a.delta, a.D, rec, mb, rec / mb, ab,
             "<-- abelian CANNOT reach the record" if ab <= rec else ""))
    lo = a.start if a.start else rec + 1
    rows = []
    for N in range(lo, lo + a.span):
        if (a.delta & 1) and (N & 1):
            continue
        sc, out = rank(N)
        if sc:
            rows.append((sc, N, out))
    rows.sort(key=lambda r: (-r[0], r[1]))
    print("\ntop candidate orders (by largest faithful cyclic action n):")
    for sc, N, out in rows[:a.top]:
        best = sorted(out, key=lambda o: -o[1])[:4]
        desc = "  ".join("Z_%d:Z_%d(%d act)" % (m, n, nonab) for m, n, _, nonab, _ in best)
        print("  N=%-8d n_max=%-5d  %s" % (N, sc, desc))
    if not rows:
        print("  (none -- no non-abelian metacyclic group in this range)")


if __name__ == "__main__":
    main()
