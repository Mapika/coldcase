#!/usr/bin/env python3
"""Cross-check the C++ bitset ball kernel against an independent pure-Python BFS.

Randomly samples metacyclic groups Z_m rtimes_a Z_n and inverse-closed connection
sets, computes |B_k| both ways, and reports any disagreement.  Also verifies the
vertex-transitivity claim empirically: eccentricity of the identity == eccentricity
of every other vertex (so a single ball growth really does give the diameter).
"""
import random
import subprocess
import sys
import os
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("DD_BIN", os.path.join(HERE, "src", "dd_search"))


def apowers(m, n, a):
    p = [1 % m]
    for _ in range(1, n):
        p.append(p[-1] * a % m)
    return p


def mul(e1, e2, m, n, ap):
    i1, j1 = e1 % m, e1 // m
    i2, j2 = e2 % m, e2 // m
    return ((i1 + ap[j1] * i2) % m) + m * ((j1 + j2) % n)


def inv(e, m, n, ap):
    i, j = e % m, e // m
    jn = (n - j) % n
    return (((m - i % m) % m) * ap[jn] % m) + m * jn


def py_balls(m, n, a, S, D):
    N = m * n
    ap = apowers(m, n, a)
    cur = {0}
    out = []
    for _ in range(D):
        nxt = set(cur)
        for x in cur:
            for s in S:
                nxt.add(mul(x, s, m, n, ap))
        cur = nxt
        out.append(len(cur))
    return out


def py_all_ecc(m, n, a, S):
    """eccentricity from every vertex (tests vertex-transitivity)."""
    N = m * n
    ap = apowers(m, n, a)
    adj = [[mul(x, s, m, n, ap) for s in S] for x in range(N)]
    eccs = []
    for src in range(N):
        dist = [-1] * N
        dist[src] = 0
        q = deque([src])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if dist[y] < 0:
                    dist[y] = dist[x] + 1
                    q.append(y)
        eccs.append(max(dist))
    return eccs


def sample_group(rng):
    while True:
        m = rng.randint(2, 60)
        n = rng.randint(1, 12)
        cands = [a for a in range(1, m) if pow(a, n, m) == 1 % m and _gcd(a, m) == 1]
        if cands:
            return m, n, rng.choice(cands)


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def sample_S(m, n, a, rng, size):
    N = m * n
    ap = apowers(m, n, a)
    items = []
    seen = set()
    for e in range(1, N):
        if e in seen:
            continue
        ei = inv(e, m, n, ap)
        seen.add(e)
        seen.add(ei)
        items.append([e] if ei == e else [e, ei])
    rng.shuffle(items)
    S = []
    for it in items:
        if len(S) + len(it) <= size:
            S.extend(it)
        if len(S) == size:
            break
    return S if len(S) == size else None


def main():
    rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    bad = 0
    vt_checked = 0
    for t in range(trials):
        m, n, a = sample_group(rng)
        N = m * n
        if N > 700:
            continue
        hi = min(8, N - 1)
        if hi < 2:
            continue
        deg = rng.randint(2, hi)
        S = sample_S(m, n, a, rng, deg)
        if not S:
            continue
        D = rng.randint(1, 6)
        py = py_balls(m, n, a, S, D)
        spec = "%d,%d,%d:%s" % (m, n, a, ",".join(map(str, S)))
        r = subprocess.run([BIN, "--eval", spec, "--D", str(D)],
                           capture_output=True, text=True)
        cpp = [int(x.split("=")[1]) for x in r.stdout.split() if x.startswith("|B")]
        if cpp != py:
            bad += 1
            print("MISMATCH m=%d n=%d a=%d S=%s D=%d\n  py =%s\n  cpp=%s" % (m, n, a, S, D, py, cpp))
        # vertex-transitivity spot check on small instances
        if N <= 220 and vt_checked < 40:
            eccs = py_all_ecc(m, n, a, S)
            vt_checked += 1
            if len(set(eccs)) != 1:
                bad += 1
                print("VT VIOLATION m=%d n=%d a=%d S=%s eccs=%s" % (m, n, a, S, sorted(set(eccs))))
    print("crosscheck: %d trials, %d mismatches, %d vertex-transitivity checks" % (trials, bad, vt_checked))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
