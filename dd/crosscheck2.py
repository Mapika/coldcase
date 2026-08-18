#!/usr/bin/env python3
"""Cross-check the affine2 model of dd_search2 against a pure-Python group law.

G = (Z_q x Z_q) rtimes_A Z_n, A in GL_2(Z_q) of order n (q need not be prime).
element ((x,y), j) -> index (j*p + x)*p + y
((x1,y1),j1)*((x2,y2),j2) = ((x1,y1) + A^{j1}(x2,y2), j1+j2 mod n)
"""
import random
import subprocess
import sys
import os
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("DD_BIN", os.path.join(HERE, "src", "dd_search2"))


def matmul(X, Y, p):
    return [(X[0] * Y[0] + X[1] * Y[2]) % p, (X[0] * Y[1] + X[1] * Y[3]) % p,
            (X[2] * Y[0] + X[3] * Y[2]) % p, (X[2] * Y[1] + X[3] * Y[3]) % p]


def mat_order(A, p, bound=200):
    C = list(A)
    I = [1 % p, 0, 0, 1 % p]
    for k in range(1, bound + 1):
        if C == I:
            return k
        C = matmul(C, A, p)
    return 0


def powers(A, p, n):
    out = [[1 % p, 0, 0, 1 % p]]
    for _ in range(1, n):
        out.append(matmul(out[-1], A, p))
    return out


def dec(e, p):
    y = e % p
    r = e // p
    return r % p, y, r // p


def mul(e1, e2, p, n, Ap):
    x1, y1, j1 = dec(e1, p)
    x2, y2, j2 = dec(e2, p)
    X = Ap[j1]
    nx = (x1 + X[0] * x2 + X[1] * y2) % p
    ny = (y1 + X[2] * x2 + X[3] * y2) % p
    return ((j1 + j2) % n * p + nx) * p + ny


def inv(e, p, n, Ap):
    x, y, j = dec(e, p)
    jn = (n - j) % n
    X = Ap[jn]
    return (jn * p + (-(X[0] * x + X[1] * y)) % p) * p + (-(X[2] * x + X[3] * y)) % p


def main():
    rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    bad = 0
    done = 0
    vt = 0
    while done < trials:
        p = rng.choice([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])   # base need not be prime
        A = [rng.randrange(p) for _ in range(4)]
        from math import gcd
        if gcd((A[0] * A[3] - A[1] * A[2]) % p, p) != 1:
            continue
        n = mat_order(A, p)
        if n < 2 or n > 12:
            continue
        N = p * p * n
        if N > 700:
            continue
        Ap = powers(A, p, n)
        # inverse-closed connection set
        seen, items = set(), []
        for e in range(1, N):
            if e in seen:
                continue
            ei = inv(e, p, n, Ap)
            seen.add(e)
            seen.add(ei)
            items.append([e] if ei == e else [e, ei])
        rng.shuffle(items)
        deg = rng.randint(2, 8)
        S = []
        for it in items:
            if len(S) + len(it) <= deg:
                S.extend(it)
            if len(S) == deg:
                break
        if len(S) != deg:
            continue
        # sanity: group law closure + inverse consistency
        for _ in range(30):
            a1, b1 = rng.randrange(N), rng.randrange(N)
            assert mul(a1, inv(a1, p, n, Ap), p, n, Ap) == 0, "inverse wrong"
            assert 0 <= mul(a1, b1, p, n, Ap) < N
        D = rng.randint(1, 5)
        cur = {0}
        py = []
        for _ in range(D):
            cur = cur | {mul(x, s, p, n, Ap) for x in cur for s in S}
            py.append(len(cur))
        spec = "%d,%d,%d,%d,%d,%d:%s" % (p, n, A[0], A[1], A[2], A[3], ",".join(map(str, S)))
        r = subprocess.run([BIN, "--eval2", spec, "--D", str(D)], capture_output=True, text=True)
        cpp = [int(x.split("=")[1]) for x in r.stdout.split() if x.startswith("|B")]
        if cpp != py:
            bad += 1
            print("MISMATCH p=%d n=%d A=%s S=%s D=%d\n  py =%s\n  cpp=%s" % (p, n, A, S, D, py, cpp))
        # vertex transitivity
        if N <= 250 and vt < 25:
            adj = [[mul(x, s, p, n, Ap) for s in S] for x in range(N)]
            eccs = set()
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
                eccs.add(max(dist))
            vt += 1
            if len(eccs) != 1:
                bad += 1
                print("VT VIOLATION p=%d n=%d A=%s eccs=%s" % (p, n, A, eccs))
        done += 1
    print("crosscheck2 (affine2): %d trials, %d mismatches, %d VT checks" % (done, bad, vt))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
