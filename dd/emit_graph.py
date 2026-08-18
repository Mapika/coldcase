#!/usr/bin/env python3
"""Expand a dd_search hit (JSON line) into an edge list + JSON sidecar in dd/results/.

Hit format: {"delta":D,"D":d,"N":n,"m":m,"n":n2,"a":a,"S":[...]}
Element e encodes (i,j) = (e % m, e // m) in G = Z_m rtimes_a Z_n2.

Also re-derives the graph independently of the C++ engine (pure Python group law),
so the edge list is not just a copy of the searcher's internal state.
"""
import argparse
import json
import os
import sys
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def group_mul(e1, e2, m, n, apow):
    i1, j1 = e1 % m, e1 // m
    i2, j2 = e2 % m, e2 // m
    return ((i1 + apow[j1] * i2) % m) + m * ((j1 + j2) % n)


def matmul(X, Y, p):
    return [(X[0] * Y[0] + X[1] * Y[2]) % p, (X[0] * Y[1] + X[1] * Y[3]) % p,
            (X[2] * Y[0] + X[3] * Y[2]) % p, (X[2] * Y[1] + X[3] * Y[3]) % p]


def build_graph_affine2(hit):
    """G = (Z_p x Z_p) rtimes_A Z_n; index ((x,y),j) = (j*p+x)*p+y."""
    p, n, A, N = hit["p"], hit["n"], hit["A"], hit["N"]
    assert p * p * n == N, "p^2*n != N"
    Ap = [[1 % p, 0, 0, 1 % p]]
    for _ in range(1, n):
        Ap.append(matmul(Ap[-1], A, p))
    assert matmul(Ap[-1], A, p) == [1 % p, 0, 0, 1 % p], "A^n != I -- not a valid group"

    def mul(e1, e2):
        y1, r1 = e1 % p, e1 // p
        x1, j1 = r1 % p, r1 // p
        y2, r2 = e2 % p, e2 // p
        x2, j2 = r2 % p, r2 // p
        X = Ap[j1]
        return (((j1 + j2) % n) * p + (x1 + X[0] * x2 + X[1] * y2) % p) * p \
               + (y1 + X[2] * x2 + X[3] * y2) % p

    S = hit["S"]
    assert len(set(S)) == len(S) and 0 not in S
    inv = {}
    for e in S:
        for f in range(N):
            if mul(e, f) == 0:
                inv[e] = f
                break
    Sset = set(S)
    for e in S:
        assert inv[e] in Sset, "S not closed under inverse (%d)" % e
    edges = set()
    for x in range(N):
        for s in S:
            y = mul(x, s)
            edges.add((x, y) if x < y else (y, x))
    return N, sorted(edges), S


def build_graph(hit):
    if hit.get("model") == "affine2":
        return build_graph_affine2(hit)
    m, n, a, N = hit["m"], hit["n"], hit["a"], hit["N"]
    assert m * n == N, "m*n != N"
    apow = [1 % m]
    for _ in range(1, n):
        apow.append(apow[-1] * a % m)
    assert apow[-1] * a % m == 1 % m, "a^n != 1 mod m -- not a valid group"
    S = hit["S"]
    assert len(set(S)) == len(S), "repeated generator"
    assert 0 not in S, "identity in S"
    # inverse-closure check
    Sset = set(S)
    inv = {}
    for e in S:
        for f in range(N):
            if group_mul(e, f, m, n, apow) == 0:
                inv[e] = f
                break
    for e in S:
        assert inv[e] in Sset, "S not closed under inverse (%d)" % e
    edges = set()
    for x in range(N):
        for s in S:
            y = group_mul(x, s, m, n, apow)
            edges.add((x, y) if x < y else (y, x))
    return N, sorted(edges), S


def diameter(N, edges):
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    best = 0
    for s in range(N):
        dist = [-1] * N
        dist[s] = 0
        q = deque([s])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if dist[y] < 0:
                    dist[y] = dist[x] + 1
                    q.append(y)
        if min(dist) < 0:
            return -1
        best = max(best, max(dist))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hitfile", help="file with one JSON hit per line, or '-' for stdin")
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", default=RESULTS)
    ap.add_argument("--check-diameter", action="store_true")
    a = ap.parse_args()

    src = sys.stdin if a.hitfile == "-" else open(a.hitfile)
    os.makedirs(a.outdir, exist_ok=True)
    written = []
    for line in src:
        line = line.strip()
        if not line:
            continue
        hit = json.loads(line)
        N, edges, S = build_graph(hit)
        deg = [0] * N
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        assert max(deg) == hit["delta"], "degree %d != delta %d" % (max(deg), hit["delta"])
        if hit.get("model") == "affine2":
            name = "d%d_D%d_N%d_aff_p%d_n%d%s" % (
                hit["delta"], hit["D"], N, hit["p"], hit["n"],
                ("_" + a.tag) if a.tag else "")
        else:
            name = "d%d_D%d_N%d_m%d_n%d_a%d%s" % (
                hit["delta"], hit["D"], N, hit["m"], hit["n"], hit["a"],
                ("_" + a.tag) if a.tag else "")
        ep = os.path.join(a.outdir, name + ".edges")
        with open(ep, "w") as f:
            if hit.get("model") == "affine2":
                f.write("# Cayley graph of (Z_%d x Z_%d) rtimes_A Z_%d, A=%s, degree %d, diameter %d\n"
                        % (hit["p"], hit["p"], hit["n"], hit["A"], hit["delta"], hit["D"]))
                f.write("# element ((x,y),j) <-> index (j*%d+x)*%d+y ; S = %s\n" % (hit["p"], hit["p"], S))
            else:
                f.write("# Cayley graph of Z_%d rtimes_%d Z_%d, degree %d, target diameter %d\n"
                        % (hit["m"], hit["a"], hit["n"], hit["delta"], hit["D"]))
                f.write("# element (i,j) <-> index j*%d+i ; connection set S = %s\n" % (hit["m"], S))
            f.write("N %d\n" % N)
            for u, v in edges:
                f.write("%d %d\n" % (u, v))
        meta = dict(hit)
        meta["edges"] = len(edges)
        if hit.get("model") == "affine2":
            meta["group"] = "(Z_%d x Z_%d) rtimes_A Z_%d, A=%s" % (hit["p"], hit["p"], hit["n"], hit["A"])
            meta["construction"] = "Cayley graph of an affine-type group (Z_p^2 : Z_n)"
        else:
            meta["group"] = "Z_%d rtimes_%d Z_%d" % (hit["m"], hit["a"], hit["n"])
            meta["construction"] = "Cayley graph of metacyclic group, connection set S"
        if a.check_diameter:
            meta["verified_diameter"] = diameter(N, edges)
        json.dump(meta, open(os.path.join(a.outdir, name + ".json"), "w"), indent=1)
        written.append(ep)
        print(ep)
    if not written:
        print("(no hits)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
