#!/usr/bin/env python3
"""verify_vt.py -- exact verifier for large vertex-transitive candidates.

Pure Python, standard library only.

An exhaustive all-pairs BFS is O(N*E), which is hours once N reaches tens of
thousands.  This verifier is exact and near-linear instead, by *proving* vertex-
transitivity from the edge list rather than assuming it:

  1. read the edge list; reject self-loops, repeated edges, degree > Delta,
     disconnected graphs;
  2. build two candidate permutations of the vertex set from the construction data
     in the sidecar -- left translation by the two generators of
     G = Z_m rtimes_a Z_n:
         L_x(i,j) = (i+1 mod m, j)          [ = (1,0) . (i,j) ]
         L_y(i,j) = (a*i mod m, j+1 mod n)  [ = (0,1) . (i,j) ]
  3. CHECK, against the edge list alone, that each permutation really is a graph
     automorphism: it is a bijection and it maps every edge to an edge;
  4. CHECK that the group they generate acts transitively on the vertices (orbit of
     vertex 0 is everything);
  5. conclude that all eccentricities are equal, so diameter = ecc(0), and compute
     ecc(0) with a single BFS.

Steps 3-5 use only the edge list -- the sidecar merely *suggests* the permutations,
and a wrong suggestion fails step 3 rather than producing a wrong answer.  If any
check fails the script reports FAIL and makes no diameter claim.

    python3 verify_vt.py FILE.edges
"""
import argparse
import json
import os
import sys
from collections import deque


def read_edges(path):
    n_decl = None
    edges = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if p[0].upper() == "N" and len(p) == 2:
                n_decl = int(p[1])
                continue
            edges.append((int(p[0]), int(p[1])))
    return n_decl, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edgelist")
    ap.add_argument("--delta", type=int)
    ap.add_argument("--D", type=int)
    ap.add_argument("--N", type=int)
    a = ap.parse_args()

    side = os.path.splitext(a.edgelist)[0] + ".json"
    meta = json.load(open(side)) if os.path.exists(side) else {}
    delta = a.delta or meta.get("delta")
    D = a.D or meta.get("D")
    Nc = a.N or meta.get("N")
    if delta is None or D is None:
        print("FAIL: need --delta and --D")
        return 1
    if meta.get("model") == "affine2":
        print("FAIL: this verifier only knows the metacyclic construction")
        return 1
    m, n, aa = meta.get("m"), meta.get("n"), meta.get("a")
    if m is None:
        print("FAIL: sidecar has no group data (m, n, a)")
        return 1
    if pow(aa, n, m) != 1 % m:
        print("FAIL: a^n != 1 (mod m); Z_%d rtimes_%d Z_%d is not a group" % (m, aa, n))
        return 1

    n_decl, edges = read_edges(a.edgelist)
    maxv = max(max(u, v) for u, v in edges) + 1
    N = n_decl if n_decl is not None else (Nc if Nc is not None else maxv)
    if N < maxv:
        print("FAIL: declared N=%d < max vertex index+1 = %d" % (N, maxv))
        return 1
    if Nc is not None and N != Nc:
        print("FAIL: vertex count %d != claimed %d" % (N, Nc))
        return 1
    if m * n != N:
        print("FAIL: m*n = %d != N = %d" % (m * n, N))
        return 1

    # ---- 1. simplicity, degrees, connectivity
    adj = [[] for _ in range(N)]
    seen = set()
    for u, v in edges:
        if u == v:
            print("FAIL: self-loop at %d" % u)
            return 1
        k = (u, v) if u < v else (v, u)
        if k in seen:
            print("FAIL: repeated edge %s" % (k,))
            return 1
        seen.add(k)
        adj[u].append(v)
        adj[v].append(u)
    degs = [len(x) for x in adj]
    if max(degs) > delta:
        print("FAIL: max degree %d > Delta=%d" % (max(degs), delta))
        return 1

    # ---- 2. candidate automorphisms: left translation by the two group generators
    def Lx(e):
        i, j = e % m, e // m
        return (i + 1) % m + m * j

    def Ly(e):
        i, j = e % m, e // m
        return (aa * i) % m + m * ((j + 1) % n)

    perms = [("L_x = left mult by (1,0)", [Lx(e) for e in range(N)]),
             ("L_y = left mult by (0,1)", [Ly(e) for e in range(N)])]

    for name, p in perms:
        if sorted(p) != list(range(N)):
            print("FAIL: %s is not a permutation of the vertex set" % name)
            return 1
        for u, v in edges:
            x, y = p[u], p[v]
            if ((x, y) if x < y else (y, x)) not in seen:
                print("FAIL: %s does not preserve the edge (%d,%d)" % (name, u, v))
                return 1

    # ---- 3. the generated group acts transitively (orbit of 0 is everything)
    orbit = {0}
    stack = [0]
    while stack:
        x = stack.pop()
        for _, p in perms:
            y = p[x]
            if y not in orbit:
                orbit.add(y)
                stack.append(y)
    if len(orbit) != N:
        print("FAIL: the verified automorphisms act with orbit size %d < %d, "
              "so vertex-transitivity is not established" % (len(orbit), N))
        return 1

    # ---- 4. single BFS: eccentricity of vertex 0 == diameter
    dist = [-1] * N
    dist[0] = 0
    q = deque([0])
    reached = 1
    ecc = 0
    while q:
        x = q.popleft()
        d = dist[x] + 1
        for y in adj[x]:
            if dist[y] < 0:
                dist[y] = d
                ecc = d if d > ecc else ecc
                reached += 1
                q.append(y)
    if reached != N:
        print("FAIL: graph is disconnected (%d of %d reachable)" % (reached, N))
        return 1
    if ecc > D:
        print("FAIL: eccentricity %d > D=%d" % (ecc, D))
        return 1

    mb = 1 + delta * ((delta - 1) ** D - 1) // (delta - 2) if delta > 2 else 2 * D + 1
    print("PASS")
    print("  file            : %s" % os.path.abspath(a.edgelist))
    print("  vertices N      : %d" % N)
    print("  edges           : %d" % len(edges))
    print("  degree min/max  : %d / %d   (limit Delta=%d)" % (min(degs), max(degs), delta))
    print("  regular         : %s" % (min(degs) == max(degs)))
    print("  connected       : yes")
    print("  automorphisms   : 2 verified against the edge list, orbit of vertex 0 = all %d" % N)
    print("  => vertex-transitive, so every eccentricity equals ecc(0)")
    print("  diameter        : %d   (limit D=%d)" % (ecc, D))
    print("  distance profile: %s" % (sorted_profile(dist),))
    print("  Moore bound     : %d   (N/Moore = %.4f)" % (mb, N / mb))
    return 0


def sorted_profile(dist):
    from collections import Counter
    c = Counter(dist)
    return [c[k] for k in range(max(c) + 1)]


if __name__ == "__main__":
    sys.exit(main())
