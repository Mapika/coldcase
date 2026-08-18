#!/usr/bin/env python3
"""verify_dd.py -- standalone exact verifier for degree/diameter graph claims.

Pure Python, standard library only.  Reads an edge list and checks:
  * vertex count           N
  * simplicity             (no loops, no repeated edges)
  * maximum degree         <= Delta
  * connectivity
  * diameter               <= D    (exhaustive BFS from EVERY vertex)

Usage:
    python3 verify_dd.py EDGELIST.txt --delta 16 --D 3 --N 1611
    python3 verify_dd.py EDGELIST.txt            # reads params from the .json sidecar

Edge list format: one edge per line, "u v" (0-based ints).  Blank lines and lines
starting with '#' are ignored.  An optional first non-comment line "N <count>" fixes
the vertex count (needed only if some vertex is isolated).

Exit status 0 = PASS, 1 = FAIL.
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
            parts = line.split()
            if parts[0].upper() == "N" and len(parts) == 2:
                n_decl = int(parts[1])
                continue
            if len(parts) != 2:
                raise ValueError("bad edge line: %r" % line)
            edges.append((int(parts[0]), int(parts[1])))
    return n_decl, edges


def build(n, edges):
    adj = [[] for _ in range(n)]
    seen = set()
    for u, v in edges:
        if u == v:
            raise ValueError("self-loop at %d" % u)
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError("vertex out of range in edge (%d,%d), N=%d" % (u, v, n))
        key = (u, v) if u < v else (v, u)
        if key in seen:
            raise ValueError("duplicate edge %s" % (key,))
        seen.add(key)
        adj[u].append(v)
        adj[v].append(u)
    return adj


def ecc(adj, src, n):
    """Eccentricity of src; returns (ecc, n_reached)."""
    dist = [-1] * n
    dist[src] = 0
    q = deque([src])
    far = 0
    reached = 1
    while q:
        x = q.popleft()
        dx = dist[x] + 1
        for y in adj[x]:
            if dist[y] < 0:
                dist[y] = dx
                if dx > far:
                    far = dx
                reached += 1
                q.append(y)
    return far, reached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edgelist")
    ap.add_argument("--delta", type=int, default=None)
    ap.add_argument("--D", type=int, default=None)
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    delta, D, N = a.delta, a.D, a.N
    side = os.path.splitext(a.edgelist)[0] + ".json"
    if os.path.exists(side):
        meta = json.load(open(side))
        if delta is None:
            delta = meta.get("delta")
        if D is None:
            D = meta.get("D")
        if N is None:
            N = meta.get("N")
    if delta is None or D is None:
        print("FAIL: need --delta and --D (no sidecar found)")
        return 1

    n_decl, edges = read_edges(a.edgelist)
    maxv = max(max(u, v) for u, v in edges) + 1 if edges else 0
    n = n_decl if n_decl is not None else (N if N is not None else maxv)
    if n < maxv:
        print("FAIL: declared N=%d < max vertex index+1 =%d" % (n, maxv))
        return 1

    try:
        adj = build(n, edges)
    except ValueError as e:
        print("FAIL: %s" % e)
        return 1

    if N is not None and n != N:
        print("FAIL: vertex count %d != claimed %d" % (n, N))
        return 1

    degs = [len(x) for x in adj]
    dmax = max(degs)
    if dmax > delta:
        print("FAIL: max degree %d > Delta=%d" % (dmax, delta))
        return 1

    diam = 0
    argmax = -1
    for s in range(n):
        e, reached = ecc(adj, s, n)
        if reached != n:
            print("FAIL: graph is disconnected (vertex %d reaches %d of %d)" % (s, reached, n))
            return 1
        if e > diam:
            diam, argmax = e, s
        if e > D:
            print("FAIL: eccentricity of vertex %d is %d > D=%d" % (s, e, D))
            return 1

    if not a.quiet:
        print("PASS")
        print("  file            : %s" % os.path.abspath(a.edgelist))
        print("  vertices N      : %d" % n)
        print("  edges           : %d" % len(edges))
        print("  degree min/max  : %d / %d   (limit Delta=%d)" % (min(degs), dmax, delta))
        print("  regular         : %s" % (min(degs) == dmax))
        print("  connected       : yes")
        print("  diameter        : %d   (limit D=%d)   worst-case source %d" % (diam, D, argmax))
        mb = 1 + delta * ((delta - 1) ** D - 1) // (delta - 2) if delta > 2 else 2 * D + 1
        print("  Moore bound     : %d   (N/Moore = %.4f)" % (mb, n / mb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
