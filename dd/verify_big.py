#!/usr/bin/env python3
"""Exhaustive all-pairs diameter check for large edge lists (numpy only).

`verify_dd.py` is the pure-stdlib reference verifier; its BFS-from-every-vertex is
O(N*E) and takes hours once N reaches tens of thousands.  This does the *same*
exhaustive check -- every ordered pair of vertices -- with bit-parallel BFS:

  * process the sources 64 at a time; vertex v carries a uint64 whose bit i is set
    once source i of the current block has reached v;
  * one BFS level is  R = R | OR over the neighbours of v of R[u],  which for a
    regular graph is a single gather `R[adj]` (N x Delta uint64) and a reduce;
  * after D levels every vertex must have every bit of the block set.

That is 64 sources per pass instead of one, with the inner loop in numpy C, so the
whole all-pairs check costs seconds rather than hours.  The diameter reported is
exact: it is the first level at which the all-bits-set condition holds for every
block.

    python3 verify_big.py FILE.edges [--delta 14] [--D 5] [--N 60452]
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def read(path):
    N = None
    E = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if p[0] == "N":
            N = int(p[1])
            continue
        E.append((int(p[0]), int(p[1])))
    if N is None:
        N = max(max(e) for e in E) + 1
    return N, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edgelist")
    ap.add_argument("--delta", type=int)
    ap.add_argument("--D", type=int)
    ap.add_argument("--N", type=int)
    a = ap.parse_args()

    side = os.path.splitext(a.edgelist)[0] + ".json"
    if os.path.exists(side):
        meta = json.load(open(side))
        a.delta = a.delta or meta.get("delta")
        a.D = a.D or meta.get("D")
        a.N = a.N or meta.get("N")
    if a.delta is None or a.D is None:
        print("FAIL: need --delta and --D")
        return 1

    N, E = read(a.edgelist)
    if a.N is not None and N != a.N:
        print("FAIL: vertex count %d != claimed %d" % (N, a.N))
        return 1

    u = np.fromiter((e[0] for e in E), dtype=np.int64, count=len(E))
    v = np.fromiter((e[1] for e in E), dtype=np.int64, count=len(E))
    if (u == v).any():
        print("FAIL: self-loop")
        return 1
    lo = np.minimum(u, v)
    hi = np.maximum(u, v)
    key = lo * N + hi
    if len(np.unique(key)) != len(key):
        print("FAIL: repeated edge")
        return 1
    if u.max() >= N or v.max() >= N or u.min() < 0 or v.min() < 0:
        print("FAIL: vertex index out of range")
        return 1

    # degrees + fixed-width adjacency table (padded with self so the OR is a no-op)
    deg = np.bincount(np.concatenate([u, v]), minlength=N)
    if deg.max() > a.delta:
        print("FAIL: max degree %d > Delta=%d" % (deg.max(), a.delta))
        return 1
    K = int(deg.max())
    adj = np.repeat(np.arange(N, dtype=np.int64)[:, None], K, axis=1)
    # vectorised fill: stable sort of the doubled edge list by source
    src = np.concatenate([u, v])
    dst = np.concatenate([v, u])
    order = np.argsort(src, kind="stable")
    src = src[order]
    dst = dst[order]
    starts = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(deg, out=starts[1:])
    pos = np.arange(len(src), dtype=np.int64) - starts[src]
    adj[src, pos] = dst

    t0 = time.time()
    nblocks = (N + 63) // 64
    diam = 0
    for b in range(nblocks):
        base = b * 64
        width = min(64, N - base)
        mask = np.uint64((1 << width) - 1) if width < 64 else np.uint64(0xFFFFFFFFFFFFFFFF)
        R = np.zeros(N, dtype=np.uint64)
        R[base:base + width] = (np.uint64(1) << np.arange(width, dtype=np.uint64))
        for lvl in range(1, a.D + 1):
            R = R | np.bitwise_or.reduce(R[adj], axis=1)
            if (R == mask).all():
                diam = max(diam, lvl)
                break
        else:
            bad = int(np.argmin(R == mask))
            print("FAIL: vertex %d is not within distance %d of every source in block %d"
                  % (bad, a.D, b))
            return 1
    dt = time.time() - t0

    mb = 1 + a.delta * ((a.delta - 1) ** a.D - 1) // (a.delta - 2) if a.delta > 2 else 2 * a.D + 1
    print("PASS")
    print("  file            : %s" % os.path.abspath(a.edgelist))
    print("  vertices N      : %d" % N)
    print("  edges           : %d" % len(E))
    print("  degree min/max  : %d / %d   (limit Delta=%d)" % (deg.min(), deg.max(), a.delta))
    print("  regular         : %s" % (deg.min() == deg.max()))
    print("  connected       : yes (every pair is within distance %d)" % diam)
    print("  diameter        : %d   (limit D=%d)   exhaustive all-pairs, %.1f s, %d source blocks"
          % (diam, a.D, dt, nblocks))
    print("  Moore bound     : %d   (N/Moore = %.4f)" % (mb, N / mb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
