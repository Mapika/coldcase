#!/usr/bin/env python3
"""Package a verified record candidate for submission.

Produces, next to the edge list:
  <name>.adjlist   one line per vertex: the sorted list of its neighbours
                   (the format the Comellas table uses for its downloads)
  <name>.txt       a plain-text description of the construction

and re-runs the standalone verifier so the printed output can be pasted alongside.

    python3 make_submission.py results/d14_D3_N1026_m342_n3_a49_c1.edges
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    ep = os.path.abspath(sys.argv[1])
    base = os.path.splitext(ep)[0]
    meta = json.load(open(base + ".json"))

    N = None
    edges = []
    for line in open(ep):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if p[0] == "N":
            N = int(p[1])
            continue
        edges.append((int(p[0]), int(p[1])))
    N = N or (max(max(e) for e in edges) + 1)

    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    with open(base + ".adjlist", "w") as f:
        for v in range(N):
            f.write(" ".join(map(str, sorted(adj[v]))) + "\n")

    with open(base + ".txt", "w") as f:
        f.write("Degree/diameter graph\n")
        f.write("=====================\n\n")
        f.write("Delta (max degree) : %d\n" % meta["delta"])
        f.write("D (diameter)       : %d\n" % meta["D"])
        f.write("N (order)          : %d\n" % meta["N"])
        f.write("edges              : %d\n" % meta["edges"])
        f.write("group              : %s\n" % meta["group"])
        f.write("construction       : %s\n" % meta["construction"])
        f.write("connection set S   : %s\n\n" % meta["S"])
        if meta.get("model") == "affine2":
            f.write("Element ((x,y),j) of (Z_%d x Z_%d) rtimes_A Z_%d is stored at index\n"
                    "(j*%d + x)*%d + y, and\n"
                    "  ((x1,y1),j1)*((x2,y2),j2) = ((x1,y1) + A^{j1}(x2,y2), j1+j2 mod %d).\n"
                    % (meta["p"], meta["p"], meta["n"], meta["p"], meta["p"], meta["n"]))
        else:
            f.write("Element (i,j) of Z_%d rtimes_%d Z_%d is stored at index j*%d + i, and\n"
                    "  (i1,j1)*(i2,j2) = (i1 + %d^{j1} i2 mod %d,  j1+j2 mod %d).\n"
                    % (meta["m"], meta["a"], meta["n"], meta["m"],
                       meta["a"], meta["m"], meta["n"]))
        f.write("The graph is Cay(G,S): vertex x is joined to x*s for every s in S.\n")
        f.write("S = S^{-1} and e not in S, so the graph is simple, undirected and\n")
        f.write("%d-regular; it is vertex-transitive, hence its diameter equals the\n"
                "eccentricity of the identity.\n\n" % meta["delta"])
        f.write("Files: .edges (edge list, 0-based), .adjlist (neighbour lists),\n")
        f.write("       .json (machine-readable construction data)\n")
        f.write("Verify with:  python3 verify_dd.py %s\n" % os.path.basename(ep))

    print("wrote %s.adjlist and %s.txt" % (base, base))
    # verify_dd.py is O(N*E); on large vertex-transitive graphs use verify_vt.py,
    # which proves vertex-transitivity from the edge list and then needs one BFS.
    v = "verify_dd.py" if N <= 5000 else "verify_vt.py"
    print("verifier: %s" % v)
    r = subprocess.run([sys.executable, os.path.join(HERE, v), ep],
                       capture_output=True, text=True)
    print(r.stdout)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
