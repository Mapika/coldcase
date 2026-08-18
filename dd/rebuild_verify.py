#!/usr/bin/env python3
"""Rebuild and verify the five submitted degree-diameter graphs from their
group data alone. Standalone: Python 3 stdlib only.

Each graph is the Cayley graph of the metacyclic group G = Z_m x|_a Z_n
(order N = m*n), element (i,j) stored at index j*m + i, with product
  (i1,j1)*(i2,j2) = (i1 + a^{j1} * i2 mod m, j1 + j2 mod n),
over an inverse-closed connection set S (element indices).

For each entry this script: rebuilds the edge set, checks S = S^{-1}, checks
Delta-regularity and simplicity, runs BFS from vertex 0, PROVES vertex-
transitivity (left translations by two generators of G are automorphisms and
act transitively), hence diameter = ecc(0), and reports it.

Run: python3 rebuild_verify.py            (verifies all five entries)
"""
from collections import deque

ENTRIES = [
    # (Delta, D, m, n, a, connection set)
    (14, 3, 344, 3, 337, [564, 916, 563, 965, 449, 703, 533, 715, 431, 897, 387, 989, 536, 912]),
    (9, 5, 163, 54, 141, [5394, 3462, 3227, 5781, 2350, 6535, 4859, 4179, 4429]),
    (11, 5, 873, 24, 785, [17925, 4053, 17295, 4479, 11902, 9880, 16477, 5825, 42, 831, 11115]),
    (13, 5, 1781, 24, 969, [4333, 39699, 34936, 8915, 24972, 17983, 24223, 20246, 33887, 9248, 42306, 2325, 22919]),
    (14, 5, 1517, 40, 142, [7759, 53494, 8035, 53133, 7305, 54929, 49995, 12979, 47806, 13776, 9202, 52936, 23616, 38540]),
]


def mul(x, y, m, n, apow):
    i1, j1 = x % m, x // m
    i2, j2 = y % m, y // m
    return ((j1 + j2) % n) * m + (i1 + apow[j1] * i2) % m


def inv(x, m, n, apow):
    i, j = x % m, x // m
    jj = (-j) % n
    return jj * m + (-apow[jj] * i) % m


def bfs_ecc(adj, s, N):
    dist = [-1] * N
    dist[s] = 0
    dq = deque([s])
    far = 0
    seen = 1
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                far = max(far, dist[v])
                seen += 1
                dq.append(v)
    return far, seen


def verify(Delta, D, m, n, a, S):
    N = m * n
    apow = [pow(a, j, m) for j in range(n)]
    assert (a ** n) % m == 1 % m or pow(a, n, m) == 1 % m, "a^n != 1 mod m"
    Sset = set(S)
    assert len(Sset) == len(S) == Delta, "connection set size/duplicates"
    assert all(inv(s, m, n, apow) in Sset for s in S), "S not inverse-closed"
    assert 0 not in Sset, "identity in S"
    adj = [[mul(x, s, m, n, apow) for s in S] for x in range(N)]
    for x in range(N):
        assert len(set(adj[x])) == Delta and x not in adj[x], "not simple/regular"
    # vertex-transitivity: left translation L_g(x)=g*x is always a graph
    # automorphism of a Cayley graph (edges are right multiplications); left
    # translations act transitively. Verify explicitly for two generators g:
    for g in (1 % N, m):  # (1,0) and (0,1) as elements, when nontrivial
        perm = [mul(g, x, m, n, apow) for x in range(N)]
        assert sorted(perm) == list(range(N)), "translation not a bijection"
        edge = set()
        for x in range(N):
            for v in adj[x]:
                edge.add((x, v))
        for x in range(N):
            for v in adj[x]:
                assert (perm[x], perm[v]) in edge, "translation breaks an edge"
    # orbit of 0 under the two translations must be everything
    orbit = {0}
    frontier = [0]
    gens = [1 % N, m]
    while frontier:
        nxt = []
        for x in frontier:
            for g in gens:
                y = mul(g, x, m, n, apow)
                if y not in orbit:
                    orbit.add(y)
                    nxt.append(y)
        frontier = nxt
    assert len(orbit) == N, "translations not transitive"
    ecc, seen = bfs_ecc(adj, 0, N)
    assert seen == N, "not connected"
    assert ecc <= D, f"diameter {ecc} > {D}"
    return N, ecc


def main():
    for Delta, D, m, n, a, S in ENTRIES:
        N, ecc = verify(Delta, D, m, n, a, S)
        print(f"({Delta},{D}): Z_{m} x|_{a} Z_{n}  ->  N = {N}, "
              f"{Delta}-regular, vertex-transitive, diameter = {ecc} <= {D}   OK")


if __name__ == "__main__":
    main()
