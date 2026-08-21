#!/usr/bin/env python3
"""Correctness gates for the distance-count transform engine.

Gate (i): transform vs brute-force N_d on random small grids, exhaustive.
Gate (ii): gain/loss maps + ball ops vs direct CPU counts on a mid cell.
Refuses to run if another compute process owns the GPU.
"""
import subprocess
import sys

import numpy as np

from dctlib import DCT


def gpu_busy():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                          "--format=csv,noheader"], capture_output=True,
                         text=True)
    pids = [p.split(",")[0].strip() for p in out.stdout.strip().splitlines()]
    import os
    live = [p for p in pids if p and os.path.exists(f"/proc/{p}")]
    return bool(live)


def brute_nd(q, n, R, S_idx, space):
    """CPU brute force: N_d(x) for all x, d=0..R, from scratch."""
    digits = np.empty((space, n), dtype=np.int8)
    x = np.arange(space)
    for i in range(n):
        digits[:, i] = x % q
        x = x // q
    sd = digits[S_idx]                       # (|S|, n)
    Nd = np.zeros((R + 1, space), dtype=np.int32)
    for u in sd:
        d = (digits != u[None, :]).sum(axis=1)
        for dd in range(R + 1):
            Nd[dd] += (d == dd)
    return Nd


def gate1():
    rng = np.random.default_rng(7)
    cases = [(3, 4, 2), (3, 5, 3), (4, 4, 3), (2, 6, 2), (5, 4, 2), (6, 3, 2),
             (4, 6, 4), (3, 6, 5)]
    for (q, n, R) in cases:
        space = q ** n
        for rep in range(3):
            k = int(rng.integers(1, max(2, space // 7)))
            S_idx = rng.choice(space, size=k, replace=False).astype(np.int64)
            # include multiplicities sometimes
            if rep == 2 and k > 2:
                S_idx = np.concatenate([S_idx, S_idx[:3]])
            eng = DCT(q, n, R, extract_cap=1 << 16)
            try:
                eng.recount(S_idx)          # code multiset -> cnt map
                cnt = eng.read_cnt().astype(np.int64)
                Nd = brute_nd(q, n, R, S_idx, space)
                want_cnt = Nd.sum(axis=0)
                assert (cnt == np.minimum(want_cnt, 65535)).all(), \
                    f"cnt mismatch q={q} n={n} R={R} rep={rep}"
                # gain map: S = uncovered set
                eng.gain_map()
                gain = eng.map_read_at(np.arange(space))
                unc = (want_cnt == 0)
                NdU = brute_nd(q, n, R, np.flatnonzero(unc), space)
                assert (gain == NdU.sum(axis=0)).all(), \
                    f"gain mismatch q={q} n={n} R={R} rep={rep}"
                # loss map: S = [cnt==1]
                eng.loss_map()
                loss = eng.map_read_at(np.arange(space))
                NdP = brute_nd(q, n, R, np.flatnonzero(want_cnt == 1), space)
                assert (loss == NdP.sum(axis=0)).all(), \
                    f"loss mismatch q={q} n={n} R={R} rep={rep}"
                # ball ops
                probe = rng.choice(space, size=min(20, space), replace=False)
                g = eng.ball_gather(probe.astype(np.int64), 0)
                assert (g == NdU.sum(axis=0)[probe]).all(), "ball_gather(0)"
                w = int(rng.integers(space))
                eng.ball_update(np.array([w], dtype=np.int64), +1)
                cnt2 = eng.read_cnt().astype(np.int64)
                Nd1 = brute_nd(q, n, R, np.array([w]), space)
                assert (cnt2 == want_cnt + Nd1.sum(axis=0)).all(), "ball_update"
                assert eng.uncovered() == int((cnt2 == 0).sum()), "count_eq"
            finally:
                eng.close()
        print(f"gate1 ok q={q} n={n} R={R}")


def gate2():
    """Mid cell K6(6,3): transform maps vs independent CPU ball counting."""
    q, n, R = 6, 6, 3
    space = q ** n
    rng = np.random.default_rng(11)
    M = 41
    code = rng.integers(0, q, size=(M, n)).astype(np.int64)
    eng = DCT(q, n, R, extract_cap=1 << 20)
    try:
        idx = eng.word_index(code)
        eng.recount(idx)
        cnt = eng.read_cnt().astype(np.int64)
        # independent CPU cnt by explicit ball enumeration per codeword
        digits = np.empty((space, n), dtype=np.int8)
        x = np.arange(space)
        for i in range(n):
            digits[:, i] = x % q
            x = x // q
        cpu_cnt = np.zeros(space, dtype=np.int64)
        for c in code:
            d = (digits != c[None, :]).sum(axis=1)
            cpu_cnt += (d <= R)
        assert (cnt == cpu_cnt).all(), "K6(6,3) cnt map"
        eng.gain_map()
        probes = rng.choice(space, size=200, replace=False)
        gains = eng.map_read_at(probes.astype(np.int64))
        unc = cpu_cnt == 0
        for p, g in zip(probes, gains):
            d = (digits != digits[p][None, :]).sum(axis=1)
            assert g == int((unc & (d <= R)).sum()), "gain probe"
        # extraction consistency
        mx = eng.map_max()
        oi, ov, found = eng.map_extract(mx)
        full = eng.map_read_at(np.arange(space))
        assert mx == full.max() and found == int((full >= mx).sum())
        assert set(oi.tolist()) <= set(np.flatnonzero(full >= mx).tolist())
        print(f"gate2 ok K6(6,3): cnt exact, 200 gain probes exact, "
              f"max={mx} extract={found}")
    finally:
        eng.close()


if __name__ == "__main__":
    if gpu_busy():
        print("YIELD: GPU busy")
        sys.exit(2)
    gate1()
    gate2()
    print("ALL GATES PASS")
