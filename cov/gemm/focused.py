#!/usr/bin/env python3
"""
focused.py -- the GEMM formulation of the *focused* covering-code move.

The CPU solver (cov/NOTES.md sec.6, cov/opt/METHODS.md) picks a random uncovered
word `u` and evaluates the moves that would cover it: every codeword `c` near
`u`, nudged one coordinate towards `u`.  That is `~R+1` (or with `--wide`,
`n(q-1)`) target positions per codeword.

Here the target positions are *every* word of the ball `B_R(u)` -- the complete
set of positions from which a codeword covers `u` -- for every candidate
codeword.  The neighbourhood is |B_R| x n_cand moves (8.7 million on K6(8,4)
with all 169 codewords) instead of ~10^3, and it is evaluated exactly, by two
dense integer GEMMs and nothing else.

Derivation
----------
Let cnt[w] be the number of codewords covering w, U = {w : cnt[w] = 0} and, for
a codeword c, S1(c) = {w : cnt[w] = 1 and c covers w} -- the words c alone
covers.  Moving c to x leaves uncovered exactly

    (|U| + |S1(c)|) - #{ w in U u S1(c) : d(w,x) <= R }.                   (*)

Build the column set  W = U ++ S1(c_1) ++ ... ++ S1(c_k)  (the S1 sets are
disjoint, because cnt[w]=1 means a unique owner) and the 0/1 ownership matrix

    O[j, i] = 1  iff  column j is in U, or column j is in S1(c_i).

Then with the indicator of ball membership

    Ind[x, j] = [ <phi(x), phi(w_j)> >= n-R ]        (GEMM 1, thresholded)

the count in (*) is exactly

    P[x, i] = (Ind @ O)[x, i]                        (GEMM 2)

and the objective to maximise is  P[x,i] - |S1(c_i)|, whose maximiser gives the
best move; the resulting uncovered count is |U| - max(P - |S1|).  Both products
are integer GEMMs on tensor cores; the only scatter left in the whole iteration
is the pair of |B_R|-sized `index_add_`s that update cnt after the commit.

Exactness: GEMM 1 has 0/1 inputs and inner dimension n*q with values <= n, so
fp16 is exact (covgemm.py).  GEMM 2 has 0/1 inputs and its values can reach |U|,
which need not be < 2048, so it is done in int8/int32 (`torch._int_mm`), which
is exact for any size.
"""
import argparse, itertools, json, math, os, sys, time
import numpy as np
import torch

from covgemm import Space, digits_of, index_of, onehot_t
from gemmsolve import read_code, write_code


# ------------------------------------------------------- ball pattern table
def ball_patterns(n, q, R):
    """All digit-difference patterns of Hamming weight <= R.  (|B_R|, n) int8."""
    blocks = [np.zeros((1, n), dtype=np.int8)]
    for r in range(1, R + 1):
        combs = list(itertools.combinations(range(n), r))
        vals = np.array(list(itertools.product(range(1, q), repeat=r)),
                        dtype=np.int8)
        blk = np.zeros((len(combs) * len(vals), n), dtype=np.int8)
        for k, cmb in enumerate(combs):
            blk[k * len(vals):(k + 1) * len(vals), list(cmb)] = vals
        blocks.append(blk)
    return np.concatenate(blocks, 0)


# ------------------------------------------------------------------ kernels
def _ind_raw(X, W, thr):
    """Ball-membership indicator, written straight out as int8 (1 byte/pair)."""
    return ((X @ W.t()) >= thr).to(torch.int8)


def _pow2(x, lo=256):
    """Next power of two >= max(x, lo).  Keeps the number of distinct GEMM
    shapes -- and therefore of torch.compile specialisations -- to a handful."""
    x = max(int(x), lo)
    return 1 << (x - 1).bit_length()


class Solver:
    def __init__(self, n, q, R, code, seed=1, device="cuda",
                 compile_=True, tile_bytes=256 << 20, cand=0):
        assert R < n
        self.n, self.q, self.R = n, q, R
        self.N, self.nq, self.thr = q ** n, n * q, n - R
        self.dev = device
        self.cand = cand
        self.tile_bytes = tile_bytes
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.ind = torch.compile(_ind_raw, dynamic=False,
                                 mode="max-autotune-no-cudagraphs") \
            if compile_ else _ind_raw
        patt = ball_patterns(n, q, R)
        self.BR = patt.shape[0]
        self.patt = torch.as_tensor(patt.astype(np.int64), device=device)
        self.pw = torch.tensor([q ** (n - 1 - j) for j in range(n)],
                               dtype=torch.int64, device=device)
        self.code = torch.as_tensor(np.asarray(code, dtype=np.int64),
                                    device=device)
        self.M = self.code.shape[0]
        self.cnt = torch.zeros(self.N, dtype=torch.int16, device=device)
        # Digit table for the whole space (N x n int8, 13 MB on K6(8,4)).  It
        # turns "digits of these word indices" -- needed three times per
        # iteration -- from n kernel launches into a single gather, which
        # matters because the loop is latency-bound, not flop-bound.
        self.dig_all = torch.empty((self.N, n), dtype=torch.uint8, device=device)
        t = torch.arange(self.N, device=device, dtype=torch.int64)
        for j in range(n - 1, -1, -1):
            self.dig_all[:, j] = (t % q).to(torch.uint8)
            t = torch.div(t, q, rounding_mode="floor")
        self.colofs = (torch.arange(n, device=device) * q)
        self.space = None
        self.stats = dict(pairs=0, moves_eval=0)
        self.reset_cnt()
        self.Aoh = self._oh(self.code)          # kept in step with self.code

    # ---------------------------------------------------------- primitives
    def _oh(self, dig, dtype=torch.float16):
        return onehot_t(dig, self.q, dtype, self.dev)

    def _digits(self, idx):
        return self.dig_all[idx].to(torch.int64)

    def _oh_idx(self, idx, dtype=torch.float16):
        """One-hot rows for word indices, straight from the digit table."""
        cols = self.dig_all[idx].to(torch.int64) + self.colofs
        oh = torch.zeros((idx.numel(), self.nq), dtype=dtype, device=self.dev)
        return oh.scatter_(1, cols, 1)

    def ball_idx(self, cdig):
        """Flat indices of B_R(c) for a single centre, by pure arithmetic."""
        return (((self.patt + cdig[None, :]) % self.q) * self.pw).sum(1)

    def tile_for(self, Cp):
        """Row-tile height: fixed for a given column bucket (so the compiled
        kernel is reused), and chosen to waste as few padded rows as possible."""
        nt = max(1, -(-self.BR * Cp // self.tile_bytes))
        return max(32, -(-(-(-self.BR // nt)) // 32) * 32)

    def reset_cnt(self):
        """cnt from scratch, by ball marking (index_add over M balls)."""
        self.cnt.zero_()
        one = torch.ones(self.BR, dtype=torch.int16, device=self.dev)
        for i in range(self.M):
            self.cnt.index_add_(0, self.ball_idx(self.code[i]), one)

    def uncovered(self):
        return int((self.cnt == 0).sum())

    # ------------------------------------------------------------- the move
    def step(self):
        n, q, R, thr = self.n, self.q, self.R, self.thr
        # {cnt<=1} in one pass: U0 = the uncovered words, U1 = the singly
        # covered ones.  Two nonzero() calls would cost two host syncs.
        low = torch.nonzero(self.cnt <= 1, as_tuple=False).squeeze(1)
        isz = self.cnt[low] == 0
        U0, U1 = low[isz], low[~isz]
        u_n = int(U0.numel())
        if u_n == 0:
            return dict(unc=0, done=True)
        # 1. a random uncovered word, and the ball of positions covering it
        j = int(torch.randint(0, u_n, (1,), generator=self.g, device=self.dev))
        udig = self._digits(U0[j:j + 1])[0]
        rdig = (self.patt + udig[None, :]) % q                 # (BR, n)
        A = self.Aoh

        # 2. candidate codewords: all of them, or the `cand` closest to u
        if self.cand and self.cand < self.M:
            agr = (A @ self._oh(udig[None, :]).t()).squeeze(1)  # = n - d(c,u)
            sel = torch.topk(agr, self.cand).indices
        else:
            sel = torch.arange(self.M, device=self.dev)
        k = int(sel.numel())

        # 3. columns: U, then the singly-covered words owned by a candidate
        if U1.numel():
            ohU1 = self._oh_idx(U1)
            own = ((ohU1 @ A[sel].t()) >= thr)                  # (|U1|, k)
            keep = own.any(1)
            U1 = U1[keep]
            own = own[keep].to(torch.int8)
        else:
            own = torch.zeros((0, k), dtype=torch.int8, device=self.dev)
        s_c = own.sum(0, dtype=torch.int32)                     # |S1(c_i)|
        cols = torch.cat([U0, U1])
        C = int(cols.numel())
        # Bucket the two free dimensions to powers of two and zero-pad.  A
        # zero column of W is the all-zero one-hot vector, whose agreement with
        # everything is 0 < thr, so it never enters an indicator; a zero column
        # of O contributes nothing to any P entry and is sliced off.
        Cp, kp = _pow2(C), _pow2(k, 32)
        Wc = self._oh_idx(cols)
        Wc = torch.cat([Wc, torch.zeros((Cp - C, self.nq), dtype=Wc.dtype,
                                        device=self.dev)], 0)
        O = torch.zeros((Cp, kp), dtype=torch.int8, device=self.dev)
        O[:u_n, :k] = 1
        if own.numel():
            O[u_n:C, :k] = own

        # 4. the two GEMMs, tiled over the rows of B_R(u) at a FIXED tile size
        tile = self.tile_for(Cp)
        best_val, best_row, best_col = None, 0, 0
        obj_off = s_c.to(torch.int32)
        for t0 in range(0, self.BR, tile):
            t1 = min(t0 + tile, self.BR)
            Xt = self._oh(rdig[t0:t1])
            if t1 - t0 != tile:                      # keep one fixed shape
                Xt = torch.cat([Xt, torch.zeros((tile - (t1 - t0), self.nq),
                                                dtype=Xt.dtype, device=self.dev)], 0)
            Ind = self.ind(Xt, Wc, thr)
            P = torch._int_mm(Ind, O)[:, :k]
            key = (P - obj_off).to(torch.float32)
            key += torch.rand(key.shape, generator=self.g, device=self.dev)
            if t1 - t0 != tile:
                key[t1 - t0:] = -1e30                # mask the padded rows
            v, fl = key.flatten().max(0)
            v = float(v)
            if best_val is None or v > best_val:
                best_val = v
                fl = int(fl)
                best_row, best_col = t0 + fl // k, fl % k
            self.stats["pairs"] += (t1 - t0) * C
            self.stats["moves_eval"] += (t1 - t0) * k

        # 5. commit
        ci = int(sel[best_col])
        newdig = rdig[best_row]
        old = self.ball_idx(self.code[ci])
        new = self.ball_idx(newdig)
        one = torch.ones(self.BR, dtype=torch.int16, device=self.dev)
        self.cnt.index_add_(0, old, -one)
        self.cnt.index_add_(0, new, one)
        self.code[ci] = newdig
        self.Aoh[ci] = 0
        self.Aoh[ci, newdig + self.colofs] = 1
        return dict(unc=self.uncovered(), done=False, c=ci, cols=C, k=k)

    def code_np(self):
        return self.code.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-q", type=int, required=True)
    p.add_argument("-n", type=int, required=True)
    p.add_argument("-R", type=int, required=True)
    p.add_argument("-M", type=int, default=0)
    p.add_argument("--in", dest="inp", default="")
    p.add_argument("--out", default="")
    p.add_argument("-s", "--seed", type=int, default=1)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--time", type=float, default=1e9)
    p.add_argument("--target", type=int, default=0)
    p.add_argument("--cand", type=int, default=0, help="0 = all codewords")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--tile-mb", type=int, default=256)
    p.add_argument("--warmup", type=int, default=0,
                   help="steps run before the clock starts, on a saved copy of "
                        "the state, purely to trigger torch.compile; the state "
                        "and the RNG are restored afterwards")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--every", type=int, default=25)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    rng = np.random.default_rng(a.seed)
    if a.inp:
        code = read_code(a.inp, a.q)
        M = a.M or code.shape[0]
        if code.shape[0] > M:
            code = code[:M]
        elif code.shape[0] < M:
            code = np.concatenate([code, rng.integers(0, a.q, (M - code.shape[0], a.n))])
    else:
        M = a.M
        code = rng.integers(0, a.q, (M, a.n))

    tb0 = time.perf_counter()
    S = Solver(a.n, a.q, a.R, code, seed=a.seed, compile_=not a.no_compile,
               tile_bytes=a.tile_mb << 20, cand=a.cand)
    torch.cuda.synchronize()
    tb = time.perf_counter() - tb0
    tw = 0.0
    if a.warmup:
        tw0 = time.perf_counter()
        save = (S.cnt.clone(), S.code.clone(), S.Aoh.clone(), S.g.get_state())
        for _ in range(a.warmup):
            S.step()
        S.cnt, S.code, S.Aoh = save[0], save[1], save[2]
        S.g.set_state(save[3])
        torch.cuda.synchronize()
        tw = time.perf_counter() - tw0
    unc = S.uncovered()
    best, best_code = unc, S.code_np().copy()
    if not a.quiet:
        print(f"# |B_R|={S.BR} build {tb:.2f}s start uncovered={unc}", flush=True)
    t0 = time.perf_counter(); ttt = 0.0 if (a.target and unc <= a.target) else None
    ttt_it = 0 if ttt == 0.0 else None
    it = 0
    dts = []
    while it < a.iters and time.perf_counter() - t0 < a.time:
        _t = time.perf_counter()
        r = S.step(); it += 1
        dts.append(time.perf_counter() - _t)
        unc = r["unc"]
        if a.selftest:
            ref = S.cnt.clone(); S.reset_cnt()
            assert torch.equal(S.Aoh, S._oh(S.code)), "Aoh drifted from code" 
            if not torch.equal(ref, S.cnt):
                print(f"SELFTEST FAIL at iter {it}: "
                      f"{int((ref!=S.cnt).sum())} counters differ"); sys.exit(3)
        if unc < best:
            best, best_code = unc, S.code_np().copy()
        if a.target and ttt is None and best <= a.target:
            ttt = time.perf_counter() - t0
            ttt_it = it
        if unc == 0:
            break
        if not a.quiet and it % a.every == 0:
            el = time.perf_counter() - t0
            print(f"  it {it:6d} unc {unc:7d} best {best:7d} cols {r.get('cols',0):6d}"
                  f" {el:8.2f}s {it/el:8.2f} it/s", flush=True)
    el = time.perf_counter() - t0
    if a.out:
        write_code(a.out, best_code, a.q)
    print(f"RESULT q={a.q} n={a.n} R={a.R} M={M} uncovered={best} iters={it} "
          f"time={el:.3f} build={tb:.3f} rate={it/max(el,1e-9):.3f} "
          f"target={a.target} ttt_wall={'' if ttt is None else f'{ttt:.4f}'} "
          f"ttt_it={'' if ttt_it is None else ttt_it} warmup={tw:.3f} "
          f"ms_med={1e3*float(np.median(dts)) if dts else 0:.3f} "
          f"pairs={S.stats['pairs']} moves_eval={S.stats['moves_eval']}")


if __name__ == "__main__":
    main()
