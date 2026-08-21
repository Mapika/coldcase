#!/usr/bin/env python3
"""
gemmsolve.py -- a covering-code local search whose entire inner loop is one
dense GEMM on tensor cores.  No scattered counter updates anywhere.

The move: RELOCATE.  Pick a codeword c and move it to the best position in the
*whole* space Z_q^n.  Formally, with cnt[w] = #codewords covering w,

    U_{-c} = { w : cnt[w] - [d(w,c) <= R] = 0 }        (uncovered without c)
    score(x) = #{ w in U_{-c} : d(w,x) <= R }
    uncovered after moving c to x  =  |U_{-c}| - score(x).

`score` for *every* x in Z_q^n at once is exactly one thresholded matrix
product between the one-hot matrix of the whole space and the one-hot matrix of
the |U_{-c}| witnesses (see covgemm.py for the identity and the exactness
argument).  So one GEMM evaluates a neighbourhood of q^n moves exactly, and the
whole state update is dense vector arithmetic over a q^n array:

    cnt <- cnt - [d(.,c) <= R] + [d(.,x) <= R]

which is a second (2-column) pass of the same kernel.  Nothing in the loop
gathers, scatters or branches per word, which is what killed the first CUDA
port (NOTES.md sec. 7).

Correctness discipline
----------------------
`--selftest` recomputes cnt from scratch by an independent full GEMM after
every move and aborts on any disagreement with the incrementally maintained
array; the emitted code file is meant to be handed to cov/verify_cov.py, and
run_bench.py does exactly that.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

from covgemm import Space, digits_of, index_of, onehot_t, space_digits

# ----------------------------------------------------------------- kernels
# One fused kernel does all the work.  It is compiled once per column-count
# bucket; witness sets are zero-padded up to the bucket size, and a zero column
# is the all-zero one-hot vector, whose agreement with everything is 0 < thr,
# so padding can never be counted.  (thr = n - R >= 1 is asserted at start-up.)

def _score_raw(X, W, thr):
    return ((X @ W.t()) >= thr).sum(1, dtype=torch.int32)


def _delta_raw(X, W2, thr, sign):
    return (((X @ W2.t()) >= thr).to(torch.int32) * sign).sum(1, dtype=torch.int32)


class Kernels:
    def __init__(self, compile_=True):
        self.score = torch.compile(_score_raw, dynamic=False,
                                   mode="max-autotune-no-cudagraphs") \
            if compile_ else _score_raw
        self.delta = torch.compile(_delta_raw, dynamic=False,
                                   mode="max-autotune-no-cudagraphs") \
            if compile_ else _delta_raw


def bucket(b, lo=32):
    """Round a witness count up to a power of two >= lo (limits recompiles)."""
    return max(lo, 1 << (b - 1).bit_length())


# ------------------------------------------------------------------ solver

class GemmSolver:
    def __init__(self, n, q, R, M, device="cuda", dtype=torch.float16,
                 seed=1, chunk=1 << 21, cache_gb=6.0, col_cap=2048,
                 compile_=True, tile_bytes=512 << 20):
        assert R < n, "thr = n-R must be >= 1 for zero-padding to be inert"
        self.n, self.q, self.R, self.M = n, q, R, M
        self.N = q ** n
        self.nq = n * q
        self.thr = n - R
        self.dev, self.dtype = device, dtype
        self.col_cap, self.tile_bytes = col_cap, tile_bytes
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.space = Space(n, q, device, dtype, chunk=chunk,
                           cache_bytes=int(cache_gb * (1 << 30)))
        self.K = Kernels(compile_)
        self.cnt = torch.zeros(self.N, dtype=torch.int16, device=device)
        self.code = torch.zeros((M, n), dtype=torch.int64, device=device)
        self.pw = torch.tensor([q ** (n - 1 - j) for j in range(n)],
                               dtype=torch.int64, device=device)
        self.sign = torch.tensor([-1, 1], dtype=torch.int32, device=device)
        self.stats = dict(gemm_cols=0, iters=0)

    # -- helpers ---------------------------------------------------------
    def _oh(self, dig):
        return onehot_t(dig, self.q, self.dtype, self.dev)

    def _pad(self, W, nb):
        if W.shape[0] == nb:
            return W
        return torch.cat([W, torch.zeros((nb - W.shape[0], self.nq),
                                         dtype=W.dtype, device=W.device)], 0)

    def _score_all(self, W):
        """score[x] for every x in Z_q^n, accumulating over column blocks."""
        out = torch.zeros(self.N, dtype=torch.int32, device=self.dev)
        B = W.shape[0]
        for c0 in range(0, B, self.col_cap):
            Wc = self._pad(W[c0:c0 + self.col_cap], bucket(min(self.col_cap, B - c0)))
            self.stats["gemm_cols"] += Wc.shape[0]
            for lo, hi, X in self._tiles(Wc.shape[0]):
                out[lo:hi] += self.K.score(X, Wc, self.thr)
        return out

    def _tiles(self, ncol):
        """Row tiles of the space small enough for the intermediate."""
        it = 2 if self.dtype != torch.int8 else 4
        tile = max(1 << 15, min(self.N, self.tile_bytes // max(1, ncol * it)))
        for lo, hi, X in self.space.chunks():
            for t0 in range(0, hi - lo, tile):
                t1 = min(t0 + tile, hi - lo)
                yield lo + t0, lo + t1, X[t0:t1]

    def ball(self, dig_rows, sign=None):
        """Signed sum of ball indicators of the given codewords, over Z_q^n."""
        W = self._pad(self._oh(dig_rows), 32)
        s = torch.zeros(32, dtype=torch.int32, device=self.dev)
        s[:dig_rows.shape[0]] = sign if sign is not None else 1
        out = torch.empty(self.N, dtype=torch.int32, device=self.dev)
        for lo, hi, X in self._tiles(32):
            out[lo:hi] = self.K.delta(X, W, self.thr, s)
        return out

    # -- state -----------------------------------------------------------
    def recount(self, code=None):
        """cnt from scratch: one GEMM of the whole space against the code."""
        code = self.code if code is None else code
        cnt = torch.zeros(self.N, dtype=torch.int32, device=self.dev)
        A = self._oh(code)
        for c0 in range(0, code.shape[0], self.col_cap):
            Ac = self._pad(A[c0:c0 + self.col_cap],
                           bucket(min(self.col_cap, code.shape[0] - c0)))
            for lo, hi, X in self._tiles(Ac.shape[0]):
                cnt[lo:hi] += self.K.score(X, Ac, self.thr)
        return cnt

    def set_code(self, code_np):
        self.code = torch.as_tensor(np.asarray(code_np, dtype=np.int64),
                                    device=self.dev)
        self.M = self.code.shape[0]
        self.cnt = self.recount().to(torch.int16)

    def uncovered(self):
        return int((self.cnt == 0).sum())

    def code_np(self):
        return self.code.cpu().numpy()

    # -- the move --------------------------------------------------------
    def step(self, c=None, allow_stay=False):
        n, q, R = self.n, self.q, self.R
        if c is None:
            c = int(torch.randint(0, self.M, (1,), generator=self.g,
                                  device=self.dev).item())
        # U_{-c}: the uncovered words plus the words singly covered by c.
        # {cnt==1} is small, so testing d(w,c)<=R on it is a tiny GEMV; the
        # full-space ball indicator of c is never needed here.
        u0 = torch.nonzero(self.cnt == 0, as_tuple=False).squeeze(1)
        u1 = torch.nonzero(self.cnt == 1, as_tuple=False).squeeze(1)
        cdig = self.code[c:c + 1]
        if u1.numel():
            d1 = self._digits(u1)
            ag = self._oh(d1) @ self._oh(cdig).t()
            keep = u1[(ag[:, 0] >= self.thr)]
        else:
            keep = u1
        Um = torch.cat([u0, keep]) if keep.numel() else u0
        m = int(Um.numel())
        if m == 0:
            return dict(done=True, unc=0, c=c, m=0)
        W = self._oh(self._digits(Um))
        score = self._score_all(W)
        newunc = (m - score).to(torch.float32)
        if not allow_stay:
            cur = int((self.code[c] * self.pw).sum())
            newunc[cur] = float(self.N)          # forbid the no-op
        # uniform random choice among the argmin, without a host round trip
        key = newunc + torch.rand(self.N, generator=self.g, device=self.dev)
        x = int(torch.argmin(key).item())
        best = int(newunc[x].item())
        xdig = self._digits(torch.tensor([x], device=self.dev))
        pair = torch.cat([self.code[c:c + 1], xdig], 0)
        self.cnt += self.ball(pair, self.sign).to(torch.int16)
        self.code[c] = xdig[0]
        self.stats["iters"] += 1
        return dict(done=False, unc=best, c=c, x=x, m=m)

    def _digits(self, idx):
        dig = torch.empty((idx.numel(), self.n), dtype=torch.int64,
                          device=self.dev)
        t = idx.clone()
        for j in range(self.n - 1, -1, -1):
            dig[:, j] = t % self.q
            t = torch.div(t, self.q, rounding_mode="floor")
        return dig


# ------------------------------------------------------------------- I/O

def read_code(path, q):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if any(ch in line for ch in " ,\t"):
            rows.append([int(t) for t in line.replace(",", " ").split()])
        else:
            rows.append([int(ch, 36) for ch in line])
    a = np.array(rows, dtype=np.int64)
    assert a.min() >= 0 and a.max() < q, "digit out of range"
    return a


def write_code(path, code, q):
    with open(path, "w") as f:
        if q <= 36:
            for row in code:
                f.write("".join(np.base_repr(int(d), 36).lower() for d in row) + "\n")
        else:
            for row in code:
                f.write(" ".join(str(int(d)) for d in row) + "\n")


# ------------------------------------------------------------------- main

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
    p.add_argument("--time", type=float, default=1e9, help="wall-clock budget, s")
    p.add_argument("--target", type=int, default=0)
    p.add_argument("--allow-stay", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--cache-gb", type=float, default=6.0)
    p.add_argument("--col-cap", type=int, default=2048)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "int8"])
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--every", type=int, default=50)
    a = p.parse_args()

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    dt = torch.float16 if a.dtype == "fp16" else torch.int8
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

    t_build0 = time.perf_counter()
    S = GemmSolver(a.n, a.q, a.R, M, dtype=dt, seed=a.seed,
                   cache_gb=a.cache_gb, col_cap=a.col_cap,
                   compile_=not a.no_compile)
    S.set_code(code)
    torch.cuda.synchronize()
    t_build = time.perf_counter() - t_build0

    unc = S.uncovered()
    best, best_code = unc, S.code_np().copy()
    t0 = time.perf_counter()
    ttt = None
    it = 0
    if not a.quiet:
        print(f"# build {t_build:.2f}s  space {S.space.bytes_resident()/2**20:.0f} MiB"
              f"  cached={S.space.cached}  start uncovered={unc}", flush=True)
    if a.target and unc <= a.target:
        ttt = 0.0
    while it < a.iters and (time.perf_counter() - t0) < a.time:
        r = S.step(allow_stay=a.allow_stay)
        it += 1
        if r["done"]:
            unc = 0
        else:
            unc = r["unc"]
        if a.selftest:
            ref = S.recount().to(torch.int16)
            if not torch.equal(ref, S.cnt):
                bad = int((ref != S.cnt).sum())
                print(f"SELFTEST FAIL at iter {it}: {bad} counters differ")
                sys.exit(3)
            if int((ref == 0).sum()) != unc:
                print(f"SELFTEST FAIL at iter {it}: uncovered {unc} vs "
                      f"{int((ref==0).sum())}")
                sys.exit(3)
        if unc < best:
            best, best_code = unc, S.code_np().copy()
        if a.target and ttt is None and best <= a.target:
            ttt = time.perf_counter() - t0
        if unc == 0:
            break
        if not a.quiet and it % a.every == 0:
            el = time.perf_counter() - t0
            print(f"  it {it:6d}  unc {unc:7d}  best {best:7d}  "
                  f"{el:8.2f}s  {it/el:7.2f} it/s", flush=True)
    el = time.perf_counter() - t0
    if a.out:
        write_code(a.out, best_code, a.q)
    print(f"RESULT q={a.q} n={a.n} R={a.R} M={M} uncovered={best} iters={it} "
          f"time={el:.3f} build={t_build:.3f} rate={it/max(el,1e-9):.3f} "
          f"target={a.target} ttt_wall={'' if ttt is None else f'{ttt:.4f}'} "
          f"gemm_cols={S.stats['gemm_cols']}")


if __name__ == "__main__":
    main()
