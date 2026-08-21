#!/usr/bin/env python3
"""Throughput of the one kernel the whole GEMM solver is made of.

The kernel is: given the one-hot matrix X of (a chunk of) Z_q^n and a set W of
`B` witness words, produce for every x in the chunk

    score[x] = #{ w in W : d(x,w) <= R }   =   #{ w : <phi(x),phi(w)> >= n-R }.

That is one GEMM (K = n*q, small) followed by a threshold and a row reduction.
The reduction is the whole story: without it the GEMM writes N*B values, and
N*B * (2 or 4) bytes of HBM traffic at ~4 TB/s is what sets the rate, not the
2*N*B*nq tensor-core operations.  We measure the eager torch path (which does
materialise the N x B intermediate) and, where it compiles, a torch.compile
version that can fuse the epilogue into the matmul template.

Usage:  python3 bench_kernel.py [--quick]
"""
import argparse, json, os, sys, time
import torch
from covgemm import Space, int_mm, onehot_t, space_digits

CELLS = {
    "K6(8,4)":  dict(n=8, q=6, R=4),
    "K6(9,5)":  dict(n=9, q=6, R=5),
    "K8(9,4)":  dict(n=9, q=8, R=4),
}


def make_W(n, q, B, dtype, dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    dig = torch.randint(0, q, (B, n), device=dev, generator=g, dtype=torch.int64)
    return onehot_t(dig, q, dtype, dev)


def run(space, W, thr, mode, tile_bytes=512 << 20, out=None):
    """One full pass over Z_q^n; returns the score vector.

    The N x B agreement matrix is never materialised whole: rows are processed
    in tiles small enough that the intermediate stays under `tile_bytes`.
    """
    N, B = space.N, W.shape[0]
    ob = 4 if mode == "int8" else 2
    tile = max(1024, min(N, tile_bytes // max(1, B * ob)))
    if out is None:
        out = torch.empty(N, dtype=torch.int32, device=space.device)
    for lo, hi, X in space.chunks():
        for t0 in range(0, hi - lo, tile):
            t1 = min(t0 + tile, hi - lo)
            Xt = X[t0:t1]
            G = int_mm(Xt, W.t()) if mode == "int8" else Xt @ W.t()
            out[lo + t0:lo + t1] = (G >= thr).sum(1, dtype=torch.int32)
    return out


def timeit(fn, warmup=1, iters=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="K6(8,4)")
    ap.add_argument("--batches", default="16,64,256,1024,4096")
    ap.add_argument("--chunk", type=int, default=1 << 21)
    ap.add_argument("--cache-gb", type=float, default=4.0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    rows = []
    for cname in [s.strip() for s in a.cells.split(";")]:
        c = CELLS[cname]
        n, q, R = c["n"], c["q"], c["R"]
        for mode, dt in (("fp16", torch.float16), ("int8", torch.int8)):
            sp = Space(n, q, "cuda", dt, chunk=a.chunk,
                       cache_bytes=int(a.cache_gb * (1 << 30)))
            for B in [int(x) for x in a.batches.split(",")]:
                W = make_W(n, q, B, dt, "cuda")
                t = timeit(lambda: run(sp, W, n - R, mode))
                pairs = sp.N * B
                r = dict(cell=cname, mode=mode, N=sp.N, nq=n * q, B=B,
                         cached=sp.cached, ms=t * 1e3,
                         pairs_per_s=pairs / t,
                         tops=2 * pairs * n * q / t / 1e12,
                         out_gbps=pairs * (4 if mode == "int8" else 2) / t / 1e9)
                rows.append(r)
                print(f"{cname:9s} {mode:5s} N={sp.N:>9d} B={B:>5d} "
                      f"{t*1e3:9.3f} ms  {r['pairs_per_s']/1e9:7.2f} Gpair/s "
                      f"{r['tops']:6.1f} TOPS  out {r['out_gbps']:6.0f} GB/s",
                      flush=True)
            del sp
            torch.cuda.empty_cache()
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
