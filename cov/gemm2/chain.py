#!/usr/bin/env python3
"""ctypes driver + validation for the chain-batched GPU solver."""
import ctypes, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MAXN = 16


def load():
    lib = ctypes.CDLL(os.path.join(HERE, "libchain.so"))
    lib.chain_run.restype = ctypes.c_int
    return lib


def read_code(path, n):
    words = []
    for line in open(path):
        line = line.strip()
        if line:
            words.append([int(c) for c in line])
    return np.array(words, dtype=np.uint8)


def random_code(q, n, M, rng):
    return rng.integers(0, q, size=(M, n), dtype=np.uint8)


def run(q, n, R, code, nchains=16, iters=20000, seed=1, threads=256,
        perturb=0, rng=None):
    """code: (M,n) uint8 start; each chain starts from it (optionally with
    `perturb` random codeword replacements). Returns dict."""
    lib = load()
    M = code.shape[0]
    rng = rng or np.random.default_rng(seed)
    codes = np.zeros((nchains, M, MAXN), dtype=np.uint8)
    for c in range(nchains):
        cc = code.copy()
        for _ in range(perturb):
            cc[rng.integers(0, M)] = rng.integers(0, q, size=n, dtype=np.uint8)
        codes[c, :, :n] = cc
    best = np.zeros_like(codes)
    bu = np.zeros(nchains, dtype=np.int32)
    sv = np.zeros(nchains, dtype=np.int32)
    io = np.zeros(nchains, dtype=np.int64)
    rc = lib.chain_run(q, n, R, M, nchains,
                       ctypes.c_longlong(iters), ctypes.c_ulonglong(seed),
                       threads,
                       codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                       best.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                       bu.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                       sv.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                       io.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
    if rc != 0:
        raise RuntimeError(f"chain_run rc={rc}")
    return {"best": best[:, :, :], "best_u": bu, "solved": sv, "iters": io}


def cpu_uncovered(q, n, R, code):
    """Independent CPU recount by dilation."""
    arr = np.zeros((q,) * n, dtype=bool)
    for wd in code:
        arr[tuple(int(x) for x in wd[:n])] = True
    for _ in range(R):
        base = arr
        new = base.copy()
        for ax in range(n):
            new |= base.any(axis=ax, keepdims=True)
        arr = new
    return int(q ** n - arr.sum())
