#!/usr/bin/env python3
"""Randomised correctness tests for the GEMM formulation.

Everything the solver believes has to come out of a tensor-core matrix product,
so every count that product yields is checked here against an independent
integer computation on the CPU (numpy int64, no floating point anywhere).

Run:  python3 test_gemm.py            (needs a GPU; <10 s, <1 GB)
"""
import sys, time
import numpy as np
import torch

from covgemm import (Space, agree, count_in_ball, digits_of, index_of,
                     onehot_np, onehot_t, ref_agree, ref_counts, space_digits)

DEV = "cuda"
rng = np.random.default_rng(20260820)
fails = []


def check(name, ok, extra=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + ("  " + extra if extra else ""))
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------- 1. encoding
def t_encoding():
    for (n, q) in [(3, 2), (4, 6), (5, 3), (2, 11), (6, 6)]:
        N = q ** n
        idx = rng.integers(0, N, size=min(N, 500))
        dig = digits_of(idx, n, q)
        ok = np.array_equal(index_of(dig, q), idx)
        # cross-check against the verifier's own convention: most significant
        # digit first, base q.
        ok &= all(int("".join(np.base_repr(d, q) for d in row), q) == i
                  for row, i in zip(dig[:50], idx[:50]))
        check(f"digit<->index roundtrip q={q} n={n}", ok)
        # torch space_digits must agree with numpy digits_of
        lo, hi = 0, min(N, 4096)
        sd = space_digits(lo, hi, n, q, DEV).cpu().numpy()
        check(f"space_digits q={q} n={n}",
              np.array_equal(sd, digits_of(np.arange(lo, hi), n, q)))
        # one-hot builders must agree
        oh_np = onehot_np(dig, q)
        oh_t = onehot_t(torch.as_tensor(dig, device=DEV), q,
                        torch.float16, DEV).cpu().numpy().astype(np.uint8)
        check(f"onehot numpy==torch q={q} n={n}", np.array_equal(oh_np, oh_t))


# ------------------------------------------------- 2. the agreement identity
def t_agreement():
    for (n, q, P, B) in [(4, 6, 37, 129), (8, 6, 169, 1024), (9, 8, 64, 777),
                         (11, 3, 81, 512), (16, 2, 50, 500), (5, 12, 40, 400)]:
        da = rng.integers(0, q, size=(P, n))
        dw = rng.integers(0, q, size=(B, n))
        ref = ref_agree(da, dw)                       # int64, exact
        A16 = onehot_t(torch.as_tensor(da, device=DEV), q, torch.float16, DEV)
        W16 = onehot_t(torch.as_tensor(dw, device=DEV), q, torch.float16, DEV)
        G16 = agree(A16, W16).cpu().numpy()
        ok16 = np.array_equal(G16.astype(np.int64), ref)
        A8 = onehot_t(torch.as_tensor(da, device=DEV), q, torch.int8, DEV)
        W8 = onehot_t(torch.as_tensor(dw, device=DEV), q, torch.int8, DEV)
        G8 = agree(A8, W8).cpu().numpy()
        ok8 = np.array_equal(G8.astype(np.int64), ref)
        # and the identity itself: agreement == n - Hamming distance
        dham = (da[:, None, :] != dw[None, :, :]).sum(-1)
        ok_id = np.array_equal(ref, n - dham)
        check(f"agreement fp16 exact q={q} n={n} ({P}x{B})", ok16)
        check(f"agreement int8 exact q={q} n={n} ({P}x{B})", ok8)
        check(f"agreement == n - d_H  q={q} n={n}", ok_id)


# --------------------------------------------------- 3. ball counts, chunked
def t_ball_counts():
    for (n, q, R, P, B) in [(4, 6, 2, 20, 300), (8, 6, 4, 32, 2000),
                            (9, 8, 4, 16, 1500), (11, 3, 4, 24, 900)]:
        da = rng.integers(0, q, size=(P, n))
        dw = rng.integers(0, q, size=(B, n))
        ref = ((da[:, None, :] != dw[None, :, :]).sum(-1) <= R).sum(1)
        A = onehot_t(torch.as_tensor(da, device=DEV), q, torch.float16, DEV)
        W = onehot_t(torch.as_tensor(dw, device=DEV), q, torch.float16, DEV)
        for cc in (None, 64, 257):
            got = count_in_ball(A, W, n - R, chunk_cols=cc).cpu().numpy()
            check(f"count_in_ball q={q} n={n} R={R} chunk={cc}",
                  np.array_equal(got, ref))


# ------------------------------------------ 4. full-space coverage from GEMM
def t_space_counts():
    """cnt[w] over the whole space, from the GEMM, vs. exhaustive ball marking."""
    for (n, q, R, M) in [(4, 6, 2, 12), (6, 6, 3, 41), (5, 3, 2, 7),
                         (8, 2, 3, 9), (3, 7, 1, 25)]:
        N = q ** n
        code = rng.integers(0, q, size=(M, n))
        ref = ref_counts(code, n, q, R)
        sp = Space(n, q, DEV, torch.float16, chunk=1 << 14)
        A = onehot_t(torch.as_tensor(code, device=DEV), q, torch.float16, DEV)
        cnt = torch.empty(N, dtype=torch.int32, device=DEV)
        for lo, hi, X in sp.chunks():
            G = X @ A.t()
            cnt[lo:hi] = (G >= n - R).sum(1, dtype=torch.int32)
        check(f"space cnt q={q} n={n} R={R} M={M}",
              np.array_equal(cnt.cpu().numpy(), ref),
              f"uncovered={int((ref == 0).sum())}")


# --------------------------------------- 5. the move identity used by solver
def t_relocate_identity():
    """Moving codeword c to x leaves exactly |U_-c| - score(x) words uncovered,
    where U_-c = {w : cnt[w] - [d(w,c)<=R] = 0} and score(x)=|B_R(x) cap U_-c|.
    Checked against a from-scratch recount of the moved code."""
    for (n, q, R, M) in [(4, 6, 2, 10), (6, 6, 3, 41), (5, 3, 2, 6)]:
        code = rng.integers(0, q, size=(M, n))
        cnt = ref_counts(code, n, q, R)
        dig = digits_of(np.arange(q ** n), n, q)
        for trial in range(6):
            c = int(rng.integers(0, M))
            covc = ((dig != code[c][None, :]).sum(1) <= R)
            Um = np.flatnonzero(cnt - covc == 0)
            x = rng.integers(0, q, size=n)
            score = int(((dig[Um] != x[None, :]).sum(1) <= R).sum())
            newcode = code.copy(); newcode[c] = x
            ref = int((ref_counts(newcode, n, q, R) == 0).sum())
            check(f"relocate identity q={q} n={n} R={R} t={trial}",
                  len(Um) - score == ref, f"{len(Um)}-{score} vs {ref}")


# --------------------------------------------------- 6. fp16 exactness limit
def t_fp16_limit():
    """The exactness argument says fp16 is safe while n <= 2048.  Demonstrate
    it holds at n far beyond anything we run, and that it is the *inner
    dimension count*, not nq, that bounds the value."""
    n, q = 1024, 2
    P, B = 8, 8
    da = rng.integers(0, q, size=(P, n)); dw = rng.integers(0, q, size=(B, n))
    ref = ref_agree(da, dw)
    A = onehot_t(torch.as_tensor(da, device=DEV), q, torch.float16, DEV)
    W = onehot_t(torch.as_tensor(dw, device=DEV), q, torch.float16, DEV)
    G = agree(A, W).cpu().numpy().astype(np.int64)
    check(f"fp16 exact at n={n} (nq={n*q}, values up to {ref.max()})",
          np.array_equal(G, ref))


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    t0 = time.time()
    t_encoding(); t_agreement(); t_ball_counts()
    t_space_counts(); t_relocate_identity(); t_fp16_limit()
    print(f"\n{len(fails)} failures in {time.time()-t0:.1f}s")
    sys.exit(1 if fails else 0)
