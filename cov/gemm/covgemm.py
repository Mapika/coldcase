#!/usr/bin/env python3
"""
covgemm.py -- dense-GEMM primitives for q-ary covering codes on tensor cores.

The one identity everything rests on
------------------------------------
Encode a word w in Z_q^n as the concatenation of its n one-hot digit vectors,

    phi(w) in {0,1}^{n q},   phi(w)[i*q + d] = 1 iff w_i = d.

Then for any two words a, b

    <phi(a), phi(b)> = #{ i : a_i = b_i } = n - d_H(a,b),

so a single dense matrix product

    G = Phi(A) @ Phi(W)^T          (M x nq) @ (nq x B)  ->  M x B

carries the exact Hamming distance between every codeword of A and every word
of W, in integer arithmetic, with no gathers and no branches.  Ball membership
is the threshold  G >= n - R.

Exactness of the fp16 path
--------------------------
The inputs are 0/1, so every product is 0 or 1 and is exact in any format with
at least one mantissa bit.  Every partial sum is a non-negative integer bounded
by the full inner product, which is at most n.  IEEE binary16 represents every
integer in [0, 2048] exactly (11-bit significand), so for n <= 2048 -- every
cell in this project has n <= 16 -- no partial sum and no final value is ever
rounded, whatever order the accumulation happens in and whether the accumulator
is fp16 or fp32.  The fp16 GEMM is therefore *bit-exact integer arithmetic*
here, not an approximation, and `allow_fp16_reduced_precision_reduction` is
irrelevant to it.  (We still assert this against an integer reference in
test_gemm.py rather than trusting the argument alone.)

The int8 path (torch._int_mm) is exact by construction; it costs 4-byte outputs
instead of 2-byte ones, which matters because these GEMMs are output-bandwidth
bound (see NOTES.md sec. 3).
"""
import math
import numpy as np
import torch

# ---------------------------------------------------------------- encodings

def digits_of(idx, n, q, dtype=np.int64):
    """Row-major digit expansion: idx = sum_j d_j q^(n-1-j).  (B,) -> (B,n)."""
    idx = np.asarray(idx, dtype=np.int64)
    out = np.empty((idx.shape[0], n), dtype=dtype)
    for j in range(n - 1, -1, -1):
        out[:, j] = idx % q
        idx = idx // q
    return out


def index_of(dig, q):
    """(B,n) digits -> (B,) flat index, the inverse of digits_of."""
    dig = np.asarray(dig, dtype=np.int64)
    n = dig.shape[1]
    w = (q ** np.arange(n - 1, -1, -1, dtype=np.int64))
    return dig @ w


def onehot_np(dig, q):
    """(B,n) digit array -> (B, n*q) uint8 one-hot."""
    dig = np.asarray(dig, dtype=np.int64)
    B, n = dig.shape
    oh = np.zeros((B, n * q), dtype=np.uint8)
    cols = dig + (np.arange(n, dtype=np.int64) * q)[None, :]
    oh[np.arange(B)[:, None], cols] = 1
    return oh


def onehot_t(dig, q, dtype=torch.float16, device="cuda"):
    """(B,n) int64 torch digit tensor -> (B, n*q) one-hot in `dtype`."""
    B, n = dig.shape
    oh = torch.zeros((B, n * q), dtype=dtype, device=dig.device)
    cols = dig.to(torch.int64) + torch.arange(n, device=dig.device) * q
    oh.scatter_(1, cols, 1)
    return oh.to(device) if str(oh.device) != device else oh


def space_digits(lo, hi, n, q, device):
    """Digits of the flat indices [lo,hi) of Z_q^n, computed on `device`."""
    idx = torch.arange(lo, hi, device=device, dtype=torch.int64)
    dig = torch.empty((hi - lo, n), device=device, dtype=torch.int64)
    for j in range(n - 1, -1, -1):
        dig[:, j] = idx % q
        idx = torch.div(idx, q, rounding_mode="floor")
    return dig


# ------------------------------------------------------------------- space

class Space:
    """The one-hot matrix of all of Z_q^n, cached whole or produced in chunks.

    Rows are flat indices in the same convention as verify_cov.py / covsearch:
    w = sum_j d_j q^(n-1-j).
    """

    def __init__(self, n, q, device="cuda", dtype=torch.float16,
                 chunk=1 << 21, cache_bytes=6 << 30):
        self.n, self.q, self.device, self.dtype = n, q, device, dtype
        self.N = q ** n
        self.nq = n * q
        self.chunk = min(chunk, self.N)
        itemsize = torch.tensor([], dtype=dtype).element_size()
        self.cached = (self.N * self.nq * itemsize) <= cache_bytes
        self._full = None
        if self.cached:
            buf = torch.empty((self.N, self.nq), dtype=dtype, device=device)
            for lo in range(0, self.N, self.chunk):
                hi = min(lo + self.chunk, self.N)
                buf[lo:hi] = onehot_t(space_digits(lo, hi, n, q, device), q,
                                      dtype, device)
            self._full = buf
        else:
            self._buf = torch.empty((self.chunk, self.nq), dtype=dtype,
                                    device=device)

    def chunks(self):
        """Yield (lo, hi, onehot rows [lo,hi) of Z_q^n)."""
        for lo in range(0, self.N, self.chunk):
            hi = min(lo + self.chunk, self.N)
            if self.cached:
                yield lo, hi, self._full[lo:hi]
            else:
                dig = space_digits(lo, hi, self.n, self.q, self.device)
                v = self._buf[: hi - lo]
                v.zero_()
                cols = dig + torch.arange(self.n, device=self.device) * self.q
                v.scatter_(1, cols, 1)
                yield lo, hi, v

    def bytes_resident(self):
        it = torch.tensor([], dtype=self.dtype).element_size()
        return (self.N if self.cached else self.chunk) * self.nq * it


# -------------------------------------------------------------- primitives

def _pad_to(t, dim, mult):
    """Zero-pad `t` along `dim` up to a multiple of `mult` (no copy if already)."""
    k = t.shape[dim]
    r = (-k) % mult
    if r == 0:
        return t, k
    shape = list(t.shape); shape[dim] = r
    return torch.cat([t, torch.zeros(shape, dtype=t.dtype, device=t.device)],
                     dim=dim), k


def int_mm(A, Wt):
    """torch._int_mm with the cuBLASLt shape constraints handled by padding.

    Measured constraints of torch 2.7 / cuBLASLt on sm90 (probed, not guessed):
    M must be > 16 and a multiple of 32, K and N multiples of 8.  Zero padding
    is harmless -- a padded row/column is the all-zero vector, whose inner
    product with anything is 0 -- and the padded entries are sliced away.
    """
    A2, P = _pad_to(A, 0, 32)
    A2, _ = _pad_to(A2, 1, 8)
    W2, B = _pad_to(Wt, 1, 8)
    W2, _ = _pad_to(W2, 0, 8)
    return torch._int_mm(A2.contiguous(), W2.contiguous())[:P, :B]


def agree(A, W, out_dtype=None):
    """Exact agreement matrix <phi(a),phi(w)> for one-hot A (P,nq), W (B,nq).

    A and W must already be one-hot in the same dtype (fp16/bf16) or int8.
    """
    if A.dtype == torch.int8:
        return int_mm(A, W.t())
    G = A @ W.t()
    return G if out_dtype is None else G.to(out_dtype)


def count_in_ball(A, W, thr, chunk_cols=None):
    """For every row a of A: #{ rows w of W : <phi(a),phi(w)> >= thr }.

    thr = n - R gives the number of words of W inside the Hamming ball of a.
    Returned as int32 on the same device.
    """
    P = A.shape[0]
    out = torch.zeros(P, dtype=torch.int32, device=A.device)
    B = W.shape[0]
    step = chunk_cols or B
    for c0 in range(0, B, step):
        Wc = W[c0:c0 + step]
        G = agree(A, Wc)
        out += (G >= thr).sum(1, dtype=torch.int32)
    return out


# -------------------------------------------------------- CPU reference

def ref_agree(dig_a, dig_w):
    """Integer reference: (P,n) x (B,n) -> (P,B) coordinate agreements."""
    a = np.asarray(dig_a)[:, None, :]
    w = np.asarray(dig_w)[None, :, :]
    return (a == w).sum(-1).astype(np.int64)


def ref_counts(code_dig, n, q, R):
    """Exact cnt[w] over all of Z_q^n by direct ball marking (small cells)."""
    N = q ** n
    dig = digits_of(np.arange(N), n, q)
    cnt = np.zeros(N, dtype=np.int32)
    for c in code_dig:
        d = (dig != c[None, :]).sum(1)
        cnt += (d <= R)
    return cnt
