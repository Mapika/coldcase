#!/usr/bin/env python3
"""ctypes wrapper for libdct.so — the distance-count transform engine.

Also builds the packed ball-pattern table (host side): one uint64 per pattern,
up to R (pos<<4 | delta) bytes, zero byte terminates.  Patterns for r = 0..R
concatenated; entry 0 is the empty pattern (radius 0).
"""
import ctypes
import os
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(os.path.join(HERE, "libdct.so"))
        _lib.dct_init.restype = ctypes.c_int
        _lib.dct_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_uint64),
                                  ctypes.c_longlong, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_longlong)]
        _lib.dct_free.restype = ctypes.c_int
        _lib.dct_set_code.restype = ctypes.c_int
        _lib.dct_transform.restype = ctypes.c_int
        _lib.dct_ball_update.restype = ctypes.c_int
        _lib.dct_ball_gather.restype = ctypes.c_int
        _lib.dct_count_eq.restype = ctypes.c_longlong
        _lib.dct_map_max.restype = ctypes.c_int
        _lib.dct_map_hist.restype = ctypes.c_int
        _lib.dct_map_extract.restype = ctypes.c_longlong
        _lib.dct_map_read_at.restype = ctypes.c_int
        _lib.dct_read_cnt.restype = ctypes.c_int
    return _lib


def ball_patterns(q, n, R):
    """uint64 array of packed patterns for radii 0..R (0 = empty pattern)."""
    pats = [np.zeros(1, dtype=np.uint64)]          # r = 0
    for r in range(1, R + 1):
        subs = np.array(list(combinations(range(n), r)), dtype=np.uint64)
        nsub = len(subs)
        # value tuples: (q-1)^r combinations of deltas 1..q-1
        deltas = np.indices((q - 1,) * r).reshape(r, -1).T.astype(np.uint64) + 1
        nval = len(deltas)
        # pack: byte k = pos<<4 | delta for pair k
        out = np.zeros((nsub, nval), dtype=np.uint64)
        for k in range(r):
            byte = (subs[:, k][:, None] << np.uint64(4)) | deltas[None, :, k]
            out |= byte << np.uint64(8 * k)
        pats.append(out.ravel())
    return np.concatenate(pats)


class DCT:
    """One (q,n,R) cell resident on the GPU."""

    def __init__(self, q, n, R, extract_cap=1 << 22):
        self.q, self.n, self.R = q, n, R
        self.space = q ** n
        self.pow_q = np.array([q ** i for i in range(n)], dtype=np.int64)
        pats = ball_patterns(q, n, R)
        self.ball = len(pats)
        self.extract_cap = extract_cap
        by = ctypes.c_longlong(0)
        rc = lib().dct_init(q, n, R,
                            pats.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                            len(pats), extract_cap, ctypes.byref(by))
        if rc != 0:
            raise RuntimeError(f"dct_init rc={rc}")
        self.gpu_bytes = by.value
        self._closed = False

    def close(self):
        if not self._closed:
            lib().dct_free()
            self._closed = True

    # ---------------- code handling ----------------
    def word_index(self, words):
        """(k,n) uint8 digit array -> int64 word indices."""
        w = np.asarray(words, dtype=np.int64)
        return w @ self.pow_q

    def index_word(self, idx):
        """int64 indices -> (k,n) uint8 digits."""
        idx = np.asarray(idx, dtype=np.int64).ravel()
        out = np.empty((len(idx), self.n), dtype=np.uint8)
        x = idx.copy()
        for i in range(self.n):
            out[:, i] = x % self.q
            x //= self.q
        return out

    def set_code(self, idx):
        idx = np.ascontiguousarray(idx, dtype=np.int64)
        rc = lib().dct_set_code(
            idx.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)), len(idx))
        if rc != 0:
            raise RuntimeError(f"dct_set_code rc={rc}")

    # ---------------- transforms ----------------
    def recount(self, idx):
        """Exact coverage multiplicity map cnt from the code (one transform)."""
        self.set_code(idx)
        rc = lib().dct_transform(0, 2)
        if rc != 0:
            raise RuntimeError(f"transform(code->cnt) rc={rc}")

    def gain_map(self):
        """A0 <- exact gain map (uncovered words newly covered per position)."""
        rc = lib().dct_transform(1, 1)
        if rc != 0:
            raise RuntimeError(f"transform(uncov->sum) rc={rc}")

    def loss_map(self):
        """A0 <- exact private-coverage map (evaluate at codewords)."""
        rc = lib().dct_transform(2, 1)
        if rc != 0:
            raise RuntimeError(f"transform(cnt1->sum) rc={rc}")

    # ---------------- incremental ops ----------------
    def ball_update(self, idx, delta):
        idx = np.ascontiguousarray(np.atleast_1d(idx), dtype=np.int64)
        rc = lib().dct_ball_update(
            idx.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            len(idx), delta)
        if rc != 0:
            raise RuntimeError(f"ball_update rc={rc}")

    def ball_gather(self, idx, target):
        """Exact count of ball words with cnt==target, per position."""
        idx = np.ascontiguousarray(np.atleast_1d(idx), dtype=np.int64)
        out = np.empty(len(idx), dtype=np.int32)
        rc = lib().dct_ball_gather(
            idx.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)), len(idx),
            target, out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if rc != 0:
            raise RuntimeError(f"ball_gather rc={rc}")
        return out

    def count_eq(self, target):
        return int(lib().dct_count_eq(target))

    def uncovered(self):
        return self.count_eq(0)

    # ---------------- map queries (valid after gain_map/loss_map) ----------
    def map_max(self):
        v = ctypes.c_int32(0)
        lib().dct_map_max(ctypes.byref(v))
        return int(v.value)

    def map_hist(self, nbins, vmax):
        out = np.zeros(nbins, dtype=np.int32)
        lib().dct_map_hist(nbins, vmax,
                           out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        return out

    def map_extract(self, thr, cap=None):
        cap = min(cap or self.extract_cap, self.extract_cap)
        oi = np.empty(cap, dtype=np.uint32)
        ov = np.empty(cap, dtype=np.int32)
        found = lib().dct_map_extract(
            int(thr), oi.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ov.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), cap)
        m = min(int(found), cap)
        return oi[:m].astype(np.int64), ov[:m], int(found)

    def map_read_at(self, idx):
        idx = np.ascontiguousarray(np.atleast_1d(idx), dtype=np.int64)
        out = np.empty(len(idx), dtype=np.int32)
        rc = lib().dct_map_read_at(
            idx.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)), len(idx),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if rc != 0:
            raise RuntimeError(f"map_read_at rc={rc}")
        return out

    def read_cnt(self, off=0, length=None):
        length = length if length is not None else self.space - off
        out = np.empty(length, dtype=np.uint16)
        rc = lib().dct_read_cnt(off, length,
                                out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
        if rc != 0:
            raise RuntimeError(f"read_cnt rc={rc}")
        return out
