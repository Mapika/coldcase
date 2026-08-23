#!/usr/bin/env python3
"""ctypes wrapper for libdct.so — the distance-count transform engine.

Also builds the packed ball-pattern table (host side): one uint64 per pattern,
up to R (pos<<4 | delta) bytes, zero byte terminates.  Patterns for r = 0..R
concatenated; entry 0 is the empty pattern (radius 0).

Host-coherent extension (libdct_hc.so, env GPUDCT_SO to select): per-array
memory modes (0 = HBM cudaMalloc, 1 = LPDDR plain malloc via GH200 ATS,
2 = LPDDR managed+preferred-CPU), 64-bit extraction for q^n > 2^32, split
ball kernels, owner-trick losspass, ox-format ball table (mask+nibbles).
Auto policy: cells whose arrays exceed HBM put the A_d layers in LPDDR and
keep cnt (the random-access array) in HBM.
"""
import ctypes
import os
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

_lib = None
_libpath = None

# HBM budget for the auto placement policy (bytes).  GH200: 96 GB HBM; leave
# room for ball tables, extraction buffers and other tenants.
HBM_BUDGET = int(os.environ.get("GPUDCT_HBM_BUDGET", 80e9))


def lib():
    global _lib, _libpath
    path = os.path.join(HERE, os.environ.get("GPUDCT_SO", "libdct.so"))
    if _lib is None or _libpath != path:
        _lib = ctypes.CDLL(path)
        _libpath = path
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
        if hasattr(_lib, "dct_init2"):
            _lib.dct_init2.restype = ctypes.c_int
            _lib.dct_init2.argtypes = _lib.dct_init.argtypes + [
                ctypes.c_int, ctypes.c_int, ctypes.c_int]
            _lib.dct_init3.restype = ctypes.c_int
            _lib.dct_init3.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                ctypes.c_int, ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_longlong, ctypes.c_int,
                ctypes.POINTER(ctypes.c_longlong),
                ctypes.c_int, ctypes.c_int, ctypes.c_int]
            _lib.dct_map_extract64.restype = ctypes.c_longlong
            _lib.dct_loss_owner.restype = ctypes.c_int
            _lib.dct_set_tbl2.restype = ctypes.c_int
            _lib.dct_use_fmt.restype = ctypes.c_int
            _lib.dct_host_bytes.restype = ctypes.c_longlong
    return _lib


def has_hc():
    return hasattr(lib(), "dct_init2")


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


def _delta_sets(axes, R, problem):
    """Per-position admissible deltas for the support (ball) table."""
    if problem == "hamming":
        return [list(range(1, a)) for a in axes]
    if problem == "torus_linf":
        return [list(range(1, R + 1)) + list(range(a - R, a))
                for a in axes]
    raise ValueError(problem)


def support_patterns(axes, R, problem="hamming"):
    """Generic packed support table (byte k = pos<<4 | delta).

    hamming: subsets of <= R positions x nonzero deltas (Hamming ball).
    torus_linf: subsets of ANY size (<= 8 positions) x deltas with circular
    distance <= R per axis (Chebyshev ball on the torus)."""
    n = len(axes)
    if problem == "hamming" and len(set(axes)) == 1:
        return ball_patterns(axes[0], n, R)     # fast numpy path
    from itertools import product
    ds = _delta_sets(axes, R, problem)
    maxw = R if problem == "hamming" else n
    if maxw > 8:
        raise ValueError("support patterns cap at 8 changed positions")
    pats = [0]
    for w in range(1, maxw + 1):
        for subs in combinations(range(n), w):
            for deltas in product(*[ds[p] for p in subs]):
                v = 0
                for k, (p, dl) in enumerate(zip(subs, deltas)):
                    v |= ((p << 4) | dl) << (8 * k)
                pats.append(v)
    return np.array(pats, dtype=np.uint64)


def support_patterns_ox(axes, R, problem="hamming"):
    """Generic ox-format table (mask u16, nibble offsets u64)."""
    n = len(axes)
    if problem == "hamming" and len(set(axes)) == 1:
        return ball_patterns_ox(axes[0], n, R)
    from itertools import product
    ds = _delta_sets(axes, R, problem)
    maxw = R if problem == "hamming" else n
    masks, offs = [0], [0]
    for w in range(1, maxw + 1):
        for subs in combinations(range(n), w):
            for deltas in product(*[ds[p] for p in subs]):
                m, o = 0, 0
                for p, dl in zip(subs, deltas):
                    m |= 1 << p
                    o |= dl << (4 * p)
                masks.append(m)
                offs.append(o)
    return (np.array(masks, dtype=np.uint16), np.array(offs, dtype=np.uint64))


def ball_patterns_ox(q, n, R):
    """ox/gpucov format: (mask uint16, nibble-offset uint64) per ball word.

    Same enumeration order as ball_patterns.  Nibble i of offs = delta at
    position i (1..q-1); requires q <= 16 wraps -> q <= 10 here fits."""
    masks = [np.zeros(1, dtype=np.uint16)]
    offs = [np.zeros(1, dtype=np.uint64)]
    for r in range(1, R + 1):
        subs = np.array(list(combinations(range(n), r)), dtype=np.uint64)
        deltas = np.indices((q - 1,) * r).reshape(r, -1).T.astype(np.uint64) + 1
        m = np.zeros((len(subs), len(deltas)), dtype=np.uint16)
        o = np.zeros((len(subs), len(deltas)), dtype=np.uint64)
        for k in range(r):
            m |= (np.uint16(1) << subs[:, k].astype(np.uint16))[:, None]
            o |= deltas[None, :, k] << (np.uint64(4) * subs[:, k][:, None])
        masks.append(m.ravel())
        offs.append(o.ravel())
    return np.concatenate(masks), np.concatenate(offs)


PROBLEM_IDS = {"hamming": 0, "torus_linf": 1}


class DCT:
    """One cell resident on the GPU (or host-coherent LPDDR).

    Classic covering-code form: DCT(q, n, R).  Generic form:
    DCT(axes=[...], R=..., problem="hamming"|"torus_linf") — heterogeneous
    axis sizes allowed (mixed-radix covering codes / torus grids).

    layers_mode / cnt_mode: None = auto policy, else 0 HBM / 1 ATS / 2 managed.
    use_owner: allocate owner[] for the owner-trick losspass (dct_loss_owner).
    """

    def __init__(self, q=None, n=None, R=None, extract_cap=1 << 22,
                 layers_mode=None, cnt_mode=None, use_owner=False,
                 problem="hamming", axes=None):
        if axes is None:
            axes = [q] * n
        axes = [int(a) for a in axes]
        n = len(axes)
        self.q, self.n, self.R = (q if q else axes[0]), n, R
        self.axes, self.problem = axes, problem
        self.space = 1
        for a in axes:
            self.space *= a
        self.pow_q = np.cumprod([1] + axes[:-1]).astype(np.int64)
        pats = support_patterns(axes, R, problem)
        self.ball = len(pats)
        self.extract_cap = extract_cap
        nlayers = (R + 1) if problem == "hamming" else 1
        # auto placement: everything in HBM if it fits the budget, else
        # layers -> LPDDR (streamed by transforms), cnt stays in HBM.
        total = self.space * (4 * nlayers + 2)
        if layers_mode is None:
            layers_mode = 0 if total <= HBM_BUDGET else 1
        if cnt_mode is None:
            cnt_mode = 0 if (layers_mode == 0 or self.space * 2 <= 40e9) else 1
        self.layers_mode, self.cnt_mode = layers_mode, cnt_mode
        self.use_owner = use_owner
        by = ctypes.c_longlong(0)
        needs_hc = (layers_mode or cnt_mode or use_owner
                    or self.space > 2**32 - 2 or problem != "hamming"
                    or len(set(axes)) != 1)
        if needs_hc and not has_hc():
            raise RuntimeError(
                f"cell needs host-coherent lib (layers_mode={layers_mode}, "
                f"space={self.space:.3g}, problem={problem}); "
                f"set GPUDCT_SO=libdct_hc.so")
        pp = pats.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        if has_hc():
            ax = (ctypes.c_int * n)(*axes)
            rc = lib().dct_init3(PROBLEM_IDS[problem], n, ax, R, pp,
                                 len(pats), extract_cap, ctypes.byref(by),
                                 layers_mode, cnt_mode, 1 if use_owner else 0)
        else:
            rc = lib().dct_init(axes[0], n, R, pp, len(pats), extract_cap,
                                ctypes.byref(by))
        if rc != 0:
            raise RuntimeError(f"dct_init rc={rc}")
        self.gpu_bytes = by.value
        self._closed = False

    def close(self):
        if not self._closed:
            lib().dct_free()
            self._closed = True

    # ---------------- optional ox-format ball table ----------------
    def enable_tbl2(self):
        m, o = support_patterns_ox(self.axes, self.R, self.problem)
        rc = lib().dct_set_tbl2(
            m.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            o.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), len(m))
        if rc != 0:
            raise RuntimeError(f"dct_set_tbl2 rc={rc}")

    def use_fmt(self, fmt):
        rc = lib().dct_use_fmt(int(fmt))
        if rc != 0:
            raise RuntimeError(f"dct_use_fmt rc={rc}")

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
            out[:, i] = x % self.axes[i]
            x //= self.axes[i]
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

    def loss_owner(self, idx):
        """Owner-trick loss: rebuild cnt+owner by ball marking, scan cnt==1.

        idx must be the FULL current code.  Returns int32 loss per codeword.
        Requires use_owner=True."""
        idx = np.ascontiguousarray(idx, dtype=np.int64)
        out = np.empty(len(idx), dtype=np.int32)
        rc = lib().dct_loss_owner(
            idx.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)), len(idx),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if rc != 0:
            raise RuntimeError(f"dct_loss_owner rc={rc}")
        return out

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
        ov = np.empty(cap, dtype=np.int32)
        if self.space > 2**32 - 2:
            oi = np.empty(cap, dtype=np.int64)
            found = lib().dct_map_extract64(
                int(thr), oi.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                ov.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), cap)
            m = min(int(found), cap)
            return oi[:m], ov[:m], int(found)
        oi = np.empty(cap, dtype=np.uint32)
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
