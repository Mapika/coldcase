"""Host driver for the GPU constant-weight-code tabu search (ctypes, no deps)."""
import ctypes, os, subprocess
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(SRC, "libcwc.so")


def build(force=False):
    if force or not os.path.exists(LIB) or (
        os.path.getmtime(os.path.join(SRC, "cwc_tabu.cu")) > os.path.getmtime(LIB)
    ):
        subprocess.run([
            "nvcc", "-O3", "-arch=sm_90", "--shared", "-Xcompiler", "-fPIC",
            "-o", LIB, os.path.join(SRC, "cwc_tabu.cu")], check=True)
    return LIB


def random_cw_words(n, w, count, rng):
    """count random weight-w n-bit words as uint64 (vectorized)."""
    # argsort of random matrix -> random w-subsets
    r = rng.random((count, n)).argsort(axis=1)[:, :w]
    out = np.zeros(count, dtype=np.uint64)
    for k in range(w):
        out |= np.uint64(1) << r[:, k].astype(np.uint64)
    return out


def make_inits(n, w, M, nchains, rng, seed_code=None, keep_frac=1.0):
    """Initial states; optionally seed a prefix from an incumbent code."""
    inits = np.zeros((nchains, M), dtype=np.uint64)
    base = None
    if seed_code is not None:
        base = np.array(seed_code, dtype=np.uint64)
    for c in range(nchains):
        if base is not None:
            k = min(int(len(base) * keep_frac), M)
            if k > 0:
                idx = rng.choice(len(base), size=k, replace=False) if k < len(base) else np.arange(len(base))
                inits[c, :k] = base[idx]
            if M > k:
                inits[c, k:] = random_cw_words(n, w, M - k, rng)
        else:
            inits[c] = random_cw_words(n, w, M, rng)
    return inits


def run_chains(n, d, w, M, inits, mode=0, iters=200_000, tenure_lo=8, tenure_span=8,
               seed=None, threads=128):
    """Launch nchains tabu chains. Returns dict with found flags, costs, words."""
    assert 1 <= n <= 64 and M <= 2048
    lib = ctypes.CDLL(build())
    lib.cwc_run.restype = ctypes.c_int
    nchains = int(inits.shape[0])
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "little")

    inits = np.ascontiguousarray(inits, dtype=np.uint64)
    words_out = np.zeros_like(inits)
    best_words_out = np.zeros_like(inits)
    found = np.zeros(nchains, dtype=np.int32)
    best_cost = np.zeros(nchains, dtype=np.int32)
    iters_out = np.zeros(nchains, dtype=np.int64)

    u64p = ctypes.POINTER(ctypes.c_uint64)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64p = ctypes.POINTER(ctypes.c_longlong)
    rc = lib.cwc_run(
        n, d, w, M, mode, nchains,
        ctypes.c_longlong(iters), tenure_lo, tenure_span,
        ctypes.c_ulonglong(seed), threads,
        inits.ctypes.data_as(u64p), words_out.ctypes.data_as(u64p),
        best_words_out.ctypes.data_as(u64p),
        found.ctypes.data_as(i32p), best_cost.ctypes.data_as(i32p),
        iters_out.ctypes.data_as(i64p))
    if rc != 0:
        raise RuntimeError(f"cwc_run failed rc={rc}")
    return {"found": found, "best_cost": best_cost, "iters": iters_out,
            "words": words_out, "best_words": best_words_out, "seed": seed}
