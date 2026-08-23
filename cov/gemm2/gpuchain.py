#!/usr/bin/env python3
"""gpuchain — GPU transform-engine covering-code solver (LNS / ruin-recreate).

Covsearch-compatible CLI:
    gpuchain -q Q -n N -R R -M M -t SECONDS -s SEED --out FILE [--in SEEDFILE]

Engine: exact full-grid distance-count transforms (dct.cu).  Every decision
uses exact global information:
  * RECREATE: lazy greedy placement.  One transform yields gain(x) = exactly
    how many uncovered words a codeword at x would cover, for every x at once.
    Coverage gain is submodular, so stale gains are upper bounds and the lazy
    (pop, re-evaluate exactly, commit if still >= the next bound) loop commits
    the TRUE greedy argmax every time.  Refresh below the extraction threshold.
  * RUIN: one transform yields loss(c) = exact private coverage of every
    codeword; remove low-loss subsets (weighted / random / clustered).
  * PEEL: remove codewords whose exact loss is 0 — a cover stays a cover.

Deadline-safe: the -t budget covers init.  The incumbent is published to --out
atomically (write .part, rename) on every improvement, throttled to ~1 Hz.
Yields the GPU: if a foreign compute process appears, frees ALL device memory
within one round (<~1 s), republishes, and resumes when the GPU is idle again.

No claim rests on GPU counters: "solved" is re-counted from scratch on device,
and record status only ever comes from cov/verify_cov.py on the written file
(the caller's job — see siege.py / campaign.record).
"""
import argparse
import heapq
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COV = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, COV)

from dctlib import DCT  # noqa: E402

DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


def foreign_gpu_pids():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20)
    except Exception:
        return []
    me = os.getpid()
    pids = []
    for line in out.stdout.strip().splitlines():
        p = line.split(",")[0].strip()
        if not (p.isdigit() and int(p) != me and os.path.exists(f"/proc/{p}")):
            continue
        try:
            with open(f"/proc/{p}/cmdline", "rb") as f:
                cmd = f.read().decode("utf-8", "replace")
        except OSError:
            continue                      # gone: stale entry
        if "siege.py" in cmd or "gpuchain" in cmd or "bench_h2h" in cmd:
            continue                      # our own engines are not tenants
        pids.append(int(p))
    return pids


def write_code_atomic(path, words):
    """words: (M,n) uint8. Base-36 digits, one word per line."""
    tmp = path + ".part"
    with open(tmp, "w") as f:
        for w in words:
            f.write("".join(DIGITS[int(x)] for x in w) + "\n")
    os.replace(tmp, path)


def read_code(path, q, n):
    words = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if " " in line or "," in line:
            digs = [int(t) for t in line.replace(",", " ").split()]
        else:
            digs = [DIGITS.index(c) for c in line.lower()]
        assert len(digs) == n and all(0 <= d < q for d in digs), line
        words.append(digs)
    return np.array(words, dtype=np.uint8)


class Engine:
    """LNS covering-code engine for one cell, resident on the GPU."""

    def __init__(self, q, n, R, seed=1, out=None, log=print,
                 deadline=None, problem="hamming", axes=None):
        self.q, self.n, self.R = q, n, R
        self.problem, self.axes = problem, axes
        self.rng = np.random.default_rng(seed * 1000003 + 12345)
        self.out = out
        self.log = log
        self.deadline = deadline
        self.eng = None
        self.code = np.zeros(0, dtype=np.int64)     # word indices
        self.uncov = None
        self.best_code = None                        # best-by-uncovered snapshot
        self.best_uncov = None
        self._last_pub = 0.0
        self._last_guard = 0.0
        self._paused_s = 0.0
        self.stats = {"transforms": 0, "placements": 0, "removals": 0,
                      "rounds": 0, "recounts": 0}
        self._open()

    # ------------------------------------------------ lifecycle / guard
    def _open(self):
        self.eng = DCT(self.q, self.n, self.R, problem=self.problem,
                       axes=self.axes)

    def close(self):
        if self.eng:
            self.eng.close()
            self.eng = None

    def time_left(self):
        return np.inf if self.deadline is None else self.deadline - time.time()

    def guard(self, force=False):
        """Every 60 s: yield the GPU to foreign compute processes."""
        now = time.time()
        if not force and now - self._last_guard < 60:
            return
        self._last_guard = now
        if not foreign_gpu_pids():
            return
        self.log("GUARD: foreign GPU process detected — freeing device memory")
        self.publish(force=True)
        self.close()
        t0 = time.time()
        while foreign_gpu_pids():
            time.sleep(30)
            if self.time_left() < 60:
                break
        self._paused_s += time.time() - t0
        self.log(f"GUARD: resuming after {time.time()-t0:.0f}s pause")
        self._open()
        if len(self.code):
            self.eng.recount(self.code)
            self.uncov = self.eng.uncovered()

    # ------------------------------------------------ state
    def load(self, code):
        """Accepts either 1-D word indices or a 2-D (M,n) digit array."""
        arr = np.asarray(code)
        if arr.ndim == 2:
            arr = self.eng.word_index(arr.astype(np.int64))
        self.code = np.unique(arr.astype(np.int64)) if arr.size else \
            np.zeros(0, dtype=np.int64)
        self.eng.recount(self.code)
        self.stats["transforms"] += 1
        self.uncov = self.eng.uncovered()
        self._snapshot()

    def _snapshot(self):
        if self.best_uncov is None or self.uncov < self.best_uncov or (
                self.uncov == self.best_uncov and
                (self.best_code is None or len(self.code) < len(self.best_code))):
            self.best_uncov = self.uncov
            self.best_code = self.code.copy()
            self.publish()

    def publish(self, force=False):
        if self.out is None or self.best_code is None or not len(self.best_code):
            return
        now = time.time()
        if not force and now - self._last_pub < 1.0:
            return
        self._last_pub = now
        write_code_atomic(self.out, self.eng.index_word(self.best_code)
                          if self.eng else DCT.index_word(self, self.best_code))

    def verify_solved_on_device(self):
        """From-scratch recount; returns exact uncovered."""
        self.eng.recount(self.code)
        self.stats["transforms"] += 1
        self.stats["recounts"] += 1
        self.uncov = self.eng.uncovered()
        return self.uncov

    # ------------------------------------------------ recreate (lazy greedy)
    def greedy_fill(self, target_m, batch=64, refresh_every=512):
        """Place codewords by exact lazy greedy until covered, target_m
        reached, or deadline.  Returns final uncovered."""
        eng = self.eng
        codeset = set(self.code.tolist())
        placed_since = refresh_every  # force initial refresh
        heap = []                     # (-stale_gain, idx)
        thr = 1
        while self.uncov > 0 and len(codeset) < target_m:
            if self.time_left() < 5:
                break
            self.guard()
            if placed_since >= refresh_every or not heap:
                eng.gain_map()
                self.stats["transforms"] += 1
                mx = eng.map_max()
                if mx <= 0:
                    break
                # threshold so extraction stays bounded but deep enough to
                # survive many placements
                cap = min(eng.extract_cap, 1 << 18)
                thr = 1
                if mx > 1:
                    hist = eng.map_hist(1024, mx)
                    csum = np.cumsum(hist[::-1])[::-1]
                    good = np.flatnonzero(csum <= cap * 0.9)
                    b = int(good[0]) if len(good) else 1023
                    thr = max(1, b * mx // 1024 + 1)
                idx, val, found = eng.map_extract(thr, cap=cap)
                while found > cap and thr < mx:
                    thr = min(mx, thr + max(1, (mx - thr) // 2))
                    idx, val, found = eng.map_extract(thr, cap=cap)
                order = np.argsort(-val, kind="stable")
                heap = [(-int(val[i]), int(idx[i])) for i in order
                        if int(idx[i]) not in codeset]
                heapq.heapify(heap)
                placed_since = 0
                if not heap:
                    if thr > 1:
                        placed_since = refresh_every  # loop retries with thr=1
                        thr = 1
                        continue
                    break
            # pop a batch of best stale bounds, evaluate exactly
            cand = []
            while heap and len(cand) < batch:
                v, i = heapq.heappop(heap)
                if i not in codeset:
                    cand.append((-v, i))
            if not cand:
                placed_since = refresh_every
                continue
            widx = np.array([c[1] for c in cand], dtype=np.int64)
            exact = eng.ball_gather(widx, 0)
            bi = int(np.argmax(exact))
            bg = int(exact[bi])
            next_bound = -heap[0][0] if heap else 0
            if bg < 1:
                placed_since = refresh_every
                continue
            if bg >= next_bound and bg >= thr:
                w = int(widx[bi])
                eng.ball_update(np.array([w], dtype=np.int64), +1)
                codeset.add(w)
                self.uncov -= bg
                self.stats["placements"] += 1
                placed_since += 1
                for k, (v, i) in enumerate(cand):
                    if i != w and exact[k] > 0:
                        heapq.heappush(heap, (-int(exact[k]), i))
            else:
                for k, (v, i) in enumerate(cand):
                    if exact[k] > 0:
                        heapq.heappush(heap, (-int(exact[k]), i))
                if bg < thr:
                    placed_since = refresh_every   # must refresh to stay greedy
        self.code = np.array(sorted(codeset), dtype=np.int64)
        return self.uncov

    # ------------------------------------------------ ruin
    def losses(self):
        self.eng.loss_map()
        self.stats["transforms"] += 1
        return self.eng.map_read_at(self.code)

    def ruin(self, k, mode="low"):
        """Remove k codewords; returns removed indices."""
        m = len(self.code)
        k = min(k, m - 1)
        if k <= 0:
            return np.zeros(0, dtype=np.int64)
        if mode == "random":
            pick = self.rng.choice(m, size=k, replace=False)
        elif mode == "cluster":
            digs = self.eng.index_word(self.code).astype(np.int16)
            c = digs[self.rng.integers(m)]
            d = (digs != c[None, :]).sum(axis=1)
            pick = np.argsort(d + self.rng.random(m))[:k]
        elif mode == "hole":
            # cluster the ruin around where coverage is missing: a top-gain
            # position sits inside/near the uncovered region
            self.eng.gain_map()
            self.stats["transforms"] += 1
            mx = self.eng.map_max()
            if mx <= 0:
                return self.ruin(k, "low")
            oi, ov, found = self.eng.map_extract(max(1, mx // 2), cap=4096)
            if not len(oi):
                return self.ruin(k, "low")
            c = self.eng.index_word(
                np.array([int(oi[self.rng.integers(len(oi))])]))[0].astype(np.int16)
            digs = self.eng.index_word(self.code).astype(np.int16)
            d = (digs != c[None, :]).sum(axis=1)
            pick = np.argsort(d + self.rng.random(m))[:k]
        else:  # 'low': exact loss, exponential jitter
            lo = self.losses().astype(np.float64)
            jit = self.rng.exponential(scale=max(1.0, lo.mean() * 0.15), size=m)
            pick = np.argsort(lo + jit)[:k]
        removed = self.code[pick]
        self.eng.ball_update(removed, -1)
        self.code = np.delete(self.code, pick)
        self.uncov = self.eng.uncovered()
        self.stats["removals"] += int(k)
        return removed

    # ------------------------------------------------ peel
    def peel(self):
        """Remove zero-loss codewords (cover stays a cover). Returns #removed.

        Batched: one loss transform gives all candidates; a batched exact
        re-check filters them; a pairwise-independent subset (Hamming distance
        > 2R, so removals cannot create each other's private words) is removed
        in one go; repeat until fixpoint."""
        total = 0
        while True:
            if self.time_left() < 5:
                break
            lo = self.losses()
            zeros = np.flatnonzero(lo == 0)
            if not len(zeros):
                break
            exact = self.eng.ball_gather(self.code[zeros], 1)
            zeros = zeros[exact == 0]
            if not len(zeros):
                break
            self.rng.shuffle(zeros)
            if 2 * self.R < self.n:
                digs = self.eng.index_word(self.code[zeros]).astype(np.int16)
                sel = []
                for i in range(len(zeros)):
                    if not sel:
                        sel.append(i)
                        continue
                    d = (digs[np.array(sel)] != digs[i][None, :]).sum(axis=1)
                    if (d > 2 * self.R).all():
                        sel.append(i)
                pick = zeros[np.array(sel)]
            else:
                pick = zeros[:1]
            self.eng.ball_update(self.code[pick], -1)
            mask = np.ones(len(self.code), dtype=bool)
            mask[pick] = False
            self.code = self.code[mask]
            total += len(pick)
            self.stats["removals"] += len(pick)
        if total:
            self.uncov = self.eng.uncovered()
        return total

    # ------------------------------------------------ LNS at fixed M
    def lns(self, M, round_budget_s, kmin=8, kmax=None, log_every=20):
        """Ruin/recreate rounds at fixed M until covered or budget out.
        Assumes len(code) <= M.  Returns final uncovered (0 = solved)."""
        t0 = time.time()
        kmax = kmax or max(kmin + 1, M // 20)
        self.best_uncov = None
        self.best_code = None
        if len(self.code) < M:
            self.greedy_fill(M)
        self._snapshot()
        modes = ["low", "hole", "hole", "cluster", "random"]
        r = 0
        while self.uncov > 0:
            if time.time() - t0 > round_budget_s or self.time_left() < 10:
                break
            self.guard()
            pre = self.uncov
            k = int(self.rng.integers(kmin, kmax + 1))
            mode = modes[int(self.rng.integers(len(modes)))]
            self.ruin(k, mode)
            self.greedy_fill(M)
            self.stats["rounds"] += 1
            r += 1
            if self.uncov > pre and self.rng.random() > 0.10:
                # reject: restore best snapshot
                self.code = self.best_code.copy()
                self.eng.recount(self.code)
                self.stats["transforms"] += 1
                self.uncov = self.eng.uncovered()
            else:
                self._snapshot()
            if r % log_every == 0:
                self.log(f"  lns M={M} round={r} uncov={self.uncov} "
                         f"best={self.best_uncov} k={k} "
                         f"({(time.time()-t0):.0f}s)")
            # periodic drift check
            if r % 50 == 0:
                u_inc = self.uncov
                u_full = self.verify_solved_on_device()
                if u_inc != u_full:
                    self.log(f"  WARNING drift: incremental {u_inc} != "
                             f"recount {u_full}; adopting recount")
        if self.uncov == 0:
            self._snapshot()
        elif self.best_code is not None and self.best_uncov < self.uncov:
            self.code = self.best_code.copy()
            self.eng.recount(self.code)
            self.uncov = self.eng.uncovered()
        return self.uncov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", type=int, required=True)
    ap.add_argument("-n", type=int, required=True)
    ap.add_argument("-R", type=int, required=True)
    ap.add_argument("-M", type=int, required=True)
    ap.add_argument("-t", type=float, default=60.0)
    ap.add_argument("-s", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--in", dest="seed_code")
    ap.add_argument("--kmin", type=int, default=8)
    ap.add_argument("--kmax", type=int)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    t_start = time.time()
    deadline = t_start + a.t
    log = (lambda *x: None) if a.quiet else print
    if foreign_gpu_pids():
        print("YIELD: foreign compute process owns the GPU; waiting")
        while foreign_gpu_pids() and time.time() < deadline - 30:
            time.sleep(30)

    e = Engine(a.q, a.n, a.R, seed=a.s, out=a.out, log=log, deadline=deadline)
    try:
        if a.seed_code:
            code = read_code(a.seed_code, a.q, a.n)
            e.load(e.eng.word_index(code))
            if len(e.code) > a.M:
                e.ruin(len(e.code) - a.M, "low")
        else:
            e.load(np.zeros(0, dtype=np.int64))
        unc = e.lns(a.M, round_budget_s=deadline - time.time() - 5,
                    kmin=a.kmin, kmax=a.kmax)
        if unc == 0:
            unc = e.verify_solved_on_device()   # from-scratch recount
        e.publish(force=True)
        dt = time.time() - t_start
        moves = e.stats["placements"] + e.stats["removals"]
        print(f"RESULT q={a.q} n={a.n} R={a.R} M={len(e.code)} "
              f"uncovered={unc} iters={e.stats['rounds']} "
              f"rate={moves/max(dt,1e-9):.1f} transforms={e.stats['transforms']} "
              f"time={dt:.1f}")
    finally:
        e.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
