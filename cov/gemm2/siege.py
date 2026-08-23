#!/usr/bin/env python3
"""siege.py — descend-from-incumbent siege on monster cells with the GPU
transform engine (gpuchain.Engine).

Per cell: seed from the best recorded file in cov/results/ (or exact-greedy
from scratch), then remove-and-repair descent: ruin to M-step, LNS-repair,
record every solved M through record_gate.py (= campaign.record, the one
verification gate) in a niced background process, and keep descending while
verification runs.  Nothing is called a record here: record_gate decides.

Logs to cov/gemm2/SIEGE.md.  Yields the GPU to foreign processes (Engine.guard).

Usage:
  python3 siege.py -q 7 -n 10 -R 5 --hours 2 [--scratch] [--floor 800]
"""
import argparse
import datetime
import glob
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COV = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, COV)

from gpuchain import Engine, read_code, write_code_atomic  # noqa: E402

RESULTS = os.path.join(COV, "results")
SIEGE_LOG = os.path.join(HERE, "SIEGE.md")
SCRATCH = os.environ.get(
    "GPUCHAIN_SCRATCH",
    "/tmp/claude-1000/-lambda-nfs-new-fs-longshots/"
    "3ae6d111-07fa-4880-8ce4-cca841595d21/scratchpad/gpc")


def log_md(line):
    ts = datetime.datetime.now().strftime("%m-%d %H:%M")
    with open(SIEGE_LOG, "a") as f:
        f.write(f"- `{ts}` {line}\n")
    print(f"[siege] {line}", flush=True)


def best_recorded(q, n, R):
    best = None
    for p in glob.glob(os.path.join(RESULTS, f"K{q}_{n}_{R}_M*.txt")):
        m = re.search(r"_M(\d+)\.txt$", p)
        if m:
            M = int(m.group(1))
            if best is None or M < best[0]:
                best = (M, p)
    return best


class Recorder:
    """Fire-and-track background record_gate.py verifications.

    Never blocks the descent: at most MAXRUN gates run concurrently; excess
    candidates wait in a small queue where SUPERSEDED entries (a larger M
    while a smaller cover is already queued) are dropped — they were never
    claimed as records, and the verified trail keeps every M that matters.
    (On 1e10 cells one verification is ~an hour; the old <=2-running
    blocking loop would have stalled the siege behind the verifiers.)"""

    MAXRUN = 3
    MAXWAIT = 2

    def __init__(self, q, n, R):
        self.q, self.n, self.R = q, n, R
        self.procs = []          # (M, path, Popen)
        self.waiting = []        # (M, path) FIFO
        self.done = []           # (M, ok)

    def submit(self, M, src_path):
        dst = os.path.join(SCRATCH, f"cand_K{self.q}_{self.n}_{self.R}_M{M}.txt")
        shutil.copy(src_path, dst)
        self.waiting.append((M, dst))
        self.reap()

    def _pump(self):
        while len(self.waiting) > self.MAXWAIT + 1:
            drop = self.waiting.pop(0)      # superseded by a smaller cover
            log_md(f"K{self.q}({self.n},{self.R}) M={drop[0]}: gate submission "
                   f"superseded (never claimed); file kept at {drop[1]}")
        while self.waiting and \
                sum(1 for (_, _, p) in self.procs if p.poll() is None) < \
                self.MAXRUN:
            M, dst = self.waiting.pop(0)
            p = subprocess.Popen(
                [sys.executable, os.path.join(HERE, "record_gate.py"),
                 str(self.q), str(self.n), str(self.R), str(M), dst],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.procs.append((M, dst, p))

    def reap(self, wait=False):
        while True:
            rest = []
            for (M, dst, p) in self.procs:
                if p.poll() is None and not wait:
                    rest.append((M, dst, p))
                    continue
                out, _ = p.communicate()
                ok = p.returncode == 0
                self.done.append((M, ok))
                tag = "VERIFIED+RECORDED" if ok else "REJECTED BY GATE"
                log_md(f"K{self.q}({self.n},{self.R}) M={M}: {tag}")
                if not ok:
                    log_md(f"  gate output tail: "
                           f"{out.strip().splitlines()[-3:]}")
            self.procs = rest
            self._pump()
            if not wait or (not self.procs and not self.waiting):
                break
            time.sleep(5)


def keri_ub(q, n, R):
    try:
        import constructions as C
        e = C.load_bounds().get((q, n, R))
        return e["ub"] if e else None
    except Exception:
        return None


def descend(q, n, R, hours, scratch_build, floor, seed, notch_budget,
            seed_file=None, step0=4, step_cap=1024, ruin_extra=0,
            kmin=8, kmax=None):
    t_end = time.time() + hours * 3600
    outp = os.path.join(SCRATCH, f"live_K{q}_{n}_{R}.txt")
    rec = Recorder(q, n, R)
    ub = keri_ub(q, n, R)
    br = best_recorded(q, n, R)
    record_below = min(x for x in [ub, br[0] if br else None, 10 ** 9] if x)
    log_md(f"K{q}({n},{R}): gate submissions only below M={record_below} "
           f"(Keri ub={ub}, best recorded={br[0] if br else None})")
    e = Engine(q, n, R, seed=seed, out=outp, log=print, deadline=t_end)
    solved_words = None          # host snapshot of last verified-on-device cover
    try:
        start = best_recorded(q, n, R)
        if seed_file:
            log_md(f"K{q}({n},{R}): seeding from {os.path.basename(seed_file)}")
            e.load(e.eng.word_index(read_code(seed_file, q, n)))
        elif start and not scratch_build:
            M0, path = start
            log_md(f"K{q}({n},{R}): seeding from {os.path.basename(path)}")
            e.load(e.eng.word_index(read_code(path, q, n)))
        else:
            log_md(f"K{q}({n},{R}): exact-greedy build from scratch")
            e.load(np.zeros(0, dtype=np.int64))
            e.greedy_fill(target_m=10 ** 7)
        if e.uncov != 0:
            log_md(f"K{q}({n},{R}): start not a cover (uncov={e.uncov}); "
                   f"LNS to repair at M={len(e.code)}")
            e.lns(len(e.code), round_budget_s=notch_budget * 4)
        if e.uncov != 0:
            log_md(f"K{q}({n},{R}): could not establish a cover; aborting cell")
            return
        e.peel()
        assert e.verify_solved_on_device() == 0
        solved_words = e.eng.index_word(e.code)
        M = len(e.code)
        write_code_atomic(outp, solved_words)
        log_md(f"K{q}({n},{R}): cover established at M={M}")
        if M < record_below:
            rec.submit(M, outp)
            record_below = M

        step, fails_at_1 = step0, 0
        while time.time() < t_end and M > floor and fails_at_1 < 3:
            target = M - step
            t0 = time.time()
            # ruin past the deficit so the first repair has real slots to
            # re-optimize, instead of being forced into pure swaps
            e.ruin(len(e.code) - target + ruin_extra,
                   "cluster" if ruin_extra else "low")
            budget = min(notch_budget * (2 ** min(fails_at_1, 2)),
                         t_end - time.time() - 30)
            if budget < 20:
                break
            unc = e.lns(target, round_budget_s=budget, kmin=kmin, kmax=kmax)
            if unc == 0 and e.verify_solved_on_device() == 0:
                e.peel()
                if e.verify_solved_on_device() != 0:
                    log_md("  PEEL BROKE COVER (bug!) — restoring")
                    e.load(solved_words)
                    break
                solved_words = e.eng.index_word(e.code)
                M = len(e.code)
                write_code_atomic(outp, solved_words)
                dt = time.time() - t0
                sub = M < record_below
                log_md(f"K{q}({n},{R}): SOLVED M={M} in {dt:.0f}s "
                       f"(step={step}){'; gate submitted' if sub else ''}")
                if sub:
                    rec.submit(M, outp)
                    record_below = M
                fails_at_1 = 0
                if dt < budget * 0.25:
                    step = min(step * 2, step_cap)
            else:
                log_md(f"K{q}({n},{R}): notch M={target} FAILED "
                       f"(best uncov={e.best_uncov}, step={step}, "
                       f"budget={budget:.0f}s)")
                if step == 1:
                    fails_at_1 += 1
                step = max(1, step // 2)
                e.load(solved_words)          # restore last solved cover
            rec.reap()
        log_md(f"K{q}({n},{R}): descent ended at M={M} "
               f"(floor={floor}, wall_fails={fails_at_1}, "
               f"stats={e.stats}, paused={e._paused_s:.0f}s)")
    finally:
        e.close()
        rec.reap(wait=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", type=int, required=True)
    ap.add_argument("-n", type=int, required=True)
    ap.add_argument("-R", type=int, required=True)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--floor", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scratch", action="store_true",
                    help="ignore recorded seeds; greedy from scratch")
    ap.add_argument("--notch-budget", type=float, default=300.0)
    ap.add_argument("--seed-file")
    ap.add_argument("--step0", type=int, default=4)
    ap.add_argument("--ruin-extra", type=int, default=0)
    ap.add_argument("--kmin", type=int, default=8)
    ap.add_argument("--kmax", type=int)
    a = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)
    os.nice(5)
    descend(a.q, a.n, a.R, a.hours, a.scratch, a.floor, a.seed,
            a.notch_budget, seed_file=a.seed_file, step0=a.step0,
            ruin_extra=a.ruin_extra, kmin=a.kmin, kmax=a.kmax)


if __name__ == "__main__":
    main()
