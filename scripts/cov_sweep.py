#!/usr/bin/env python3
"""q=8..21 covering-code sweep driver.

Sequentially attacks cells from cov/sweep_targets.json (pre-sorted by gap
ratio) via cov/campaign.py descend-from-UB runs. First pass gives each cell a
modest budget; cells that improve are queued for a deepening pass.

Constraints honored: CPU-only, nice(10), workers memory-guarded (2*q^n bytes
per worker, total budget 350 GB, leaving ample host RAM for other users).
Logs to results/cov_sweep.log; per-cell outcome to results/cov_sweep_state.json.
"""
import json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
TARGETS = json.load(open("cov/sweep_targets.json"))
STATE_F = "results/cov_sweep_state.json"
LOG = open("results/cov_sweep.log", "a", buffering=1)

state = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {}


def log(msg):
    LOG.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}\n")


def best_recorded(q, n, R):
    import glob, re
    best = None
    for f in glob.glob(f"cov/results/K{q}_{n}_{R}_M*.txt"):
        m = int(re.search(r"_M(\d+)\.txt", f).group(1))
        best = m if best is None else min(best, m)
    return best


def run_cell(t, budget_s, rounds):
    q, n, R, ub = t["q"], t["n"], t["R"], t["ub"]
    key = f"{q},{n},{R}"
    start_from = best_recorded(q, n, R) or ub
    workers = min(t["workers"], 52)
    per_run_t = max(240, budget_s // max(1, rounds))
    log(f"CELL K{q}({n},{R}) lb={t['lb']} ub={ub} start={start_from} "
        f"workers={workers} budget={budget_s}s")
    cmd = ["nice", "-n", "10", "python3", "cov/campaign.py",
           "-q", str(q), "-n", str(n), "-R", str(R),
           "--descend-from", str(start_from),
           "--workers", str(workers), "-t", str(per_run_t),
           "--rounds", str(rounds),
           "--extra", "--preset", "p5b"]
    if t["space"] > 5e8:
        cmd += ["--st", "2"]
    env = dict(os.environ, COVSEARCH_BIN=os.path.join(ROOT, "cov", "opt", "covsearch2"))
    try:
        subprocess.run(cmd, timeout=budget_s + 300, env=env,
                       stdout=LOG, stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired:
        log(f"  timeout K{q}({n},{R})")
    after = best_recorded(q, n, R)
    improved = after is not None and after < ub
    state[key] = {"ub": ub, "achieved": after, "improved": improved,
                  "ts": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(state, open(STATE_F, "w"), indent=0)
    if improved:
        log(f"  *** RECORD K{q}({n},{R}): {ub} -> {after}")
    return improved


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "pass1"
    if phase == "pass1":
        for t in TARGETS:
            key = f"{t['q']},{t['n']},{t['R']}"
            if key in state:
                continue
            # budget scaled by search-space size
            budget = 480 if t["space"] < 1e8 else (720 if t["space"] < 1e9 else 1200)
            run_cell(t, budget, rounds=2)
        log("PASS1 COMPLETE")
    elif phase == "deepen":
        movers = [t for t in TARGETS
                  if state.get(f"{t['q']},{t['n']},{t['R']}", {}).get("improved")]
        log(f"DEEPEN: {len(movers)} movers")
        for t in movers:
            run_cell(t, 3600, rounds=3)
        log("DEEPEN COMPLETE")


if __name__ == "__main__":
    main()
