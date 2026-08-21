#!/usr/bin/env python3
"""Turn results_micro.csv / results_ttt.csv into the tables used in METHODS.md.

Statistics used, and why:

* Medians and quartiles, not means.  Time-to-target distributions of
  stochastic local search are strongly right-skewed; a mean is dominated by
  the tail and is not the quantity a practitioner cares about.
* A distribution-free bootstrap confidence interval for the ratio of
  medians (10000 resamples), because the two samples are independent runs
  and no parametric form is warranted.
* The Mann-Whitney U statistic as a rank-based effect size
  (P(X < Y), the probability that a random run of the faster variant beats a
  random run of the slower one), which needs no distributional assumption
  and tolerates the censored runs as long as they are ranked last.
"""
import csv, math, os, random, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def q(v, p):
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * p
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def boot_ratio_ci(a, b, n=10000, seed=1):
    """CI for median(a)/median(b) by the percentile bootstrap."""
    rng = random.Random(seed)
    if not a or not b:
        return (float("nan"), float("nan"))
    out = []
    for _ in range(n):
        ma = statistics.median([a[rng.randrange(len(a))] for _ in a])
        mb = statistics.median([b[rng.randrange(len(b))] for _ in b])
        if mb > 0:
            out.append(ma / mb)
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def mannwhitney_p_less(a, b):
    """P(X_a < X_b) + 0.5 P(=), the common-language effect size."""
    if not a or not b:
        return float("nan")
    c = 0.0
    for x in a:
        for y in b:
            c += 1.0 if x < y else (0.5 if x == y else 0.0)
    return c / (len(a) * len(b))


MINK = 4   # never trust an E(t) estimate built on fewer than this many successes


def restart_E(times, t, n):
    """Expected total CPU time to reach the target when the search is
    restarted every t CPU seconds:  E(t) = E[min(T,t)] / P(T <= t).
    Censored runs enter correctly as "not solved by t" for any t <= budget."""
    solved = [x for x in times if x <= t]
    k = len(solved)
    if k < MINK:
        return None
    return ((sum(solved) + (n - k) * t) / n) / (k / n)


def restart_best(times, budget):
    """min over a quantile grid of E(t), and the no-restart baseline.

    Both the grid restriction and the MINK floor exist because the naive
    estimator is badly biased downwards at small t: with n runs, E(t) at
    t = the smallest observation equals n times that observation, and
    minimising over a fine grid then selects the luckiest order statistic.
    """
    n = len(times)
    obs = sorted(x for x in times if x <= budget)
    if len(obs) < MINK:
        return None, None, None
    grid = sorted(set(q(obs, p) for p in [i / 20 for i in range(1, 21)]) | {budget})
    cands = [(t, restart_E(times, t, n)) for t in grid]
    cands = [(t, e) for t, e in cands if e is not None]
    if not cands:
        return None, None, None
    bt, be = min(cands, key=lambda z: z[1])
    e_inf = restart_E(times, budget, n)
    return bt, be, e_inf


def restart_gain_null(times, budget, nsim=2000, seed=3):
    """Calibrate the apparent restart gain against the memoryless null.

    min_t E(t) <= E(budget) holds by construction, so the "gain" is always
    >= 1 and a plain bootstrap interval can never exclude 1.  What must be
    asked instead is whether the gain exceeds what the SAME procedure
    extracts from an exponential (memoryless) run-time distribution with the
    same mean, for which restarts are exactly neutral in truth.  Returns
    (95th percentile of the null gain, one-sided p-value).
    """
    ok = [t for t in times if t <= budget]
    if len(ok) < MINK:
        return None
    mean = statistics.mean(ok)
    rng = random.Random(seed)
    null = []
    for _ in range(nsim):
        s = [rng.expovariate(1.0 / mean) for _ in times]
        s = [x if x <= budget else budget * 1e6 for x in s]
        bt, be, e_inf = restart_best(s, budget)
        if be and e_inf:
            null.append(e_inf / be)
    if not null:
        return None
    null.sort()
    bt, be, e_inf = restart_best(times, budget)
    obs = e_inf / be if be else 1.0
    p = sum(1 for x in null if x >= obs) / len(null)
    return null[int(0.95 * len(null))], p


def micro():
    path = os.path.join(HERE, "results_micro.csv")
    if not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["cpu"] = float(r["cpu"])
    cells = []
    for r in rows:
        if r["cell"] not in cells:
            cells.append(r["cell"])
    variants = []
    for r in rows:
        if r["variant"] not in variants:
            variants.append(r["variant"])
    print("\n## Table 1 -- fixed-iteration CPU time (identical search trajectory)\n")
    lines = []
    hdr = "| variant | " + " | ".join(f"{c} s (x)" for c in cells) + " |"
    print(hdr)
    print("|" + "---|" * (len(cells) + 1))
    base = {}
    for c in cells:
        v = [r["cpu"] for r in rows if r["cell"] == c and r["variant"] == "base"]
        base[c] = statistics.median(v) if v else float("nan")
    for var in variants:
        cols = []
        for c in cells:
            v = [r["cpu"] for r in rows if r["cell"] == c and r["variant"] == var]
            if not v:
                cols.append("--")
                continue
            m = statistics.median(v)
            cols.append(f"{m:.3f} ({base[c]/m:.2f}x)")
            lines.append(dict(cell=c, variant=var, median_cpu=round(m, 4),
                              q1=round(q(v, .25), 4), q3=round(q(v, .75), 4),
                              n=len(v), speedup=round(base[c] / m, 3)))
        print(f"| {var} | " + " | ".join(cols) + " |")
    with open(os.path.join(HERE, "table1_micro.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lines[0]))
        w.writeheader(); w.writerows(lines)
    # spread
    print("\nspread (IQR/median) of the CPU measurement, per cell:")
    for c in cells:
        sp = []
        for var in variants:
            v = [r["cpu"] for r in rows if r["cell"] == c and r["variant"] == var]
            if len(v) > 3:
                sp.append((q(v, .75) - q(v, .25)) / statistics.median(v))
        if sp:
            print(f"  {c}: median IQR/median over variants = {statistics.median(sp):.3f}"
                  f" (max {max(sp):.3f}, n_runs/variant={len(v)})")


def ttt():
    path = os.path.join(HERE, "results_ttt.csv")
    if not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["ttt_cpu"] = float(r["ttt_cpu"]); r["ttt_it"] = int(r["ttt_it"])
        r["budget"] = float(r["budget"]); r["best"] = int(r["best"])
    cells, variants = [], []
    for r in rows:
        if r["cell"] not in cells: cells.append(r["cell"])
        if r["variant"] not in variants: variants.append(r["variant"])

    print("\n## Table 2 -- CPU time to target, 20 seeds per configuration\n")
    print("| cell | variant | solved/n | median | Q1 | Q3 | speedup vs base "
          "| 95% CI | P(faster) | median iters |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    out = []
    for c in cells:
        bt = [r["ttt_cpu"] for r in rows
              if r["cell"] == c and r["variant"] == "base" and r["ttt_cpu"] >= 0]
        bmed = statistics.median(bt) if bt else float("nan")
        for var in variants:
            rs = [r for r in rows if r["cell"] == c and r["variant"] == var]
            if not rs:
                continue
            ok = [r["ttt_cpu"] for r in rs if r["ttt_cpu"] >= 0]
            its = [r["ttt_it"] for r in rs if r["ttt_it"] >= 0]
            # censored runs get the budget as a lower bound, ranked last
            cens = [r["budget"] for r in rs if r["ttt_cpu"] < 0]
            allv = ok + cens
            med = statistics.median(ok) if len(ok) > len(rs) / 2 else float("nan")
            lo, hi = boot_ratio_ci(bt, ok) if (bt and ok) else (float("nan"),) * 2
            pf = mannwhitney_p_less(ok + cens, bt + [r["budget"] for r in rows
                 if r["cell"] == c and r["variant"] == "base" and r["ttt_cpu"] < 0])
            sp = bmed / med if med == med and med > 0 else float("nan")
            # boot_ratio_ci(bt, ok) resamples median(base)/median(variant),
            # which IS the speedup, so the interval is used as-is.
            print(f"| {c} | {var} | {len(ok)}/{len(rs)} | {med:.3f} | "
                  f"{q(ok,.25):.3f} | {q(ok,.75):.3f} | "
                  f"{sp:.2f}x | [{lo:.2f},{hi:.2f}] | "
                  f"{pf:.2f} | {statistics.median(its) if its else float('nan'):.0f} |")
            out.append(dict(cell=c, variant=var, n=len(rs), solved=len(ok),
                            median_ttt_cpu=round(med, 4) if med == med else "",
                            q1=round(q(ok, .25), 4) if ok else "",
                            q3=round(q(ok, .75), 4) if ok else "",
                            speedup_vs_base=round(sp, 3) if sp == sp else "",
                            ci_lo=round(lo, 3) if lo == lo else "",
                            ci_hi=round(hi, 3) if hi == hi else "",
                            p_faster=round(pf, 3) if pf == pf else "",
                            median_ttt_iters=statistics.median(its) if its else ""))
    with open(os.path.join(HERE, "table2_ttt.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)

    # ---- paired analysis -------------------------------------------------
    # Implementation-only variants follow the SAME trajectory as base from the
    # same seed, so ttt_it matches seed by seed and the per-seed CPU ratio is a
    # paired measurement with the search variance removed entirely.
    print("\n## Table 2b -- paired per-seed speedup (trajectory-identical variants)\n")
    print("| cell | variant | seeds paired | iters identical | median ratio | "
          "min | max | 95% CI of median ratio |")
    print("|---|---|---|---|---|---|---|---|")
    for c in cells:
        bmap = {r["seed"]: r for r in rows
                if r["cell"] == c and r["variant"] == "base" and r["ttt_cpu"] >= 0}
        for var in variants:
            if var == "base":
                continue
            rat, same = [], 0
            for r in rows:
                if r["cell"] != c or r["variant"] != var or r["ttt_cpu"] < 0:
                    continue
                b = bmap.get(r["seed"])
                if not b:
                    continue
                rat.append(b["ttt_cpu"] / r["ttt_cpu"])
                same += (b["ttt_it"] == r["ttt_it"])
            if len(rat) < 3:
                continue
            rng = random.Random(5)
            bs = sorted(statistics.median([rat[rng.randrange(len(rat))]
                                           for _ in rat]) for _ in range(5000))
            print(f"| {c} | {var} | {len(rat)} | {same}/{len(rat)} | "
                  f"{statistics.median(rat):.2f}x | {min(rat):.2f} | {max(rat):.2f} | "
                  f"[{bs[int(.025*len(bs))]:.2f},{bs[int(.975*len(bs))]:.2f}] |")
    print("\nThe `iters identical` column is a correctness statement, not a "
          "performance one: it counts the seeds on which the variant reached "
          "the target at exactly the same iteration as the reference "
          "implementation. Where it is n/n the two searches are the same "
          "search and the ratio is a pure engineering speedup.")

    print("\n## Restart analysis (empirical run-time distribution)\n")
    print("| cell | variant | n | cv | best restart interval | E(best) | E(no restart) "
          "| gain | null 95% | p |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        for var in ("base", "p5"):
            rs = [r for r in rows if r["cell"] == c and r["variant"] == var]
            if not rs:
                continue
            budget = rs[0]["budget"]
            # censored runs enter as "not solved by any t <= budget"
            times = [r["ttt_cpu"] if r["ttt_cpu"] >= 0 else budget * 1e6 for r in rs]
            bt, be, e_inf = restart_best(times, budget)
            ok = [t for t in times if t <= budget]
            cv = (statistics.pstdev(ok) / statistics.mean(ok)) if len(ok) > 2 else float("nan")
            if be is None:
                print(f"| {c} | {var} | {len(rs)} | {cv:.2f} | -- | -- | -- | -- | -- | -- |")
                continue
            nn = restart_gain_null(times, budget)
            print(f"| {c} | {var} | {len(rs)} | {cv:.2f} | {bt:.2f}s | {be:.2f}s | "
                  f"{e_inf:.2f}s | {e_inf/be:.2f}x | "
                  + (f"{nn[0]:.2f}x | {nn[1]:.3f} |" if nn else "-- | -- |"))
    print("\n`cv` is the coefficient of variation of the (uncensored) "
          "time-to-target sample; cv = 1 is the exponential/memoryless case, in "
          "which restarts are exactly neutral, and cv > 1 means a heavy right "
          "tail that restarts could cut. `gain` = E(no restart)/E(best restart "
          "interval). Because min_t E(t) <= E(budget) by construction, the gain "
          "is always >= 1 even for data where restarts truly do nothing, so it "
          "is calibrated against the SAME procedure applied to simulated "
          "exponential run times with the same mean: `null 95%` is the 95th "
          "percentile of that null and `p` the one-sided p-value. p > 0.05 "
          "means the apparent gain is just the selection bias of minimising "
          "over a grid.")


if __name__ == "__main__":
    micro()
    ttt()
