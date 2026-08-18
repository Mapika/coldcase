#!/usr/bin/env python3
"""Campaign runner: hunt improved lower bounds for A(n,d,w).

For a target cell (n,d,w) with current best-known lower bound LB:
  1. build seed pool: incumbent code file (data/raw/cwc/...), plus
     lengthen/shorten transforms of neighbours, plus any of our own finds;
  2. try M = LB+1 with rounds of seeded GPU tabu chains; on stagnation,
     re-seed next round from best chain states (population style);
  3. on success: independently verify, save code + certificate, ladder to M+1.

Every found code is written to results/cwc/a{n}.{d}.{w}.{M} (0/1 lines,
same format as Brouwer's site) with a JSON sidecar.
"""
import argparse, json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "verify"))
import cwc
from verify_cwc import verify as verify_code

RESULTS = os.path.join(ROOT, "results", "cwc")
os.makedirs(RESULTS, exist_ok=True)


def load_code_file(path, n):
    words = []
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            if set(line) <= {"0", "1"}:
                words.append(int(line[::-1], 2))
            else:
                words.append(int(line, 0))
    return [w for w in words]


def find_incumbent(n, d, w):
    """Best available seed code for (n,d,w) from mirror or our results."""
    best = None
    for base in (os.path.join(ROOT, "data", "seeds"), RESULTS):
        if not os.path.isdir(base):
            continue
        for fn in os.listdir(base):
            parts = fn.lstrip("a").split(".")
            if len(parts) < 4:
                continue
            try:
                fn_n, fn_d, fn_w = int(parts[0]), int(parts[1]), int(parts[2])
                fn_m = int("".join(ch for ch in parts[3] if ch.isdigit()))
            except ValueError:
                continue
            if (fn_n, fn_d, fn_w) == (n, d, w):
                if best is None or fn_m > best[0]:
                    best = (fn_m, os.path.join(base, fn))
    return best  # (size, path) or None


def neighbour_seeds(n, d, w, rng):
    """Echols-style transforms: shorten (n+1 -> n) and lengthen (n-1 -> n)."""
    seeds = []
    inc = find_incumbent(n + 1, d, w)
    if inc:
        code = load_code_file(inc[1], n + 1)
        # shorten: keep words with bit b == 0, delete that column
        for b in range(n + 1):
            kept = []
            for x in code:
                if not (x >> b) & 1:
                    low = x & ((1 << b) - 1)
                    high = x >> (b + 1)
                    kept.append(low | (high << b))
            if len(kept) >= 3:
                seeds.append(("shorten", kept))
    inc = find_incumbent(n - 1, d, w - 1)
    if inc:
        code = load_code_file(inc[1], n - 1)
        # lengthen: append a 1-bit to every word (weight w-1 -> w, dist keeps >= d)
        seeds.append(("lengthen", [x | (1 << (n - 1)) for x in code]))
    return seeds


def save_result(n, d, w, M, words, meta):
    fn = os.path.join(RESULTS, f"a{n}.{d}.{w}.{M}")
    with open(fn, "w") as f:
        for x in words:
            f.write(format(x, f"0{n}b")[::-1] + "\n")
    with open(fn + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    return fn


def hunt_cell(n, d, w, lb, max_rounds=30, nchains=2048, iters=None,
              tenure_lo=5, tenure_span=10, time_budget_s=600,
              round_secs=45, log=print):
    """Try to push A(n,d,w) beyond lb. Returns highest M achieved (or None).

    If iters is None, per-round iterations are auto-scaled so one round takes
    roughly round_secs (empirical rate ~5e11 candidate-pair ops/s)."""
    rng = np.random.default_rng()
    achieved = None
    M = lb + 1
    t_start = time.time()
    while M <= 2048:
        # ---- build seed pool for this M ----
        pool = []
        inc = find_incumbent(n, d, w)
        if inc:
            pool.append(("incumbent", load_code_file(inc[1], n)))
        pool.extend(neighbour_seeds(n, d, w, rng))
        pool = [(tag, c) for tag, c in pool if len(c) >= min(8, M // 2)]

        best_states = None  # np array of chain states to reseed from
        success = False
        for rnd in range(max_rounds):
            if time.time() - t_start > time_budget_s:
                log(f"  [{n},{d},{w}] M={M}: time budget exhausted")
                return achieved
            inits = np.zeros((nchains, M), dtype=np.uint64)
            ns = 0
            if best_states is not None:
                # 60% of chains resume from best previous states, perturbed
                ns = int(nchains * 0.6)
                idx = rng.integers(0, len(best_states), ns)
                inits[:ns] = best_states[idx]
                # perturb: replace a few random words
                nrepl = max(1, M // 50)
                for c in range(ns):
                    repl = rng.choice(M, size=nrepl, replace=False)
                    inits[c, repl] = cwc.random_cw_words(n, w, nrepl, rng)
            # rest: seeded from pool / random
            for c in range(ns, nchains):
                if pool and rng.random() < 0.85:
                    tag, code = pool[rng.integers(0, len(pool))]
                    arr = np.array(code, dtype=np.uint64)
                    k = min(len(arr), M)
                    keep = rng.choice(len(arr), size=k, replace=False)
                    inits[c, :k] = arr[keep]
                    if M > k:
                        inits[c, k:] = cwc.random_cw_words(n, w, M - k, rng)
                else:
                    inits[c] = cwc.random_cw_words(n, w, M, rng)

            it_budget = iters
            if it_budget is None:
                cand = max(1, w * (n - w))
                it_budget = int(round_secs * 5e11 / (nchains * cand * max(M, 1)))
                it_budget = max(20_000, min(500_000, it_budget))
            r = cwc.run_chains(n, d, w, M, inits, iters=it_budget,
                               tenure_lo=tenure_lo, tenure_span=tenure_span)
            nf = int(r["found"].sum())
            bc = int(r["best_cost"].min())
            log(f"  [{n},{d},{w}] M={M} round {rnd}: found={nf} best_cost={bc}")
            if nf > 0:
                c = int(np.argmax(r["found"]))
                words = [int(x) for x in r["best_words"][c]]
                errors, mind = verify_code(words, n, d, w)
                if errors:
                    log(f"  !! verifier rejected GPU find: {errors}")
                else:
                    meta = {"n": n, "d": d, "w": w, "M": M, "min_dist": mind,
                            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "method": "seeded GPU tabu (longshots)",
                            "prev_lb": lb, "round": rnd, "seed": int(r["seed"])}
                    fn = save_result(n, d, w, M, words, meta)
                    log(f"  ** A({n},{d},{w}) >= {M} verified, saved {fn}")
                    achieved = M
                    success = True
                    break
            # reseed from best chains
            order = np.argsort(r["best_cost"])
            top = order[: max(32, nchains // 16)]
            best_states = r["best_words"][top]
        if not success:
            return achieved
        M += 1
    return achieved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("d", type=int)
    ap.add_argument("w", type=int)
    ap.add_argument("lb", type=int, help="current best-known lower bound")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--chains", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=300_000)
    ap.add_argument("--budget", type=int, default=600, help="seconds")
    args = ap.parse_args()
    res = hunt_cell(args.n, args.d, args.w, args.lb, max_rounds=args.rounds,
                    nchains=args.chains, iters=args.iters,
                    time_budget_s=args.budget)
    print(f"RESULT A({args.n},{args.d},{args.w}): "
          f"{'improved to ' + str(res) if res else 'no improvement'} (lb was {args.lb})")


if __name__ == "__main__":
    main()
