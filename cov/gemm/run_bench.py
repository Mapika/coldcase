#!/usr/bin/env python3
"""Matched-budget comparison: GEMM-on-tensor-cores vs. the CPU state of the art.

Protocol
--------
* Both engines start from the SAME fixed code file (`cov/opt/seeds_*.txt`), so
  every run of every engine begins in the identical state and the seed drives
  only the search.  This is the protocol of cov/opt/METHODS.md sec.F.
* Two quantities per run: time to a fixed target uncovered count, and the best
  uncovered count reached inside a fixed budget.
* The CPU budget is process CPU time (`--cpu`), because the host is shared with
  a production sweep (load average > 100); the GPU budget is wall clock,
  because there is no such thing as "GPU CPU time" and the GPU is likewise
  shared.  Both are therefore *pessimistic for the shared resource*, and the
  contention is reported alongside.
* Every code file either engine writes is re-read from disk and checked by
  cov/verify_cov.py with both exhaustive methods; a run whose reported
  uncovered count disagrees with the verifier is a hard failure.

Usage:
  python3 run_bench.py --cell K6_8_4 --seeds 5 --budget 25 --out results.csv
"""
import argparse, csv, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
COV = os.path.dirname(HERE)
OPT = os.path.join(COV, "opt")

CELLS = {
    # name: q n R M seedfile target  (targets from cov/opt/METHODS.md sec.F)
    "K6_6_3": dict(q=6, n=6, R=3, M=41, seed="seeds_K6_6_3_M41.txt", target=30),
    "K6_8_4": dict(q=6, n=8, R=4, M=169, seed="seeds_K6_8_4_M169.txt", target=20),
    "K8_9_4": dict(q=8, n=9, R=4, M=2944, seed="seeds_K8_9_4_M2944.txt",
                   target=10800),
}

RES = re.compile(r"RESULT (.*)")


def parse(out):
    m = RES.search(out)
    if not m:
        return None
    d = {}
    for kv in m.group(1).split():
        k, _, v = kv.partition("=")
        d[k] = v
    return d


def verify(path, q, n, R):
    """Both exhaustive methods must agree; returns the uncovered count."""
    p = subprocess.run([sys.executable, os.path.join(COV, "verify_cov.py"), path,
                        "-q", str(q), "-n", str(n), "-R", str(R),
                        "--method", "both"], capture_output=True, text=True)
    vals = re.findall(r"^method [a-z]+ *: uncovered=(\d+)", p.stdout, re.M)
    if not vals or len(set(vals)) != 1:
        return None
    return int(vals[0])


def run_cpu(c, seed, budget, variant, outdir):
    out = os.path.join(outdir, f"cpu_{variant.replace(' ','')}_{seed}.txt")
    cmd = ["nice", "-n", "12", os.path.join(OPT, "covsearch2"),
           "-q", str(c["q"]), "-n", str(c["n"]), "-R", str(c["R"]),
           "-M", str(c["M"]), "--in", os.path.join(OPT, c["seed"])] \
        + variant.split() + \
        ["--target", str(c["target"]), "--cpu", str(budget), "-t", "999999",
         "-s", str(seed), "--threads", "1", "--quiet", "--out", out]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=OPT)
    wall = time.perf_counter() - t0
    d = parse(p.stdout)
    if d is None:
        sys.stderr.write(p.stdout + p.stderr)
        return None
    return dict(engine="cpu " + variant, seed=seed, best=int(d["uncovered"]),
                iters=int(d["iters"]), cost=float(d["cpu"]), wall=wall,
                ttt=float(d["ttt_cpu"]) if d.get("ttt_cpu") else None,
                warmup=0.0, file=out)


def run_gpu(c, seed, budget, cand, outdir, warmup=12, extra=()):
    out = os.path.join(outdir, f"gpu_c{cand}_{seed}.txt")
    cmd = [sys.executable, os.path.join(HERE, "focused.py"),
           "-q", str(c["q"]), "-n", str(c["n"]), "-R", str(c["R"]),
           "-M", str(c["M"]), "--in", os.path.join(OPT, c["seed"]),
           "--target", str(c["target"]), "--time", str(budget),
           "--iters", "100000000", "-s", str(seed), "--cand", str(cand),
           "--warmup", str(warmup), "--quiet", "--out", out] + list(extra)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    wall = time.perf_counter() - t0
    d = parse(p.stdout)
    if d is None:
        sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:])
        return None
    return dict(engine=f"gpu cand={cand}", seed=seed, best=int(d["uncovered"]),
                iters=int(d["iters"]), cost=float(d["time"]), wall=wall,
                ttt=float(d["ttt_wall"]) if d.get("ttt_wall") else None,
                warmup=float(d.get("warmup", 0)),
                ms_med=float(d.get("ms_med", 0)),
                moves=int(d.get("moves_eval", 0)), file=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="K6_8_4")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=float, default=25)
    ap.add_argument("--cand", default="32")
    ap.add_argument("--cpu-variants", default="--preset p5b --wide;--preset p5b")
    ap.add_argument("--outdir", default="/tmp/covgemm_bench")
    ap.add_argument("--csv", default="")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    c = CELLS[a.cell]
    os.makedirs(a.outdir, exist_ok=True)
    rows = []
    jobs = []
    for s in range(1, a.seeds + 1):
        for v in a.cpu_variants.split(";"):
            if v.strip():
                jobs.append(("cpu", v.strip(), s))
        for cd in a.cand.split(","):
            jobs.append(("gpu", int(cd), s))
    for kind, var, s in jobs:
        r = run_cpu(c, s, a.budget, var, a.outdir) if kind == "cpu" \
            else run_gpu(c, s, a.budget, var, a.outdir)
        if r is None:
            print(f"  FAILED {kind} {var} seed {s}")
            continue
        if not a.no_verify:
            v = verify(r["file"], c["q"], c["n"], c["R"])
            r["verified"] = v
            if v != r["best"]:
                print(f"  *** VERIFIER DISAGREES: {r['engine']} seed {s} "
                      f"solver={r['best']} verifier={v}")
                sys.exit(2)
        rows.append(r)
        print(f"  {r['engine']:20s} s={s} best={r['best']:6d} "
              f"iters={r['iters']:7d} cost={r['cost']:7.2f}s "
              f"ttt={r['ttt']} verified=ok", flush=True)
    if a.csv:
        keys = sorted({k for r in rows for k in r})
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
        print("wrote", a.csv)

    import statistics as st
    print(f"\n{a.cell}  budget={a.budget}  target={c['target']}")
    print(f"{'engine':22s} {'n':>3s} {'med best':>9s} {'med ttt':>9s} "
          f"{'solved':>7s} {'med iters':>10s}")
    for e in sorted({r["engine"] for r in rows}):
        g = [r for r in rows if r["engine"] == e]
        tt = [r["ttt"] for r in g if r["ttt"] is not None]
        print(f"{e:22s} {len(g):3d} {st.median(r['best'] for r in g):9.1f} "
              f"{(st.median(tt) if tt else float('nan')):9.3f} "
              f"{len(tt):3d}/{len(g):<3d} "
              f"{st.median(r['iters'] for r in g):10.0f}")


if __name__ == "__main__":
    main()
