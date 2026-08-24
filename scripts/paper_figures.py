#!/usr/bin/env python3
"""Generate the paper's figures from the verified result state.

Fig 1 (fig_slack.pdf):  UB improvement % vs slack of the previous bound
                        over the sphere-covering bound, marker by method.
Fig 2 (fig_family82.pdf): the (8,2) family q=6..21 -- previous LB, new
                        certified LB, relative to the sphere-covering bound.

Inputs: cov/results/final_records.json, cov/bounds.json,
        cov/lb/results/lb_master.json
Output: paper/covering/fig_slack.pdf, paper/covering/fig_family82.pdf
"""
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# cells produced by the transform engine (Section 5); the rest are Section 4
GPU_CELLS = {(7,10,5), (8,10,5), (9,9,5), (9,10,5), (10,9,5), (10,10,5),
             (10,10,6)}

def ball(q, n, R):
    return sum(math.comb(n, i) * (q - 1) ** i for i in range(R + 1))

def main():
    recs = json.load(open("cov/results/final_records.json"))
    bounds = {(e["q"], e["n"], e["R"]): e for e in
              json.load(open("cov/bounds.json"))["entries"]}

    # ---- Fig 1: improvement vs slack ----
    xs_cpu, ys_cpu, xs_gpu, ys_gpu = [], [], [], []
    for r in recs:
        q, n, R, M = r["q"], r["n"], r["R"], r["ours"]
        e = bounds[(q, n, R)]
        slack = e["ub"] * ball(q, n, R) / q ** n     # previous UB / sphere bound
        pct = 100.0 * (e["ub"] - M) / e["ub"]
        if (q, n, R) in GPU_CELLS:
            xs_gpu.append(slack); ys_gpu.append(pct)
        else:
            xs_cpu.append(slack); ys_cpu.append(pct)

    fig, ax = plt.subplots(figsize=(4.9, 3.3))
    ax.scatter(xs_cpu, ys_cpu, s=26, marker="o", facecolors="none",
               edgecolors="black", linewidths=0.9, label="local search (Sec. 4)")
    ax.scatter(xs_gpu, ys_gpu, s=30, marker="s", color="black",
               label="transform engine (Sec. 5)")
    ax.set_xscale("log")
    ax.set_xlabel(r"slack of previous bound: $\mathrm{UB}_{\mathrm{prev}}\cdot|B_R|/q^n$")
    ax.set_ylabel("improvement (%)")
    ax.set_xticks([1, 2, 4, 8, 12])
    ax.set_xticklabels(["1", "2", "4", "8", "12"])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("paper/covering/fig_slack.pdf")
    print("wrote paper/covering/fig_slack.pdf "
          f"({len(xs_cpu)}+{len(xs_gpu)} points)")

    # ---- Fig 2: the (8,2) family ----
    lb_rows = [r for r in json.load(open("cov/lb/results/lb_master.json"))
               if r.get("improves_best_known") and r["n"] == 8 and r["R"] == 2]
    lb_rows.sort(key=lambda r: r["q"])
    qs = [r["q"] for r in lb_rows]
    sphere = [q ** 8 / ball(q, 8, 2) for q in qs]
    prev = [r["best_known_lb"] for r in lb_rows]
    new = [r["K_lower_bound"] for r in lb_rows]

    fig, ax = plt.subplots(figsize=(4.9, 3.3))
    ax.plot(qs, [p / s for p, s in zip(prev, sphere)], "o--", color="grey",
            mfc="none", ms=5, lw=0.9, label="previous LB")
    ax.plot(qs, [v / s for v, s in zip(new, sphere)], "s-", color="black",
            ms=4, lw=1.1, label="new certified LB")
    ax.axhline(1.0, color="black", lw=0.6, ls=":")
    ax.text(qs[-1], 1.001, "sphere-covering bound", ha="right", va="bottom",
            fontsize=7)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"bound / sphere-covering bound")
    ax.set_xticks(qs[::3] + [qs[-1]])
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("paper/covering/fig_family82.pdf")
    print(f"wrote paper/covering/fig_family82.pdf ({len(qs)} q values)")

if __name__ == "__main__":
    main()
