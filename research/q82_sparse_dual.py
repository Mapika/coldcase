#!/usr/bin/env python3
"""Search for small dual supports that already prove a weak q-family bound.

This is a discovery script, not a proof.  It keeps the exact PSD block rays from
an existing certified dual, re-optimizes their nonnegative scales together with
all linear multipliers, and asks how few PSD blocks are needed to reach

    sphere(q) * (1 + 1/(6 q^2)).

For each selected support the final value is re-rounded and checked by the
existing exact checker through solve_hp.lp_repair.
"""
import json
import math
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LB = os.path.join(ROOT, "cov", "lb")
sys.path.insert(0, LB)
import certify
import solve_hp

N, R = 8, 2
QS = [6, 8, 12, 16, 21]


def load(q):
    for base in ("certs_hp", "certs_all"):
        p = os.path.join(LB, base, f"cert_q{q}_n8_R2.json")
        if os.path.exists(p):
            with open(p) as fh:
                return json.load(fh), p
    raise FileNotFoundError(q)


def target(q):
    V = 28*q*q - 48*q + 21
    sphere = q**8 / V
    return sphere * (1.0 + 1.0/(6*q*q))


def block_meta():
    # build_model emits 25 blocks in each family, in this (a,k) order.
    ak = []
    for a in range(N+1):
        for k in range(a, N+1):
            if 2*k <= N+a:
                ak.append((a, k))
    assert len(ak) == 25
    out = []
    for fam, off in (("M'", 0), ("M''", 25), ("N", 50)):
        for j, x in enumerate(ak):
            out.append((fam, x[0], x[1], off+j))
    out.sort(key=lambda x: x[3])
    return out


META = block_meta()


def meta(b):
    fam,a,k,idx = META[b]
    assert idx == b
    return f"{b}:{fam}({a},{k})"


class RayLP:
    def __init__(self, model, cert, targ):
        self.model = model
        self.targ = targ
        D = int(cert["den"])
        Bs = [[ [int(x) for x in row] for row in B] for B in cert["dual_psd"]]
        self.D = D
        self.Bs = Bs
        self.a_orig = [int(x) for x in cert["dual_lin"]]
        nv = model.nvars
        nb = len(model.psd)
        nl = len(model.lin)
        self.nb, self.nl = nb, nl

        # Exact contributions of each PSD block ray, then convert to floats only
        # for the discovery LP.  Rows are divided by c_v, making RHS == 1.
        g = [dict() for _ in range(nb)]
        g0 = [0] * nb
        for b, X in enumerate(model.psd):
            B = Bs[b]
            for r in range(len(X)):
                for c in range(len(X)):
                    y = B[r][c]
                    if y == 0:
                        continue
                    f, c0 = X[r][c]
                    for v, cc in f.items():
                        g[b][v] = g[b].get(v, 0) + y*cc
                    g0[b] += y*c0

        cols = []
        objs = []
        for b in range(nb):
            d = {v: (gv/D)/model.obj[v] for v,gv in g[b].items()}
            cols.append(d)
            objs.append(-g0[b]/D)
        for k,(f,c0) in enumerate(model.lin):
            d = {v: cc/model.obj[v] for v,cc in f.items()}
            cols.append(d)
            objs.append(-float(c0))

        # Scale each variable so its largest row coefficient is O(1).
        self.scales=[]
        rr=[]; cc=[]; vv=[]
        for j,d in enumerate(cols):
            m=max((abs(x) for x in d.values()), default=0.0)
            s=1.0/m if m > 0 else 1.0
            self.scales.append(s)
            for v,x in d.items():
                if x:
                    rr.append(v); cc.append(j); vv.append(x*s)
        self.A=sp.csc_matrix((vv,(rr,cc)), shape=(nv,nb+nl))
        self.obj=np.array([objs[j]*self.scales[j]/(targ**3) for j in range(nb+nl)])
        self.rhs=np.ones(nv)

        # Significant blocks by actual max entry, excluding certification dust.
        mags=[]
        for B in Bs:
            mags.append(max((abs(x)/D for row in B for x in row), default=0.0))
        mm=max(mags)
        self.sig=[b for b,x in enumerate(mags) if x > mm*1e-8]
        self.mags=mags

    def solve(self, keep):
        ks=set(keep)
        bounds=[]
        for b in range(self.nb):
            bounds.append((0,None) if b in ks else (0,0))
        bounds += [(0,None)]*self.nl
        sol=linprog(-self.obj, A_ub=self.A, b_ub=self.rhs,
                    bounds=bounds, method="highs")
        if not sol.success:
            return None
        val=float(self.obj @ sol.x) * self.targ**3
        root=val**(1/3) if val > 0 else 0.0
        return root, sol

    def exact(self, keep):
        ks=set(keep)
        Bsel=[]
        for b,B in enumerate(self.Bs):
            if b in ks:
                Bsel.append(B)
            else:
                Bsel.append([[0]*len(B) for _ in range(len(B))])
        rep=solve_hp.lp_repair(self.model, self.a_orig, Bsel, self.D,
                               verbose=False)
        if rep is None:
            return None
        a2,B2,D2,_=rep
        res=certify.evaluate_certificate(self.model,D2,a2,B2)
        if not res.get("ok"):
            return None
        # compact linear support summary
        vals=[x/D2 for x in a2]
        mx=max(vals) if vals else 0.0
        top=sorted(((v/mx if mx else 0.0,i,self.model.lin_tags[i])
                    for i,v in enumerate(vals) if v>0), reverse=True)[:12]
        return res, top


def family_sets():
    return {
        "linear": [],
        "M'": list(range(0,25)),
        "M''": list(range(25,50)),
        "N": list(range(50,75)),
        "M'+M''": list(range(0,50)),
        "M'+N": list(range(0,25))+list(range(50,75)),
        "M''+N": list(range(25,75)),
        "all": list(range(75)),
    }


for q in QS:
    cert,path=load(q)
    pr=cert["problem"]
    model=certify.build_model(q,N,R,lam=[int(x) for x in pr["lambda"]],
                              beta=int(pr["beta"]))
    t=target(q)
    lp=RayLP(model,cert,t)
    print("\n"+"="*78)
    print(f"q={q} target_root={t:.9f} significant_blocks={len(lp.sig)}")
    print("sig:", " ".join(meta(b) for b in lp.sig))
    print("FAMILY ABLATION")
    for name,keep in family_sets().items():
        ans=lp.solve(keep)
        root=ans[0] if ans else float('nan')
        print(f"  {name:8s} root={root:.9f} ratio_to_target={root/t:.9f}")

    # single-block ranking among significant rays
    singles=[]
    for b in lp.sig:
        ans=lp.solve([b])
        if ans:
            singles.append((ans[0],b))
    singles.sort(reverse=True)
    print("TOP SINGLE BLOCKS")
    for root,b in singles[:12]:
        print(f"  {meta(b):16s} root={root:.9f} ratio={root/t:.9f}")

    # forward greedy support among the significant rays.
    keep=[]
    current=lp.solve(keep)[0]
    print(f"GREEDY start root={current:.9f}")
    for step in range(min(10,len(lp.sig))):
        cand=[]
        for b in lp.sig:
            if b in keep: continue
            ans=lp.solve(keep+[b])
            if ans: cand.append((ans[0],b))
        if not cand: break
        best,b=max(cand)
        keep.append(b); current=best
        print(f"  step={step+1} add={meta(b):16s} root={current:.9f} ratio={current/t:.9f}")
        if current >= t*(1-1e-9):
            break
    print("GREEDY support:", " ".join(meta(b) for b in keep))
    ex=lp.exact(keep)
    if ex:
        res,top=ex
        print(f"EXACT repaired support: root={res['cube_root_float']:.9f} K={res['K_lower_bound']} target_ratio={res['cube_root_float']/t:.9f}")
        print("TOP LINEAR ROWS IN EXACT REPAIR")
        for rel,i,tag in top:
            print(f"  rel={rel:.3e} idx={i:4d} tag={tag}")
    else:
        print("EXACT repaired support: FAILED")
