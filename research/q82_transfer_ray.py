#!/usr/bin/env python3
"""Exact-after-the-fact tests for sparse PSD rays and transfer across q.

A PSD dual block taken from one certified q0 instance remains PSD for any q;
only the primal block coefficients change.  We therefore use fixed certified
PSD rays, optimize nonnegative ray scales and linear multipliers numerically,
then reinterpret every returned float as an *exact rational*, compute all dual
inequalities in Fraction arithmetic, and apply one exact global shrink theta.
The final reported value is consequently a genuine dual-feasible rational
value for the target q (although this script is still a discovery tool, not a
closed-form proof).
"""
import json
import math
import os
import sys
from fractions import Fraction

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LB=os.path.join(ROOT,"cov","lb")
sys.path.insert(0,LB)
import certify

N,R=8,2


def load_cert(q):
    for base in ("certs_hp","certs_all"):
        p=os.path.join(LB,base,f"cert_q{q}_n8_R2.json")
        if os.path.exists(p):
            with open(p) as f: return json.load(f)
    raise FileNotFoundError(q)


def model(q):
    return certify.build_model(q,N,R,lam=[1,1,1,0,0,0,0,0,0],beta=1)


def sphere(q): return q**8/(28*q*q-48*q+21)

def coeff(q,root): return q*q*(root/sphere(q)-1.0)


def exact_ray_solve(q, srcq, keep):
    m=model(q)
    cert=load_cert(srcq)
    D=int(cert["den"])
    Bs=[[[int(x) for x in row] for row in B] for B in cert["dual_psd"]]
    assert len(Bs)==len(m.psd)==75
    for b in keep:
        ok,why=certify.is_psd_exact(Bs[b])
        assert ok,(srcq,b,why)

    nb=len(Bs); nl=len(m.lin); nv=m.nvars
    g=[{} for _ in range(nb)]; g0=[0]*nb
    for b in keep:
        X=m.psd[b]; B=Bs[b]; gb=g[b]
        for r in range(len(X)):
            for c in range(len(X)):
                z=B[r][c]
                if not z: continue
                f,c0=X[r][c]
                for v,a in f.items(): gb[v]=gb.get(v,0)+z*a
                g0[b]+=z*c0

    # Columns: requested PSD rays followed by all linear multipliers.  Divide
    # each inequality by c_v and then scale each column to max abs coefficient 1.
    cols=[]; objs=[]; kind=[]
    for b in keep:
        cols.append({v:Fraction(x,D)/m.obj[v] for v,x in g[b].items()})
        objs.append(Fraction(-g0[b],D)); kind.append(("B",b))
    for k,(f,c0) in enumerate(m.lin):
        cols.append({v:Fraction(a,m.obj[v]) for v,a in f.items()})
        objs.append(Fraction(-c0)); kind.append(("L",k))

    rr=[];cc=[];vv=[]; scales=[]
    for j,d in enumerate(cols):
        mf=max((abs(float(x)) for x in d.values()),default=0.0)
        s=1.0/mf if mf else 1.0
        scales.append(s)
        for v,x in d.items():
            xf=float(x)*s
            if xf:
                rr.append(v);cc.append(j);vv.append(xf)
    A=sp.csc_matrix((vv,(rr,cc)),shape=(nv,len(cols)))
    t=sphere(q)*(1+1/(6*q*q))
    obj=np.array([float(objs[j])*scales[j]/(t**3) for j in range(len(cols))])
    opt={"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9}
    sol=linprog(-obj,A_ub=A,b_ub=np.ones(nv),bounds=(0,None),method="highs-ds",options=opt)
    if not sol.success:
        return {"ok":False,"message":sol.message}
    normviol=max(float(x) for x in (A@sol.x-np.ones(nv)))

    # Exact rational reinterpretation of the numerical solution.
    true=[]
    for j,z in enumerate(sol.x):
        x=float(z)*scales[j]
        true.append(Fraction.from_float(max(0.0,x)))
    d=[Fraction(0) for _ in range(nv)]; d0=Fraction(0)
    for x,(typ,idx) in zip(true,kind):
        if not x: continue
        if typ=="B":
            for v,z in g[idx].items(): d[v]+=x*Fraction(z,D)
            d0+=x*Fraction(g0[idx],D)
        else:
            f,c0=m.lin[idx]
            for v,a in f.items(): d[v]+=x*a
            d0+=x*c0
    theta=Fraction(1)
    worst=Fraction(1)
    for v,dv in enumerate(d):
        cv=Fraction(m.obj[v])
        if dv>cv:
            th=cv/dv
            if th<theta: theta=th
    val=-theta*d0
    root=float(val)**(1/3) if val>0 else 0.0
    # top exact linear multipliers after shrink (relative only)
    linvals=[]
    for x,(typ,idx) in zip(true,kind):
        if typ=="L" and x:
            linvals.append((float(theta*x),idx,m.lin_tags[idx]))
    linvals.sort(reverse=True)
    mx=linvals[0][0] if linvals else 1.0
    top=[(z/mx,i,tag) for z,i,tag in linvals[:8]]
    return {"ok":True,"root":root,"theta":float(theta),"normviol":normviol,
            "a2":coeff(q,root),"top":top,"npos":sum(x>0 for x in true)}


def line(q,src,keep,res):
    if not res["ok"]:
        print(f"q={q:3d} src={src:2d} keep={keep} FAILED {res['message']}")
        return
    tar=sphere(q)*(1+1/(6*q*q))
    print(f"q={q:3d} src={src:2d} keep={keep!s:28s} root={res['root']:.6f} "
          f"target_ratio={res['root']/tar:.9f} a2={res['a2']:.6f} "
          f"theta={res['theta']:.9g} normviol={res['normviol']:.3e} npos={res['npos']}")

print("SELF-Q SINGLE N(0,0) RAY -- exact rationalized")
for q in range(6,22):
    r=exact_ray_solve(q,q,[50]); line(q,q,[50],r)

print("\nFIXED q=21 RAYS TRANSFERRED TO OTHER q")
supports={
    "N00":[50],
    "N00_N33":[50,63],
    "N00_M00":[50,0],
    "N00_M0k":[50,0,1,2,3],
    "small":[50,0,1,2,3,5,6,7],
    "q21sig":[0,1,2,3,5,6,7,9,10,13,15,25,50,63],
}
for q in [6,8,10,12,14,16,18,21,24,30,40,64,100]:
    print(f"-- target q={q} --")
    for name,keep in supports.items():
        r=exact_ray_solve(q,21,keep)
        if r.get("ok"):
            tar=sphere(q)*(1+1/(6*q*q))
            print(f" {name:10s} root/target={r['root']/tar:.9f} a2={r['a2']:.6f} theta={r['theta']:.4g}")
        else:
            print(f" {name:10s} FAILED")

# Print top linear rows for the most interesting fixed template at large q.
print("\nTOP LINEAR ROWS FOR q21sig SOURCE AT q=21,40,100")
for q in [21,40,100]:
    r=exact_ray_solve(q,21,supports["q21sig"])
    print(f"q={q} root={r.get('root',0):.6f} a2={r.get('a2',0):.6f}")
    for rel,i,tag in r.get("top",[]): print(f"  rel={rel:.3e} idx={i:4d} tag={tag}")
