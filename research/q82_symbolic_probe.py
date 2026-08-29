#!/usr/bin/env python3
import json, math, os, sys
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "cov", "lb"))
import certify

QS = list(range(6, 22))
N, R = 8, 2

def load_cert(q):
    for base in ("certs_hp", "certs_all"):
        p = os.path.join(ROOT, "cov", "lb", base, f"cert_q{q}_n8_R2.json")
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f), p
    raise FileNotFoundError(q)

def ceil_frac(num, den):
    return (num + den - 1) // den

def candidate(q, c=40):
    V = 28*q*q - 48*q + 21
    num = q**4 * (q**4 + 2*q*q - c)
    return ceil_frac(num, V)

def jaccard(a,b):
    u=a|b
    return len(a&b)/len(u) if u else 1.0

def cosine(a,b):
    dot=sum(x*y for x,y in zip(a,b))
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na and nb else 0.0

def rank_int(A):
    A=[[Fraction(x) for x in row] for row in A]
    m=len(A); n=len(A[0]) if m else 0
    r=0
    for c in range(n):
        piv=next((i for i in range(r,m) if A[i][c]), None)
        if piv is None: continue
        A[r],A[piv]=A[piv],A[r]
        p=A[r][c]
        for j in range(c,n): A[r][j] /= p
        for i in range(m):
            if i!=r and A[i][c]:
                f=A[i][c]
                for j in range(c,n): A[i][j] -= f*A[r][j]
        r += 1
        if r==m: break
    return r

def lin_sig(vals, rel=1e-8):
    mx=max(vals) if vals else 0.0
    return {i for i,x in enumerate(vals) if x > mx*rel} if mx else set()

def block_sig(blocks, den, rel=1e-8):
    mags=[]
    for B in blocks:
        mags.append(max((abs(int(x))/den for row in B for x in row), default=0.0))
    mx=max(mags) if mags else 0.0
    return {i for i,x in enumerate(mags) if x > mx*rel}, mags

rows=[]; linvecs={}; linsupp={}; psdsupp={}; tags0=None; block_sizes0=None; certs={}; models={}
print("Q82 SYMBOLIC PROBE")
print("candidate: ceil(q^4 (q^4 + 2 q^2 - 40)/(28 q^2 - 48 q + 21))")
print()
for q in QS:
    cert,path=load_cert(q); certs[q]=cert
    pr=cert["problem"]
    model=certify.build_model(q,N,R,lam=[int(x) for x in pr["lambda"]], beta=int(pr["beta"]))
    models[q]=model
    den=int(cert["den"])
    dl=[int(x) for x in cert["dual_lin"]]
    dp=[[[int(x) for x in row] for row in B] for B in cert["dual_psd"]]
    res=certify.evaluate_certificate(model,den,dl,dp)
    assert res["ok"], (q,res)
    assert len(dl)==len(model.lin) and len(dp)==len(model.psd)
    tags=tuple(model.lin_tags); bs=tuple(len(B) for B in model.psd)
    if tags0 is None: tags0=tags; block_sizes0=bs
    else:
        assert tags==tags0, f"linear row structure changed at q={q}"
        assert bs==block_sizes0, f"PSD block structure changed at q={q}"
    vals=[x/den for x in dl]
    linvecs[q]=vals; linsupp[q]=lin_sig(vals)
    psdsupp[q], bmags=block_sig(dp,den)
    V=28*q*q-48*q+21
    sphere=q**8/V
    root=res["cube_root_float"]
    creq=q**4*(1+2/q**2-root/sphere)
    cand=candidate(q)
    rows.append((q,cand,res["K_lower_bound"],root,sphere,creq,len(linsupp[q]),len(psdsupp[q]),path))

print("FINITE CERTIFIED CHECK q=6..21")
print("q  candidate  certLB  margin  c_required  lin_sig  psd_sig")
for q,cand,lb,root,sph,creq,ls,ps,pp in rows:
    print(f"{q:2d} {cand:10d} {lb:8d} {lb-cand:7d} {creq:10.5f} {ls:8d} {ps:8d}")
assert all(c <= lb for _,c,lb,*_ in rows)
print("PASS: candidate integer bound is implied by every exact certificate q=6..21")
print()

print("MODEL STRUCTURE")
print(f"linear rows: {len(tags0)}; PSD blocks: {len(block_sizes0)}; block sizes: {block_sizes0}")
print("structure identical for every q=6..21")
print()

print("LINEAR-DUAL SUPPORT STABILITY (relative threshold 1e-8 of per-q max)")
inter=set.intersection(*(linsupp[q] for q in QS)); union=set.union(*(linsupp[q] for q in QS))
print(f"intersection={len(inter)} union={len(union)} Jaccard(all)={len(inter)/len(union) if union else 1:.4f}")
print("adjacent Jaccard:", " ".join(f"{q}-{q+1}:{jaccard(linsupp[q],linsupp[q+1]):.3f}" for q in QS[:-1]))
print("q6-v-q21 Jaccard:", f"{jaccard(linsupp[6],linsupp[21]):.4f}")
print("linear direction cosines:", " ".join(f"{q}-{q+1}:{cosine(linvecs[q],linvecs[q+1]):.4f}" for q in QS[:-1]))
print("q6-v-q21 cosine:", f"{cosine(linvecs[6],linvecs[21]):.4f}")
print()

freq=[]
for i,tag in enumerate(tags0):
    f=sum(i in linsupp[q] for q in QS)
    if f:
        rels=[]
        for q in QS:
            v=linvecs[q][i]; mx=max(linvecs[q])
            rels.append(v/mx if mx else 0)
        avg=sum(rels)/len(rels)
        freq.append((f,avg,i,tag))
freq.sort(reverse=True)
print("MOST STABLE/SIGNIFICANT LINEAR ROWS")
for f,avg,i,tag in freq[:30]:
    print(f"freq={f:2d}/16 avg_rel={avg:.3e} idx={i:4d} tag={tag}")
print()

print("PSD-BLOCK SUPPORT STABILITY (relative threshold 1e-8)")
pinter=set.intersection(*(psdsupp[q] for q in QS)); punion=set.union(*(psdsupp[q] for q in QS))
print(f"intersection={len(pinter)} union={len(punion)} Jaccard(all)={len(pinter)/len(punion) if punion else 1:.4f}")
print("adjacent Jaccard:", " ".join(f"{q}-{q+1}:{jaccard(psdsupp[q],psdsupp[q+1]):.3f}" for q in QS[:-1]))
print("q6-v-q21 Jaccard:", f"{jaccard(psdsupp[6],psdsupp[21]):.4f}")
print()

print("EXACT RANKS OF SIGNIFICANT DUAL PSD BLOCKS (note certification adds interior displacement)")
for q in (6,8,10,12,14,16,18,21):
    den=int(certs[q]["den"]); dp=[[[int(x) for x in row] for row in B] for B in certs[q]["dual_psd"]]
    ss=sorted(psdsupp[q])
    rr=[(i,len(dp[i]),rank_int(dp[i])) for i in ss]
    print(f"q={q}: {rr}")
print()

# Fit asymptotic ratio from exact certified continuous roots.
# r(q)=root/sphere. Fit r-1 ~ a2/q^2+a3/q^3+a4/q^4+a5/q^5 using normal equations.
def solve_linear(A,b):
    A=[list(map(float,row))+[float(y)] for row,y in zip(A,b)]
    n=len(A[0])-1
    for c in range(n):
        p=max(range(c,n), key=lambda r: abs(A[r][c])); A[c],A[p]=A[p],A[c]
        z=A[c][c]
        for j in range(c,n+1): A[c][j]/=z
        for r in range(n):
            if r==c: continue
            z=A[r][c]
            for j in range(c,n+1): A[r][j]-=z*A[c][j]
    return [A[i][-1] for i in range(n)]
xs=[]; ys=[]
for q,cand,lb,root,sph,creq,*_ in rows:
    if q>=10:
        xs.append([q**-2,q**-3,q**-4,q**-5]); ys.append(root/sph-1)
# normal equations
AtA=[[sum(x[i]*x[j] for x in xs) for j in range(4)] for i in range(4)]
Aty=[sum(x[i]*y for x,y in zip(xs,ys)) for i in range(4)]
coef=solve_linear(AtA,Aty)
print("ASYMPTOTIC FIT on q=10..21: root/sphere - 1 ~= a2/q^2+a3/q^3+a4/q^4+a5/q^5")
print("coefficients:", " ".join(f"a{i+2}={coef[i]:.8f}" for i in range(4)))
print("The leading coefficient should be compared with the candidate value a2=2.")
