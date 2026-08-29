#!/usr/bin/env python3
import json, os, sys
import numpy as np

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LB=os.path.join(ROOT,'cov','lb')


def load(q):
    for d in ('certs_hp','certs_all'):
        p=os.path.join(LB,d,f'cert_q{q}_n8_R2.json')
        if os.path.exists(p):
            with open(p) as f:return json.load(f)
    raise FileNotFoundError(q)

print('N(0,0) DUAL BLOCK = block 50, basis [border, weight 0..8]')
for q in [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]:
    c=load(q); D=float(int(c['den']))
    B=np.array([[float(int(x))/D for x in row] for row in c['dual_psd'][50]],dtype=float)
    B=(B+B.T)/2
    w,V=np.linalg.eigh(B)
    order=np.argsort(w)[::-1]; w=w[order]; V=V[:,order]
    mx=max(abs(w[0]),1e-300)
    ratios=w/mx
    erank=int(np.sum(w>mx*1e-8))
    v=V[:,0]
    j=int(np.argmax(np.abs(v)))
    if v[j]<0:v=-v
    v=v/max(abs(v))
    # Two natural congruence scalings to look for a stable vector pattern.
    s=np.sqrt(q-1.0)
    vsqrt=np.array([v[0]]+[v[i+1]*(s**i) for i in range(9)])
    if max(abs(vsqrt)):vsqrt/=max(abs(vsqrt))
    vpow=np.array([v[0]]+[v[i+1]*((q-1.0)**i) for i in range(9)])
    if max(abs(vpow)):vpow/=max(abs(vpow))
    print(f'q={q:2d} erank1e-8={erank} eig/max='+' '.join(f'{x:.3e}' for x in ratios[:6]))
    print('  v      = '+' '.join(f'{x:+.4e}' for x in v))
    print('  v*sqrt = '+' '.join(f'{x:+.4e}' for x in vsqrt))
    print('  v*pow  = '+' '.join(f'{x:+.4e}' for x in vpow))
