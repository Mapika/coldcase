#!/usr/bin/env python3
"""Compare the N(0,0) dual block across q after dual congruences.

The exact checker uses the integral alpha normalization.  Its source notes that
this differs from the paper normalization by A_int = D A_paper D with
D_i=(q-1)^(i/2) on weight i.  Hence a dual matrix transforms as
Y_paper = D Y_int D.  We compare raw, paper, and inverse congruences and their
dominant rank-2 subspaces.
"""
import json, os
import numpy as np

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LB=os.path.join(ROOT,'cov','lb')
QS=list(range(6,22))

def load(q):
    for d in ('certs_hp','certs_all'):
        p=os.path.join(LB,d,f'cert_q{q}_n8_R2.json')
        if os.path.exists(p):
            with open(p) as f:return json.load(f)
    raise FileNotFoundError(q)

def block(q):
    c=load(q); den=float(int(c['den']))
    Y=np.array([[float(int(x))/den for x in row] for row in c['dual_psd'][50]],float)
    return (Y+Y.T)/2

def trans(Y,q,mode):
    s=q-1.0
    d=np.array([1.0,1.0]+[s**(i/2) for i in range(1,9)])
    if mode=='raw': z=np.ones_like(d)
    elif mode=='paper': z=d
    elif mode=='inverse': z=1/d
    else: raise ValueError(mode)
    A=(z[:,None]*Y)*z[None,:]
    n=np.linalg.norm(A,'fro')
    return A/n if n else A

def eig(A):
    w,V=np.linalg.eigh(A); o=np.argsort(w)[::-1]; return w[o],V[:,o]

def fcos(A,B): return float(np.sum(A*B)/(np.linalg.norm(A)*np.linalg.norm(B)))
def subcos(A,B,r=2):
    _,U=eig(A); _,V=eig(B)
    sv=np.linalg.svd(U[:,:r].T@V[:,:r],compute_uv=False)
    return sv

for mode in ('raw','paper','inverse'):
    As={q:trans(block(q),q,mode) for q in QS}
    print('\nMODE',mode)
    print('adjacent Frobenius cos:', ' '.join(f'{q}-{q+1}:{fcos(As[q],As[q+1]):.5f}' for q in QS[:-1]))
    print('13-v-21 Fcos',f'{fcos(As[13],As[21]):.6f}',' 6-v-21',f'{fcos(As[6],As[21]):.6f}')
    print('rank2 principal cos 13-v-21', ' '.join(f'{x:.6f}' for x in subcos(As[13],As[21],2)))
    print('rank2 principal cos 16-v-21', ' '.join(f'{x:.6f}' for x in subcos(As[16],As[21],2)))
    for q in (10,13,16,18,21):
        w,V=eig(As[q]); mx=max(abs(w[0]),1e-300)
        v=V[:,0]
        if v[np.argmax(abs(v))]<0:v=-v
        v=v/max(abs(v))
        print(f'q={q:2d} eigrel='+' '.join(f'{x/mx:.3e}' for x in w[:4]))
        print('  topv '+' '.join(f'{x:+.5e}' for x in v))
