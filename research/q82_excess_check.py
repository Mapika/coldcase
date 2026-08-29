#!/usr/bin/env python3
"""Independent finite/algebra checks for q82_excess_theorem.md.

No solver and no third-party package.  This does NOT prove the theorem; it
checks every finite arithmetic/combinatorial identity used in the proof over a
wide q range and reproduces the q=6..21 consequence table exactly.
"""
from fractions import Fraction
from itertools import combinations, product
from math import ceil, comb


def V(q): return 28*q*q-48*q+21
def V1(q): return 8*q-7

def theorem_fraction(q, lam=9):
    v=V(q); v1=V1(q)
    return Fraction(q**8*(v+2*lam), v*v+lam*v+lam*v1)

def iceil(x): return (x.numerator+x.denominator-1)//x.denominator

# Published/table values used only for consequence comparison.
OLD={6:2276,7:5457,8:11766,9:23184,10:42772,11:74415,12:123772,
     13:197563,14:305294,15:457584,16:669207,17:955978,18:1339650,
     19:1842639,20:2495614,21:3329193}
SDP={6:2367,7:5631,8:12033,9:23642,10:43423,11:75448,12:125156,
     13:199633,14:307909,15:461294,16:673723,17:962145,18:1346931,
     19:1852296,20:2506759,21:3343629}
EXPECT={6:2293,7:5498,8:11812,9:23289,10:42876,11:74630,12:123976,
        13:197981,14:305659,15:458297,16:669813,17:957133,18:1340601,
        19:1844404,20:2497038,21:3331782}

print('THEOREM TABLE')
for q in range(6,22):
    b=iceil(theorem_fraction(q))
    assert b==EXPECT[q], (q,b,EXPECT[q])
    assert OLD[q] < b < SDP[q], (q,OLD[q],b,SDP[q])
    bo=iceil(theorem_fraction(q,12)) if q%2 else None
    print(q, OLD[q], b, SDP[q], ('odd12='+str(bo)) if bo else '')

# Expanded denominator identity.
for q in range(2,100):
    lhs=V(q)**2+9*V(q)+9*V1(q)
    rhs=784*q**4-2688*q**3+3732*q**2-2376*q+567
    assert lhs==rhs
    assert V(q)+18==28*q*q-48*q+39

# Relative correction is strictly stronger than 1+1/(6q^2) for q>=6.
for q in range(6,10000):
    v=V(q); d=v*v+9*v+9*V1(q)
    rel=Fraction(9*(v-V1(q)), d)
    assert rel > Fraction(1,6*q*q)


def ball2_zero(q):
    n=8
    out=[(0,)*n]
    # weight 1
    for i in range(n):
        for a in range(1,q):
            z=[0]*n; z[i]=a; out.append(tuple(z))
    # weight 2
    for i,j in combinations(range(n),2):
        for a in range(1,q):
            for b in range(1,q):
                z=[0]*n; z[i]=a; z[j]=b; out.append(tuple(z))
    return out

def hd(a,b): return sum(x!=y for x,y in zip(a,b))

# Intersection formulas: enumerate only B2(0), so O(q^2), not q^8.
for q in range(3,31):
    B=ball2_zero(q)
    assert len(B)==V(q)
    got=[]
    for d in range(2,6):
        c=(1,)*d+(0,)*(8-d)
        got.append(sum(hd(y,c)<=2 for y in B))
    expect=[q*q+12*q-12,6*(q-1),6,0]
    assert got==expect,(q,got,expect)

    # B1(0) intersection with B2(center at distance 2,3).
    B1=[z for z in B if sum(x!=0 for x in z)<=1]
    c2=(1,1,0,0,0,0,0,0)
    c3=(1,1,1,0,0,0,0,0)
    assert sum(hd(y,c2)<=2 for y in B1)==2*q-1
    assert sum(hd(y,c3)<=2 for y in B1)==3

# Common weight-2 neighbour lemma.  A shared coordinate can have equal or
# unequal nonzero symbols. q=3 already realizes every equality pattern.
q=3
for u in (0,1,2,3):
    # supports: b uses 0,1,2; b' shares first u and has fresh coordinates.
    supp1={0,1,2}
    fresh=list(range(3,3+(3-u)))
    supp2=set(range(u))|set(fresh)
    for samebits in product((0,1),repeat=u):
        b=[0]*8; bp=[0]*8
        for i in supp1:b[i]=1
        for i in supp2:bp[i]=1
        for i,same in enumerate(samebits):
            bp[i]=1 if same else 2
        common=0
        for i,j in combinations(range(8),2):
            for a in (1,2):
                for z in (1,2):
                    y=[0]*8;y[i]=a;y[j]=z
                    if hd(y,b)<=2 and hd(y,bp)<=2:common+=1
        assert common>=2*comb(u,2),(u,samebits,common)

# Algebraic local-lemma case checks.  Smin is the convex minimum degree-pair
# count in the exact cases used in the proof.
def upperW(t): return comb(t+1,2)
for q in range(6,501):
    s=q-1
    if q%2==0:
        # L2=3, h=0 or1
        low0=2*(6*comb(s,2)-comb(2*s,2))
        assert low0>upperW(3)
        low1=2*(6*comb(s,2)+1-comb(2*s+1,2))
        assert low1>0
    else:
        # L2=0
        low=2*(6*comb(s,2)-comb(2*s,2))
        assert low>0
        # L2=6: h=0,1,2
        assert low>upperW(6)
        low1=2*(6*comb(s,2)-comb(2*s+1,2))
        assert low1>upperW(3)
        low2=2*(6*comb(s,2)+6-comb(2*s+2,2))
        assert low2>0

print('PASS: all exact identities/local cases checked')
