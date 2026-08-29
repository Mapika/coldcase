#!/usr/bin/env python3
"""Independent exact checks for q82_excess_theorem.md.

No solver and no third-party package.  This does NOT replace the written proof;
it checks every finite arithmetic/combinatorial identity used there, including
the strengthened local lemma L_2 >= q-1.
"""
from fractions import Fraction
from itertools import combinations, product
from math import comb


def V(q): return 28*q*q-48*q+21
def V1(q): return 8*q-7

def theorem_fraction(q, lam=None):
    if lam is None: lam=q-1
    v=V(q); v1=V1(q)
    return Fraction(q**8*(v+2*lam), v*v+lam*(v+v1))

def iceil(x): return (x.numerator+x.denominator-1)//x.denominator

OLD={6:2276,7:5457,8:11766,9:23184,10:42772,11:74415,12:123772,
     13:197563,14:305294,15:457584,16:669207,17:955978,18:1339650,
     19:1842639,20:2495614,21:3329193}
SDP={6:2367,7:5631,8:12033,9:23642,10:43423,11:75448,12:125156,
     13:199633,14:307909,15:461294,16:673723,17:962145,18:1346931,
     19:1852296,20:2506759,21:3343629}
EXPECT={6:2282,7:5484,8:11796,9:23277,10:42876,11:74655,12:124045,
        13:198121,14:305905,15:458696,16:670424,17:958031,18:1341877,
        19:1846170,20:2499426,21:3334952}

print('STRENGTHENED THEOREM TABLE')
for q in range(6,22):
    b=iceil(theorem_fraction(q))
    assert b==EXPECT[q], (q,b,EXPECT[q])
    assert OLD[q] < b < SDP[q], (q,OLD[q],b,SDP[q])
    print(q, OLD[q], b, SDP[q])

# Closed-form algebra.
for q in range(2,1000):
    s=q-1; v=V(q); v1=V1(q)
    assert v+2*s == 28*q*q-46*q+19
    assert v*v+s*(v+v1) == 784*q**4-2660*q**3+3412*q**2-1962*q+427
    # Relative correction over sphere is exactly s(V-V1)/den = 28s^3/den.
    den=v*v+s*(v+v1)
    assert s*(v-v1)==28*s**3
    rel=Fraction(28*s**3,den)
    # q*rel -> 1/28; this broad rational sandwich is enough to catch signs/orders.
    assert rel > 0
    if q>=100:
        assert Fraction(1,40*q) < rel < Fraction(1,20*q)


def ball2_zero(q):
    n=8
    out=[(0,)*n]
    for i in range(n):
        for a in range(1,q):
            z=[0]*n; z[i]=a; out.append(tuple(z))
    for i,j in combinations(range(n),2):
        for a in range(1,q):
            for b in range(1,q):
                z=[0]*n; z[i]=a; z[j]=b; out.append(tuple(z))
    return out

def hd(a,b): return sum(x!=y for x,y in zip(a,b))

# Ball-intersection formulas used to derive L1 and L2.
for q in range(3,31):
    B=ball2_zero(q)
    assert len(B)==V(q)
    got=[]
    for d in range(2,6):
        c=(1,)*d+(0,)*(8-d)
        got.append(sum(hd(y,c)<=2 for y in B))
    expect=[q*q+12*q-12,6*(q-1),6,0]
    assert got==expect,(q,got,expect)
    B1=[z for z in B if sum(x!=0 for x in z)<=1]
    c2=(1,1,0,0,0,0,0,0)
    c3=(1,1,1,0,0,0,0,0)
    assert sum(hd(y,c2)<=2 for y in B1)==2*q-1
    assert sum(hd(y,c3)<=2 for y in B1)==3

# Pair-overlap lemma: two weight-3 words with support intersection u have at
# least 2*C(u,2) common weight-2 radius-2 neighbours. q=3 realizes every
# equality/difference pattern relevant to the argument.
q=3
for u in (0,1,2,3):
    supp1={0,1,2}
    fresh=list(range(3,3+(3-u)))
    supp2=set(range(u))|set(fresh)
    for samebits in product((0,1),repeat=u):
        b=[0]*8; bp=[0]*8
        for i in supp1:b[i]=1
        for i in supp2:bp[i]=1
        for i,same in enumerate(samebits):bp[i]=1 if same else 2
        common=0
        for i,j in combinations(range(8),2):
            for a in (1,2):
                for z in (1,2):
                    y=[0]*8;y[i]=a;y[j]=z
                    if hd(y,b)<=2 and hd(y,bp)<=2:common+=1
        assert common>=2*comb(u,2),(u,samebits,common)

# Exact integer version of the strengthened local contradiction.  If L2<s,
# write L1=3h.  We compare the best possible degree-convex lower bound on W
# with the worst possible upper bound C(s-3h,2).  Check many q directly; the
# written proof then reduces the same difference to s(19s-84)/18 > 0.
for q in range(6,2001):
    s=q-1
    for h in range((s-1)//3+1):
        e=3*h
        a=e//2; b=e-a
        degree_pairs=6*comb(s,2)+comb(a,2)+comb(b,2)
        B=2*s+h
        wlow=2*(degree_pairs-comb(B,2))
        wup=comb(s-3*h,2)
        assert wlow>wup,(q,s,h,wlow,wup)

# Check the analytic lower bound used in the final line of the proof.
for s in range(5,100000):
    assert s*(19*s-84)>0

print('PASS: strengthened theorem identities and local lemma checks')
