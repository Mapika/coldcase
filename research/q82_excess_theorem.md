# A combinatorial infinite-family lower bound for `K_q(8,2)`

**Status:** proof draft; all algebraic and finite checks pass.  Literature-novelty
check against Chen--Honkala (1990), especially their Theorem 4, is still required.

Let `C subset Z_q^8` have covering radius 2 and size `M`. Put

- `s = q-1`,
- `V = |B_2| = 1+8s+28s^2 = 28q^2-48q+21`,
- `V1 = |B_1| = 1+8s = 8q-7`,
- `m(x) = |C cap B_2(x)|`,
- `e(x) = m(x)-1 >= 0`,
- `E = sum_x e(x) = MV-q^8`.

The key point is a local excess lemma whose lower bound grows with q.  The
proof is direct and does not use the SDP.

## Theorem

For every integer `q >= 6`, with `s=q-1`,

```
K_q(8,2) >= ceil( q^8 (V+2s) / (V^2 + s(V+V1)) ).
```

Equivalently,

```
K_q(8,2) >= ceil(
 q^8 (28q^2 - 46q + 19)
 / (784q^4 - 2660q^3 + 3412q^2 - 1962q + 427)
).
```

Relative to the sphere-covering bound this is exactly

```
K_q(8,2) >= (q^8/V) *
  (1 + 28(q-1)^3 / (V^2 + (q-1)(V+V1))).
```

Thus, as `q -> infinity`,

```
K_q(8,2) >= (q^8/V) * (1 + 1/(28q) + O(q^-2)).
```

This is materially stronger asymptotically than the first constant-excess
bound found during the SDP probe (`L_2 >= 9`, which only gives an `O(q^-2)`
relative correction).

## 1. A local excess lemma

For `x in Z_q^8`, write

```
A_i(x) = |C cap S_i(x)|,
L_r(x) = sum_{y in B_r(x)} e(y).
```

Call `x` a *unique boundary point* if

```
A_0(x)=A_1(x)=0,  A_2(x)=1.
```

### Lemma

If `q >= 6`, `s=q-1`, and `x` is a unique boundary point, then

```
L_2(x) >= s.
```

### Proof

Move `x` to zero by an isometry and let `c` be its unique codeword at
distance 2.

The required ball-intersection numbers are

```
|B_2(z) cap B_2(x)| = q^2+12q-12 = s^2+14s+1,   d(x,z)=2,
                     = 6s,                       d(x,z)=3,
                     = 6,                        d(x,z)=4,
                     = 0,                        d(x,z)>=5.
```

Consequently

```
L_2 = 6s A_3 + 6 A_4 - 27s^2 + 6s.                 (1)
```

Similarly,

```
|B_2(z) cap B_1(x)| = 2q-1 = 2s+1,   d(x,z)=2,
                     = 3,             d(x,z)=3,
```

so

```
L_1 = 3(A_3-2s).                                      (2)
```

In particular there is an integer `h >= 0` such that

```
A_3 = 2s+h,   L_1=3h.                                 (3)
```

Assume for a contradiction that

```
L_2 <= s-1.
```

Let

```
T = C cap S_3(x),
B = |T| = 2s+h,
t = L_2-L_1 = L_2-3h.
```

The number `t` is exactly the excess on the weight-2 shell `S_2(x)`.  Since
`t >= 0`,

```
0 <= 3h <= L_2 <= s-1.                                (4)
```

Let `R` be the six coordinates outside `supp(c)`.  For every `i in R`, all
`s` singleton words supported on coordinate i must be covered by words of T:
`c` is at distance 3 from them, and the assumptions `A_0=A_1=0`, `A_2=1`
leave no other codeword of weight at most 2.  A weight-3 word covers such a
singleton only when it contains coordinate i with the same nonzero symbol.
Hence, if `d_i` is the number of supports of T containing coordinate i,

```
d_i >= s  for each of the six coordinates in R.       (5)
```

For `y in S_2(x)` define

```
r_y = |{b in T : d(b,y) <= 2}|,
W = sum_{y in S_2(x)} binom(r_y,2).
```

Since `r_y <= m(y)`, whenever `r_y>0` we have `r_y-1 <= e(y)`.
The sum of `e(y)` over `S_2(x)` is t.  Convexity therefore gives

```
W <= binom(t+1,2).                                     (6)
```

For the reverse estimate, take two words `b,b' in T`.  If their supports
intersect in `u` coordinates, they have at least

```
2 binom(u,2)
```

common radius-2 neighbours of weight 2.  For every two common support
coordinates there are at least two valid symbol assignments: if the words
differ, use the two crossed assignments; if they agree in both coordinates,
use the common assignment and change one of the two symbols (possible since
`q>=3`).  Different support pairs give different weight-2 words.  Hence

```
W >= 2 sum_{b<b'} binom(|supp(b) cap supp(b')|,2)
  >= 2( sum_i binom(d_i,2) - binom(B,2) ).             (7)
```

The last inequality uses `binom(u,2) >= u-1` and

```
sum_{b<b'} |supp(b) cap supp(b')| = sum_i binom(d_i,2).
```

There are `3B=6s+3h` support incidences in T.  By (4), `3h<s`.
Subject to the six lower bounds in (5), the convex sum
`sum_i binom(d_i,2)` is therefore minimized by keeping those six degrees at s
and distributing the remaining `3h` incidences as evenly as possible over the
two coordinates of `supp(c)`.  Put

```
a=floor(3h/2),  b=ceil(3h/2).
```

Then

```
sum_i binom(d_i,2)
 >= 6 binom(s,2) + binom(a,2)+binom(b,2)
 >= 6 binom(s,2) + (9h^2-6h)/4.                       (8)
```

Combining (7) and (8), while (4) gives
`t <= s-1-3h`, yields

```
W >= 2(6 binom(s,2) + (9h^2-6h)/4 - binom(2s+h,2)),
W <= binom(t+1,2) <= binom(s-3h,2).                   (9)
```

The difference between the lower and upper expressions in (9) is at least

```
(3s^2 - 7s - 2h^2 - 2hs - 7h)/2.                     (10)
```

Finally `h <= s/3`, so

```
2h^2+2hs+7h <= 8s^2/9 + 7s/3.
```

The numerator in (10) is therefore at least

```
s(19s-84)/9,
```

which is strictly positive for `s>=5`, i.e. `q>=6`.  This contradicts (9).
Thus `L_2(x) >= s`. □

## 2. Global double count

Let X be the set of unique boundary points.  Every point with multiplicity at
least two contributes at least one unit to E, so at least `q^8-E` points are
covered exactly once.  Among these uniquely covered points, at most `M V1`
have their unique covering codeword at distance 0 or 1.  Hence

```
|X| >= q^8 - E - M V1.                                (11)
```

Every excess unit is counted in exactly V radius-2 balls, so

```
sum_x L_2(x) = V E.                                    (12)
```

The local lemma gives `L_2(x)>=s` on X.  Therefore

```
V E >= s |X| >= s(q^8-E-MV1).                         (13)
```

Substituting `E=MV-q^8` gives

```
V(MV-q^8) >= s(2q^8-M(V+V1)),
```

and hence

```
M [V^2+s(V+V1)] >= q^8(V+2s).
```

This is the theorem. □

## 3. Numerical consequences for the rejected-paper range

The single closed-form theorem improves every previous lower bound in the
complete family `6 <= q <= 21`:

| q | previous LB | theorem LB | certified SDP LB |
|---:|---:|---:|---:|
| 6 | 2276 | 2282 | 2367 |
| 7 | 5457 | 5484 | 5631 |
| 8 | 11766 | 11796 | 12033 |
| 9 | 23184 | 23277 | 23642 |
| 10 | 42772 | 42876 | 43423 |
| 11 | 74415 | 74655 | 75448 |
| 12 | 123772 | 124045 | 125156 |
| 13 | 197563 | 198121 | 199633 |
| 14 | 305294 | 305905 | 307909 |
| 15 | 457584 | 458696 | 461294 |
| 16 | 669207 | 670424 | 673723 |
| 17 | 955978 | 958031 | 962145 |
| 18 | 1339650 | 1341877 | 1346931 |
| 19 | 1842639 | 1846170 | 1852296 |
| 20 | 2495614 | 2499426 | 2506759 |
| 21 | 3329193 | 3334952 | 3343629 |

The SDP remains stronger throughout this finite range, but the theorem now
provides a genuine q-parametric mathematical result rather than a collection
of numerical cells.

## 4. Stronger local constants are available

The argument above deliberately uses the clean threshold `L_2>=q-1`.  Keeping
the exact congruence of (1) modulo 6 and the exact integer inequality (9)
gives larger local constants.  For q=6,...,21 the first local values not ruled
out by this pair-overlap argument are

```
9,12,15,18,21,18,21,24,27,30,33,30,33,36,39,42.
```

Using those constants in (13) gives stronger finite bounds, still below the
certified SDP values.  They are not used in the theorem because `q-1` gives a
much cleaner closed form and already supplies an asymptotic improvement of
order `1/q` over the sphere density.

## 5. Checks still needed before claiming novelty

1. Compare the closed form against Chen--Honkala (1990), especially their
   radius-two excess result [their Theorem 4].  The 1995 survey explicitly says
   that theorem is omitted there because its notation is too long.
2. Compare against the nonbinary lower-bound formulas in Chapter 6 of
   *Covering Codes* (1997) and the later Bhandari--Durairajan/Bhandari et al.
   lower bounds.
3. Have the pair-overlap proof independently re-read line by line.
4. Only after the literature comparison should this be called a new theorem
   rather than a theorem candidate.
