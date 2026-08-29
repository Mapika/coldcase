# A combinatorial infinite-family lower bound for `K_q(8,2)`

**Status:** proof draft, algebra checked; literature-novelty check still required.

Let `C subset Z_q^8` have covering radius 2 and size `M`. Put

- `s = q-1`,
- `V = |B_2| = 1+8s+28s^2 = 28q^2-48q+21`,
- `V1 = |B_1| = 1+8s = 8q-7`,
- `m(x) = |C cap B_2(x)|`,
- `e(x) = m(x)-1 >= 0`,
- `E = sum_x e(x) = MV-q^8`.

The main local lemma below gives a direct excess argument, independent of the
SDP.

## Theorem

For every integer `q >= 6`,

```
K_q(8,2) >= ceil( q^8 (V+18) / (V^2 + 9V + 9V1) ).
```

Equivalently,

```
K_q(8,2) >= ceil(
 q^8 (28q^2 - 48q + 39)
 / (784q^4 - 2688q^3 + 3732q^2 - 2544q + 630)
).
```

Relative to the sphere-covering bound this is

```
K_q(8,2) >= (q^8/V) * (1 + 9(V-V1)/(V^2+9V+9V1)),
```

so

```
K_q(8,2) >= (q^8/V) * (1 + 9/(28q^2) + O(q^-3)).
```

A slightly stronger parity statement follows from the proof: the local
constant 9 can be replaced by 12 when q is odd.

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

If `q >= 6` and `x` is a unique boundary point, then

```
L_2(x) >= 9.
```

If q is odd, in fact `L_2(x) >= 12`.

### Proof

Set `s=q-1` and move `x` to zero by an isometry. Let `c` be the unique
codeword at distance 2 from x.

The needed ball-intersection numbers are

```
|B_2(z) cap B_2(x)| = q^2+12q-12 = s^2+14s+1,   d(x,z)=2,
                     = 6s,                       d(x,z)=3,
                     = 6,                        d(x,z)=4,
                     = 0,                        d(x,z)>=5.
```

Hence

```
L_2 = 6s A_3 + 6 A_4 - 27s^2 + 6s.                 (1)
```

Similarly, using

```
|B_2(z) cap B_1(x)| = 2q-1 = 2s+1,   d(x,z)=2,
                     = 3,             d(x,z)=3,
```

we get

```
L_1 = 3(A_3-2s).                                      (2)
```

In particular `A_3 >= 2s`. Also, from (1),

```
L_2 == 0 (mod 6)  if q is odd,
L_2 == 3 (mod 6)  if q is even.                       (3)
```

We now rule out every value below 9 (and also 6 in the odd case).

Let

```
h = A_3-2s = L_1/3,
T = C cap S_3(x),
B = |T| = 2s+h,
t = L_2-L_1.
```

Thus `t` is the excess on the weight-2 shell `S_2(x)`.

Let `R` be the six coordinates outside `supp(c)`. For each `i in R`, at
least `s` words of T contain coordinate i: c covers no singleton word on such
a coordinate, and all s singleton symbols there must be covered by T. If
`d_i` is the number of supports of T containing coordinate i, then

```
d_i >= s  for all i in R.                             (4)
```

For `y in S_2(x)` define

```
r_y = |{b in T : d(b,y) <= 2}|,
W = sum_{y in S_2(x)} binom(r_y,2).
```

Since `r_y <= m(y)`, we have `r_y-1 <= e(y)` whenever `r_y>0`. Therefore,
for fixed total shell excess t, convexity gives

```
W <= binom(t+1,2).                                     (5)
```

For the reverse estimate, consider two words `b,b' in T`. If their supports
intersect in `u` coordinates, then they have at least

```
2 binom(u,2)
```

common radius-2 neighbours of weight 2. Indeed, for every two common support
coordinates, if the two words agree in one of them choose that common symbol;
otherwise choose the symbol of b in one coordinate and the symbol of b' in
the other. Interchanging the choices gives two common weight-2 neighbours.
Consequently

```
W >= 2 sum_{b<b'} binom(|supp(b) cap supp(b')|,2)
  >= 2( sum_i binom(d_i,2) - binom(B,2) ).             (6)
```

The last inequality uses `2 binom(u,2) >= 2(u-1)` and
`sum_{b<b'} |supp(b) cap supp(b')| = sum_i binom(d_i,2)`.
We may use all eight coordinate degrees in (6); (4) supplies lower bounds on
six of them.

#### q even

Here s is odd and at least 5. By (3), the only possible value below 9 is
`L_2=3`. Since `0 <= L_1 <= L_2` and L1 is a multiple of 3, there are two
cases.

- `h=0`: then `B=2s`, `t=3`, and the six coordinates of R already account
  for all `3B=6s` support incidences. Thus their degrees are exactly s and
  the other two degrees are zero. From (6),

  ```
  W >= 2(6 binom(s,2)-binom(2s,2)) = 2s(s-2) >= 30,
  ```

  while (5) gives `W <= binom(4,2)=6`, a contradiction.

- `h=1`: then `B=2s+1` and `t=0`. There are `6s+3` support incidences.
  Subject to the six lower bounds `d_i>=s`, the convex sum
  `sum_i binom(d_i,2)` is minimized by degrees

  ```
  s,s,s,s,s,s,2,1,
  ```

  so it is at least `6 binom(s,2)+1`. Hence

  ```
  W >= 2(6 binom(s,2)+1-binom(2s+1,2))
    = 2(s^2-4s+1) > 0
  ```

  for `s>=5`, whereas `t=0` makes (5) give W=0. Contradiction.

Thus `L_2>=9` for even q.

#### q odd

Here s is even and at least 6. By (3), values below 12 are 0 and 6.

If `L_2=0`, then h=0 and t=0. Equation (6) gives

```
W >= 2s(s-2)>0,
```

contradicting (5).

Now suppose `L_2=6`. Since `L_1=3h`, we have `h in {0,1,2}`.

- `h=0`: `B=2s`, `t=6`, and (6) gives
  `W>=2s(s-2)>=48`, while (5) gives `W<=binom(7,2)=21`.

- `h=1`: `B=2s+1`, `t=3`. Using just the six degree lower bounds in (4),

  ```
  W >= 2(6 binom(s,2)-binom(2s+1,2))
    = 2s(s-4) >= 24,
  ```

  while (5) gives `W<=6`.

- `h=2`: `B=2s+2`, `t=0`. There are `6s+6` support incidences. The convex
  degree sum is minimized by

  ```
  s,s,s,s,s,s,3,3,
  ```

  and is at least `6 binom(s,2)+6`. Thus

  ```
  W >= 2(6 binom(s,2)+6-binom(2s+2,2))
    = 2(s-1)(s-5) > 0,
  ```

  contradicting W=0.

Hence `L_2>=12` for odd q. This proves the lemma. □

## 2. Global double count

Let X be the set of unique boundary points. Since every point is covered,

```
E = sum_x (m(x)-1).
```

At most E points have multiplicity at least two, so at least `q^8-E` points
are covered exactly once. Among points covered exactly once, at most `M V1`
have their unique covering codeword at distance 0 or 1. Therefore

```
|X| >= q^8 - E - M V1.                                (7)
```

On the other hand, every excess unit is counted in exactly V radius-2 balls:

```
sum_x L_2(x) = V E.                                    (8)
```

By the local lemma, `L_2(x)>=9` for x in X. Combining (7) and (8),

```
V E >= 9 |X| >= 9(q^8-E-MV1).                          (9)
```

Since `E=MV-q^8`, (9) becomes

```
V(MV-q^8) >= 9(2q^8-M(V+V1)),
```

and hence

```
M (V^2+9V+9V1) >= q^8(V+18).
```

This is the claimed bound. □

For odd q, replacing 9 by the local constant 12 gives the sharper bound

```
K_q(8,2) >= ceil( q^8(V+24)/(V^2+12V+12V1) ).
```

## 3. Numerical consequences for the rejected-paper range

The uniform theorem already beats every previous lower bound in the complete
family `6 <= q <= 21`:

| q | previous LB | theorem LB | certified SDP LB |
|---:|---:|---:|---:|
| 6 | 2276 | 2293 | 2367 |
| 7 | 5457 | 5498 | 5631 |
| 8 | 11766 | 11812 | 12033 |
| 9 | 23184 | 23289 | 23642 |
| 10 | 42772 | 42876 | 43423 |
| 11 | 74415 | 74630 | 75448 |
| 12 | 123772 | 123976 | 125156 |
| 13 | 197563 | 197981 | 199633 |
| 14 | 305294 | 305659 | 307909 |
| 15 | 457584 | 458297 | 461294 |
| 16 | 669207 | 669813 | 673723 |
| 17 | 955978 | 957133 | 962145 |
| 18 | 1339650 | 1340601 | 1346931 |
| 19 | 1842639 | 1844404 | 1852296 |
| 20 | 2495614 | 2497038 | 2506759 |
| 21 | 3329193 | 3331782 | 3343629 |

For odd q the parity-refined constant 12 gives still stronger values.

## 4. Checks still needed before claiming novelty

1. Compare the closed form against Chen--Honkala (1990), van Wee's q-ary
   excess bounds, and the formulas summarized in Chapter 6 of *Covering Codes*.
2. Have the local pair-overlap argument independently re-read line by line.
3. Add a small enumerator that verifies the radius-2/radius-1 intersection
   numbers and the two-common-weight-2-neighbours claim for generic q symbols.
4. Only after (1) should this be called a new theorem rather than a derived
   theorem candidate.
