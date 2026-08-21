/*
 * lincov.c -- search for covering codes that are UNIONS OF COSETS OF A LINEAR
 *             CODE over GF(q).   (arena entry "structure")
 *
 * Idea.  Fix a linear [n,k]_q code C0 with parity check matrix H (m = n-k rows).
 * Let S = { H e : wt(e) <= R } be the set of syndromes reachable by an error of
 * weight at most R.  For a union of cosets  C = U_{i<c} (v_i + C0)  with
 * syndromes sigma_i = H v_i, a word w is covered iff
 *
 *      exists i :  H w - sigma_i  in  S      <=>   H w  in  sigma_i + S.
 *
 * So the whole covering question collapses from q^n words to q^m syndromes:
 *
 *      C covers Z_q^n at radius R   <=>   U_i (sigma_i + S) = GF(q)^m.
 *
 * That is the entire point of this program.  For K_3(11,4) at M = 81 = 3^4 the
 * search space drops from 177147 words to 2187 syndromes and one 7x4 matrix,
 * i.e. by a factor of 81 in the objective and by 3^28 vs 3^(11*81) in the state.
 *
 * WLOG (row operations do not change the code; a coordinate permutation does
 * not change the covering radius) H is systematic, H = [ I_m | A ], so the only
 * free variables are the m*k entries of A together with the c-1 coset syndromes
 * (sigma_0 = 0 WLOG, by translating the whole code).
 *
 * k is a knob, not a constant:  k = n - m controls how much symmetry we impose.
 * M = c * q^k, so large k means few cosets and a highly structured (small,
 * rigid) search space, small k means many cosets and a search that is closer to
 * free.  The driver walks k downwards -- that is the "symmetry then free"
 * schedule at the level of the algebra.
 *
 * Exit codes:  0 = full cover found, 1 = best partial written, 3 = not
 * applicable to these parameters (q not a prime power, q^k does not divide M,
 * syndrome space too large, ...).
 *
 * Build: gcc -O3 -march=native -o lincov lincov.c -lm
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define QMAX 64

static int q, n, R, M, kk, cc, mm;
static long long QM;                    /* q^mm */

/* ---------------- GF(q) ---------------- */
static uint8_t ADDT[QMAX][QMAX], MULT[QMAX][QMAX], NEGT[QMAX];

static int gf_build(int qq)
{
    int p = 0, e = 0;
    for (int b = 2; b <= qq; b++) if (qq % b == 0) { p = b; break; }
    if (!p) return 0;
    { int t = qq; while (t % p == 0) { t /= p; e++; } if (t != 1) return 0; }
    if (e == 1) {
        for (int a = 0; a < qq; a++)
            for (int b = 0; b < qq; b++) {
                ADDT[a][b] = (uint8_t)((a + b) % p);
                MULT[a][b] = (uint8_t)((a * b) % p);
            }
    } else {
        /* elements are polynomials, digit i (base p) = coefficient of x^i */
        int pe[8]; pe[0] = 1; for (int i = 1; i <= e; i++) pe[i] = pe[i-1] * p;
        for (int a = 0; a < qq; a++)
            for (int b = 0; b < qq; b++) {
                int r = 0;
                for (int i = 0; i < e; i++) {
                    int da = (a / pe[i]) % p, db = (b / pe[i]) % p;
                    r += ((da + db) % p) * pe[i];
                }
                ADDT[a][b] = (uint8_t)r;
            }
        /* find a modulus polynomial f = x^e + sum F_i x^i that makes a field */
        int found = 0;
        for (int F = 0; F < qq && !found; F++) {
            for (int a = 0; a < qq; a++)
                for (int b = 0; b < qq; b++) {
                    /* multiply a by b modulo f, MSB-first shift-and-add */
                    int acc = 0;
                    for (int i = e - 1; i >= 0; i--) {
                        /* acc *= x  (mod f) */
                        int over = (acc / pe[e-1]) % p;
                        acc = (acc % pe[e-1]) * p;
                        if (over) {
                            int sub = 0;
                            for (int j = 0; j < e; j++) {
                                int fj = (F / pe[j]) % p;
                                int cj = (acc / pe[j]) % p;
                                sub += ((cj - over * fj) % p + p) % p * pe[j];
                            }
                            acc = sub;
                        }
                        /* acc += b_i * a */
                        int bi = (b / pe[i]) % p;
                        if (bi) {
                            int s2 = 0;
                            for (int j = 0; j < e; j++) {
                                int aj = (a / pe[j]) % p, cj = (acc / pe[j]) % p;
                                s2 += ((cj + bi * aj) % p) * pe[j];
                            }
                            acc = s2;
                        }
                    }
                    MULT[a][b] = (uint8_t)acc;
                }
            /* field test: every nonzero element invertible */
            found = 1;
            for (int a = 1; a < qq && found; a++) {
                int inv = 0;
                for (int b = 1; b < qq; b++) if (MULT[a][b] == 1) { inv = 1; break; }
                if (!inv) found = 0;
            }
        }
        if (!found) return 0;
    }
    for (int a = 0; a < qq; a++)
        for (int b = 0; b < qq; b++) if (ADDT[a][b] == 0) NEGT[a] = (uint8_t)b;
    return 1;
}

/* ---------------- syndrome arithmetic ---------------- */
/* A syndrome is an index in [0,QM); digit i has weight q^i. */

static long long *qpow;
static long long *Tdelta;          /* [(i*q+d)*q+v] -> index delta of d->d+v   */
static uint8_t  *digtab;           /* digtab[i*QM+s] = digit i of s            */

static inline long long addsyn(long long a, long long b)
{
    long long r = a;
    for (int i = 0; i < mm; i++)
        r += Tdelta[(((long long)i * q) + digtab[(long long)i * QM + a]) * q
                    + digtab[(long long)i * QM + b]];
    return r;
}

/* ---------------- state ---------------- */

static uint8_t *A;                 /* mm x kk                                   */
static uint8_t *colvec;            /* n x mm : the columns of H                 */
static uint32_t **addcol;          /* [j*(q-1)+a-1][s] = s + a*h_j              */
static long long *sigma;           /* cc coset syndromes, sigma[0] = 0          */

static uint32_t *stampS, curS;
static uint32_t *stampC, curC;
static uint32_t *Slist; static long long Sn;
static long long ncov;
static int collectS;

/* rng */
typedef struct { uint64_t s[4]; } rng_t;
static rng_t G;
static inline uint64_t rotl(uint64_t x, int k){ return (x<<k)|(x>>(64-k)); }
static uint64_t rnext(rng_t *r){ uint64_t *s=r->s,res=rotl(s[0]+s[3],23)+s[0],t=s[1]<<17;
    s[2]^=s[0];s[3]^=s[1];s[1]^=s[2];s[0]^=s[3];s[2]^=t;s[3]=rotl(s[3],45);return res; }
static void rseed(rng_t *r,uint64_t sd){ for(int i=0;i<4;i++){ sd+=0x9E3779B97F4A7C15ULL;
    uint64_t z=sd; z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL;
    r->s[i]=z^(z>>31);} }
static inline uint64_t rmod(rng_t *r,uint64_t m){ return rnext(r)%m; }

/* rebuild the addition tables for column j from colvec[j] */
static void rebuild_col(int j)
{
    uint8_t vv[QMAX];
    int *d = malloc(sizeof(int) * mm);
    for (int a = 1; a < q; a++) {
        for (int i = 0; i < mm; i++) vv[i] = MULT[a][colvec[(size_t)j*mm + i]];
        uint32_t *tab = addcol[j*(q-1) + a - 1];
        long long delta = 0;
        for (int i = 0; i < mm; i++) { d[i] = 0; delta += Tdelta[(((long long)i*q)+0)*q + vv[i]]; }
        for (long long s = 0; s < QM; s++) {
            tab[s] = (uint32_t)(s + delta);
            int i = 0;
            while (i < mm) {
                delta -= Tdelta[(((long long)i*q)+d[i])*q + vv[i]];
                d[i]++;
                if (d[i] < q) { delta += Tdelta[(((long long)i*q)+d[i])*q + vv[i]]; break; }
                d[i] = 0; delta += Tdelta[(((long long)i*q)+0)*q + vv[i]];
                i++;
            }
        }
    }
    free(d);
}

static void set_A(int r, int j, int val)
{
    A[(size_t)r*kk + j] = (uint8_t)val;
    colvec[(size_t)(mm+j)*mm + r] = (uint8_t)val;
    rebuild_col(mm + j);
}

/* enumerate all syndromes of vectors of weight <= R */
static void rec(int j0, int rem, long long s)
{
    if (stampS[s] != curS) {
        stampS[s] = curS; ncov++;
        if (collectS) Slist[Sn++] = (uint32_t)s;
    }
    if (rem == 0) return;
    for (int j = j0; j < n; j++) {
        uint32_t **base = addcol + j*(q-1);
        for (int a = 0; a < q-1; a++) rec(j+1, rem-1, base[a][s]);
    }
}

static long long evaluate(void)
{
    curS++; ncov = 0; Sn = 0; collectS = (cc > 1);
    rec(0, R, 0);
    if (cc == 1) return QM - ncov;
    curC++;
    long long cov2 = 0;
    for (int i = 0; i < cc; i++) {
        long long sg = sigma[i];
        if (sg == 0) {
            for (long long t = 0; t < Sn; t++) {
                long long s2 = Slist[t];
                if (stampC[s2] != curC) { stampC[s2] = curC; cov2++; }
            }
        } else {
            for (long long t = 0; t < Sn; t++) {
                long long s2 = addsyn(sg, Slist[t]);
                if (stampC[s2] != curC) { stampC[s2] = curC; cov2++; }
            }
        }
    }
    return QM - cov2;
}

/* ---------------- output ---------------- */

static const char *DIG = "0123456789abcdefghijklmnopqrstuvwxyz";

static void write_code(const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return; }
    uint8_t *w = malloc(n), *u = malloc(kk ? kk : 1);
    long long qk = 1; for (int i = 0; i < kk; i++) qk *= q;
    for (int i = 0; i < cc; i++) {
        for (long long t = 0; t < qk; t++) {
            long long x = t;
            for (int j = 0; j < kk; j++) { u[j] = (uint8_t)(x % q); x /= q; }
            /* w = v_i + sum_j u_j * G_j , G_j = ( -A[0][j..], e_j ) */
            for (int r = 0; r < mm; r++) {
                int acc = (int)((sigma[i] / qpow[r]) % q);
                for (int j = 0; j < kk; j++)
                    acc = ADDT[acc][MULT[u[j]][NEGT[A[(size_t)r*kk + j]]]];
                w[r] = (uint8_t)acc;
            }
            for (int j = 0; j < kk; j++) w[mm + j] = u[j];
            for (int j = 0; j < n; j++) fputc(DIG[w[j]], f);
            fputc('\n', f);
        }
    }
    free(w); free(u);
    fclose(f);
}

/* ---------------- driver ---------------- */

static double now(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(int argc, char **argv)
{
    double tlimit = 30.0;
    uint64_t seed = 1;
    const char *outpath = NULL;
    int want_k = -1;
    double memcap = 1.5e9;
    int verbose = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        #define ARG(x) (!strcmp(a,x) && i+1 < argc)
        if      (ARG("-q")) q = atoi(argv[++i]);
        else if (ARG("-n")) n = atoi(argv[++i]);
        else if (ARG("-R")) R = atoi(argv[++i]);
        else if (ARG("-M")) M = atoi(argv[++i]);
        else if (ARG("-k")) want_k = atoi(argv[++i]);
        else if (ARG("-t")) tlimit = atof(argv[++i]);
        else if (ARG("-s")) seed = strtoull(argv[++i], NULL, 10);
        else if (ARG("--out")) outpath = argv[++i];
        else if (!strcmp(a, "-v")) verbose = 1;
        else { fprintf(stderr, "bad option %s\n", a); return 2; }
        #undef ARG
    }
    if (q < 2 || n < 1 || R < 0 || M < 1) {
        fprintf(stderr, "usage: lincov -q Q -n N -R R -M M [-k K] [-t sec] [-s seed] [--out f]\n");
        return 2;
    }
    if (q > QMAX) return 3;
    if (!gf_build(q)) { if (verbose) fprintf(stderr, "q=%d is not a prime power\n", q); return 3; }

    /* pick k */
    if (want_k < 0) {
        want_k = 0;
        long long p2 = 1;
        for (int k = 1; k <= n - 1; k++) {
            p2 *= q;
            if (p2 > M) break;
            if (M % p2 == 0) want_k = k;
        }
    }
    kk = want_k;
    if (kk < 1 || kk >= n) return 3;
    { long long p2 = 1;
      for (int i = 0; i < kk; i++) { if (p2 > M) return 3; p2 *= q; }
      if (p2 > M || M % p2) return 3;
      cc = (int)(M / p2); }
    mm = n - kk;
    QM = 1; for (int i = 0; i < mm; i++) { QM *= q; if (QM > 2e8) return 3; }

    /* necessary condition: c * |{wt<=R}| >= q^m */
    {
        double P = 0, cb = 1;
        for (int i = 0; i <= R && i <= n; i++) {
            cb = 1; for (int j = 0; j < i; j++) cb = cb * (n - j) / (j + 1);
            double tp = cb; for (int j = 0; j < i; j++) tp *= (q - 1);
            P += tp;
        }
        if ((double)cc * P < (double)QM) {
            if (verbose) fprintf(stderr, "sphere bound rules out k=%d\n", kk);
            return 3;
        }
        /* keep the coset marking loop affordable */
        if (cc > 1 && (double)cc * P * mm > 4e9) return 3;
    }
    /* memory for the addition tables */
    if ((double)n * (q-1) * QM * 4.0 > memcap) return 3;

    qpow = malloc(sizeof(long long) * (mm + 1));
    qpow[0] = 1; for (int i = 1; i <= mm; i++) qpow[i] = qpow[i-1] * q;
    Tdelta = malloc(sizeof(long long) * (size_t)mm * q * q);
    for (int i = 0; i < mm; i++)
        for (int d = 0; d < q; d++)
            for (int v = 0; v < q; v++)
                Tdelta[(((long long)i*q)+d)*q + v] = ((long long)ADDT[d][v] - d) * qpow[i];
    digtab = malloc((size_t)mm * QM);
    for (int i = 0; i < mm; i++)
        for (long long s = 0; s < QM; s++) digtab[(size_t)i*QM + s] = (uint8_t)((s / qpow[i]) % q);

    A = calloc((size_t)mm * kk, 1);
    colvec = calloc((size_t)n * mm, 1);
    addcol = malloc(sizeof(uint32_t*) * (size_t)n * (q-1));
    for (int j = 0; j < n*(q-1); j++) addcol[j] = malloc(sizeof(uint32_t) * (size_t)QM);
    sigma = calloc(cc, sizeof(long long));
    stampS = calloc((size_t)QM, sizeof(uint32_t));
    stampC = calloc((size_t)QM, sizeof(uint32_t));
    Slist  = malloc(sizeof(uint32_t) * (size_t)QM);
    curS = curC = 0;

    /* H = [ I_m | A ] : the first m columns are the unit vectors */
    for (int j = 0; j < mm; j++) { colvec[(size_t)j*mm + j] = 1; rebuild_col(j); }

    rseed(&G, seed * 6364136223846793005ULL + 1442695040888963407ULL);

    uint8_t *bestA = malloc((size_t)mm * kk);
    long long *bestSig = malloc(sizeof(long long) * cc);
    long long best = -1;
    double t0 = now();
    long long nrestart = 0, nev = 0;

    while (now() - t0 < tlimit) {
        /* ---- random start ---- */
        for (int r = 0; r < mm; r++)
            for (int j = 0; j < kk; j++) A[(size_t)r*kk + j] = (uint8_t)rmod(&G, q);
        for (int j = 0; j < kk; j++) {
            for (int r = 0; r < mm; r++) colvec[(size_t)(mm+j)*mm + r] = A[(size_t)r*kk + j];
            rebuild_col(mm + j);
        }
        for (int i = 1; i < cc; i++) sigma[i] = (long long)rmod(&G, (uint64_t)QM);
        long long f = evaluate(); nev++;
        nrestart++;
        int stall = 0;

        while (f > 0 && now() - t0 < tlimit) {
            int improved = 0;
            /* ---- coordinate descent over the entries of A ---- */
            int nent = mm * kk;
            for (int step = 0; step < nent; step++) {
                int idx = (int)rmod(&G, (uint64_t)nent);
                int r = idx / kk, j = idx % kk;
                int old = A[(size_t)r*kk + j];
                long long bf = f; int bv = old, nt = 1;
                for (int v = 0; v < q; v++) {
                    if (v == old) continue;
                    set_A(r, j, v);
                    long long g = evaluate(); nev++;
                    if (g < bf) { bf = g; bv = v; nt = 1; }
                    else if (g == bf && v != old) { nt++; if (rmod(&G, (uint64_t)nt) == 0) bv = v; }
                }
                set_A(r, j, bv);
                if (bf < f) improved = 1;
                f = bf;
                if (f == 0) break;
            }
            if (f == 0) break;
            /* ---- coset syndromes ---- */
            for (int i = 1; i < cc && f > 0; i++) {
                long long old = sigma[i], bf = f, bv = old;
                int tries = 24;
                for (int t = 0; t < tries; t++) {
                    sigma[i] = (long long)rmod(&G, (uint64_t)QM);
                    long long g = evaluate(); nev++;
                    if (g < bf) { bf = g; bv = sigma[i]; }
                }
                sigma[i] = bv;
                if (bf < f) improved = 1;
                f = bf;
            }
            if (f < best || best < 0) {
                best = f;
                memcpy(bestA, A, (size_t)mm*kk);
                memcpy(bestSig, sigma, sizeof(long long)*cc);
                if (verbose) fprintf(stderr, "# k=%d c=%d best=%lld syndromes (%lld words) t=%.1f\n",
                                     kk, cc, best, best * (M / cc), now() - t0);
            }
            if (!improved) {
                if (++stall > 2) break;              /* restart */
                /* kick: randomise a few entries */
                for (int t = 0; t < 1 + mm*kk/6; t++) {
                    int r = (int)rmod(&G, (uint64_t)mm), j = (int)rmod(&G, (uint64_t)kk);
                    set_A(r, j, (int)rmod(&G, (uint64_t)q));
                }
                f = evaluate(); nev++;
            } else stall = 0;
        }
        if (f == 0) { best = 0;
                      memcpy(bestA, A, (size_t)mm*kk);
                      memcpy(bestSig, sigma, sizeof(long long)*cc);
                      break; }
    }

    memcpy(A, bestA, (size_t)mm*kk);
    memcpy(sigma, bestSig, sizeof(long long)*cc);
    long long qk = M / cc;
    printf("LINRESULT q=%d n=%d R=%d M=%d k=%d c=%d m=%d syndromes=%lld "
           "uncovered_syndromes=%lld uncovered_words=%lld restarts=%lld evals=%lld time=%.2f\n",
           q, n, R, M, kk, cc, mm, QM, best, best * qk, nrestart, nev, now() - t0);
    if (outpath) write_code(outpath);
    return best == 0 ? 0 : 1;
}
