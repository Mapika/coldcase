/*
 * symsearch.c -- local search for covering codes that are INVARIANT under the
 *                diagonal translation  x -> x + 1^n  of Z_q^n.
 *                (arena entry "structure")
 *
 * The group.  G = < x -> x + (1,1,...,1) >  acts on Z_q^n by isometries, freely
 * (no fixed points), so every orbit has exactly q elements.  If the code C is a
 * union of such orbits then the coverage multiplicity function
 *
 *      cnt[w] = #{ c in C : d(w,c) <= R }
 *
 * is constant on orbits.  So the whole problem descends to the quotient
 * Z_q^n / G, which is q times smaller.
 *
 * The quotient, concretely.  Map w to (w_0, d) with d_i = w_i - w_0 (i = 1..n-1);
 * this is a bijection Z_q^n -> Z_q x Z_q^{n-1} under which G acts only on the
 * first factor.  So the quotient IS Z_q^{n-1}, indexed by d.  Normalise each
 * orbit of codewords by its member with first coordinate 0, i.e. represent an
 * orbit by e in Z_q^{n-1}; the orbit is  {(j, e_1+j, ..., e_{n-1}+j) : j in Z_q}.
 * A short computation gives, with delta = j - w_0,
 *
 *      d(w, c^{(j)}) = [delta != 0] + #{ i : d_i != e_i + delta }.
 *
 * Hence the reduced counter is
 *
 *      rcnt[d] = sum over orbits e of #{ delta : #{i : d_i != e_i+delta}
 *                                                  <= R - [delta != 0] }
 *
 * i.e. **one orbit behaves exactly like q ordinary Hamming balls in Z_q^{n-1}**:
 * one of radius R centred at e, and q-1 of radius R-1 centred at e + delta*1.
 * Everything from the free solver (incremental counters, the sphere-difference
 * move, greedy init, kicks) carries over verbatim, one radius at a time.
 *
 * What this buys, measured on K_8(9,4):
 *   - counters: 8^8 = 16.7 M cells instead of 8^9 = 134 M   (33 MB vs 268 MB)
 *   - one move relocates q = 8 codewords for the price of one baseline move
 *     (sum_delta C(n-2,R_delta)(q-1)^R_delta  ==  C(n-1,R)(q-1)^R exactly)
 *   - the search has M/q free variables instead of M.
 *
 * The risk is over-constraint: q | M is forced and the best invariant code may
 * be larger than the best code.  That is what the driver's symmetry-then-free
 * schedule is for -- this program produces the invariant part, the free solver
 * then breaks the symmetry with the leftover words.
 *
 * Build: gcc -O3 -march=native -fopenmp -o symsearch symsearch.c -lm
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static int q, n, R, NR;            /* NR = n-1 = dimension of the quotient   */
static int NORB;                   /* number of orbits                        */
static long long NTOT;             /* q^NR                                    */
static long long pw[32];

static uint16_t *cnt;
static long long uncovered;        /* uncovered points of the QUOTIENT        */

static uint8_t *rep;               /* NORB * NR                               */
static uint8_t *bestrep;
static long long best_uncovered;
static long long *cidx;            /* NORB * q  : index of centre (e+delta)   */

static int NCOMB[8];               /* per radius r: C(NR-1, r)                */
static uint8_t *combs[8];          /* per radius r: NR * NCOMB[r] * r         */
static unsigned nthreads = 1;

/* ---------------- rng ---------------- */
typedef struct { uint64_t s[4]; } rng_t;
static rng_t grng;
static inline uint64_t rotl(uint64_t x,int k){return (x<<k)|(x>>(64-k));}
static inline uint64_t rnext(rng_t *r){uint64_t *s=r->s,res=rotl(s[0]+s[3],23)+s[0],t=s[1]<<17;
    s[2]^=s[0];s[3]^=s[1];s[1]^=s[2];s[0]^=s[3];s[2]^=t;s[3]=rotl(s[3],45);return res;}
static void rseed(rng_t *r,uint64_t sd){for(int i=0;i<4;i++){sd+=0x9E3779B97F4A7C15ULL;
    uint64_t z=sd;z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;z=(z^(z>>27))*0x94D049BB133111EBULL;
    r->s[i]=z^(z>>31);}}
static inline uint64_t rmod(rng_t *r,uint64_t m){return rnext(r)%m;}

/* ---------------- combinatorics ---------------- */
static long long ipow(long long b,int e){long long r=1;while(e-->0)r*=b;return r;}
static long long binom(int a,int b){ if(b<0||b>a) return 0; long long r=1;
    if(b>a-b) b=a-b; for(int i=0;i<b;i++) r=r*(a-i)/(i+1); return r; }

static void build_combs(int r)
{
    if (r < 0 || combs[r]) return;
    NCOMB[r] = (int)binom(NR - 1, r);
    combs[r] = malloc((size_t)NR * NCOMB[r] * (r ? r : 1));
    int stride = r ? r : 1;
    for (int p = 0; p < NR; p++) {
        int rest[32], nr2 = 0;
        for (int j = 0; j < NR; j++) if (j != p) rest[nr2++] = j;
        int c[16];
        for (int i = 0; i < r; i++) c[i] = i;
        for (int kkk = 0; kkk < NCOMB[r]; kkk++) {
            for (int i = 0; i < r; i++)
                combs[r][((size_t)p * NCOMB[r] + kkk) * stride + i] = (uint8_t)rest[c[i]];
            int i = r - 1;
            while (i >= 0 && c[i] == nr2 - r + i) i--;
            if (i < 0) break;
            c[i]++;
            for (int j = i + 1; j < r; j++) c[j] = c[j-1] + 1;
        }
    }
}

static void make_offs(const uint8_t *c, long long *offs)
{
    for (int j = 0; j < NR; j++) {
        int k = 0;
        for (int v = 0; v < q; v++)
            if (v != c[j]) offs[(size_t)j*q + (k++)] = (long long)(v - c[j]) * pw[j];
    }
}

static long long widx(const uint8_t *w)
{
    long long x = 0;
    for (int j = 0; j < NR; j++) x += (long long)w[j] * pw[j];
    return x;
}
static void word_of(long long idx, uint8_t *w)
{
    for (int j = 0; j < NR; j++) w[j] = (uint8_t)((idx / pw[j]) % q);
}

/* ---------------- full ball marking (init / kick) ---------------- */
static long long ball_rec(long long base, const long long *offs, int startpos,
                          int rem, int delta)
{
    long long dunc = 0;
    if (rem == 0) return 0;
    for (int j = startpos; j < NR; j++)
        for (int k = 0; k < q-1; k++) {
            long long w = base + offs[(size_t)j*q + k];
            if (delta > 0) { if (cnt[w]++ == 0) dunc--; }
            else           { if (--cnt[w] == 0) dunc++; }
            dunc += ball_rec(w, offs, j+1, rem-1, delta);
        }
    return dunc;
}
static void ball_apply(const uint8_t *c, long long base, int rad, int delta)
{
    if (rad < 0) return;
    long long offs[32*64];
    make_offs(c, offs);
    long long dunc = 0;
    if (delta > 0) { if (cnt[base]++ == 0) dunc--; }
    else           { if (--cnt[base] == 0) dunc++; }
    dunc += ball_rec(base, offs, 0, rad, delta);
    uncovered += dunc;
}

/* apply (or remove) the whole q-fold structure of orbit i */
static void orbit_apply(int i, int delta)
{
    uint8_t c[32];
    for (int d = 0; d < q; d++) {
        int rad = R - (d != 0);
        if (rad < 0) continue;
        for (int j = 0; j < NR; j++) c[j] = (uint8_t)((rep[(size_t)i*NR + j] + d) % q);
        long long b = widx(c);
        cidx[(size_t)i*q + d] = b;
        ball_apply(c, b, rad, delta);
    }
}

/* ---------------- the move ---------------- */
static inline long long sphere_walk(const uint8_t *pos, long long base,
                                    const long long *offs, long long dv,
                                    int rad, int mode)
{
    long long dunc = 0;
    int k[16];
    long long partial[17];
    if (rad == 0) {
        long long a = base, b = base + dv;
        if (mode) { if (--cnt[a]==0) dunc++; if (cnt[b]++==0) dunc--; }
        else      { if (cnt[a]==1)  dunc++; if (cnt[b]==0)  dunc--; }
        return dunc;
    }
    for (int i = 0; i < rad; i++) k[i] = 0;
    partial[0] = base;
    for (int i = 0; i < rad; i++)
        partial[i+1] = partial[i] + offs[(size_t)pos[i]*q + 0];
    for (;;) {
        long long a = partial[rad], b = a + dv;
        if (mode) { if (--cnt[a]==0) dunc++; if (cnt[b]++==0) dunc--; }
        else      { if (cnt[a]==1)  dunc++; if (cnt[b]==0)  dunc--; }
        int i = rad - 1;
        while (i >= 0 && k[i] == q-2) { k[i] = 0; i--; }
        if (i < 0) break;
        k[i]++;
        for (int j = i; j < rad; j++)
            partial[j+1] = partial[j] + offs[(size_t)pos[j]*q + k[j]];
    }
    return dunc;
}

/* Move orbit i: rep[i][p] -> v.  All q centres shift their p-th coordinate.
 * mode 0 = read-only estimate (used only for ranking candidates),
 * mode 1 = commit (exact: every sub-move updates cnt and its own bookkeeping). */
static long long move_delta(int i, int p, int v, int commit, int par)
{
    long long dunc = 0;
    uint8_t c[32];
    long long op[32*64];
    int oldv = rep[(size_t)i*NR + p];
    for (int d = 0; d < q; d++) {
        int rad = R - (d != 0);
        if (rad < 0) continue;
        for (int j = 0; j < NR; j++) c[j] = (uint8_t)((rep[(size_t)i*NR + j] + d) % q);
        long long base = cidx[(size_t)i*q + d];
        int nv = (v + d) % q;
        long long dv = (long long)(nv - c[p]) * pw[p];
        make_offs(c, op);
        const uint8_t *cb = combs[rad] + (size_t)p * NCOMB[rad] * (rad ? rad : 1);
        int stride = rad ? rad : 1;
        long long acc = 0;
        if (par) {
#pragma omp parallel for schedule(static) reduction(+:acc)
            for (int kk = 0; kk < NCOMB[rad]; kk++)
                acc += sphere_walk(cb + (size_t)kk*stride, base, op, dv, rad, commit);
        } else {
            for (int kk = 0; kk < NCOMB[rad]; kk++)
                acc += sphere_walk(cb + (size_t)kk*stride, base, op, dv, rad, commit);
        }
        dunc += acc;
        if (commit) cidx[(size_t)i*q + d] = base + dv;
    }
    if (commit) {
        rep[(size_t)i*NR + p] = (uint8_t)v;
        uncovered += dunc;
        (void)oldv;
    }
    return dunc;
}

/* ---------------- output ---------------- */
static const char *DIG = "0123456789abcdefghijklmnopqrstuvwxyz";
static void write_code(const char *path, const uint8_t *rp)
{
    char tmpn[4096];
    snprintf(tmpn, sizeof tmpn, "%s.part", path);
    FILE *f = fopen(tmpn, "w");
    if (!f) { perror(tmpn); return; }
    for (int i = 0; i < NORB; i++)
        for (int j = 0; j < q; j++) {
            fputc(DIG[j], f);
            for (int t = 0; t < NR; t++)
                fputc(DIG[(rp[(size_t)i*NR + t] + j) % q], f);
            fputc('\n', f);
        }
    fclose(f);
    rename(tmpn, path);
}

static long long pick_uncovered(rng_t *r)
{
    for (int t = 0; t < 64; t++) {
        long long w = (long long)(rnext(r) % (uint64_t)NTOT);
        if (cnt[w] == 0) return w;
    }
    long long start = (long long)(rnext(r) % (uint64_t)NTOT);
    for (long long d = 0; d < NTOT; d++) {
        long long w = start + d; if (w >= NTOT) w -= NTOT;
        if (cnt[w] == 0) return w;
    }
    return -1;
}

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + 1e-9*t.tv_nsec; }

int main(int argc, char **argv)
{
    double tlimit = 60.0;
    uint64_t seed = 12345;
    const char *outpath = NULL;
    int M = 0, kick = 2, verbose = 0, want_orb = 0;
    long long stall_limit = 3000;
    int candcap = 100000;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        #define ARG(x) (!strcmp(a,x) && i+1<argc)
        if      (ARG("-q")) q = atoi(argv[++i]);
        else if (ARG("-n")) n = atoi(argv[++i]);
        else if (ARG("-R")) R = atoi(argv[++i]);
        else if (ARG("-M")) M = atoi(argv[++i]);
        else if (ARG("-t")) tlimit = atof(argv[++i]);
        else if (ARG("-s")) seed = strtoull(argv[++i],NULL,10);
        else if (ARG("--out")) outpath = argv[++i];
        else if (ARG("--orbits")) want_orb = atoi(argv[++i]);
        else if (ARG("--kick")) kick = atoi(argv[++i]);
        else if (ARG("--stall")) stall_limit = atoll(argv[++i]);
        else if (ARG("--cand")) candcap = atoi(argv[++i]);
        else if (ARG("--threads")) nthreads = atoi(argv[++i]);
        else if (!strcmp(a,"-v")) verbose = 1;
        else { fprintf(stderr,"bad option %s\n",a); return 2; }
        #undef ARG
    }
    if (q < 2 || n < 2 || R < 0 || R > 7 || M < q) {
        fprintf(stderr,"usage: symsearch -q Q -n N -R R -M M [-t sec] [-s seed] [--out f]\n");
        return 2;
    }
    /* clock starts before the allocation and the greedy initialisation */
    double t0 = now();
    NORB = want_orb > 0 ? want_orb : M / q;
    if (NORB < 1 || NORB > M / q) return 3;
    NR = n - 1;
#ifdef _OPENMP
    if (nthreads) omp_set_num_threads(nthreads);
    nthreads = omp_get_max_threads();
#endif
    NTOT = 1;
    for (int i = 0; i < NR; i++) {
        if (NTOT > (long long)4e9 / q) { fprintf(stderr,"quotient too large\n"); return 3; }
        NTOT *= q;
    }
    for (int p = 0; p < NR; p++) pw[p] = ipow(q, NR-1-p);
    if ((long long)NORB * q > 65000) { fprintf(stderr,"M too large for uint16\n"); return 3; }

    cnt = malloc(sizeof(uint16_t) * (size_t)NTOT);
    if (!cnt) { fprintf(stderr,"cannot allocate counters\n"); return 3; }
    rep = malloc((size_t)NORB * NR);
    bestrep = malloc((size_t)NORB * NR);
    cidx = malloc(sizeof(long long) * (size_t)NORB * q);
    build_combs(R);
    if (R >= 1) build_combs(R-1);
    rseed(&grng, seed * 0x9E3779B97F4A7C15ULL + 12345);

    /* greedy init: put each orbit representative on a still-uncovered point */
    memset(cnt, 0, sizeof(uint16_t) * (size_t)NTOT);
    uncovered = NTOT;
    {
        uint8_t tmp[32];
        for (int i = 0; i < NORB; i++) {
            /* Initialisation is itself deadline-bounded: on a heavily loaded
             * machine placing the orbits can outlast the whole budget, and a
             * short invariant code that exists beats a full one that does not.
             * The driver's finisher tops the answer back up to M words. */
            if (i > 0 && now() - t0 > tlimit) { NORB = i; break; }
            long long w = pick_uncovered(&grng);
            if (w < 0) w = (long long)rmod(&grng,(uint64_t)NTOT);
            word_of(w, tmp);
            memcpy(rep + (size_t)i*NR, tmp, NR);
            orbit_apply(i, +1);
        }
    }
    memcpy(bestrep, rep, (size_t)NORB*NR);
    best_uncovered = uncovered;
    if (verbose) {
        printf("# symsearch q=%d n=%d R=%d M=%d orbits=%d quotient=%lld init_unc=%lld(x%d words)\n",
               q,n,R,M,NORB,NTOT,uncovered,q);
        fflush(stdout);
    }

    int par_cand = (NCOMB[R] < 128);
    uint8_t *u = malloc(NR), *tmpw = malloc(NR);
    int maxc = NORB * q * NR + 8;
    int *ci = malloc(sizeof(int)*maxc), *cp = malloc(sizeof(int)*maxc), *cv = malloc(sizeof(int)*maxc);
    long long *cd = malloc(sizeof(long long)*maxc);
    uint8_t *cenbuf = malloc((size_t)NORB*q*NR);

    long long it = 0, since_improve = 0, nkicks = 0;
    double last_flush = -1e9;
    int flush_pending = 1;
    if (outpath) write_code(outpath, bestrep);      /* never leave --out empty */

    while (uncovered > 0) {
        double el = now() - t0;
        if (el > tlimit) break;
        if (outpath && flush_pending && el - last_flush > 1.0) {
            write_code(outpath, bestrep); last_flush = el; flush_pending = 0;
        }
        it++;
        long long uw = pick_uncovered(&grng);
        if (uw < 0) break;
        word_of(uw, u);

        /* effective distance of every (orbit, delta) pair to u */
        int dmin = n + 2;
        for (int i = 0; i < NORB; i++)
            for (int d = 0; d < q; d++) {
                int rad = R - (d != 0);
                if (rad < 0) continue;
                const uint8_t *e = rep + (size_t)i*NR;
                uint8_t *cb = cenbuf + ((size_t)i*q + d)*NR;
                int dd = 0;
                for (int j = 0; j < NR; j++) {
                    int cvl = (e[j] + d) % q;
                    cb[j] = (uint8_t)cvl;
                    dd += (cvl != u[j]);
                }
                int eff = dd + (d != 0);
                if (eff < dmin) dmin = eff;
            }
        int dtar = (dmin <= R+1) ? R+1 : dmin;

        int nc = 0;
        for (int i = 0; i < NORB && nc < maxc-1; i++)
            for (int d = 0; d < q; d++) {
                int rad = R - (d != 0);
                if (rad < 0) continue;
                const uint8_t *cb = cenbuf + ((size_t)i*q + d)*NR;
                int dd = 0;
                for (int j = 0; j < NR; j++) dd += (cb[j] != u[j]);
                if (dd + (d != 0) != dtar) continue;
                for (int p = 0; p < NR && nc < maxc-1; p++) {
                    if (cb[p] == u[p]) continue;
                    ci[nc] = i; cp[nc] = p; cv[nc] = (u[p] - d % q + q) % q; nc++;
                }
            }
        if (nc == 0) {
            ci[0] = (int)rmod(&grng,(uint64_t)NORB);
            cp[0] = (int)rmod(&grng,(uint64_t)NR);
            cv[0] = (int)rmod(&grng,(uint64_t)q);
            nc = 1;
        }
        int take = nc < candcap ? nc : candcap;
        for (int s = 0; s < take; s++) {
            int j = s + (int)rmod(&grng,(uint64_t)(nc-s));
            int a=ci[s],b=cp[s],c2=cv[s];
            ci[s]=ci[j];cp[s]=cp[j];cv[s]=cv[j];
            ci[j]=a;cp[j]=b;cv[j]=c2;
        }

#pragma omp parallel for schedule(dynamic,1) if(take > 3 && par_cand)
        for (int s = 0; s < take; s++)
            cd[s] = move_delta(ci[s], cp[s], cv[s], 0, 0);

        int bi = -1, bp = -1, bv = -1, nties = 0;
        long long bestd = 1LL<<60;
        for (int s = 0; s < take; s++) {
            if (cv[s] == rep[(size_t)ci[s]*NR + cp[s]]) continue;
            if (cd[s] < bestd) { bestd = cd[s]; bi=ci[s]; bp=cp[s]; bv=cv[s]; nties=1; }
            else if (cd[s] == bestd) { nties++; if (rmod(&grng,(uint64_t)nties)==0){bi=ci[s];bp=cp[s];bv=cv[s];} }
        }
        if (bi < 0) { int s=(int)rmod(&grng,(uint64_t)take); bi=ci[s];bp=cp[s];bv=cv[s];
                      if (bv == rep[(size_t)bi*NR+bp]) bv = (bv+1)%q; }

        move_delta(bi, bp, bv, 1, !par_cand);

        if (uncovered < best_uncovered) {
            best_uncovered = uncovered;
            memcpy(bestrep, rep, (size_t)NORB*NR);
            since_improve = 0; flush_pending = 1;
            /* printed unconditionally: if the process is killed before it can
             * print SYMRESULT, this is what tells the driver what it achieved */
            printf("# improved: word_uncovered=%lld red=%lld it=%lld t=%.1f\n",
                   best_uncovered*q, best_uncovered, it, now()-t0);
            fflush(stdout);
            if (best_uncovered == 0) break;
        } else if (++since_improve > stall_limit) {
            for (int t = 0; t < kick; t++) {
                int i = (int)rmod(&grng,(uint64_t)NORB);
                orbit_apply(i, -1);
                long long w = pick_uncovered(&grng);
                if (w >= 0) {
                    word_of(w, tmpw);
                    memcpy(rep + (size_t)i*NR, tmpw, NR);
                    for (int t2 = 0; t2 < R; t2++)
                        rep[(size_t)i*NR + rmod(&grng,(uint64_t)NR)] = (uint8_t)rmod(&grng,(uint64_t)q);
                } else {
                    for (int j = 0; j < NR; j++)
                        rep[(size_t)i*NR + j] = (uint8_t)rmod(&grng,(uint64_t)q);
                }
                orbit_apply(i, +1);
            }
            nkicks++;
            since_improve = 0;
        }
    }

    double el = now() - t0;
    printf("SYMRESULT q=%d n=%d R=%d M=%d orbits=%d red_uncovered=%lld "
           "word_uncovered=%lld iters=%lld kicks=%lld time=%.2f rate=%.0f\n",
           q, n, R, NORB*q, NORB, best_uncovered, best_uncovered*q, it, nkicks, el,
           it/(el>0?el:1));
    if (outpath) write_code(outpath, bestrep);
    return best_uncovered == 0 ? 0 : 1;
}
