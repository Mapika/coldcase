/*
 * covsearch.c -- local search for q-ary covering codes.
 *
 * Problem.  K_q(n,R) is the least M such that some M-subset C of Z_q^n has
 * every word of Z_q^n within Hamming distance R of C.  This program fixes M and
 * minimises the number of UNCOVERED words; reaching 0 proves K_q(n,R) <= M.
 *
 * State.  cnt[w] = number of codewords covering the word w (uint16; M <= 65535).
 * `uncovered` counts the w with cnt[w] == 0.  Both are maintained incrementally.
 *
 * The move.  Change one coordinate of one codeword: c -> c' with c'_p = v.
 * The key identity that makes this cheap:
 *
 *     d(w,c') = d(w,c) + [w_p == c_p] - [w_p == v]
 *
 * so a word can only leave the ball of c if d(w,c) == R and w_p == c_p, and can
 * only enter the ball of c' if d(w,c') == R and w_p == v.  Both sets are indexed
 * by the SAME set of (n-1)-suffixes -- namely the words that differ from c in
 * exactly R of the coordinates other than p -- so a single enumeration of
 *
 *     S = C(n-1,R) * (q-1)^R      patterns
 *
 * yields both: for each pattern the leaving word is base+off and the entering
 * word is base+off+dv where dv = (v - c_p)*q^(n-1-p).  A move therefore costs
 * 2S memory touches instead of two full balls of volume sum_i C(n,i)(q-1)^i.
 * For (q,n,R)=(6,10,4) that is 157k touches instead of 295k, and evaluating a
 * candidate without committing costs the same S reads.
 *
 * Search.  Ostergard-style guided local search:
 *   1. pick a uniformly random uncovered word u;
 *   2. collect the moves that would cover u -- every codeword at distance
 *      exactly R+1 from u, moved one step towards u (if none is that close,
 *      fall back to codewords at the minimum distance, which make progress
 *      without covering u yet);
 *   3. evaluate a random sample of at most `cand` of them exactly, in parallel;
 *   4. commit the best non-tabu one (aspiration: tabu is overridden by a move
 *      that beats the incumbent), ties broken at random;
 *   5. forbid restoring that (codeword, position, old value) for `tabu` steps.
 * On stagnation the state is kicked by randomising a few codewords.
 *
 * Both the candidate evaluations and the commit are OpenMP-parallel over the
 * C(n-1,R) position-subsets; within one sphere every word occurs exactly once,
 * so the commit needs no atomics on cnt.
 *
 * Build:  gcc -O3 -march=native -fopenmp -o covsearch covsearch.c -lm
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

/* ------------------------------------------------------------------ */
/* parameters                                                          */
/* ------------------------------------------------------------------ */

static int q, n, R, M;
static long long NTOT;             /* q^n                              */
static long long pw[32];           /* pw[p] = q^(n-1-p)                */

static uint16_t *cnt;              /* coverage multiplicity per word    */
static long long uncovered;

static uint8_t *code;              /* M*n digits                        */
static long long *cidx;            /* M word indices                    */

static uint8_t *bestcode;
static long long best_uncovered;

/* position subsets: for each excluded position p, all C(n-1,R) ways to
 * choose R of the remaining positions. */
static int NCOMB;
static uint8_t *combs;             /* n * NCOMB * R                     */

static unsigned nthreads = 1;

/* ------------------------------------------------------------------ */
/* rng (xoshiro256++, one state per thread)                            */
/* ------------------------------------------------------------------ */

typedef struct { uint64_t s[4]; } rng_t;
static rng_t grng;

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t rnext(rng_t *r) {
    uint64_t *s = r->s, res = rotl(s[0] + s[3], 23) + s[0], t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return res;
}
static void rseed(rng_t *r, uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        r->s[i] = z ^ (z >> 31);
    }
}
static inline uint64_t rmod(rng_t *r, uint64_t m) { return rnext(r) % m; }

/* ------------------------------------------------------------------ */
/* combinatorial helpers                                               */
/* ------------------------------------------------------------------ */

static long long ipow(long long b, int e) {
    long long r = 1; while (e-- > 0) r *= b; return r;
}
static long long binom(int a, int b) {
    if (b < 0 || b > a) return 0;
    long long r = 1;
    if (b > a - b) b = a - b;
    for (int i = 0; i < b; i++) r = r * (a - i) / (i + 1);
    return r;
}

static void build_combs(void) {
    NCOMB = (int)binom(n - 1, R);
    combs = malloc((size_t)n * NCOMB * (R ? R : 1));
    for (int p = 0; p < n; p++) {
        int rest[32], nr = 0;
        for (int j = 0; j < n; j++) if (j != p) rest[nr++] = j;
        int c[16];
        for (int i = 0; i < R; i++) c[i] = i;
        for (int k = 0; k < NCOMB; k++) {
            for (int i = 0; i < R; i++)
                combs[((size_t)p * NCOMB + k) * (R ? R : 1) + i] = (uint8_t)rest[c[i]];
            /* next combination */
            int i = R - 1;
            while (i >= 0 && c[i] == nr - R + i) i--;
            if (i < 0) break;
            c[i]++;
            for (int j = i + 1; j < R; j++) c[j] = c[j - 1] + 1;
        }
    }
}

/* offs for a codeword: offs[j*q + k] = (k-th value != c_j, minus c_j) * pw[j] */
static void make_offs(const uint8_t *c, long long *offs) {
    for (int j = 0; j < n; j++) {
        int k = 0;
        for (int v = 0; v < q; v++)
            if (v != c[j]) offs[(size_t)j * q + (k++)] = (long long)(v - c[j]) * pw[j];
    }
}

/* ------------------------------------------------------------------ */
/* full-ball marking (initialisation only)                             */
/* ------------------------------------------------------------------ */

static long long ball_apply_rec(long long base, const long long *offs,
                                const uint8_t *c, int startpos, int rem, int delta)
{
    long long dunc = 0;
    if (rem == 0) return 0;
    for (int j = startpos; j < n; j++) {
        for (int k = 0; k < q - 1; k++) {
            long long w = base + offs[(size_t)j * q + k];
            if (delta > 0) { if (cnt[w]++ == 0) dunc--; }
            else           { if (--cnt[w] == 0) dunc++; }
            dunc += ball_apply_rec(w, offs, c, j + 1, rem - 1, delta);
        }
    }
    return dunc;
}

/* add (delta=+1) or remove (delta=-1) the whole radius-R ball around c */
static void ball_apply(const uint8_t *c, long long base, int delta)
{
    long long *offs = malloc(sizeof(long long) * n * q);
    make_offs(c, offs);
    long long dunc = 0;
    if (delta > 0) { if (cnt[base]++ == 0) dunc--; }
    else           { if (--cnt[base] == 0) dunc++; }
    dunc += ball_apply_rec(base, offs, c, 0, R, delta);
    uncovered += dunc;
    free(offs);
}

/* ------------------------------------------------------------------ */
/* the move: sphere enumeration                                        */
/* ------------------------------------------------------------------ */

/* Walk the (q-1)^R value assignments for one position-subset.
 * mode 0: evaluate (read-only), accumulate delta-uncovered into *acc.
 * mode 1: commit. */
static inline long long sphere_walk(const uint8_t *pos, long long base,
                                    const long long *offs, long long dv, int mode)
{
    long long dunc = 0;
    int k[16];
    long long partial[17];
    if (R == 0) {
        long long a = base, b = base + dv;
        if (mode) {
            if (--cnt[a] == 0) dunc++;
            if (cnt[b]++ == 0) dunc--;
        } else {
            if (cnt[a] == 1) dunc++;
            if (cnt[b] == 0) dunc--;
        }
        return dunc;
    }
    for (int i = 0; i < R; i++) k[i] = 0;
    partial[0] = base;
    for (int i = 0; i < R; i++)
        partial[i + 1] = partial[i] + offs[(size_t)pos[i] * q + 0];
    for (;;) {
        long long a = partial[R], b = a + dv;
        if (mode) {
            if (--cnt[a] == 0) dunc++;
            if (cnt[b]++ == 0) dunc--;
        } else {
            if (cnt[a] == 1) dunc++;
            if (cnt[b] == 0) dunc--;
        }
        /* odometer over the last coordinate first */
        int i = R - 1;
        while (i >= 0 && k[i] == q - 2) { k[i] = 0; i--; }
        if (i < 0) break;
        k[i]++;
        for (int j = i; j < R; j++)
            partial[j + 1] = partial[j] + offs[(size_t)pos[j] * q + k[j]];
    }
    return dunc;
}

/* Evaluate or commit the move "codeword i, position p, new value v".
 * Returns the change in `uncovered` (negative is good).
 * `par` enables OpenMP over the position-subsets; use it only for the commit
 * and for instances where a single sphere is large, because the candidate
 * loop in the search is itself parallelised and nesting the two is a loss. */
static long long move_delta(int i, int p, int v, int commit, int par)
{
    const uint8_t *c = code + (size_t)i * n;
    long long dv = (long long)(v - c[p]) * pw[p];
    long long base = cidx[i];

    long long op[32 * 32];
    make_offs(c, op);

    long long dunc = 0;
    const uint8_t *cb = combs + (size_t)p * NCOMB * (R ? R : 1);
    int stride = R ? R : 1;

    if (par) {
#pragma omp parallel for schedule(static) reduction(+:dunc)
        for (int k = 0; k < NCOMB; k++)
            dunc += sphere_walk(cb + (size_t)k * stride, base, op, dv, commit);
    } else {
        for (int k = 0; k < NCOMB; k++)
            dunc += sphere_walk(cb + (size_t)k * stride, base, op, dv, commit);
    }

    if (commit) {
        uint8_t *cw = code + (size_t)i * n;
        cidx[i] += dv;
        cw[p] = (uint8_t)v;
        uncovered += dunc;
    }
    return dunc;
}

/* ------------------------------------------------------------------ */
/* code I/O                                                           */
/* ------------------------------------------------------------------ */

static const char *DIG = "0123456789abcdefghijklmnopqrstuvwxyz";

static void write_code(const char *path, const uint8_t *cw)
{
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return; }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < n; j++) {
            if (q <= 36) fputc(DIG[cw[(size_t)i * n + j]], f);
            else fprintf(f, "%d%s", cw[(size_t)i * n + j], j + 1 < n ? " " : "");
        }
        fputc('\n', f);
    }
    fclose(f);
}

static int read_code(const char *path, uint8_t *cw, int maxM)
{
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    char line[4096];
    int m = 0;
    while (fgets(line, sizeof line, f)) {
        char *h = strchr(line, '#'); if (h) *h = 0;
        int len = 0; uint8_t tmp[64];
        int spaced = (strchr(line, ' ') || strchr(line, ',') || strchr(line, '\t'));
        if (spaced) {
            char *tok = strtok(line, " ,\t\r\n");
            while (tok && len < n) { tmp[len++] = (uint8_t)atoi(tok); tok = strtok(NULL, " ,\t\r\n"); }
        } else {
            for (char *s = line; *s && *s != '\n' && *s != '\r'; s++) {
                const char *d = strchr(DIG, *s);
                if (!d) continue;
                if (len < n) tmp[len++] = (uint8_t)(d - DIG);
            }
        }
        if (len == 0) continue;
        if (len != n) { fprintf(stderr, "%s: bad codeword length %d\n", path, len); exit(1); }
        if (m >= maxM) { fprintf(stderr, "%s: more than M=%d codewords\n", path, maxM); exit(1); }
        memcpy(cw + (size_t)m * n, tmp, n);
        m++;
    }
    fclose(f);
    return m;
}

static int count_code(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    char line[4096];
    int m = 0;
    while (fgets(line, sizeof line, f)) {
        char *h = strchr(line, '#'); if (h) *h = 0;
        int any = 0;
        for (char *s2 = line; *s2; s2++) if (strchr(DIG, *s2)) { any = 1; break; }
        if (any) m++;
    }
    fclose(f);
    return m;
}

static long long widx(const uint8_t *w)
{
    long long x = 0;
    for (int j = 0; j < n; j++) x += (long long)w[j] * pw[j];
    return x;
}

/* ------------------------------------------------------------------ */
/* remove-and-repair: how many words does codeword i cover ALONE?      */
/* ------------------------------------------------------------------ */

static long long unique_rec(long long base, const long long *offs,
                            int startpos, int rem)
{
    long long u = 0;
    if (rem == 0) return 0;
    for (int j = startpos; j < n; j++)
        for (int k = 0; k < q - 1; k++) {
            long long w = base + offs[(size_t)j * q + k];
            u += (cnt[w] == 1);
            u += unique_rec(w, offs, j + 1, rem - 1);
        }
    return u;
}

static long long unique_cover(const uint8_t *c, long long base)
{
    long long offs[32 * 32];
    make_offs(c, offs);
    return (cnt[base] == 1) + unique_rec(base, offs, 0, R);
}

/* ------------------------------------------------------------------ */
/* full rebuild of cnt from `code`                                     */
/* ------------------------------------------------------------------ */

static void rebuild(void)
{
    memset(cnt, 0, sizeof(uint16_t) * (size_t)NTOT);
    uncovered = NTOT;
    for (int i = 0; i < M; i++) {
        cidx[i] = widx(code + (size_t)i * n);
        ball_apply(code + (size_t)i * n, cidx[i], +1);
    }
}

/* ------------------------------------------------------------------ */
/* search                                                              */
/* ------------------------------------------------------------------ */

static long long pick_uncovered(rng_t *r)
{
    /* Reservoir-free: sample uniformly at random until an uncovered word is
     * hit, but fall back to a linear scan from a random offset when uncovered
     * words are rare (which is exactly the endgame). */
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

static void word_of(long long idx, uint8_t *w)
{
    for (int j = 0; j < n; j++) { w[j] = (uint8_t)((idx / pw[j]) % q); }
}

int main(int argc, char **argv)
{
    double tlimit = 60.0;
    uint64_t seed = 12345;
    const char *inpath = NULL, *outpath = NULL;
    int cand = 100000;          /* evaluate this many candidate moves per step */
    int tabu_len = 0;           /* tabu tenure; measured to hurt, so off       */
    int kick = 2;               /* codewords teleported on stagnation          */
    int noise = 0;              /* WalkSAT noise in per-mille; measured to hurt */
    int init_greedy = 1;        /* place initial codewords on uncovered words   */
    int sa_mode = 0;            /* --sa: focused simulated annealing            */
    double t0temp = 0.6, cool = 0.999999, tmin = 0.02, T = 0.6;
    long long maxiter = -1;
    int verbose = 1;
    long long stall_limit = 5000;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        #define ARG(x) (!strcmp(a, x) && i + 1 < argc)
        if      (ARG("-q")) q = atoi(argv[++i]);
        else if (ARG("-n")) n = atoi(argv[++i]);
        else if (ARG("-R")) R = atoi(argv[++i]);
        else if (ARG("-M")) M = atoi(argv[++i]);
        else if (ARG("-t")) tlimit = atof(argv[++i]);
        else if (ARG("-s")) seed = strtoull(argv[++i], NULL, 10);
        else if (ARG("--in")) inpath = argv[++i];
        else if (ARG("--out")) outpath = argv[++i];
        else if (ARG("--cand")) cand = atoi(argv[++i]);
        else if (ARG("--tabu")) tabu_len = atoi(argv[++i]);
        else if (ARG("--kick")) kick = atoi(argv[++i]);
        else if (ARG("--noise")) noise = atoi(argv[++i]);
        else if (ARG("--iters")) maxiter = atoll(argv[++i]);
        else if (ARG("--stall")) stall_limit = atoll(argv[++i]);
        else if (ARG("--threads")) nthreads = atoi(argv[++i]);
        else if (!strcmp(a, "--random-init")) init_greedy = 0;
        else if (!strcmp(a, "--sa")) sa_mode = 1;
        else if (ARG("--t0")) t0temp = atof(argv[++i]);
        else if (ARG("--cool")) cool = atof(argv[++i]);
        else if (ARG("--tmin")) tmin = atof(argv[++i]);
        else if (!strcmp(a, "--quiet")) verbose = 0;
        else { fprintf(stderr, "unknown/incomplete option %s\n", a); return 2; }
        #undef ARG
    }
    if (q < 2 || n < 1 || R < 0 || M < 1) {
        fprintf(stderr,
          "usage: covsearch -q Q -n N -R R -M M [-t sec] [-s seed] [--in f] [--out f]\n"
          "                 [--cand K] [--tabu T] [--kick K] [--noise PERMILLE] [--iters I]\n"
          "                 [--stall S] [--sa --t0 T0 --cool C --tmin TMIN]\n"
          "                 [--random-init]\n"
          "                 [--threads T] [--quiet]\n");
        return 2;
    }
#ifdef _OPENMP
    if (nthreads) omp_set_num_threads(nthreads);
    nthreads = omp_get_max_threads();
#endif

    NTOT = 1;
    for (int i = 0; i < n; i++) {
        if (NTOT > (long long)4e10 / q) { fprintf(stderr, "q^n too large\n"); return 2; }
        NTOT *= q;
    }
    for (int p = 0; p < n; p++) pw[p] = ipow(q, n - 1 - p);
    if (M > 65535) { fprintf(stderr, "M > 65535 not supported (uint16 counters)\n"); return 2; }

    cnt = malloc(sizeof(uint16_t) * (size_t)NTOT);
    if (!cnt) { fprintf(stderr, "cannot allocate %.1f GB for counters\n",
                        sizeof(uint16_t) * (double)NTOT / 1e9); return 2; }
    code = malloc((size_t)M * n);
    bestcode = malloc((size_t)M * n);
    cidx = malloc(sizeof(long long) * M);
    build_combs();
    rseed(&grng, seed);

    long long S = binom(n - 1, R) * ipow(q - 1, R);
    long long V = 0;
    for (int i = 0; i <= R && i <= n; i++) V += binom(n, i) * ipow(q - 1, i);
    if (verbose) {
        printf("# covsearch q=%d n=%d R=%d M=%d  q^n=%lld  ball=%lld  sphere-move=%lld  threads=%u\n",
               q, n, R, M, NTOT, V, S, nthreads);
        printf("# sphere-covering bound: M >= %lld\n", (NTOT + V - 1) / V);
        fflush(stdout);
    }

    /* Initial code.  A seed file may hold MORE than M codewords; in that case
     * we descend by remove-and-repair: repeatedly delete the codeword that
     * covers the fewest words that nothing else covers, which is the cheapest
     * deletion in terms of newly uncovered words. */
    int have = 0;
    if (inpath) {
        int mseed = count_code(inpath);
        if (mseed > M) {
            uint8_t *seed_code = malloc((size_t)mseed * n);
            long long *seed_idx = malloc(sizeof(long long) * mseed);
            read_code(inpath, seed_code, mseed);
            memset(cnt, 0, sizeof(uint16_t) * (size_t)NTOT);
            uncovered = NTOT;
            for (int i = 0; i < mseed; i++) {
                seed_idx[i] = widx(seed_code + (size_t)i * n);
                ball_apply(seed_code + (size_t)i * n, seed_idx[i], +1);
            }
            if (verbose)
                printf("# seed has %d codewords (uncovered=%lld); dropping %d by "
                       "remove-and-repair\n", mseed, uncovered, mseed - M);
            while (mseed > M) {
                int worst = 0; long long wu = -1;
                for (int i = 0; i < mseed; i++) {
                    long long u2 = unique_cover(seed_code + (size_t)i * n, seed_idx[i]);
                    if (wu < 0 || u2 < wu) { wu = u2; worst = i; }
                }
                ball_apply(seed_code + (size_t)worst * n, seed_idx[worst], -1);
                mseed--;
                memmove(seed_code + (size_t)worst * n,
                        seed_code + (size_t)(worst + 1) * n, (size_t)(mseed - worst) * n);
                memmove(seed_idx + worst, seed_idx + worst + 1,
                        sizeof(long long) * (size_t)(mseed - worst));
            }
            memcpy(code, seed_code, (size_t)M * n);
            free(seed_code); free(seed_idx);
            have = M;
        } else {
            have = read_code(inpath, code, M);
        }
    }
    if (have < M && init_greedy) {
        /* Greedy-random initialisation: place each new codeword ON a word that
         * is still uncovered.  Every placement then removes a whole ball worth
         * of uncovered words instead of landing in already-covered territory,
         * which starts the search an order of magnitude closer than a uniform
         * random code. */
        memset(cnt, 0, sizeof(uint16_t) * (size_t)NTOT);
        uncovered = NTOT;
        for (int i = 0; i < have; i++) {
            cidx[i] = widx(code + (size_t)i * n);
            ball_apply(code + (size_t)i * n, cidx[i], +1);
        }
        for (int i = have; i < M; i++) {
            long long w = pick_uncovered(&grng);
            if (w < 0) w = (long long)rmod(&grng, (uint64_t)NTOT);
            word_of(w, code + (size_t)i * n);
            cidx[i] = w;
            ball_apply(code + (size_t)i * n, cidx[i], +1);
        }
    } else {
        for (int i = have; i < M; i++)
            for (int j = 0; j < n; j++)
                code[(size_t)i * n + j] = (uint8_t)rmod(&grng, q);
        rebuild();
    }
    memcpy(bestcode, code, (size_t)M * n);
    best_uncovered = uncovered;
    if (verbose) { printf("# init uncovered=%lld\n", uncovered); fflush(stdout); }

    /* tabu table */
    int *tabu = calloc((size_t)M * n * q, sizeof(int));

    struct timespec t0, tn;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    uint8_t *u = malloc(n), *tmpw = malloc(n);
    int *cand_i = malloc(sizeof(int) * (size_t)M * (R + 2));
    int *cand_p = malloc(sizeof(int) * (size_t)M * (R + 2));
    int *dist = malloc(sizeof(int) * M);
    long long *cand_d = malloc(sizeof(long long) * (size_t)M * (R + 2));
    /* Parallelise over candidates when there are enough of them and each
     * sphere is small; otherwise parallelise inside the sphere walk. */
    int par_cand = (NCOMB < 256);

    long long it = 0, since_improve = 0, nkicks = 0;
    double last_report = 0;
    T = t0temp;

    while (uncovered > 0) {
        if ((it & 0xFF) == 0) {
            clock_gettime(CLOCK_MONOTONIC, &tn);
            double el = (tn.tv_sec - t0.tv_sec) + 1e-9 * (tn.tv_nsec - t0.tv_nsec);
            if (el > tlimit) break;
            if (verbose && el - last_report > 10.0) {
                last_report = el;
                printf("# t=%6.1fs it=%-10lld unc=%-10lld best=%lld (%.0f it/s)\n",
                       el, it, uncovered, best_uncovered, it / el);
                fflush(stdout);
            }
        }
        if (maxiter >= 0 && it >= maxiter) break;
        it++;

        long long uw = pick_uncovered(&grng);
        if (uw < 0) break;
        word_of(uw, u);

        /* distances to all codewords */
        int dmin = n + 1;
        for (int i = 0; i < M; i++) {
            const uint8_t *c = code + (size_t)i * n;
            int d = 0;
            for (int j = 0; j < n; j++) d += (c[j] != u[j]);
            dist[i] = d;
            if (d < dmin) dmin = d;
        }
        /* target distance: R+1 if reachable in one move, else the minimum */
        int dtar = (dmin <= R + 1) ? R + 1 : dmin;

        int ncand = 0;
        for (int i = 0; i < M; i++) {
            if (dist[i] != dtar) continue;
            const uint8_t *c = code + (size_t)i * n;
            for (int p = 0; p < n; p++) {
                if (c[p] == u[p]) continue;
                cand_i[ncand] = i; cand_p[ncand] = p; ncand++;
            }
        }
        if (ncand == 0) {           /* should not happen, but stay safe */
            int i = (int)rmod(&grng, M), p = (int)rmod(&grng, n);
            cand_i[0] = i; cand_p[0] = p; ncand = 1;
        }

        /* sample at most `cand` of them */
        int take = ncand < cand ? ncand : cand;
        for (int s = 0; s < take; s++) {
            int j = s + (int)rmod(&grng, (uint64_t)(ncand - s));
            int ti = cand_i[s], tp = cand_p[s];
            cand_i[s] = cand_i[j]; cand_p[s] = cand_p[j];
            cand_i[j] = ti; cand_p[j] = tp;
        }

        int bi = -1, bp = -1, bv = -1;
        if (sa_mode) {
            /* Focused simulated annealing: propose ONE random move that covers
             * u, evaluate it, and accept by Metropolis.  One sphere walk per
             * iteration instead of |candidates|, so this trades move quality
             * for one to two orders of magnitude more moves per second. */
            int s2 = (int)rmod(&grng, (uint64_t)take);
            int i = cand_i[s2], p = cand_p[s2], v = u[p];
            long long d = move_delta(i, p, v, 0, !par_cand);
            int accept;
            if (d <= 0) accept = 1;
            else {
                double pr = exp(-(double)d / T);
                accept = (rnext(&grng) >> 11) * 0x1.0p-53 < pr;
            }
            if (accept) {
                move_delta(i, p, v, 1, !par_cand);
                if (uncovered < best_uncovered) {
                    best_uncovered = uncovered;
                    memcpy(bestcode, code, (size_t)M * n);
                    since_improve = 0;
                    if (verbose) {
                        clock_gettime(CLOCK_MONOTONIC, &tn);
                        double el2 = (tn.tv_sec - t0.tv_sec) + 1e-9 * (tn.tv_nsec - t0.tv_nsec);
                        printf("# improved: unc=%lld at it=%lld t=%.1fs T=%.3f\n",
                               best_uncovered, it, el2, T);
                        fflush(stdout);
                    }
                    if (best_uncovered == 0) break;
                }
            }
            T *= cool;
            if (T < tmin) T = t0temp;      /* reheat */
            if (++since_improve > stall_limit) { T = t0temp; since_improve = 0; nkicks++; }
            continue;
        }
        if (noise && (int)rmod(&grng, 1000) < noise) {
            /* Walksat-style noise: take a random covering move without
             * evaluating it.  Cheap, and it is what gets the search off the
             * large plateaus where every move has delta 0. */
            int s = (int)rmod(&grng, (uint64_t)take);
            bi = cand_i[s]; bp = cand_p[s]; bv = u[bp];
        } else {
            long long bestd = 1LL << 60;
            int nties = 0;
            /* Evaluate the sampled candidates in parallel: one thread per
             * candidate.  Each evaluation only READS cnt[], so no locking. */
#pragma omp parallel for schedule(dynamic, 1) if(take > 3 && par_cand)
            for (int s = 0; s < take; s++)
                cand_d[s] = move_delta(cand_i[s], cand_p[s], u[cand_p[s]], 0, 0);
            for (int s = 0; s < take; s++) {
                int i = cand_i[s], p = cand_p[s], v = u[p];
                long long d = cand_d[s];
                /* tabu forbids RESTORING a value we recently moved away from,
                 * so the test is on the value we are moving TO. */
                int is_tabu = tabu[((size_t)i * n + p) * q + v] > it;
                if (is_tabu && uncovered + d >= best_uncovered) continue;  /* aspiration */
                if (d < bestd) { bestd = d; bi = i; bp = p; bv = v; nties = 1; }
                else if (d == bestd) { nties++; if (rmod(&grng, (uint64_t)nties) == 0) { bi = i; bp = p; bv = v; } }
            }
            if (bi < 0) {   /* everything tabu: take a random candidate anyway */
                int s = (int)rmod(&grng, (uint64_t)take);
                bi = cand_i[s]; bp = cand_p[s]; bv = u[bp];
            }
        }

        int oldv = code[(size_t)bi * n + bp];
        move_delta(bi, bp, bv, 1, !par_cand);
        tabu[((size_t)bi * n + bp) * q + oldv] = (int)(it + tabu_len + rmod(&grng, 4));

        if (uncovered < best_uncovered) {
            best_uncovered = uncovered;
            memcpy(bestcode, code, (size_t)M * n);
            since_improve = 0;
            if (verbose) {
                clock_gettime(CLOCK_MONOTONIC, &tn);
                double el = (tn.tv_sec - t0.tv_sec) + 1e-9 * (tn.tv_nsec - t0.tv_nsec);
                printf("# improved: unc=%lld at it=%lld t=%.1fs\n", best_uncovered, it, el);
                fflush(stdout);
            }
            if (best_uncovered == 0) break;
        } else if (++since_improve > stall_limit) {
            /* Kick: teleport a few codewords onto (perturbed) uncovered
             * words.  Landing on an uncovered word is strictly more useful
             * than landing anywhere, and the perturbation keeps it from
             * undoing itself on the next iteration. */
            for (int t = 0; t < kick; t++) {
                int i = (int)rmod(&grng, M);
                ball_apply(code + (size_t)i * n, cidx[i], -1);
                long long w = pick_uncovered(&grng);
                if (w >= 0) {
                    word_of(w, tmpw);
                    memcpy(code + (size_t)i * n, tmpw, n);
                    for (int t2 = 0; t2 < R; t2++)
                        code[(size_t)i * n + rmod(&grng, n)] = (uint8_t)rmod(&grng, q);
                } else {
                    for (int j = 0; j < n; j++)
                        code[(size_t)i * n + j] = (uint8_t)rmod(&grng, q);
                }
                cidx[i] = widx(code + (size_t)i * n);
                ball_apply(code + (size_t)i * n, cidx[i], +1);
            }
            nkicks++;
            memset(tabu, 0, sizeof(int) * (size_t)M * n * q);
            since_improve = 0;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &tn);
    double el = (tn.tv_sec - t0.tv_sec) + 1e-9 * (tn.tv_nsec - t0.tv_nsec);
    printf("RESULT q=%d n=%d R=%d M=%d uncovered=%lld iters=%lld kicks=%lld "
           "time=%.2f rate=%.0f\n",
           q, n, R, M, best_uncovered, it, nkicks, el, it / (el > 0 ? el : 1));
    if (outpath) write_code(outpath, bestcode);
    return best_uncovered == 0 ? 0 : 1;
}
