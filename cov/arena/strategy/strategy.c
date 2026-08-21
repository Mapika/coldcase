/*
 * strategy.c -- arena entry "strategy": local search for q-ary covering codes,
 * with the SEARCH STRATEGY changed relative to cov/search/covsearch.c.
 *
 * The problem, the state (cnt[w] = coverage multiplicity), the sphere-difference
 * move and its bookkeeping are taken verbatim from covsearch.c -- they are
 * measured-correct and this entry does not try to make them faster.  What is
 * different is *which* moves the search makes and *how the time budget is spent*:
 *
 *  1. TWO move operators, chosen by an online bandit:
 *       FINE     -- covsearch's focused single-coordinate move (unchanged).
 *       TELEPORT -- ruin-and-recreate: delete the least-uniquely-useful of k_rem
 *                   sampled codewords and re-insert it at the best of k_add
 *                   sampled uncovered words, scored by how many uncovered words
 *                   its whole radius-R ball contains.  Revert if it worsened.
 *     The bandit scores each operator by RECORD improvement per second over 0.1 s
 *     blocks, with hysteresis towards FINE.  Which operator wins depends on the
 *     instance AND on how much CPU the machine is actually giving you: on
 *     K_8(9,4)@940 under load TELEPORT is 4-5x the better arm and --ops auto
 *     beats --ops fine by ~0.5 M uncovered; on an idle box the ranking reverses.
 *
 *  2. Greedy max-ball construction: each initial codeword goes on the best of
 *     k sampled uncovered words instead of a uniformly random uncovered word,
 *     with k throttled to keep the construction inside a fraction of the budget.
 *     On big instances this, not the local search, decides the score.
 *
 *  3. The uncovered word to work on is picked with covsearch's size-biased scan
 *     (isolated uncovered words preferred), which measured 17% better than
 *     uniform selection; the uniform cache is the fallback for huge instances.
 *
 *  4. The time limit is honoured every iteration (covsearch checks every 256
 *     iterations, which overruns by ~3x on instances that run at 1.5 it/s), and
 *     the best code is flushed to --out whenever it improves, so a kill at the
 *     deadline still leaves a valid file.  Repeated codewords are removed on the
 *     way out, because the judge scores a cover with duplicates as 0, not 1000.
 *
 * Build:  gcc -O3 -march=native -fopenmp -o strategy strategy.c -lm
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

static uint16_t *cnt;
static long long uncovered;

static uint8_t *code;
static long long *cidx;

static uint8_t *bestcode;
static long long best_uncovered;

static int NCOMB;
static uint8_t *combs;

static unsigned nthreads = 1;
static int par_cand = 0;           /* parallelise over FINE candidates  */
static int par_ball = 0;           /* parallelise over ball probes      */

static const char *outpath = NULL;
static int verbose = 1;

/* ------------------------------------------------------------------ */
/* rng (xoshiro256++)                                                  */
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

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* ------------------------------------------------------------------ */
/* combinatorial helpers                                               */
/* ------------------------------------------------------------------ */

static long long ipow(long long b, int e) { long long r = 1; while (e-- > 0) r *= b; return r; }
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
            int i = R - 1;
            while (i >= 0 && c[i] == nr - R + i) i--;
            if (i < 0) break;
            c[i]++;
            for (int j = i + 1; j < R; j++) c[j] = c[j - 1] + 1;
        }
    }
}

static void make_offs(const uint8_t *c, long long *offs) {
    for (int j = 0; j < n; j++) {
        int k = 0;
        for (int v = 0; v < q; v++)
            if (v != c[j]) offs[(size_t)j * q + (k++)] = (long long)(v - c[j]) * pw[j];
    }
}

static void word_of(long long idx, uint8_t *w) {
    for (int j = 0; j < n; j++) w[j] = (uint8_t)((idx / pw[j]) % q);
}
static long long widx(const uint8_t *w) {
    long long x = 0;
    for (int j = 0; j < n; j++) x += (long long)w[j] * pw[j];
    return x;
}

/* ------------------------------------------------------------------ */
/* full-ball walks                                                     */
/* ------------------------------------------------------------------ */

static long long ball_apply_rec(long long base, const long long *offs,
                                int startpos, int rem, int delta)
{
    long long dunc = 0;
    if (rem == 0) return 0;
    for (int j = startpos; j < n; j++) {
        for (int k = 0; k < q - 1; k++) {
            long long w = base + offs[(size_t)j * q + k];
            if (delta > 0) { if (cnt[w]++ == 0) dunc--; }
            else           { if (--cnt[w] == 0) dunc++; }
            dunc += ball_apply_rec(w, offs, j + 1, rem - 1, delta);
        }
    }
    return dunc;
}

static void ball_apply(const uint8_t *c, long long base, int delta)
{
    long long offs[32 * 40];
    make_offs(c, offs);
    long long dunc = 0;
    if (delta > 0) { if (cnt[base]++ == 0) dunc--; }
    else           { if (--cnt[base] == 0) dunc++; }
    if (par_ball && R > 0) {
        /* The top-level (position, value) pair is the SMALLEST changed
         * coordinate of every word in that subtree, so the subtrees are
         * disjoint and the update needs no atomics. */
        int njk = n * (q - 1);
#pragma omp parallel for schedule(dynamic, 1) reduction(+:dunc)
        for (int jk = 0; jk < njk; jk++) {
            int j = jk / (q - 1), k = jk % (q - 1);
            long long w = base + offs[(size_t)j * q + k];
            long long d = 0;
            if (delta > 0) { if (cnt[w]++ == 0) d--; }
            else           { if (--cnt[w] == 0) d++; }
            d += ball_apply_rec(w, offs, j + 1, R - 1, delta);
            dunc += d;
        }
    } else {
        dunc += ball_apply_rec(base, offs, 0, R, delta);
    }
    uncovered += dunc;
}

/* how many words in the ball of c does nothing else cover? */
static long long unique_rec(long long base, const long long *offs, int startpos, int rem)
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
    long long offs[32 * 40];
    make_offs(c, offs);
    return (cnt[base] == 1) + unique_rec(base, offs, 0, R);
}

/* how many UNCOVERED words would a codeword placed at `base` cover? */
static long long gain_rec(long long base, const long long *offs, int startpos, int rem)
{
    long long u = 0;
    if (rem == 0) return 0;
    for (int j = startpos; j < n; j++)
        for (int k = 0; k < q - 1; k++) {
            long long w = base + offs[(size_t)j * q + k];
            u += (cnt[w] == 0);
            u += gain_rec(w, offs, j + 1, rem - 1);
        }
    return u;
}
static long long ball_gain(long long base)
{
    uint8_t c[64];
    long long offs[32 * 40];
    word_of(base, c);
    make_offs(c, offs);
    return (cnt[base] == 0) + gain_rec(base, offs, 0, R);
}

/* ------------------------------------------------------------------ */
/* uncovered-word cache                                                */
/* ------------------------------------------------------------------ */

#define UCAP 4096
static long long ucache[UCAP];
static int uncache_n = 0;

static void refill_cache(rng_t *r)
{
    uncache_n = 0;
    long long start = (long long)(rnext(r) % (uint64_t)NTOT);
    /* one pass from a random offset; reservoir past the cap keeps it unbiased */
    long long seen = 0;
    for (long long d = 0; d < NTOT; d++) {
        long long w = start + d; if (w >= NTOT) w -= NTOT;
        if (cnt[w]) continue;
        seen++;
        if (uncache_n < UCAP) ucache[uncache_n++] = w;
        else {
            long long j = (long long)(rnext(r) % (uint64_t)seen);
            if (j < UCAP) ucache[j] = w;
            else if (seen > 64 * UCAP) break;   /* plenty; stop scanning */
        }
    }
}

/* pickmode 1 = covsearch's "first uncovered word at or after a random index".
 * That is NOT uniform: a word is returned with probability proportional to the
 * length of the run of covered words in front of it, i.e. isolated uncovered
 * words are preferred.  Measured on K_3(11,4)@81: 461 uncovered against 538 for
 * uniform selection over the same code, same seeds, same iteration budget.
 * Expected cost is q^n/uncovered sequential reads, so it is bounded by the
 * density of what it is looking for; the hard cap only matters when a huge
 * instance is nearly solved, and there the cache path takes over. */
static int pickmode = 1;

static long long pick_uncovered(rng_t *r)
{
    if (pickmode) {
        for (int t = 0; t < 64; t++) {
            long long w = (long long)(rnext(r) % (uint64_t)NTOT);
            if (cnt[w] == 0) return w;
        }
        long long lim = NTOT < 4000000 ? NTOT : 4000000;
        for (int t = 0; t < 2; t++) {
            long long start = (long long)(rnext(r) % (uint64_t)NTOT);
            for (long long d = 0; d < lim; d++) {
                long long w = start + d; if (w >= NTOT) w -= NTOT;
                if (cnt[w] == 0) return w;
            }
        }
        if (lim == NTOT) return -1;
    }
    /* cheap path: uniform darts, correct and unbiased while uncovered is dense */
    for (int t = 0; t < 24; t++) {
        long long w = (long long)(rnext(r) % (uint64_t)NTOT);
        if (cnt[w] == 0) return w;
    }
    for (int t = 0; t < 3; t++) {
        for (int a = 0; a < 32 && uncache_n; a++) {
            int j = (int)rmod(r, (uint64_t)uncache_n);
            long long w = ucache[j];
            if (cnt[w] == 0) return w;
            ucache[j] = ucache[--uncache_n];
        }
        refill_cache(r);
        if (uncache_n == 0) return -1;
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/* CELF lazy-greedy construction pool                                  */
/*                                                                     */
/* gain(w) = |ball_R(w) cap Uncovered| is monotonically NON-INCREASING  */
/* as codewords are added, so a gain computed at any earlier time is a  */
/* valid upper bound now.  That makes exact greedy over a pool of       */
/* candidate centres affordable: keep the pool in a max-heap keyed on   */
/* the stale gain, and only recompute the top until the top is current. */
/* Typical cost is 3-5 ball walks per placement for a pool of thousands */
/* of centres, versus k walks per placement for best-of-k sampling.     */
/* ------------------------------------------------------------------ */

static long long *pool_w, *pool_g;
static int *pool_st;
static int pool_n = 0, pool_cap = 0, gstamp = 0;

static void pool_up(int i)
{
    while (i > 0) {
        int p = (i - 1) / 2;
        if (pool_g[p] >= pool_g[i]) break;
        long long tw = pool_w[i], tg = pool_g[i]; int ts = pool_st[i];
        pool_w[i] = pool_w[p]; pool_g[i] = pool_g[p]; pool_st[i] = pool_st[p];
        pool_w[p] = tw; pool_g[p] = tg; pool_st[p] = ts;
        i = p;
    }
}
static void pool_down(int i)
{
    for (;;) {
        int l = 2 * i + 1, r = l + 1, b = i;
        if (l < pool_n && pool_g[l] > pool_g[b]) b = l;
        if (r < pool_n && pool_g[r] > pool_g[b]) b = r;
        if (b == i) break;
        long long tw = pool_w[i], tg = pool_g[i]; int ts = pool_st[i];
        pool_w[i] = pool_w[b]; pool_g[i] = pool_g[b]; pool_st[i] = pool_st[b];
        pool_w[b] = tw; pool_g[b] = tg; pool_st[b] = ts;
        i = b;
    }
}
static void pool_push(long long w, long long g, int st)
{
    if (pool_n >= pool_cap) return;
    pool_w[pool_n] = w; pool_g[pool_n] = g; pool_st[pool_n] = st;
    pool_n++; pool_up(pool_n - 1);
}
static void pool_pop(void)
{
    pool_n--;
    if (pool_n > 0) {
        pool_w[0] = pool_w[pool_n]; pool_g[0] = pool_g[pool_n]; pool_st[0] = pool_st[pool_n];
        pool_down(0);
    }
}

/* ------------------------------------------------------------------ */
/* the sphere-difference move (verbatim from covsearch.c)              */
/* ------------------------------------------------------------------ */

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
        int i = R - 1;
        while (i >= 0 && k[i] == q - 2) { k[i] = 0; i--; }
        if (i < 0) break;
        k[i]++;
        for (int j = i; j < R; j++)
            partial[j + 1] = partial[j] + offs[(size_t)pos[j] * q + k[j]];
    }
    return dunc;
}

static long long move_delta(int i, int p, int v, int commit, int par)
{
    const uint8_t *c = code + (size_t)i * n;
    long long dv = (long long)(v - c[p]) * pw[p];
    long long base = cidx[i];

    long long op[32 * 40];
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

static int cmp_ll(const void *a, const void *b)
{
    long long x = *(const long long *)a, y = *(const long long *)b;
    return x < y ? -1 : (x > y);
}

/* The judge counts DISTINCT codewords and awards the full 1000 only when that
 * count reaches M.  A repeated codeword would therefore turn a genuine cover
 * into a score of 0, so any repeat is replaced by a fresh unused word; adding a
 * codeword can never uncover anything, so the cover stays a cover. */
static void dedupe(uint8_t *cw, rng_t *r)
{
    long long *ix = malloc(sizeof(long long) * M);
    long long *srt = malloc(sizeof(long long) * M);
    for (int i = 0; i < M; i++) ix[i] = widx(cw + (size_t)i * n);
    memcpy(srt, ix, sizeof(long long) * M);
    qsort(srt, M, sizeof(long long), cmp_ll);
    int ndup = 0;
    for (int i = 1; i < M; i++) if (srt[i] == srt[i - 1]) ndup++;
    if (ndup && (long long)M < NTOT) {
        long long *extra = malloc(sizeof(long long) * (size_t)ndup);
        int nextra = 0;
        char *first = calloc((size_t)M, 1);
        char *claimed = calloc((size_t)M, 1);
        /* the run start of a value in the sorted array identifies that value */
        for (int i = 0; i < M; i++) {
            long long *f = bsearch(&ix[i], srt, M, sizeof(long long), cmp_ll);
            if (!f) { first[i] = 1; continue; }
            int pos = (int)(f - srt);
            while (pos > 0 && srt[pos - 1] == ix[i]) pos--;
            if (!claimed[pos]) { claimed[pos] = 1; first[i] = 1; }
        }
        free(claimed);
        for (int i = 0; i < M; i++) {
            if (first[i]) continue;
            for (int tries = 0; tries < 100000; tries++) {
                long long c = (long long)(rnext(r) % (uint64_t)NTOT);
                int clash = 0;
                for (int j = 0; j < M && !clash; j++) if (ix[j] == c) clash = 1;
                for (int j = 0; j < nextra && !clash; j++) if (extra[j] == c) clash = 1;
                if (clash) continue;
                ix[i] = c;
                extra[nextra++] = c;
                word_of(c, cw + (size_t)i * n);
                break;
            }
        }
        free(first); free(extra);
    }
    free(ix); free(srt);
}

static void write_code(const char *path, const uint8_t *cw)
{
    if (!path) return;
    static uint8_t *buf = NULL;
    static rng_t wr = { { 0x243F6A8885A308D3ULL, 0x13198A2E03707344ULL,
                          0xA4093822299F31D0ULL, 0x082EFA98EC4E6C89ULL } };
    if (!buf) buf = malloc((size_t)M * n);
    memcpy(buf, cw, (size_t)M * n);
    dedupe(buf, &wr);

    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "w");
    if (!f) { perror(tmp); return; }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < n; j++) {
            if (q <= 10) fputc('0' + buf[(size_t)i * n + j], f);
            else if (q <= 36) fputc(DIG[buf[(size_t)i * n + j]], f);
            else fprintf(f, "%d%s", buf[(size_t)i * n + j], j + 1 < n ? " " : "");
        }
        fputc('\n', f);
    }
    fclose(f);
    rename(tmp, path);      /* atomic: the judge never sees a half-written file */

    /* sidecar so the portfolio driver can rank chains that were killed */
    snprintf(tmp, sizeof tmp, "%s.cnt", path);
    f = fopen(tmp, "w");
    if (f) { fprintf(f, "%lld\n", best_uncovered); fclose(f); }
}

static int read_code(const char *path, uint8_t *cw, int maxM)
{
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    char line[8192];
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
        if (m >= maxM) break;
        memcpy(cw + (size_t)m * n, tmp, n);
        m++;
    }
    fclose(f);
    return m;
}

/* ------------------------------------------------------------------ */
/* operators                                                           */
/* ------------------------------------------------------------------ */

static uint8_t *uw;                 /* scratch: the picked uncovered word */
static int *cand_i, *cand_p, *dist;
static long long *cand_d;
static int cand_limit = 100000;
static int *tabu = NULL;           /* forbids restoring a value, as covsearch  */
static int tabu_len = 0;           /* covsearch default: tenure 0 + rand(0..3) */
static long long g_it = 0;
static int hardpick = 1;

/* one focused single-coordinate move (covsearch's move, unchanged) */
static long long op_fine(rng_t *r)
{
    long long before = uncovered;
    long long w = pick_uncovered(r);
    if (w < 0) return 0;
    if (hardpick > 1) {
        /* "hardest first": of a few uncovered words, work on the one whose
         * nearest codeword is furthest away. */
        int bestd = -1;
        long long bw = w;
        for (int t = 0; t < hardpick; t++) {
            long long w2 = t ? pick_uncovered(r) : w;
            if (w2 < 0) continue;
            word_of(w2, uw);
            int dm = n + 1;
            for (int i = 0; i < M; i++) {
                const uint8_t *c = code + (size_t)i * n;
                int d = 0;
                for (int j = 0; j < n; j++) d += (c[j] != uw[j]);
                if (d < dm) dm = d;
            }
            if (dm > bestd) { bestd = dm; bw = w2; }
        }
        w = bw;
    }
    word_of(w, uw);

    int dmin = n + 1;
    for (int i = 0; i < M; i++) {
        const uint8_t *c = code + (size_t)i * n;
        int d = 0;
        for (int j = 0; j < n; j++) d += (c[j] != uw[j]);
        dist[i] = d;
        if (d < dmin) dmin = d;
    }
    int dtar = (dmin <= R + 1) ? R + 1 : dmin;

    int ncand = 0;
    for (int i = 0; i < M; i++) {
        if (dist[i] != dtar) continue;
        const uint8_t *c = code + (size_t)i * n;
        for (int p = 0; p < n; p++) {
            if (c[p] == uw[p]) continue;
            cand_i[ncand] = i; cand_p[ncand] = p; ncand++;
        }
    }
    if (ncand == 0) {
        int i = (int)rmod(r, M), p = (int)rmod(r, n);
        cand_i[0] = i; cand_p[0] = p; ncand = 1;
    }
    int take = ncand < cand_limit ? ncand : cand_limit;
    if (take < ncand)
        for (int s = 0; s < take; s++) {
            int j = s + (int)rmod(r, (uint64_t)(ncand - s));
            int ti = cand_i[s], tp = cand_p[s];
            cand_i[s] = cand_i[j]; cand_p[s] = cand_p[j];
            cand_i[j] = ti; cand_p[j] = tp;
        }

#pragma omp parallel for schedule(dynamic, 1) if(take > 3 && par_cand)
    for (int s = 0; s < take; s++)
        cand_d[s] = move_delta(cand_i[s], cand_p[s], uw[cand_p[s]], 0, 0);

    long long bestd = 1LL << 60;
    int bi = -1, bp = -1, bv = -1, nties = 0;
    for (int s = 0; s < take; s++) {
        int i = cand_i[s], p = cand_p[s], v = uw[p];
        long long d = cand_d[s];
        if (tabu) {
            int is_tabu = tabu[((size_t)i * n + p) * q + v] > g_it;
            if (is_tabu && uncovered + d >= best_uncovered) continue;  /* aspiration */
        }
        if (d < bestd) { bestd = d; bi = i; bp = p; bv = v; nties = 1; }
        else if (d == bestd) { nties++; if (rmod(r, (uint64_t)nties) == 0) { bi = i; bp = p; bv = v; } }
    }
    if (bi < 0) {
        int s = (int)rmod(r, (uint64_t)take);
        bi = cand_i[s]; bp = cand_p[s]; bv = uw[bp];
    }
    int oldv = code[(size_t)bi * n + bp];
    move_delta(bi, bp, bv, 1, !par_cand);
    if (tabu) tabu[((size_t)bi * n + bp) * q + oldv] = (int)(g_it + tabu_len + rmod(r, 4));
    return before - uncovered;
}

/* ruin-and-recreate: delete the least-uniquely-useful of k_rem sampled
 * codewords, re-insert it at the best of k_add sampled uncovered words. */
static int krem = 3, kadd = 6, tel_revert = 1;

static long long op_teleport(rng_t *r)
{
    long long before = uncovered;
    int kr = krem < M ? krem : M;
    int idx[64]; long long uc[64];
    for (int t = 0; t < kr; t++) idx[t] = (int)rmod(r, M);
#pragma omp parallel for schedule(dynamic, 1) if(par_ball && kr > 1)
    for (int t = 0; t < kr; t++) uc[t] = unique_cover(code + (size_t)idx[t] * n, cidx[idx[t]]);
    int bi = idx[0]; long long bu = uc[0];
    for (int t = 1; t < kr; t++) if (uc[t] < bu) { bu = uc[t]; bi = idx[t]; }

    uint8_t oldw[64];
    memcpy(oldw, code + (size_t)bi * n, n);
    long long oldidx = cidx[bi];
    ball_apply(oldw, oldidx, -1);

    long long ws[64], gs[64];
    int nw = 0;
    for (int t = 0; t < kadd; t++) {
        long long w = pick_uncovered(r);
        if (w >= 0) ws[nw++] = w;
    }
    if (nw == 0) {                       /* fully covered after removal: undo */
        ball_apply(oldw, oldidx, +1);
        return before - uncovered;
    }
#pragma omp parallel for schedule(dynamic, 1) if(par_ball && nw > 1)
    for (int t = 0; t < nw; t++) gs[t] = ball_gain(ws[t]);
    int bt = 0;
    for (int t = 1; t < nw; t++) if (gs[t] > gs[bt]) bt = t;

    word_of(ws[bt], code + (size_t)bi * n);
    cidx[bi] = ws[bt];
    ball_apply(code + (size_t)bi * n, ws[bt], +1);

    if (tel_revert && uncovered > before) {
        ball_apply(code + (size_t)bi * n, cidx[bi], -1);
        memcpy(code + (size_t)bi * n, oldw, n);
        cidx[bi] = oldidx;
        ball_apply(oldw, oldidx, +1);
    }
    return before - uncovered;
}

/* Restart from the incumbent: throw away whatever the chain has drifted into
 * and rebuild cnt[] from the best code seen, so the next kick perturbs an elite
 * rather than a plateau wanderer.  Costs one memset of q^n plus M ball walks,
 * so it is only enabled where that is cheap. */
static void rebuild_from(const uint8_t *src)
{
    memcpy(code, src, (size_t)M * n);
    memset(cnt, 0, sizeof(uint16_t) * (size_t)NTOT);
    uncovered = NTOT;
    for (int i = 0; i < M; i++) {
        cidx[i] = widx(code + (size_t)i * n);
        ball_apply(code + (size_t)i * n, cidx[i], +1);
    }
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    double tlimit = 60.0;
    uint64_t seed = 12345;
    const char *inpath = NULL;
    int kinit = 12;                /* best-of-k construction breadth */
    int pool_size = 0;             /* CELF pool size; 0 => best-of-k (measured better) */
    double cfrac = 0.55;           /* fraction of the budget the construction may use */
    int inj = 2;                   /* fresh centres injected per placement    */
    int use_tabu = 1;
    int elite_every = 0;           /* every k-th fruitless kick, restart from best; 0=off (unvalidated, see NOTES) */
    int use_tel = 1, use_fine = 1; /* which operators the bandit may use     */
    double block = 0.10;           /* bandit block length, seconds           */
    long long stall_limit = 5000;
    int kick = 2;
    double eps = 0.03;

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
        else if (ARG("--threads")) nthreads = atoi(argv[++i]);
        else if (ARG("--kinit")) kinit = atoi(argv[++i]);
        else if (ARG("--pool")) pool_size = atoi(argv[++i]);
        else if (ARG("--cfrac")) cfrac = atof(argv[++i]);
        else if (ARG("--inj")) inj = atoi(argv[++i]);
        else if (ARG("--tabu")) tabu_len = atoi(argv[++i]);
        else if (!strcmp(a, "--no-tabu")) use_tabu = 0;
        else if (ARG("--elite")) elite_every = atoi(argv[++i]);
        else if (!strcmp(a, "--pick-cache")) pickmode = 0;
        else if (ARG("--hard")) hardpick = atoi(argv[++i]);
        else if (ARG("--krem")) krem = atoi(argv[++i]);
        else if (ARG("--kadd")) kadd = atoi(argv[++i]);
        else if (ARG("--block")) block = atof(argv[++i]);
        else if (ARG("--eps")) eps = atof(argv[++i]);
        else if (ARG("--stall")) stall_limit = atoll(argv[++i]);
        else if (ARG("--kick")) kick = atoi(argv[++i]);
        else if (ARG("--cand")) cand_limit = atoi(argv[++i]);
        else if (!strcmp(a, "--norevert")) tel_revert = 0;
        else if (!strcmp(a, "--ops") && i + 1 < argc) {
            const char *o = argv[++i];
            use_tel = !strcmp(o, "tel") || !strcmp(o, "auto");
            use_fine = !strcmp(o, "fine") || !strcmp(o, "auto");
        }
        else if (!strcmp(a, "--quiet")) verbose = 0;
        else { fprintf(stderr, "unknown/incomplete option %s\n", a); return 2; }
        #undef ARG
    }
    if (q < 2 || n < 1 || R < 0 || M < 1) {
        fprintf(stderr, "usage: strategy -q Q -n N -R R -M M [-t sec] [-s seed] [--out f]\n");
        return 2;
    }
#ifdef _OPENMP
    if (nthreads) omp_set_num_threads(nthreads);
    nthreads = omp_get_max_threads();
#endif

    double T0 = now();

    if (R > 16) { fprintf(stderr, "R > 16 not supported\n"); return 2; }
    if (krem > 60) krem = 60;
    if (kadd > 60) kadd = 60;
    if (krem < 1) krem = 1;
    if (kadd < 1) kadd = 1;
    NTOT = 1;
    for (int i = 0; i < n; i++) {
        if (NTOT > (long long)1e10 / q) { fprintf(stderr, "q^n too large for 25 GB\n"); return 2; }
        NTOT *= q;
    }
    for (int p = 0; p < n; p++) pw[p] = ipow(q, n - 1 - p);
    if (M > 65535) { fprintf(stderr, "M > 65535 not supported\n"); return 2; }

    cnt = calloc((size_t)NTOT, sizeof(uint16_t));
    if (!cnt) { fprintf(stderr, "cannot allocate counters\n"); return 2; }
    code = calloc((size_t)M * n, 1);   /* never emit uninitialised digits */
    bestcode = malloc((size_t)M * n);
    cidx = malloc(sizeof(long long) * M);
    uw = malloc(n);
    cand_i = malloc(sizeof(int) * (size_t)M * (n + 1));
    cand_p = malloc(sizeof(int) * (size_t)M * (n + 1));
    cand_d = malloc(sizeof(long long) * (size_t)M * (n + 1));
    dist = malloc(sizeof(int) * M);
    build_combs();
    rseed(&grng, seed);

    long long S = binom(n - 1, R) * ipow(q - 1, R);
    long long V = 0;
    for (int i = 0; i <= R && i <= n; i++) V += binom(n, i) * ipow(q - 1, i);

    par_cand = (NCOMB < 256) && (nthreads > 1) && (S >= 8000);
    par_ball = (nthreads > 1) && (V >= 8000);

    if (verbose) {
        printf("# strategy q=%d n=%d R=%d M=%d q^n=%lld ball=%lld sphere=%lld "
               "threads=%u par_cand=%d par_ball=%d\n",
               q, n, R, M, NTOT, V, S, nthreads, par_cand, par_ball);
        fflush(stdout);
    }

    /* ---- construction ------------------------------------------------ */
    int have = 0;
    uncovered = NTOT;
    if (inpath) {
        have = read_code(inpath, code, M);
        for (int i = 0; i < have; i++) {
            cidx[i] = widx(code + (size_t)i * n);
            ball_apply(code + (size_t)i * n, cidx[i], +1);
        }
    }
    if (pool_size > 0 && M - have > 1) {
        /* --- CELF lazy greedy ----------------------------------------- */
        int batch = (int)nthreads; if (batch < 1) batch = 1; if (batch > 32) batch = 32;
        pool_cap = pool_size + 2 * (M - have) + 64;
        pool_w = malloc(sizeof(long long) * pool_cap);
        pool_g = malloc(sizeof(long long) * pool_cap);
        pool_st = malloc(sizeof(int) * pool_cap);
        long long ws[32], gs[32];
        for (int base = 0; base < pool_size; base += batch) {
            int nb = 0;
            for (int t = 0; t < batch && base + t < pool_size; t++) {
                long long w = pick_uncovered(&grng);
                if (w >= 0) ws[nb++] = w;
            }
            if (!nb) break;
#pragma omp parallel for schedule(dynamic, 1) if(par_ball && nb > 1)
            for (int t = 0; t < nb; t++) gs[t] = ball_gain(ws[t]);
            for (int t = 0; t < nb; t++) pool_push(ws[t], gs[t], 0);
        }
        for (int i = have; i < M; i++) {
            /* inject fresh centres drawn from the CURRENT uncovered set */
            if (inj > 0 && i > have) {
                int nb = 0;
                for (int t = 0; t < inj && t < 32; t++) {
                    long long w = pick_uncovered(&grng);
                    if (w >= 0) ws[nb++] = w;
                }
#pragma omp parallel for schedule(dynamic, 1) if(par_ball && nb > 1)
                for (int t = 0; t < nb; t++) gs[t] = ball_gain(ws[t]);
                for (int t = 0; t < nb; t++) pool_push(ws[t], gs[t], gstamp);
            }
            /* lazy re-evaluation until the heap top is current */
            while (pool_n > 0 && pool_st[0] != gstamp) {
                int nb = 0;
                while (nb < batch && pool_n > 0 && pool_st[0] != gstamp) {
                    ws[nb++] = pool_w[0]; pool_pop();
                }
#pragma omp parallel for schedule(dynamic, 1) if(par_ball && nb > 1)
                for (int t = 0; t < nb; t++) gs[t] = ball_gain(ws[t]);
                for (int t = 0; t < nb; t++) pool_push(ws[t], gs[t], gstamp);
            }
            long long w;
            if (pool_n > 0) { w = pool_w[0]; pool_pop(); }
            else {
                w = pick_uncovered(&grng);
                if (w < 0) w = (long long)rmod(&grng, (uint64_t)NTOT);
            }
            word_of(w, code + (size_t)i * n);
            cidx[i] = w;
            ball_apply(code + (size_t)i * n, w, +1);
            gstamp++;
        }
    } else {
    /* Best-of-k greedy, with k throttled so the construction cannot eat the
     * whole budget: on a loaded machine one full pass of M ball walks on
     * K_8(9,4) already costs ~12 s, so a fixed k is not safe. */
    double cbudget = cfrac * tlimit, cwrite = 0;
    int k = kinit;
    for (int i = have; i < M; i++) {
        long long ws[64], gs[64];
        int nw = 0;
        if (((i - have) & 15) == 15) {
            double el = now() - T0;
            double proj = el * (double)(M - have) / (double)(i + 1 - have);
            if (proj > cbudget) k = k > 1 ? k / 2 : 0;
            else if (proj < 0.45 * cbudget && k < kinit) k = k ? k * 2 : 1;
            if (el > 0.80 * tlimit) k = 0;
            /* keep a usable partial code on disk while the construction runs:
             * a chain killed mid-construction still beats the random fallback */
            if (el - cwrite > 3.0) {
                cwrite = el;
                best_uncovered = uncovered;
                write_code(outpath, code);
            }
        }
        if (k > 64) k = 64;
        for (int t = 0; t < k; t++) {
            long long w = pick_uncovered(&grng);
            if (w >= 0) ws[nw++] = w;
        }
        long long w;
        if (nw == 0) {
            w = pick_uncovered(&grng);
            if (w < 0) w = (long long)rmod(&grng, (uint64_t)NTOT);
        }
        else if (nw == 1) w = ws[0];
        else {
#pragma omp parallel for schedule(dynamic, 1) if(par_ball)
            for (int t = 0; t < nw; t++) gs[t] = ball_gain(ws[t]);
            int bt = 0;
            for (int t = 1; t < nw; t++) if (gs[t] > gs[bt]) bt = t;
            w = ws[bt];
        }
        word_of(w, code + (size_t)i * n);
        cidx[i] = w;
        ball_apply(code + (size_t)i * n, w, +1);
    }
    }
    memcpy(bestcode, code, (size_t)M * n);
    best_uncovered = uncovered;
    write_code(outpath, bestcode);
    if (verbose) { printf("# init uncovered=%lld t=%.2f\n", uncovered, now() - T0); fflush(stdout); }

    /* ---- bandit-driven search ---------------------------------------- */
    double rate[2] = { 0, 0 }, acc_t[2] = { 0, 0 }, acc_g[2] = { 0, 0 };
    const double minmeas = 0.02;
    int tried[2] = { 0, 0 };
    long long nops[2] = { 0, 0 };
    double tops[2] = { 0, 0 };
    long long gains[2] = { 0, 0 };
    long long it = 0, since_improve = 0, nkicks = 0, kicks_since_improve = 0;
    double last_write = 0, last_report = 0;
    long long written_best = best_uncovered;
    uint8_t *tmpw = malloc(n);

    if (!use_tel && !use_fine) use_fine = 1;
    if (M < 2) use_tel = 0;
    if (use_tabu) tabu = calloc((size_t)M * n * q, sizeof(int));

    double tnow = now();
    while (uncovered > 0 && tnow - T0 < tlimit) {
        int op;
        int explore = 0;
        if (!use_tel) op = 0;
        else if (!use_fine) op = 1;
        else if (!tried[0]) op = 0;
        else if (!tried[1]) op = 1;
        else if ((rnext(&grng) >> 11) * 0x1.0p-53 < eps) { op = (int)rmod(&grng, 2); explore = 1; }
        /* Hysteresis towards FINE: it is covsearch's measured-best move, so
         * TELEPORT has to be clearly better, not marginally, to take a block.
         * And the short-horizon EMA alone is not enough evidence -- on
         * K_3(11,4)@81 TELEPORT never moves the record at all, yet EMA noise was
         * handing it 20-30% of the budget, which cost 10 uncovered words against
         * covsearch in the judge run.  So it also has to be ahead on the whole
         * run so far before it gets an exploit block. */
        else {
            int want = (rate[1] > 2.0 * rate[0] + 1e-9);
            if (want && tops[0] > 0.5 && tops[1] > 0.5) {
                double cum0 = (double)gains[0] / tops[0];
                double cum1 = (double)gains[1] / tops[1];
                want = (cum1 > 1.2 * cum0);
            }
            op = want ? 1 : 0;
        }

        double bstart = tnow;
        long long bbest = best_uncovered;
        double blen = explore ? 0.2 * block : block;
        long long bops = 0;
        do {
            g_it = it;
            long long g = op ? op_teleport(&grng) : op_fine(&grng);
            (void)g;
            it++; bops++;
            if (uncovered < best_uncovered) {
                best_uncovered = uncovered;
                memcpy(bestcode, code, (size_t)M * n);
                since_improve = 0; kicks_since_improve = 0;
                if (best_uncovered == 0) break;
            } else since_improve++;
            tnow = now();
        } while (tnow - bstart < blen && tnow - T0 < tlimit && uncovered > 0);

        /* The bandit scores an operator by how fast it moves the RECORD, not the
         * current state.  Scoring current-uncovered would bias towards TELEPORT,
         * which reverts its own damage and so never scores negative, while FINE
         * legitimately random-walks upward on plateaus. */
        /* Accumulate until at least `minmeas` seconds of this operator have been
         * observed before touching the estimate.  Updating from a single
         * exploratory move divides a whole-number gain by a microsecond and
         * produces a rate of 10^5 from one lucky hit, which then buys the
         * operator full blocks it has not earned. */
        double el = tnow - bstart;
        acc_t[op] += el;
        acc_g[op] += (double)(bbest - best_uncovered);
        if (acc_t[op] >= minmeas) {
            double g = acc_g[op] / acc_t[op];
            rate[op] = tried[op] ? 0.7 * rate[op] + 0.3 * g : g;
            tried[op] = 1;
            acc_t[op] = 0; acc_g[op] = 0;
        }
        nops[op] += bops; tops[op] += el; gains[op] += bbest - best_uncovered;

        if (best_uncovered == 0) break;

        /* stagnation kick: teleport a few codewords onto uncovered words */
        if (since_improve > stall_limit) {
            int nk = kick;
            if (elite_every > 0 && ++kicks_since_improve % elite_every == 0
                && (double)NTOT + (double)M * V < 1e8) {
                rebuild_from(bestcode);
                nk = kick * 3;
            }
            for (int t = 0; t < nk; t++) {
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
            since_improve = 0;
            if (tabu) memset(tabu, 0, sizeof(int) * (size_t)M * n * q);
        }

        if (best_uncovered < written_best && tnow - last_write > 1.0) {
            write_code(outpath, bestcode);
            written_best = best_uncovered;
            last_write = tnow;
        }
        if (verbose && tnow - T0 - last_report > 10.0) {
            last_report = tnow - T0;
            printf("# t=%6.1f it=%-9lld unc=%-10lld best=%-10lld rate[fine]=%.0f rate[tel]=%.0f "
                   "n[fine]=%lld n[tel]=%lld\n",
                   tnow - T0, it, uncovered, best_uncovered, rate[0], rate[1], nops[0], nops[1]);
            fflush(stdout);
        }
    }

    write_code(outpath, bestcode);
    double el = now() - T0;
    printf("RESULT q=%d n=%d R=%d M=%d uncovered=%lld iters=%lld kicks=%lld time=%.2f "
           "fine=%lld/%.1fs/%lld tel=%lld/%.1fs/%lld\n",
           q, n, R, M, best_uncovered, it, nkicks, el,
           nops[0], tops[0], gains[0], nops[1], tops[1], gains[1]);
    fflush(stdout);
    return best_uncovered == 0 ? 0 : 1;
}
