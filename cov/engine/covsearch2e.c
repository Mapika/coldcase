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
#include <limits.h>
#include <sys/mman.h>
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

/* ---- engineering options (all default OFF = baseline behaviour) ---- */
static int opt_st   = 0;   /* 0 = read cnt[] when evaluating, 1 = uint8 min(cnt,2)
                              side array, 2 = 2-bit packed min(cnt,2)          */
static int opt_hoist = 0;  /* hoist make_offs out of the candidate loop         */
static int opt_early = 0;  /* bound-based early exit in candidate evaluation    */
static int opt_pf    = 0;  /* software prefetch in the sphere walk              */
static int opt_huge  = 0;  /* madvise(MADV_HUGEPAGE) the big arrays             */
static int opt_fix0  = 0;  /* symmetry: freeze codeword 0 at 0^n                */
static uint8_t *st   = NULL;   /* saturating state array, see opt_st            */

/* ---- cache of known-uncovered words ------------------------------- */
/* Without it, pick_uncovered falls back to a full q^n linear scan as soon as
 * the uncovered words are rarer than 1/64 of the space, i.e. for the whole
 * endgame.  With it, one scan refills a buffer that then serves many
 * iterations, and every commit donates the words it just uncovered. */
#define UCCAP 4096
static long long ucbuf[UCCAP];
static int  ucn = 0;
static int  ucrec = 0;         /* commits donate newly-uncovered words         */
static int  opt_ucache = 0;
static int  opt_upick = 0;   /* pick the longest-uncovered word instead of a random one */
static int  opt_wide  = 0;   /* evaluate ALL n(q-1) moves of each candidate codeword */

/* saturating state accessors -------------------------------------- */
static inline int st_get8(long long w)  { return st[w]; }
static inline int st_get2(long long w)  { return (st[w >> 2] >> ((w & 3) << 1)) & 3; }
static inline void st_set(long long w, int v) {
    if (opt_st == 1) st[w] = (uint8_t)v;
    else { int sh = (int)((w & 3) << 1); uint8_t *b = &st[w >> 2];
           *b = (uint8_t)((*b & ~(3 << sh)) | (v << sh)); }
}
/* call after cnt[w] has been changed to nc */
static inline void st_upd(long long w, int nc) {
    if (!opt_st) return;
    if (nc <= 2) st_set(w, nc);
}

/* ---- profiling ---------------------------------------------------- */
#ifdef PROF
enum { P_PICK, P_CAND, P_EVAL, P_SEL, P_COMMIT, P_BEST, P_KICK, P_CLOCK, P_N };
static const char *pname[P_N] =
    {"pick_uncovered","cand_collect","cand_eval","select","commit","bestcopy","kick","clock"};
static double prof_s[P_N];
static long long prof_c[P_N];
static long long prof_patterns_eval = 0, prof_patterns_commit = 0;
static long long prof_ncand_sum = 0, prof_take_sum = 0, prof_iters = 0;
static long long prof_unc_sum = 0, prof_bestd_sum = 0, prof_early_saved = 0;
static long long prof_pick_scans = 0;
static long long prof_ncw_sum = 0, prof_ul_used = 0, prof_ul_n_sum = 0;
static long long prof_ul_rebuild = 0, prof_ul_compact = 0;
static inline double thr_cpu(void) {
    struct timespec ts; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
#define PB(i) double _pb = thr_cpu()
#define PE(i) do { prof_s[i] += thr_cpu() - _pb; prof_c[i]++; } while (0)
#else
#define PB(i) do {} while (0)
#define PE(i) do {} while (0)
#endif

static inline double cpu_now(void) {
    struct timespec ts; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
static inline double wall_now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static uint8_t *code;              /* M*n digits                        */
static long long *cidx;            /* M word indices                    */

static uint8_t *bestcode;
static long long best_uncovered;

/* ---------------------------------------------------------------------
 * cov/engine additions (see cov/engine/NOTES.md).  Three properties are
 * imported from the arena entries `lowlevel` and `structure`, and nothing
 * else about the solver changes:
 *
 *   1. the wall clock starts at process start, not after initialisation;
 *   2. the incumbent is published to --out atomically (write .part, rename)
 *      as soon as one exists and on every improvement, throttled to 1 Hz,
 *      so a SIGKILL at any moment leaves the best code so far on disk;
 *   3. initialisation is itself deadline-bounded, and what is written is
 *      always exactly M DISTINCT codewords.
 *
 * The search trajectory is untouched: no decision below reads the clock.
 * --------------------------------------------------------------------- */
static double g_t_start;           /* wall clock at process start        */
static double g_last_pub = -1e9;   /* last publication, for the 1 Hz cap  */
static const char *g_outpath = NULL;
static int g_init_aborted = 0;     /* budget died during ball marking     */
static inline double elapsed_total(void) { return wall_now() - g_t_start; }

/* position subsets: for each excluded position p, all C(n-1,R) ways to
 * choose R of the remaining positions. */
static int NCOMB;
static uint8_t *combs;             /* n * NCOMB * R                     */
/* and, for the shared-sphere evaluator, all C(n,R) subsets of ALL n
 * positions (the full distance-R sphere around a codeword). */
static int FNCOMB;
static uint8_t *fcombs;            /* FNCOMB * R                        */

static int opt_eval = 0;   /* 0 = per-candidate walk (baseline)
                              1 = shared distance-R sphere for the leaving
                                  side, per-candidate walk for the entering side
                              2 = 1, plus the uncovered-word list for the
                                  entering side                                */

/* ---- exact list of uncovered words (for opt_eval == 2) ------------- */
/* The entering side of a move only ever counts words with cnt == 0, and in
 * the regime that matters the uncovered words are a vanishing fraction of
 * Z_q^n (tens out of millions).  Enumerating them directly and testing
 * membership of the entering set in O(n) then beats walking the sphere. */
#define ULCAP (1 << 20)
static long long *ul_idx;      /* word index of each list entry            */
static uint8_t   *ul_dig;      /* ULCAP * n digits                         */
static uint64_t  *ul_bit;      /* "is in the list" bitset over Z_q^n       */
static int ul_n = 0;           /* entries, including stale (now covered)   */
static int ul_valid = 0;       /* the list is a superset of the uncovered set */

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

/* page-aligned allocation, optionally backed by transparent huge pages */
static void *xalloc(size_t bytes)
{
    void *p = NULL;
    size_t align = 2 * 1024 * 1024;
    if (posix_memalign(&p, align, (bytes + align - 1) / align * align) != 0) return NULL;
#ifdef MADV_HUGEPAGE
    if (opt_huge) madvise(p, (bytes + align - 1) / align * align, MADV_HUGEPAGE);
#endif
    return p;
}

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

static void build_fcombs(void)
{
    FNCOMB = (int)binom(n, R);
    fcombs = malloc((size_t)FNCOMB * (R ? R : 1));
    int c[16];
    for (int i = 0; i < R; i++) c[i] = i;
    for (int k = 0; k < FNCOMB; k++) {
        for (int i = 0; i < R; i++) fcombs[(size_t)k * (R ? R : 1) + i] = (uint8_t)c[i];
        int i = R - 1;
        while (i >= 0 && c[i] == n - R + i) i--;
        if (i < 0) break;
        c[i]++;
        for (int j = i + 1; j < R; j++) c[j] = c[j - 1] + 1;
    }
}

/* ---- uncovered-word list maintenance ------------------------------ */
static inline int ul_test(long long w) { return (ul_bit[w >> 6] >> (w & 63)) & 1; }
static inline void ul_setbit(long long w) { ul_bit[w >> 6] |= 1ULL << (w & 63); }
static inline void ul_clrbit(long long w) { ul_bit[w >> 6] &= ~(1ULL << (w & 63)); }

static void word_of(long long idx, uint8_t *w);

/* called when cnt[w] has just become 0 */
static inline void ul_insert(long long w)
{
    if (!ul_valid || ul_test(w)) return;
    if (ul_n >= ULCAP) { ul_valid = 0; return; }
    ul_setbit(w);
    ul_idx[ul_n] = w;
    word_of(w, ul_dig + (size_t)ul_n * n);
    ul_n++;
}

/* drop entries that have since been covered */
static void ul_compact(void)
{
    int m = 0;
    for (int e = 0; e < ul_n; e++) {
        long long w = ul_idx[e];
        if (cnt[w] == 0) {
            if (m != e) { ul_idx[m] = w; memcpy(ul_dig + (size_t)m * n,
                                                ul_dig + (size_t)e * n, n); }
            m++;
        } else ul_clrbit(w);
    }
    ul_n = m;
#ifdef PROF
    prof_ul_compact++;
#endif
}

/* rebuild the list from scratch by one pass over cnt[] */
static void ul_rebuild(void)
{
    memset(ul_bit, 0, (size_t)((NTOT + 63) / 64) * 8);
    ul_n = 0; ul_valid = 1;
    for (long long w = 0; w < NTOT; w++) {
        if (cnt[w]) continue;
        if (ul_n >= ULCAP) { ul_valid = 0; return; }
        ul_setbit(w);
        ul_idx[ul_n] = w;
        word_of(w, ul_dig + (size_t)ul_n * n);
        ul_n++;
    }
#ifdef PROF
    prof_ul_rebuild++;
#endif
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
            if (delta > 0) { int nw = ++cnt[w]; if (nw == 1) dunc--; st_upd(w, nw); }
            else           { int nw = --cnt[w]; if (nw == 0) { dunc++; ul_insert(w); } st_upd(w, nw); }
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
    if (delta > 0) { int nb = ++cnt[base]; if (nb == 1) dunc--; st_upd(base, nb); }
    else           { int nb = --cnt[base]; if (nb == 0) { dunc++; ul_insert(base); } st_upd(base, nb); }
    dunc += ball_apply_rec(base, offs, c, 0, R, delta);
    uncovered += dunc;
    free(offs);
}

/* ------------------------------------------------------------------ */
/* the move: sphere enumeration                                        */
/* ------------------------------------------------------------------ */

/* ---------------------------------------------------------------------
 * The sphere walk.
 *
 * One position-subset ps[0..R-1] (all != p) and the (q-1)^R value
 * assignments on it.  The word visited is `a` (which LEAVES the ball of c
 * when c_p changes) and `a+dv` (which ENTERS the ball of c').
 *
 * Two accumulators are kept rather than one signed delta:
 *   pos = #{a : cnt[a] == 1}  -- words that become uncovered
 *   neg = #{b : cnt[b] == 0}  -- words that become covered
 * so that delta = pos - neg.  Keeping them apart is what makes the
 * early-exit bound of `move_eval` possible (neg <= uncovered).
 *
 * The innermost odometer digit is peeled into its own (q-1)-trip loop:
 *   - the q-1 loads are independent, so the core can have them all in
 *     flight, and
 *   - the odometer branch is executed (q-1) times less often.
 * The last digit is also the one with the smallest stride, because the
 * position subsets are generated in increasing order and pw[] decreases
 * with the position index, so this is also the most cache-friendly nest.
 * ------------------------------------------------------------------- */

#define G_CNT(w) cnt[w]
#define G_ST8(w) st[w]
#define G_ST2(w) st_get2(w)

#define WALK_EVAL(NAME, GET)                                                   \
static inline void NAME(const uint8_t *ps, long long base,                     \
                        const long long *offs, long long dv,                   \
                        long long *ppos, long long *pneg)                      \
{                                                                              \
    long long pos_ = 0, neg_ = 0;                                              \
    if (R == 0) {                                                              \
        pos_ = (GET(base) == 1); neg_ = (GET(base + dv) == 0);                 \
        *ppos += pos_; *pneg += neg_; return;                                  \
    }                                                                          \
    { int k[16]; long long partial[17];                                        \
      for (int i = 0; i < R - 1; i++) k[i] = 0;                                \
      partial[0] = base;                                                       \
      for (int i = 0; i < R - 1; i++)                                          \
          partial[i + 1] = partial[i] + offs[(size_t)ps[i] * q];               \
      const long long *ol = offs + (size_t)ps[R - 1] * q;                      \
      for (;;) {                                                               \
          long long p1 = partial[R - 1];                                       \
          for (int kk = 0; kk < q - 1; kk++) {                                 \
              long long a = p1 + ol[kk];                                       \
              pos_ += (GET(a) == 1);                                           \
              neg_ += (GET(a + dv) == 0);                                      \
          }                                                                    \
          int i = R - 2;                                                       \
          while (i >= 0 && k[i] == q - 2) { k[i] = 0; i--; }                   \
          if (i < 0) break;                                                    \
          k[i]++;                                                              \
          for (int j = i; j < R - 1; j++)                                      \
              partial[j + 1] = partial[j] + offs[(size_t)ps[j] * q + k[j]];    \
      } }                                                                      \
    *ppos += pos_; *pneg += neg_;                                              \
}

WALK_EVAL(walk_eval_cnt, G_CNT)
WALK_EVAL(walk_eval_st8, G_ST8)
WALK_EVAL(walk_eval_st2, G_ST2)

/* the original (baseline) walk, kept verbatim for --walk 0 */
static inline long long sphere_walk(const uint8_t *pos, long long base,
                                    const long long *offs, long long dv, int mode)
{
    long long dunc = 0;
    int k[16];
    long long partial[17];
    if (R == 0) {
        long long a = base, b = base + dv;
        if (mode) {
            int na = --cnt[a];
            if (na == 0) { dunc++; if (ucrec && ucn < UCCAP) ucbuf[ucn++] = a; ul_insert(a); }
            st_upd(a, na);
            int nb = ++cnt[b]; if (nb == 1) dunc--; st_upd(b, nb);
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
            int na = --cnt[a];
            if (na == 0) { dunc++; if (ucrec && ucn < UCCAP) ucbuf[ucn++] = a; ul_insert(a); }
            st_upd(a, na);
            int nb = ++cnt[b]; if (nb == 1) dunc--; st_upd(b, nb);
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

static int    opt_walk = 0;        /* 1 = peeled-innermost eval walk */
static long long PPC;              /* (q-1)^R patterns per position subset */
static long long TOTPAT;           /* NCOMB * PPC = S                      */

/* Evaluate the move "codeword i, position p, new value v" without
 * committing.  `offs` may be a precomputed offset table for codeword i
 * (opt_hoist) or NULL.  If `cutoff` is not LLONG_MAX and opt_early is on,
 * the walk may abort as soon as it is certain that the move cannot beat
 * `cutoff`; the return value is then EVAL_PRUNED. */
#define EVAL_PRUNED (1LL << 40)

static long long move_eval(int i, int p, int v, const long long *offs, long long cutoff)
{
    const uint8_t *c = code + (size_t)i * n;
    long long dv = (long long)(v - c[p]) * pw[p];
    long long base = cidx[i];
    long long op[32 * 32];
    if (!offs) { make_offs(c, op); offs = op; }

    const uint8_t *cb = combs + (size_t)p * NCOMB * (R ? R : 1);
    int stride = R ? R : 1;
    long long pos = 0, neg = 0;

    for (int k = 0; k < NCOMB; k++) {
        const uint8_t *ps = cb + (size_t)k * stride;
        if (opt_pf && k + 1 < NCOMB) {
            /* prefetch the first words of the NEXT position subset; the walk
             * of the current one gives it a whole (q-1)^R block of lead time */
            const uint8_t *ns = cb + (size_t)(k + 1) * stride;
            long long nb = base;
            for (int i = 0; i < R - 1; i++) nb += offs[(size_t)ns[i] * q];
            long long a0 = nb + offs[(size_t)ns[R - 1] * q];
            if (opt_st) { __builtin_prefetch(&st[opt_st == 1 ? a0 : a0 >> 2], 0, 1);
                          __builtin_prefetch(&st[opt_st == 1 ? a0 + dv : (a0 + dv) >> 2], 0, 1); }
            else        { __builtin_prefetch(&cnt[a0], 0, 1);
                          __builtin_prefetch(&cnt[a0 + dv], 0, 1); }
        }
        if (!opt_walk)      { long long d = sphere_walk(ps, base, offs, dv, 0);
                              /* baseline walk returns pos-neg only; split is
                                 unavailable, so early exit needs opt_walk */
                              pos += d; }
        else if (opt_st == 0) walk_eval_cnt(ps, base, offs, dv, &pos, &neg);
        else if (opt_st == 1) walk_eval_st8(ps, base, offs, dv, &pos, &neg);
        else                  walk_eval_st2(ps, base, offs, dv, &pos, &neg);

        if (opt_early && opt_walk && cutoff != LLONG_MAX) {
            long long rem = TOTPAT - (long long)(k + 1) * PPC;
            long long slack = uncovered - neg; if (slack < 0) slack = 0;
            long long lb = pos - neg - (rem < slack ? rem : slack);
            if (lb > cutoff) {   /* strict: a tie is still possible when lb == cutoff,
                                    and ties are broken at random, so keep those */
#ifdef PROF
                prof_patterns_eval += (long long)(k + 1) * PPC;
                prof_early_saved   += rem;
#endif
                return EVAL_PRUNED;
            }
        }
    }
#ifdef PROF
    prof_patterns_eval += TOTPAT;
#endif
    return pos - neg;
}

/* Commit the move.  Also records words that newly became uncovered in
 * newunc_buf so that pick_uncovered can avoid rescanning q^n. */
static long long move_commit(int i, int p, int v, int par)
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
            dunc += sphere_walk(cb + (size_t)k * stride, base, op, dv, 1);
    } else {
        for (int k = 0; k < NCOMB; k++)
            dunc += sphere_walk(cb + (size_t)k * stride, base, op, dv, 1);
    }
#ifdef PROF
    prof_patterns_commit += TOTPAT;
#endif
    uint8_t *cw = code + (size_t)i * n;
    cidx[i] += dv;
    cw[p] = (uint8_t)v;
    uncovered += dunc;
    return dunc;
}

/* ---------------------------------------------------------------------
 * Shared-sphere evaluation.
 *
 * Fix a codeword c and an uncovered word u with d(u,c) = R+1 (or, in the
 * fallback branch, d(u,c) = dmin).  The candidate moves that the search
 * wants to evaluate are c -> c^(p) for each of the R+1 positions p with
 * c_p != u_p, where c^(p) agrees with c except c^(p)_p = u_p.
 *
 * Leaving side.  w leaves the ball of c under move p iff |D| = R and
 * p notin D, where D = diff(w,c).  Hence, writing
 *     T  = #{w : |diff(w,c)| = R, cnt[w] = 1}
 *     H_j= #{w : |diff(w,c)| = R, cnt[w] = 1, j in diff(w,c)}
 * we get pos_p = T - H_p for EVERY p at once.  One enumeration of the full
 * distance-R sphere, C(n,R)(q-1)^R words, replaces R+1 enumerations of
 * C(n-1,R)(q-1)^R words each: a factor (R+1)(n-R)/n on the leaving side
 * (2.00 for K6(6,3), 2.50 for K6(8,4), 2.78 for K8(9,4)).  T and H are
 * updated only when cnt[w] == 1, which is rare, so the inner loop is a
 * single load and a compare.
 *
 * Entering side.  w enters the ball of c^(p) iff |diff(w,c)| = R+1,
 * p in diff(w,c) and w_p = u_p; it counts only if cnt[w] = 0.  So neg_p
 * can be read off the list of uncovered words: for each uncovered x with
 * d(x,c) = R+1, increment neg_j for every j in diff(x,c) with x_j = u_j.
 * That is O(n) per uncovered word per codeword, and beats the sphere walk
 * whenever |U| n < (R+1) C(n-1,R) (q-1)^R -- which holds by orders of
 * magnitude in the endgame.  When it does not (or when the list has
 * overflowed) the entering side falls back to the per-position walk.
 * ------------------------------------------------------------------- */

#define WALK_POS(NAME, GET)                                                    \
static inline void NAME(const uint8_t *ps, long long base,                     \
                        const long long *offs, long long *pt1, long long *hit) \
{                                                                              \
    long long t1 = 0;                                                          \
    if (R == 0) { if (GET(base) == 1) t1++; *pt1 += t1; return; }              \
    { int k[16]; long long partial[17];                                        \
      for (int i = 0; i < R - 1; i++) k[i] = 0;                                \
      partial[0] = base;                                                       \
      for (int i = 0; i < R - 1; i++)                                          \
          partial[i + 1] = partial[i] + offs[(size_t)ps[i] * q];               \
      const long long *ol = offs + (size_t)ps[R - 1] * q;                      \
      for (;;) {                                                               \
          long long p1 = partial[R - 1];                                       \
          for (int kk = 0; kk < q - 1; kk++) {                                 \
              long long w = p1 + ol[kk];                                       \
              if (GET(w) == 1) {                                               \
                  t1++;                                                        \
                  for (int i = 0; i < R; i++) hit[ps[i]]++;                    \
              }                                                                \
          }                                                                    \
          int i = R - 2;                                                       \
          while (i >= 0 && k[i] == q - 2) { k[i] = 0; i--; }                   \
          if (i < 0) break;                                                    \
          k[i]++;                                                              \
          for (int j = i; j < R - 1; j++)                                      \
              partial[j + 1] = partial[j] + offs[(size_t)ps[j] * q + k[j]];    \
      } }                                                                      \
    *pt1 += t1;                                                                \
}

WALK_POS(walk_pos_cnt, G_CNT)
WALK_POS(walk_pos_st8, G_ST8)
WALK_POS(walk_pos_st2, G_ST2)

#define WALK_NEG(NAME, GET)                                                    \
static inline long long NAME(const uint8_t *ps, long long base,                \
                             const long long *offs, long long dv)              \
{                                                                              \
    long long neg = 0;                                                         \
    if (R == 0) return (GET(base + dv) == 0);                                  \
    { int k[16]; long long partial[17];                                        \
      for (int i = 0; i < R - 1; i++) k[i] = 0;                                \
      partial[0] = base + dv;                                                  \
      for (int i = 0; i < R - 1; i++)                                          \
          partial[i + 1] = partial[i] + offs[(size_t)ps[i] * q];               \
      const long long *ol = offs + (size_t)ps[R - 1] * q;                      \
      for (;;) {                                                               \
          long long p1 = partial[R - 1];                                       \
          for (int kk = 0; kk < q - 1; kk++) neg += (GET(p1 + ol[kk]) == 0);   \
          int i = R - 2;                                                       \
          while (i >= 0 && k[i] == q - 2) { k[i] = 0; i--; }                   \
          if (i < 0) break;                                                    \
          k[i]++;                                                              \
          for (int j = i; j < R - 1; j++)                                      \
              partial[j + 1] = partial[j] + offs[(size_t)ps[j] * q + k[j]];    \
      } }                                                                      \
    return neg;                                                                \
}

WALK_NEG(walk_neg_cnt, G_CNT)
WALK_NEG(walk_neg_st8, G_ST8)
WALK_NEG(walk_neg_st2, G_ST2)

/* Evaluate the candidate moves on codeword i at once.
 * dvec[p*q + v] receives the delta of the move c_p -> v.
 *
 * The leaving count T - H_p does not depend on v at all, and the entering
 * count read off the uncovered-word list is naturally indexed by (p, v) --
 * incrementing neg[j*q + x_j] costs exactly what incrementing neg[j] cost.
 * So ONE sphere enumeration plus ONE pass over the uncovered list yields the
 * exact delta of every one of the n(q-1) single-coordinate moves of this
 * codeword, not only the R+1 that move it towards u.  `--wide` exploits that
 * (it needs use_ul, since the fallback would have to walk n(q-1) spheres).
 * With --wide off only the entries v = u_p are ever read. */
static void eval_codeword(int i, const uint8_t *u, const long long *offs_in,
                          long long *dvec, int use_ul)
{
    const uint8_t *c = code + (size_t)i * n;
    long long base = cidx[i];
    long long op[32 * 32];
    const long long *offs = offs_in;
    if (!offs) { make_offs(c, op); offs = op; }

    long long t1 = 0, hit[64], neg[32 * 32];
    for (int j = 0; j < n; j++) hit[j] = 0;
    memset(neg, 0, sizeof(long long) * (size_t)n * q);

    int stride = R ? R : 1;
    for (int k = 0; k < FNCOMB; k++) {
        const uint8_t *ps = fcombs + (size_t)k * stride;
        if      (opt_st == 1) walk_pos_st8(ps, base, offs, &t1, hit);
        else if (opt_st == 2) walk_pos_st2(ps, base, offs, &t1, hit);
        else                  walk_pos_cnt(ps, base, offs, &t1, hit);
    }
#ifdef PROF
    prof_patterns_eval += (long long)FNCOMB * PPC;
#endif

    if (use_ul) {
        int D[64];
        for (int e = 0; e < ul_n; e++) {
            long long x = ul_idx[e];
            if (cnt[x] != 0) continue;                 /* stale entry */
            const uint8_t *xd = ul_dig + (size_t)e * n;
            int d = 0;
            for (int j = 0; j < n; j++)
                if (xd[j] != c[j]) { if (d < R + 1) D[d] = j; d++; if (d > R + 1) break; }
            if (d != R + 1) continue;
            for (int t = 0; t < R + 1; t++) {
                int j = D[t];
                neg[(size_t)j * q + xd[j]]++;
            }
        }
    } else {
        const uint8_t *cb;
        for (int p = 0; p < n; p++) {
            if (c[p] == u[p]) continue;
            long long dv = (long long)(u[p] - c[p]) * pw[p];
            cb = combs + (size_t)p * NCOMB * stride;
            long long nn = 0;
            for (int k = 0; k < NCOMB; k++) {
                const uint8_t *ps = cb + (size_t)k * stride;
                if      (opt_st == 1) nn += walk_neg_st8(ps, base, offs, dv);
                else if (opt_st == 2) nn += walk_neg_st2(ps, base, offs, dv);
                else                  nn += walk_neg_cnt(ps, base, offs, dv);
            }
            neg[(size_t)p * q + u[p]] = nn;
#ifdef PROF
            prof_patterns_eval += TOTPAT;
#endif
        }
    }

    for (int p = 0; p < n; p++) {
        long long lv = t1 - hit[p];
        for (int v = 0; v < q; v++)
            if (v != c[p]) dvec[(size_t)p * q + v] = lv - neg[(size_t)p * q + v];
    }
}

/* baseline-compatible entry point used by --sa and the noise path */
static long long move_delta(int i, int p, int v, int commit, int par)
{
    if (commit) return move_commit(i, p, v, par);
    return move_eval(i, p, v, NULL, LLONG_MAX);
}

/* ------------------------------------------------------------------ */
/* code I/O                                                           */
/* ------------------------------------------------------------------ */

static const char *DIG = "0123456789abcdefghijklmnopqrstuvwxyz";

static long long widx(const uint8_t *w);
static void word_of(long long idx, uint8_t *w);

/* Write exactly M DISTINCT codewords, atomically.
 *
 * Distinctness matters because every consumer -- the arena judge, campaign.py,
 * verify_cov.py -- counts distinct words, so a genuine cover that repeats one
 * codeword is scored as a code of size M-1 (or rejected).  Duplicates can only
 * arise from a kick teleporting two codewords onto the same word.  Adding a
 * codeword never uncovers anything, so replacing a duplicate by any unused
 * word preserves a cover.  `nw` may be less than M (a deadline-bounded init),
 * in which case the file is topped up to M distinct words as well.
 *
 * The write goes to <path>.part and is renamed into place, so a reader can
 * never observe a half-written file and a kill can never destroy the previous
 * publication. */
static void write_code_n(const char *path, const uint8_t *cw, int nw)
{
    static uint8_t *buf = NULL;
    static char *seen = NULL;          /* NTOT flags, only when affordable */
    static long long *sortidx = NULL;
    if (!buf) buf = malloc((size_t)M * n);
    if (nw > M) nw = M;

    int m = 0;
    if (!seen && NTOT <= 400000000LL) seen = calloc((size_t)NTOT, 1);
    if (seen) {
        for (int i = 0; i < nw; i++) {
            long long x = widx(cw + (size_t)i * n);
            if (seen[x]) continue;
            seen[x] = 1;
            memcpy(buf + (size_t)m * n, cw + (size_t)i * n, n);
            m++;
        }
        /* top up with unused words: walk indices from a fixed point */
        for (long long x = 0; m < M && x < NTOT; x++) {
            if (seen[x]) continue;
            seen[x] = 1;
            word_of(x, buf + (size_t)m * n);
            m++;
        }
        for (int i = 0; i < m; i++) seen[widx(buf + (size_t)i * n)] = 0;
    } else {
        /* NTOT too large for a flag array: sort the indices instead */
        if (!sortidx) sortidx = malloc(sizeof(long long) * (size_t)M);
        for (int i = 0; i < nw; i++) sortidx[i] = widx(cw + (size_t)i * n);
        for (int i = 0; i < nw; i++) {
            int dup = 0;
            for (int j = 0; j < i; j++) if (sortidx[j] == sortidx[i]) { dup = 1; break; }
            if (dup) continue;
            memcpy(buf + (size_t)m * n, cw + (size_t)i * n, n);
            sortidx[m] = sortidx[i];
            m++;
        }
        for (long long x = 0; m < M && x < NTOT; x++) {
            int dup = 0;
            for (int j = 0; j < m; j++) if (sortidx[j] == x) { dup = 1; break; }
            if (dup) continue;
            word_of(x, buf + (size_t)m * n);
            sortidx[m] = x;
            m++;
        }
    }

    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s.part", path);
    FILE *f = fopen(tmp, "w");
    if (!f) { perror(tmp); return; }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (q <= 36) fputc(DIG[buf[(size_t)i * n + j]], f);
            else fprintf(f, "%d%s", buf[(size_t)i * n + j], j + 1 < n ? " " : "");
        }
        fputc('\n', f);
    }
    fflush(f);
    fclose(f);
    if (rename(tmp, path) != 0) perror(path);
}

static void write_code(const char *path, const uint8_t *cw)
{
    write_code_n(path, cw, M);
}

/* Publish the incumbent, at most once a second unless forced. */
static void publish(const uint8_t *cw, int nw, int force)
{
    if (!g_outpath) return;
    double t = wall_now();
    if (!force && t - g_last_pub < 1.0) return;
    g_last_pub = t;
    write_code_n(g_outpath, cw, nw);
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

static void st_reset(void)
{
    if (opt_st == 1)      memset(st, 0, (size_t)NTOT);
    else if (opt_st == 2) memset(st, 0, (size_t)((NTOT + 3) / 4));
}

static void rebuild(void)
{
    memset(cnt, 0, sizeof(uint16_t) * (size_t)NTOT);
    st_reset();
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
    if (opt_upick && ul_valid) {
        /* "oldest uncovered first": the list is append-ordered, so scanning it
         * from the front returns the word that has been uncovered longest.
         * This is the cheapest form of the guided/weighted focus that the
         * literature adds to focused local search. */
        for (int e = 0; e < ul_n; e++)
            if (cnt[ul_idx[e]] == 0) return ul_idx[e];
    }
    if (opt_ucache) {
        /* serve from the cache, discarding entries that have since been
         * covered; a random slot is drawn so the choice stays unbiased over
         * the cache contents. */
        while (ucn > 0) {
            int j = (int)(rnext(r) % (uint64_t)ucn);
            long long w = ucbuf[j];
            ucbuf[j] = ucbuf[--ucn];
            if (cnt[w] == 0) return w;
        }
    }
    for (int t = 0; t < 64; t++) {
        long long w = (long long)(rnext(r) % (uint64_t)NTOT);
        if (cnt[w] == 0) return w;
    }
#ifdef PROF
    prof_pick_scans++;
#endif
    long long start = (long long)(rnext(r) % (uint64_t)NTOT);
    long long found = -1;
    for (long long d = 0; d < NTOT; d++) {
        long long w = start + d; if (w >= NTOT) w -= NTOT;
        if (cnt[w] == 0) {
            if (!opt_ucache) return w;
            if (found < 0) found = w;
            else if (ucn < UCCAP) ucbuf[ucn++] = w;
            else break;
        }
    }
    return found;
}

static void word_of(long long idx, uint8_t *w)
{
    for (int j = 0; j < n; j++) { w[j] = (uint8_t)((idx / pw[j]) % q); }
}

int main(int argc, char **argv)
{
    g_t_start = wall_now();     /* the budget covers initialisation too */
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
    long long target = -1;          /* stop once best_uncovered <= target */
    double cpulimit = 0.0;          /* CPU-seconds budget (0 = use -t wall) */

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
        /* ---- engineering options ---- */
        else if (ARG("--st"))     opt_st = atoi(argv[++i]);
        else if (ARG("--eval"))   opt_eval = atoi(argv[++i]);
        else if (ARG("--walk"))   opt_walk = atoi(argv[++i]);
        else if (!strcmp(a, "--hoist"))  opt_hoist = 1;
        else if (!strcmp(a, "--early"))  opt_early = 1;
        else if (!strcmp(a, "--pf"))     opt_pf = 1;
        else if (!strcmp(a, "--huge"))   opt_huge = 1;
        else if (!strcmp(a, "--fix0"))   opt_fix0 = 1;
        else if (!strcmp(a, "--ucache")) opt_ucache = 1;
        else if (!strcmp(a, "--upick"))  opt_upick = 1;
        else if (!strcmp(a, "--wide"))   opt_wide = 1;
        else if (ARG("--target")) target = atoll(argv[++i]);
        else if (ARG("--cpu"))    cpulimit = atof(argv[++i]);
        else if (ARG("--preset")) {
            const char *pz = argv[++i];
            if (!strcmp(pz, "base")) { /* exactly cov/search/covsearch.c */ }
            else if (!strcmp(pz, "p1")) { opt_walk = 1; opt_hoist = 1; }
            else if (!strcmp(pz, "p2")) { opt_walk = 1; opt_hoist = 1; opt_st = 1; }
            else if (!strcmp(pz, "p2b")){ opt_walk = 1; opt_hoist = 1; opt_st = 2; }
            else if (!strcmp(pz, "p3")) { opt_hoist = 1; opt_st = 1; opt_eval = 1; }
            else if (!strcmp(pz, "p4")) { opt_hoist = 1; opt_st = 1; opt_eval = 2; }
            else if (!strcmp(pz, "p5") || !strcmp(pz, "best"))
                                        { opt_hoist = 1; opt_st = 1; opt_eval = 2; opt_huge = 1; }
            else if (!strcmp(pz, "p5b")) { opt_hoist = 1; opt_st = 2; opt_eval = 2; opt_huge = 1; }
            else { fprintf(stderr, "unknown preset %s\n", pz); return 2; }
        }
        else { fprintf(stderr, "unknown/incomplete option %s\n", a); return 2; }
        #undef ARG
    }
    g_outpath = outpath;
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

    cnt = xalloc(sizeof(uint16_t) * (size_t)NTOT);
    if (!cnt) { fprintf(stderr, "cannot allocate %.1f GB for counters\n",
                        sizeof(uint16_t) * (double)NTOT / 1e9); return 2; }
    if (opt_st) {
        size_t sb = (opt_st == 1) ? (size_t)NTOT : (size_t)((NTOT + 3) / 4);
        st = xalloc(sb);
        if (!st) { fprintf(stderr, "cannot allocate state array\n"); return 2; }
    }
    code = malloc((size_t)M * n);
    bestcode = malloc((size_t)M * n);
    cidx = malloc(sizeof(long long) * M);
    build_combs();
    build_fcombs();
    if (opt_eval == 2) {
        ul_idx = malloc(sizeof(long long) * ULCAP);
        ul_dig = malloc((size_t)ULCAP * n);
        ul_bit = calloc((size_t)((NTOT + 63) / 64), 8);
        if (!ul_idx || !ul_dig || !ul_bit) {
            fprintf(stderr, "cannot allocate uncovered-word list\n"); return 2;
        }
        ul_valid = 0;   /* built lazily once the uncovered set is small enough */
    }
    rseed(&grng, seed);

    PPC = ipow(q - 1, R);
    TOTPAT = (long long)NCOMB * PPC;
    ucrec = opt_ucache;
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
            st_reset();
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
            st_reset();
        uncovered = NTOT;
        for (int i = 0; i < have; i++) {
            cidx[i] = widx(code + (size_t)i * n);
            ball_apply(code + (size_t)i * n, cidx[i], +1);
        }
        for (int i = have; i < M; i++) {
            /* Deadline-bounded initialisation (imported from `covfast`).  On
             * the biggest cells marking M balls of |B_R| words each can outrun
             * the whole budget; being killed at that point would leave nothing
             * on disk at all.  If the budget is gone, place the remaining
             * codewords without marking their balls -- the bookkeeping is
             * about to be discarded anyway -- publish a complete, valid,
             * distinct M-word code, and report uncovered as unknown (-1)
             * rather than as a number the counters no longer support. */
            if (elapsed_total() > tlimit) {
                for (; i < M; i++) {
                    long long w2 = pick_uncovered(&grng);
                    if (w2 < 0) w2 = (long long)rmod(&grng, (uint64_t)NTOT);
                    word_of(w2, code + (size_t)i * n);
                    cidx[i] = w2;
                }
                g_init_aborted = 1;
                break;
            }
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
    if (opt_fix0) {
        /* Symmetry reduction.  w -> w - c_0 (componentwise mod q) is an
         * isometry of the Hamming space, so translating the whole code so
         * that codeword 0 sits at 0^n changes neither the covering radius nor
         * the uncovered count.  Codeword 0 is then frozen for the rest of the
         * run, which quotients out the q^n translations. */
        uint8_t t0v[64];
        memcpy(t0v, code, n);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < n; j++)
                code[(size_t)i * n + j] = (uint8_t)((code[(size_t)i * n + j] + q - t0v[j]) % q);
        rebuild();
    }
    memcpy(bestcode, code, (size_t)M * n);
    best_uncovered = uncovered;
    /* From this line on there is no execution path -- overrun, SIGKILL, OOM --
     * that leaves the caller an empty or unparseable output file. */
    publish(bestcode, M, 1);
    if (g_init_aborted) {
        printf("RESULT q=%d n=%d R=%d M=%d uncovered=-1 iters=0 kicks=0 "
               "time=%.2f rate=0 cpu=%.3f cpurate=0 target=-1 "
               "ttt_cpu=-1 ttt_wall=-1 ttt_it=-1 init_aborted=1\n",
               q, n, R, M, elapsed_total(), cpu_now());
        fflush(stdout);
        return 1;
    }
    if (verbose) { printf("# init uncovered=%lld\n", uncovered); fflush(stdout); }

    /* tabu table */
    int *tabu = calloc((size_t)M * n * q, sizeof(int));

    struct timespec t0, tn;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    uint8_t *u = malloc(n), *tmpw = malloc(n);
    size_t candcap = (size_t)M * (opt_wide ? (size_t)n * q : (size_t)(R + 2));
    int *cand_i = malloc(sizeof(int) * candcap);
    int *cand_p = malloc(sizeof(int) * candcap);
    int *cand_v = malloc(sizeof(int) * candcap);
    int *dist = malloc(sizeof(int) * M);
    long long *cand_d = malloc(sizeof(long long) * candcap);
    int wide_now = 0;
    /* Parallelise over candidates when there are enough of them and each
     * sphere is small; otherwise parallelise inside the sphere walk. */
    int par_cand = (NCOMB < 256);

    long long it = 0, since_improve = 0, nkicks = 0;
    double last_report = 0;
    T = t0temp;
    double cpu0 = cpu_now();
    double ttt_cpu = -1, ttt_wall = -1;
    long long ttt_it = -1;
    if (target >= 0 && best_uncovered <= target) { ttt_cpu = 0; ttt_wall = 0; ttt_it = 0; }

    /* per-iteration offset tables, one per distinct candidate codeword */
    long long *offs_pool = malloc(sizeof(long long) * (size_t)M * n * q);
    int *offs_have = calloc((size_t)M, sizeof(int));
    long long offs_epoch = 0;
    /* shared-sphere evaluator scratch */
    long long *dvec = malloc(sizeof(long long) * (size_t)M * n * q);
    int *dvec_epoch = calloc((size_t)M, sizeof(int));
    int *cw_list = malloc(sizeof(int) * (size_t)M);

    while (uncovered > 0) {
        {
            /* The clock is read on EVERY iteration, not every 16th.  One
             * iteration of K_8(9,4) costs most of a second, so a coarse check
             * overshoots the budget by seconds -- which is how the baseline
             * turned a 60 s budget into a 120 s run and got killed with an
             * empty file.  clock_gettime is a vDSO read of ~20 ns against a
             * >=30 us iteration, i.e. under 0.1%. */
            clock_gettime(CLOCK_MONOTONIC, &tn);
            double el = (tn.tv_sec - t0.tv_sec) + 1e-9 * (tn.tv_nsec - t0.tv_nsec);
            if (cpulimit > 0.0) { if (cpu_now() - cpu0 > cpulimit) break; }
            else if (elapsed_total() > tlimit) break;
            if (verbose && el - last_report > 10.0) {
                last_report = el;
                printf("# t=%6.1fs it=%-10lld unc=%-10lld best=%lld (%.0f it/s)\n",
                       el, it, uncovered, best_uncovered, it / el);
                fflush(stdout);
            }
        }
        if (maxiter >= 0 && it >= maxiter) break;
        if (target >= 0 && best_uncovered <= target) break;
        it++;
        if (opt_eval == 2) {
            if (!ul_valid && uncovered <= ULCAP / 2) ul_rebuild();
            else if (ul_valid && ul_n > 2 * uncovered + 64) ul_compact();
        }
#ifdef PROF
        prof_iters++; prof_unc_sum += uncovered;
#endif

        long long uw;
        { PB(P_PICK); uw = pick_uncovered(&grng); PE(P_PICK); }
        if (uw < 0) break;
        word_of(uw, u);

        int ncand = 0, take = 0, dtar = 0;
        { PB(P_CAND);
        /* distances to all codewords */
        int dmin = n + 1;
        for (int i = opt_fix0; i < M; i++) {
            const uint8_t *c = code + (size_t)i * n;
            int d = 0;
            for (int j = 0; j < n; j++) d += (c[j] != u[j]);
            dist[i] = d;
            if (d < dmin) dmin = d;
        }
        /* target distance: R+1 if reachable in one move, else the minimum */
        dtar = (dmin <= R + 1) ? R + 1 : dmin;

        /* The wide neighbourhood needs the uncovered-word list, because the
         * fallback entering side would have to walk n(q-1) spheres instead of
         * R+1.  Decide before collecting, so the candidate set and the
         * evaluator agree. */
        wide_now = opt_wide && opt_eval == 2 && ul_valid &&
                   ((long long)ul_n * n < (long long)(R + 1) * TOTPAT);
        for (int i = opt_fix0; i < M; i++) {
            if (dist[i] != dtar) continue;
            const uint8_t *c = code + (size_t)i * n;
            if (wide_now) {
                for (int p = 0; p < n; p++)
                    for (int v = 0; v < q; v++) {
                        if (v == c[p]) continue;
                        cand_i[ncand] = i; cand_p[ncand] = p;
                        cand_v[ncand] = v; ncand++;
                    }
            } else {
                for (int p = 0; p < n; p++) {
                    if (c[p] == u[p]) continue;
                    cand_i[ncand] = i; cand_p[ncand] = p;
                    cand_v[ncand] = u[p]; ncand++;
                }
            }
        }
        if (ncand == 0) {           /* should not happen, but stay safe */
            int i = opt_fix0 + (int)rmod(&grng, M - opt_fix0), p = (int)rmod(&grng, n);
            cand_i[0] = i; cand_p[0] = p; cand_v[0] = u[p]; ncand = 1;
        }

        /* sample at most `cand` of them */
        take = ncand < cand ? ncand : cand;
        for (int s = 0; s < take; s++) {
            int j = s + (int)rmod(&grng, (uint64_t)(ncand - s));
            int ti = cand_i[s], tp = cand_p[s], tv = cand_v[s];
            cand_i[s] = cand_i[j]; cand_p[s] = cand_p[j]; cand_v[s] = cand_v[j];
            cand_i[j] = ti; cand_p[j] = tp; cand_v[j] = tv;
        }
        offs_epoch++;
        if (opt_hoist) {
            /* one offset table per distinct candidate codeword; the R+1
             * candidates that share a codeword then share the table. */
            for (int s = 0; s < take; s++) {
                int i = cand_i[s];
                if (offs_have[i] != (int)(offs_epoch & 0x7fffffff)) {
                    offs_have[i] = (int)(offs_epoch & 0x7fffffff);
                    make_offs(code + (size_t)i * n, offs_pool + (size_t)i * n * q);
                }
            }
        }
        PE(P_CAND); }
#ifdef PROF
        prof_ncand_sum += ncand; prof_take_sum += take;
#endif

        int bi = -1, bp = -1, bv = -1;
        if (sa_mode) {
            /* Focused simulated annealing: propose ONE random move that covers
             * u, evaluate it, and accept by Metropolis.  One sphere walk per
             * iteration instead of |candidates|, so this trades move quality
             * for one to two orders of magnitude more moves per second. */
            int s2 = (int)rmod(&grng, (uint64_t)take);
            int i = cand_i[s2], p = cand_p[s2], v = cand_v[s2];
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
            bi = cand_i[s]; bp = cand_p[s]; bv = cand_v[s];
        } else {
            long long bestd = 1LL << 60;
            int nties = 0;
            /* Evaluate the sampled candidates in parallel: one thread per
             * candidate.  Each evaluation only READS cnt[], so no locking. */
            { PB(P_EVAL);
            if (opt_eval) {
                /* group the candidates by codeword and evaluate each codeword's
                 * whole fan of moves with one shared sphere enumeration */
                int ncw = 0;
                for (int s = 0; s < take; s++) {
                    int i = cand_i[s];
                    if (dvec_epoch[i] != (int)(offs_epoch & 0x7fffffff)) {
                        dvec_epoch[i] = (int)(offs_epoch & 0x7fffffff);
                        cw_list[ncw++] = i;
                    }
                }
                int use_ul = (opt_eval == 2) && ul_valid &&
                             ((long long)ul_n * n < (long long)(R + 1) * TOTPAT);
#pragma omp parallel for schedule(dynamic, 1) if(ncw > 1 && par_cand)
                for (int z = 0; z < ncw; z++) {
                    int i = cw_list[z];
                    eval_codeword(i, u,
                                  opt_hoist ? offs_pool + (size_t)i * n * q : NULL,
                                  dvec + (size_t)i * n * q, use_ul);
                }
                for (int s = 0; s < take; s++)
                    cand_d[s] = dvec[((size_t)cand_i[s] * n + cand_p[s]) * q + cand_v[s]];
#ifdef PROF
                prof_ncw_sum += ncw; prof_ul_used += use_ul; prof_ul_n_sum += ul_n;
#endif
            } else if (opt_early) {
                /* shared incumbent so that later candidates can be pruned;
                 * `cut` is only ever lowered, so a stale read is safe (it just
                 * prunes less).  A pruned candidate returns EVAL_PRUNED, which
                 * loses every comparison below. */
                volatile long long cut = LLONG_MAX;
#pragma omp parallel for schedule(dynamic, 1) if(take > 3 && par_cand)
                for (int s = 0; s < take; s++) {
                    long long myc = cut;
                    long long d = move_eval(cand_i[s], cand_p[s], cand_v[s],
                                            opt_hoist ? offs_pool + (size_t)cand_i[s] * n * q : NULL,
                                            myc);
                    cand_d[s] = d;
                    if (d != EVAL_PRUNED && d < cut) cut = d;
                }
            } else {
#pragma omp parallel for schedule(dynamic, 1) if(take > 3 && par_cand)
                for (int s = 0; s < take; s++)
                    cand_d[s] = move_eval(cand_i[s], cand_p[s], cand_v[s],
                                          opt_hoist ? offs_pool + (size_t)cand_i[s] * n * q : NULL,
                                          LLONG_MAX);
            }
            PE(P_EVAL); }
            { PB(P_SEL);
            for (int s = 0; s < take; s++) {
                int i = cand_i[s], p = cand_p[s], v = cand_v[s];
                long long d = cand_d[s];
                if (d == EVAL_PRUNED) continue;
                /* tabu forbids RESTORING a value we recently moved away from,
                 * so the test is on the value we are moving TO. */
                int is_tabu = tabu[((size_t)i * n + p) * q + v] > it;
                if (is_tabu && uncovered + d >= best_uncovered) continue;  /* aspiration */
                if (d < bestd) { bestd = d; bi = i; bp = p; bv = v; nties = 1; }
                else if (d == bestd) { nties++; if (rmod(&grng, (uint64_t)nties) == 0) { bi = i; bp = p; bv = v; } }
            }
            if (bi < 0) {   /* everything tabu/pruned: take a random candidate */
                int s = (int)rmod(&grng, (uint64_t)take);
                bi = cand_i[s]; bp = cand_p[s]; bv = cand_v[s];
            }
#ifdef PROF
            if (bestd < (1LL << 59)) prof_bestd_sum += bestd;
#endif
            PE(P_SEL); }
        }

        int oldv = code[(size_t)bi * n + bp];
        /* the uncovered-word list is maintained inside the commit walk, which
         * would race if the walk were OpenMP-parallel, so serialise it there */
        { PB(P_COMMIT); move_commit(bi, bp, bv, opt_eval == 2 ? 0 : !par_cand); PE(P_COMMIT); }
        tabu[((size_t)bi * n + bp) * q + oldv] = (int)(it + tabu_len + rmod(&grng, 4));

        if (uncovered < best_uncovered) {
            PB(P_BEST);
            best_uncovered = uncovered;
            memcpy(bestcode, code, (size_t)M * n);
            since_improve = 0;
            if (target >= 0 && ttt_it < 0 && best_uncovered <= target) {
                clock_gettime(CLOCK_MONOTONIC, &tn);
                ttt_wall = (tn.tv_sec - t0.tv_sec) + 1e-9 * (tn.tv_nsec - t0.tv_nsec);
                ttt_cpu = cpu_now() - cpu0;
                ttt_it = it;
            }
            if (verbose) {
                clock_gettime(CLOCK_MONOTONIC, &tn);
                double el = (tn.tv_sec - t0.tv_sec) + 1e-9 * (tn.tv_nsec - t0.tv_nsec);
                printf("# improved: unc=%lld at it=%lld t=%.1fs cpu=%.4f\n",
                       best_uncovered, it, el, cpu_now() - cpu0);
                fflush(stdout);
            }
            PE(P_BEST);
            /* republish, throttled to 1 Hz; a full cover is published at once */
            publish(bestcode, M, best_uncovered == 0);
            if (best_uncovered == 0) break;
        } else if (++since_improve > stall_limit) {
            /* Kick: teleport a few codewords onto (perturbed) uncovered
             * words.  Landing on an uncovered word is strictly more useful
             * than landing anywhere, and the perturbation keeps it from
             * undoing itself on the next iteration. */
            PB(P_KICK);
            for (int t = 0; t < kick; t++) {
                int i = opt_fix0 + (int)rmod(&grng, (uint64_t)(M - opt_fix0));
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
            PE(P_KICK);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &tn);
    double el = (tn.tv_sec - t0.tv_sec) + 1e-9 * (tn.tv_nsec - t0.tv_nsec);
    double cpu = cpu_now() - cpu0;
    printf("RESULT q=%d n=%d R=%d M=%d uncovered=%lld iters=%lld kicks=%lld "
           "time=%.2f rate=%.0f cpu=%.3f cpurate=%.0f target=%lld "
           "ttt_cpu=%.4f ttt_wall=%.4f ttt_it=%lld\n",
           q, n, R, M, best_uncovered, it, nkicks, el, it / (el > 0 ? el : 1),
           cpu, it / (cpu > 0 ? cpu : 1), target, ttt_cpu, ttt_wall, ttt_it);
#ifdef PROF
    {
        double tot = 0;
        for (int i = 0; i < P_N; i++) tot += prof_s[i];
        printf("# PROF total_measured=%.3f cpu=%.3f iters=%lld\n", tot, cpu, prof_iters);
        for (int i = 0; i < P_N; i++)
            printf("# PROF %-15s %8.3f s  %6.2f%% of cpu  calls=%lld  us/call=%.2f\n",
                   pname[i], prof_s[i], 100.0 * prof_s[i] / (cpu > 0 ? cpu : 1),
                   prof_c[i], prof_c[i] ? 1e6 * prof_s[i] / prof_c[i] : 0.0);
        printf("# PROF patterns_eval=%lld patterns_commit=%lld ratio=%.1f\n",
               prof_patterns_eval, prof_patterns_commit,
               prof_patterns_commit ? (double)prof_patterns_eval / prof_patterns_commit : 0.0);
        printf("# PROF mean_ncand=%.1f mean_take=%.1f mean_uncovered=%.1f "
               "mean_bestd=%.2f pick_scans=%lld early_saved_patterns=%lld\n",
               prof_iters ? (double)prof_ncand_sum / prof_iters : 0.0,
               prof_iters ? (double)prof_take_sum / prof_iters : 0.0,
               prof_iters ? (double)prof_unc_sum / prof_iters : 0.0,
               prof_iters ? (double)prof_bestd_sum / prof_iters : 0.0,
               prof_pick_scans, prof_early_saved);
        printf("# PROF mean_ncw=%.1f ul_used_frac=%.3f mean_ul_n=%.1f "
               "ul_rebuilds=%lld ul_compacts=%lld\n",
               prof_iters ? (double)prof_ncw_sum / prof_iters : 0.0,
               prof_iters ? (double)prof_ul_used / prof_iters : 0.0,
               prof_iters ? (double)prof_ul_n_sum / prof_iters : 0.0,
               prof_ul_rebuild, prof_ul_compact);
        printf("# PROF ns_per_eval_pattern=%.3f\n",
               prof_patterns_eval ? 1e9 * prof_s[P_EVAL] / prof_patterns_eval : 0.0);
    }
#endif
    if (outpath) publish(bestcode, M, 1);
    return best_uncovered == 0 ? 0 : 1;
}
