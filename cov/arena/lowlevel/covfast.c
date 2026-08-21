/*
 * covfast.c -- arena entry "lowlevel".
 *
 * SAME search strategy as cov/search/covsearch.c (focused local search:
 * pick a random uncovered word, enumerate every codeword-move that could
 * cover it, evaluate them all exactly, commit the best, kick on stagnation).
 * Everything that changed is under the hood:
 *
 *   1. Compile-time specialisation of the sphere walk on R (1..6) and on
 *      q-1 (2, 5, 7, generic).  The baseline reads q, n, R from globals
 *      inside the innermost loop and runs a runtime odometer with a carry
 *      chain; here the (q-1)^R value enumeration is a static loop nest with
 *      loop-invariant offset pointers, so the carry logic disappears.
 *   2. Branchless accumulation.  The baseline's `if (cnt[a]==1) dunc++;`
 *      mispredicts ~30% of the time and serialises the load stream; the
 *      evaluation kernels here compute `(cnt[a]==1) - (cnt[b]==0)` with no
 *      control flow, which lets the core keep many independent misses in
 *      flight.
 *   3. NEON.  When the innermost varying coordinate is the last one its
 *      q-1 words are contiguous, so one 128-bit load + compare + masked
 *      horizontal add replaces q-1 scalar load/compare/branch triples.
 *      That covers R/(n-1) of all position subsets.
 *   4. uint8 counters whenever M <= 255 (then cnt[w] <= M can never
 *      overflow, so this is exact, not an approximation).  Halves the
 *      working set: K_6(6,3) becomes L1-resident, K_3(11,4) L2-resident.
 *   5. Early exit.  The decrease part of a move's delta is at most the
 *      number of currently uncovered words U, so a candidate whose partial
 *      delta already exceeds (best-so-far + U) can never win.  Checked once
 *      per position subset.  This prunes *exactly* the candidates that are
 *      provably not optimal, so the move actually chosen -- and the random
 *      tie-break -- are bit-identical to an unpruned search.
 *   6. O(1) uncovered-word sampling.  The baseline falls back to a full
 *      q^n linear scan once uncovered words are rare, which is exactly the
 *      endgame; here an exact uncovered-word list with an index array is
 *      maintained incrementally by the commit path.
 *   7. Cheap RNG bounding (Lemire) instead of `%`.
 *
 * Build: gcc -O3 -march=native -fopenmp -fno-stack-protector -o covfast covfast.c
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#define HAVE_NEON 1
#endif

typedef int64_t OFFT;

/* value reported for a candidate whose evaluation was cut short: larger than
 * any real delta (|delta| <= 2S), smaller than the initial bestd sentinel. */
#define PRUNED (1LL << 58)

/* ------------------------------------------------------------------ */
/* parameters and state                                                */
/* ------------------------------------------------------------------ */

static int q, n, R, M, q1;
static long long NTOT;
static long long pw[40];

static uint8_t  *cnt8;             /* one of these two is live          */
static uint16_t *cnt16;
static int use8;
/* uint8 counters are used even when M > 255: cnt[w] is EXACT while it fits,
 * and a word's counter can rise by at most one per committed move (the
 * entering set of a move contains each word once), so latching a flag when a
 * counter reaches sat_thresh and widening to uint16 before the next move can
 * never lose a count.  In practice the flag never fires -- the mean coverage
 * multiplicity is M*|B_R|/q^n, about 2.3 on K_8(9,4) -- but the fallback makes
 * the optimisation exact rather than merely likely. */
static int satflag;
static unsigned sat_thresh = 250;
static long long npromote;
static long long uncovered;

static uint8_t *code;              /* M*n digits                        */
static long long *cidx;            /* M word indices                    */
static OFFT *offtab;               /* M * n * q   value offsets         */

static uint8_t *bestcode;
static long long best_uncovered;

static int NCOMB, CSTRIDE;
static uint8_t *combs;             /* n * NCOMB * max(R,1)              */

static unsigned nthreads = 1;

/* Exact uncovered-word list: O(1) uniform sampling with no scans.  Only worth
 * its keep once uncovered words are rare -- when a large fraction of the space
 * is uncovered, random probing finds one immediately and the list just makes
 * every commit pay for thousands of insert/delete pairs.  So it is switched on
 * and off dynamically against uncovered/q^n, with hysteresis. */
static int32_t *ulist, *upos;
static long long ulen;
static int ulist_on;      /* currently maintained */
static int ulist_avail;   /* allowed (memory, index width) */

/* NEON lane masks: valid lanes are v<q and v!=clast */
static uint16_t maskv16[64][8];
static uint8_t  maskv8[64][16];
static int use_simd;

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
/* Lemire's nearly-divisionless bounded rand: one multiply, no modulo on the
 * fast path.  Distribution is uniform (the rejection branch is taken with
 * probability < m/2^64). */
static int modrng = 0;
static inline uint64_t rmod(rng_t *r, uint64_t m) {
    if (__builtin_expect(modrng, 0)) return rnext(r) % m;
    __uint128_t x = (__uint128_t)rnext(r) * m;
    uint64_t l = (uint64_t)x;
    if (__builtin_expect(l < m, 0)) {
        uint64_t t = (uint64_t)(-(int64_t)m) % m;
        while (l < t) { x = (__uint128_t)rnext(r) * m; l = (uint64_t)x; }
    }
    return (uint64_t)(x >> 64);
}

/* ------------------------------------------------------------------ */
/* combinatorics                                                       */
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
    CSTRIDE = R ? R : 1;
    combs = malloc((size_t)n * NCOMB * CSTRIDE);
    for (int p = 0; p < n; p++) {
        int rest[40], nr = 0;
        for (int j = 0; j < n; j++) if (j != p) rest[nr++] = j;
        int c[16];
        for (int i = 0; i < R; i++) c[i] = i;
        for (int k = 0; k < NCOMB; k++) {
            for (int i = 0; i < R; i++)
                combs[((size_t)p * NCOMB + k) * CSTRIDE + i] = (uint8_t)rest[c[i]];
            int i = R - 1;
            while (i >= 0 && c[i] == nr - R + i) i--;
            if (i < 0) break;
            c[i]++;
            for (int j = i + 1; j < R; j++) c[j] = c[j - 1] + 1;
        }
    }
}

/* offsets for codeword i: off[j*q+k] = (k-th value != c_j, minus c_j)*pw[j] */
static void make_offs_row(const uint8_t *c, OFFT *offs, int j) {
    int k = 0;
    for (int v = 0; v < q; v++)
        if (v != c[j]) offs[(size_t)j * q + (k++)] = (OFFT)(v - c[j]) * pw[j];
}
static void make_offs(const uint8_t *c, OFFT *offs) {
    for (int j = 0; j < n; j++) make_offs_row(c, offs, j);
}

/* ------------------------------------------------------------------ */
/* uncovered-word list                                                 */
/* ------------------------------------------------------------------ */

static inline void u_add(long long w) {
    if (!ulist_on) return;
    upos[w] = (int32_t)ulen; ulist[ulen++] = (int32_t)w;
}
static inline void u_del(long long w) {
    if (!ulist_on) return;
    int32_t i = upos[w];
    int32_t last = ulist[--ulen];
    ulist[i] = last; upos[last] = i; upos[w] = -1;
}
static void u_rebuild(void) {
    if (!ulist_on) return;
    ulen = 0;
    if (use8) { for (long long w = 0; w < NTOT; w++) if (!cnt8[w]) { upos[w] = (int32_t)ulen; ulist[ulen++] = (int32_t)w; } else upos[w] = -1; }
    else      { for (long long w = 0; w < NTOT; w++) if (!cnt16[w]) { upos[w] = (int32_t)ulen; ulist[ulen++] = (int32_t)w; } else upos[w] = -1; }
}

/* Parallel-decision calibration.  One OpenMP fork/join per iteration is only
 * worth it if an iteration is big.  On K_8(9,4) a candidate is ~25 ms and 6
 * threads are a 6x win; on K_6(6,3) a candidate is ~0.6 us and, on a box that
 * is 3x oversubscribed by other tenants, waiting for six descheduled threads at
 * the barrier costs ~1 ms of CPU per iteration -- measured 34x WORSE than
 * running serially.  Rather than guess a threshold, alternate blocks of serial
 * and parallel iterations, keep the faster, and re-measure every 20 s so the
 * decision tracks the machine's actual load. */
#define CAL_N 128
static int par_mode = 0, cal_phase = 0;
static long long cal_n = 0;
static double cal_t0, cal_begin, cal_serial, cal_par, cal_until;

static struct timespec T0;
static inline double elapsed(void) {
    struct timespec x; clock_gettime(CLOCK_MONOTONIC, &x);
    return (x.tv_sec - T0.tv_sec) + 1e-9 * (x.tv_nsec - T0.tv_nsec);
}

static inline int is_unc(long long w) { return use8 ? (cnt8[w] == 0) : (cnt16[w] == 0); }

static void promote16(void);

/* turn the list on when uncovered < q^n/32, off again above q^n/16 */
static void u_track(void)
{
    if (!ulist_avail) return;
    if (!ulist_on) {
        if (uncovered * 32 < NTOT) {
            if (!ulist) {
                ulist = malloc(sizeof(int32_t) * (size_t)NTOT);
                upos  = malloc(sizeof(int32_t) * (size_t)NTOT);
                if (!ulist || !upos) {
                    free(ulist); free(upos); ulist = upos = NULL;
                    ulist_avail = 0; return;
                }
            }
            ulist_on = 1;
            u_rebuild();
        }
    } else if (uncovered * 16 > NTOT) {
        ulist_on = 0;
    }
}

/* ------------------------------------------------------------------ */
/* full-ball apply (initialisation and kicks)                          */
/* ------------------------------------------------------------------ */

#define GEN_BALL(NAME, CT, CNT, SAT)                                           \
static long long NAME(long long base, const OFFT *offs, int startpos,          \
                      int rem, int delta)                                      \
{                                                                              \
    long long dunc = 0;                                                        \
    if (rem == 0) return 0;                                                    \
    for (int j = startpos; j < n; j++)                                         \
        for (int k = 0; k < q1; k++) {                                         \
            long long w = base + offs[(size_t)j * q + k];                      \
            if (delta > 0) { if (CNT[w]++ == 0) { dunc--; u_del(w); }          \
                             else if ((unsigned)CNT[w] >= (SAT)) satflag = 1; }\
            else           { if (--CNT[w] == 0) { dunc++; u_add(w); } }        \
            dunc += NAME(w, offs, j + 1, rem - 1, delta);                      \
        }                                                                      \
    return dunc;                                                               \
}
GEN_BALL(ball_rec8, uint8_t, cnt8, sat_thresh)
GEN_BALL(ball_rec16, uint16_t, cnt16, 65500u)

/* Marking one radius-R ball costs |B_R| touches -- 333k on K_8(9,4), times M
 * codewords -- and on that cell it can eat the entire wall-clock budget before
 * the search starts.  Within ONE ball each word is generated exactly once (the
 * recursion enumerates each (subset, assignment) pair once), and the top-level
 * loop over the first differing position partitions those subsets, so splitting
 * that loop across threads touches disjoint words and needs no atomics.  Only
 * used for the initial marking, where the uncovered-word list is switched off. */
#define GEN_BALL_PAR(NAME, REC, CNT)                                           \
static long long NAME(long long base, const OFFT *offs)                        \
{                                                                              \
    long long dunc = 0;                                                        \
    _Pragma("omp parallel for schedule(dynamic,1) reduction(+:dunc)")          \
    for (int j = 0; j < n; j++)                                                \
        for (int k = 0; k < q1; k++) {                                         \
            long long w = base + offs[(size_t)j * q + k];                      \
            if (CNT[w]++ == 0) dunc--;                                         \
            dunc += REC(w, offs, j + 1, R - 1, +1);                            \
        }                                                                      \
    return dunc;                                                               \
}
GEN_BALL_PAR(ball_par8, ball_rec8, cnt8)
GEN_BALL_PAR(ball_par16, ball_rec16, cnt16)

static int par_init = 0;

static void ball_apply(const uint8_t *c, long long base, int delta)
{
    OFFT offs[2048];
    make_offs(c, offs);
    long long dunc = 0;
    if (par_init && delta > 0 && R >= 1 && !ulist_on) {
        if (use8) { if (cnt8[base]++ == 0) dunc--; dunc += ball_par8(base, offs); }
        else      { if (cnt16[base]++ == 0) dunc--; dunc += ball_par16(base, offs); }
        uncovered += dunc;
        return;
    }
    if (use8) {
        if (delta > 0) { if (cnt8[base]++ == 0) { dunc--; u_del(base); }
                         else if (cnt8[base] >= sat_thresh) satflag = 1; }
        else           { if (--cnt8[base] == 0) { dunc++; u_add(base); } }
        dunc += ball_rec8(base, offs, 0, R, delta);
    } else {
        if (delta > 0) { if (cnt16[base]++ == 0) { dunc--; u_del(base); } }
        else           { if (--cnt16[base] == 0) { dunc++; u_add(base); } }
        dunc += ball_rec16(base, offs, 0, R, delta);
    }
    uncovered += dunc;
}

/* ------------------------------------------------------------------ */
/* the sphere kernels                                                  */
/* ------------------------------------------------------------------ */
/*
 * A move is "codeword i, position p, new value v".  The words that leave the
 * ball of c are the w with d(w,c)==R, w_p==c_p; the words that enter the ball
 * of c' are those same w with coordinate p bumped by dv.  Enumerate the R
 * differing positions (a subset of the n-1 positions != p, precomputed in
 * `combs`) and their (q-1)^R value assignments.
 *
 * The value assignment is a static nest of depth R with loop-invariant offset
 * row pointers O0..O_{R-2}; the innermost coordinate pos[R-1] is handled by
 * INNER.  combs rows are sorted ascending, so pos[R-1] is the largest position
 * and equals n-1 (stride 1, contiguous words) exactly when the subset contains
 * the last coordinate.
 */

#define DECL_O(d) const OFFT *O##d = off + (size_t)pos[d] * q;
#define LOOPB(d, prev) for (int k##d = 0; k##d < q1_; k##d++) { long long a##d = (prev) + O##d[k##d];

#define NEST_1(IN) IN(base)
#define NEST_2(IN) DECL_O(0) LOOPB(0,base) IN(a0) }
#define NEST_3(IN) DECL_O(0) DECL_O(1) LOOPB(0,base) LOOPB(1,a0) IN(a1) } }
#define NEST_4(IN) DECL_O(0) DECL_O(1) DECL_O(2) \
                   LOOPB(0,base) LOOPB(1,a0) LOOPB(2,a1) IN(a2) } } }
#define NEST_5(IN) DECL_O(0) DECL_O(1) DECL_O(2) DECL_O(3) \
                   LOOPB(0,base) LOOPB(1,a0) LOOPB(2,a1) LOOPB(3,a2) IN(a3) } } } }
#define NEST_6(IN) DECL_O(0) DECL_O(1) DECL_O(2) DECL_O(3) DECL_O(4) \
                   LOOPB(0,base) LOOPB(1,a0) LOOPB(2,a1) LOOPB(3,a2) LOOPB(4,a3) IN(a4) } } } } }

/* ---- evaluation (read-only, branchless) ---- */

#define EV_SCAL(A) { long long aa_; \
    for (int kk_ = 0; kk_ < q1_; kk_++) { aa_ = (A) + OL[kk_]; \
        dunc += (long long)(cnt_[aa_] == 1) - (long long)(cnt_[aa_ + dv] == 0); } }

#ifdef HAVE_NEON
#define EV_CONT16(A) { const uint16_t *pa_ = cnt_ + ((A) - clast); \
    uint16x8_t va_ = vld1q_u16(pa_), vb_ = vld1q_u16(pa_ + dv); \
    uint16x8_t e1_ = vandq_u16(vceqq_u16(va_, vdupq_n_u16(1)), vmsk); \
    uint16x8_t e0_ = vandq_u16(vceqzq_u16(vb_), vmsk); \
    dunc += vaddlvq_s16(vsubq_s16(vreinterpretq_s16_u16(e0_), \
                                  vreinterpretq_s16_u16(e1_))); }
#define EV_CONT8(A) { const uint8_t *pa_ = cnt_ + ((A) - clast); \
    uint8x16_t va_ = vld1q_u8(pa_), vb_ = vld1q_u8(pa_ + dv); \
    uint8x16_t e1_ = vandq_u8(vceqq_u8(va_, vdupq_n_u8(1)), vmsk); \
    uint8x16_t e0_ = vandq_u8(vceqzq_u8(vb_), vmsk); \
    dunc += vaddlvq_s8(vsubq_s8(vreinterpretq_s8_u8(e0_), \
                                vreinterpretq_s8_u8(e1_))); }
#else
#define EV_CONT16(A) EV_SCAL(A)
#define EV_CONT8(A)  EV_SCAL(A)
#endif

/* NAME(combs_p, ncomb, base, off, dv, clast, mask, thresh) */
#define GEN_EVAL(NAME, CT, CNT, RR, Q1, EV_CONT, VT)                           \
static long long NAME(const uint8_t *cb, int ncomb, long long base,            \
                      const OFFT *off, long long dv, int lastp, int clast,     \
                      const void *maskp, long long thresh)                     \
{                                                                              \
    long long dunc = 0;                                                        \
    CT *const cnt_ = CNT;                                                      \
    const int q1_ = (Q1);                                                      \
    (void)clast; (void)maskp; (void)cnt_; (void)q1_;                           \
    VT                                                                         \
    for (int s_ = 0; s_ < ncomb; s_++) {                                       \
        const uint8_t *pos = cb + (size_t)s_ * RR;                             \
        const OFFT *OL = off + (size_t)pos[RR - 1] * q;                        \
        (void)OL;                                                              \
        if (use_simd && pos[RR - 1] == lastp) { NEST_##RR(EV_CONT) }           \
        else                                  { NEST_##RR(EV_SCAL) }           \
        if (dunc > thresh) return PRUNED;                                      \
    }                                                                          \
    return dunc;                                                               \
}

#ifdef HAVE_NEON
#define VT16 const uint16x8_t vmsk = vld1q_u16((const uint16_t *)maskp);
#define VT8  const uint8x16_t vmsk = vld1q_u8((const uint8_t *)maskp);
#else
#define VT16
#define VT8
#endif

#define GEN_EVAL_SET(SUF, CT, CNT, Q1, EV_CONT, VT)                            \
    GEN_EVAL(ev##SUF##_1, CT, CNT, 1, Q1, EV_CONT, VT)                         \
    GEN_EVAL(ev##SUF##_2, CT, CNT, 2, Q1, EV_CONT, VT)                         \
    GEN_EVAL(ev##SUF##_3, CT, CNT, 3, Q1, EV_CONT, VT)                         \
    GEN_EVAL(ev##SUF##_4, CT, CNT, 4, Q1, EV_CONT, VT)                         \
    GEN_EVAL(ev##SUF##_5, CT, CNT, 5, Q1, EV_CONT, VT)                         \
    GEN_EVAL(ev##SUF##_6, CT, CNT, 6, Q1, EV_CONT, VT)

GEN_EVAL_SET(16g, uint16_t, cnt16, q1, EV_CONT16, VT16)
GEN_EVAL_SET(16a, uint16_t, cnt16,  2, EV_CONT16, VT16)
GEN_EVAL_SET(16b, uint16_t, cnt16,  5, EV_CONT16, VT16)
GEN_EVAL_SET(16c, uint16_t, cnt16,  7, EV_CONT16, VT16)
GEN_EVAL_SET(8g,  uint8_t,  cnt8,  q1, EV_CONT8,  VT8)
GEN_EVAL_SET(8a,  uint8_t,  cnt8,   2, EV_CONT8,  VT8)
GEN_EVAL_SET(8b,  uint8_t,  cnt8,   5, EV_CONT8,  VT8)
GEN_EVAL_SET(8c,  uint8_t,  cnt8,   7, EV_CONT8,  VT8)

/* ---- commit (scalar, maintains cnt and the uncovered list) ---- */

#define CM_SCAL(A) { long long aa_, bb_; \
    for (int kk_ = 0; kk_ < q1_; kk_++) { aa_ = (A) + OL[kk_]; bb_ = aa_ + dv; \
        if (--cnt_[aa_] == 0) { dunc++; u_add(aa_); } \
        if (cnt_[bb_]++ == 0) { dunc--; u_del(bb_); } \
        else if ((unsigned)cnt_[bb_] >= sat_) satflag = 1; } }

#define GEN_COMMIT(NAME, CT, CNT, RR, Q1, SAT)                                 \
static long long NAME(const uint8_t *cb, int ncomb, long long base,            \
                      const OFFT *off, long long dv)                           \
{                                                                              \
    long long dunc = 0;                                                        \
    CT *const cnt_ = CNT;                                                      \
    const int q1_ = (Q1);                                                      \
    const unsigned sat_ = (SAT);                                               \
    for (int s_ = 0; s_ < ncomb; s_++) {                                       \
        const uint8_t *pos = cb + (size_t)s_ * RR;                             \
        const OFFT *OL = off + (size_t)pos[RR - 1] * q;                        \
        NEST_##RR(CM_SCAL)                                                     \
    }                                                                          \
    return dunc;                                                               \
}

#define GEN_COMMIT_SET(SUF, CT, CNT, Q1, SAT)                                  \
    GEN_COMMIT(cm##SUF##_1, CT, CNT, 1, Q1, SAT)                               \
    GEN_COMMIT(cm##SUF##_2, CT, CNT, 2, Q1, SAT)                               \
    GEN_COMMIT(cm##SUF##_3, CT, CNT, 3, Q1, SAT)                               \
    GEN_COMMIT(cm##SUF##_4, CT, CNT, 4, Q1, SAT)                               \
    GEN_COMMIT(cm##SUF##_5, CT, CNT, 5, Q1, SAT)                               \
    GEN_COMMIT(cm##SUF##_6, CT, CNT, 6, Q1, SAT)

GEN_COMMIT_SET(16, uint16_t, cnt16, q1, 65500u)
GEN_COMMIT_SET(8,  uint8_t,  cnt8,  q1, sat_thresh)

typedef long long (*evfn_t)(const uint8_t *, int, long long, const OFFT *,
                            long long, int, int, const void *, long long);
typedef long long (*cmfn_t)(const uint8_t *, int, long long, const OFFT *, long long);

static evfn_t EVFN;
static cmfn_t CMFN;

/* R == 0 fallbacks (a move touches exactly two words) */
static long long ev_r0(const uint8_t *cb, int ncomb, long long base,
                       const OFFT *off, long long dv, int lastp, int clast,
                       const void *maskp, long long thresh)
{
    (void)cb; (void)ncomb; (void)off; (void)lastp; (void)clast; (void)maskp; (void)thresh;
    long long a = base, b = base + dv;
    if (use8) return (long long)(cnt8[a] == 1) - (long long)(cnt8[b] == 0);
    return (long long)(cnt16[a] == 1) - (long long)(cnt16[b] == 0);
}
static long long cm_r0(const uint8_t *cb, int ncomb, long long base,
                       const OFFT *off, long long dv)
{
    (void)cb; (void)ncomb; (void)off;
    long long a = base, b = base + dv, dunc = 0;
    if (use8) {
        if (--cnt8[a] == 0) { dunc++; u_add(a); }
        if (cnt8[b]++ == 0) { dunc--; u_del(b); }
        else if (cnt8[b] >= sat_thresh) satflag = 1;
    } else {
        if (--cnt16[a] == 0) { dunc++; u_add(a); }
        if (cnt16[b]++ == 0) { dunc--; u_del(b); }
    }
    return dunc;
}

static void select_kernels(void)
{
    static evfn_t ev16[4][7] = {
        {0, ev16g_1, ev16g_2, ev16g_3, ev16g_4, ev16g_5, ev16g_6},
        {0, ev16a_1, ev16a_2, ev16a_3, ev16a_4, ev16a_5, ev16a_6},
        {0, ev16b_1, ev16b_2, ev16b_3, ev16b_4, ev16b_5, ev16b_6},
        {0, ev16c_1, ev16c_2, ev16c_3, ev16c_4, ev16c_5, ev16c_6}};
    static evfn_t ev8[4][7] = {
        {0, ev8g_1, ev8g_2, ev8g_3, ev8g_4, ev8g_5, ev8g_6},
        {0, ev8a_1, ev8a_2, ev8a_3, ev8a_4, ev8a_5, ev8a_6},
        {0, ev8b_1, ev8b_2, ev8b_3, ev8b_4, ev8b_5, ev8b_6},
        {0, ev8c_1, ev8c_2, ev8c_3, ev8c_4, ev8c_5, ev8c_6}};
    static cmfn_t cm16[7] = {0, cm16_1, cm16_2, cm16_3, cm16_4, cm16_5, cm16_6};
    static cmfn_t cm8[7]  = {0, cm8_1,  cm8_2,  cm8_3,  cm8_4,  cm8_5,  cm8_6};

    if (R == 0) { EVFN = ev_r0; CMFN = cm_r0; return; }
    int qi = (q1 == 2) ? 1 : (q1 == 5) ? 2 : (q1 == 7) ? 3 : 0;
    if (R >= 1 && R <= 6) {
        EVFN = use8 ? ev8[qi][R] : ev16[qi][R];
        CMFN = use8 ? cm8[R] : cm16[R];
    } else { EVFN = NULL; CMFN = NULL; }   /* R>6: caller falls back */
}

static int nosimd_opt = 0;

static void set_simd(void)
{
    use_simd = 0;
#ifdef HAVE_NEON
    if (!nosimd_opt && R >= 1 && R <= 6 && q >= 5) {
        int lanes = use8 ? 16 : 8;
        if (q <= lanes) use_simd = 1;
    }
#endif
}

/* Widen the counters from uint8 to uint16 in place.  Called between moves,
 * never during one, so every count is still exact when it happens. */
static void promote16(void)
{
    uint16_t *c16 = malloc(sizeof(uint16_t) * ((size_t)NTOT + 32));
    if (!c16) { fprintf(stderr, "counter promotion: out of memory\n"); exit(4); }
    for (long long w = 0; w < NTOT + 32; w++) c16[w] = cnt8[w];
    free(cnt8); cnt8 = NULL; cnt16 = c16; use8 = 0; satflag = 0; npromote++;
    set_simd(); select_kernels();
}

/* ------------------------------------------------------------------ */
/* generic (unspecialised) walk, used when R > 6                       */
/* ------------------------------------------------------------------ */

static long long gen_walk(const uint8_t *cb, int ncomb, long long base,
                          const OFFT *off, long long dv, int commit)
{
    long long dunc = 0;
    int k[32]; long long partial[33];
    for (int s = 0; s < ncomb; s++) {
        const uint8_t *pos = cb + (size_t)s * R;
        for (int i = 0; i < R; i++) k[i] = 0;
        partial[0] = base;
        for (int i = 0; i < R; i++) partial[i + 1] = partial[i] + off[(size_t)pos[i] * q];
        for (;;) {
            long long a = partial[R], b = a + dv;
            if (commit) {
                if (use8) { if (--cnt8[a] == 0) { dunc++; u_add(a); }
                            if (cnt8[b]++ == 0) { dunc--; u_del(b); }
                            else if (cnt8[b] >= sat_thresh) satflag = 1; }
                else      { if (--cnt16[a] == 0) { dunc++; u_add(a); }
                            if (cnt16[b]++ == 0) { dunc--; u_del(b); } }
            } else {
                if (use8) dunc += (cnt8[a] == 1) - (cnt8[b] == 0);
                else      dunc += (cnt16[a] == 1) - (cnt16[b] == 0);
            }
            int i = R - 1;
            while (i >= 0 && k[i] == q1 - 1) { k[i] = 0; i--; }
            if (i < 0) break;
            k[i]++;
            for (int j = i; j < R; j++)
                partial[j + 1] = partial[j] + off[(size_t)pos[j] * q + k[j]];
        }
    }
    return dunc;
}

/* ------------------------------------------------------------------ */
/* move interface                                                      */
/* ------------------------------------------------------------------ */

static inline long long eval_move(int i, int p, int v, long long thresh)
{
    const uint8_t *c = code + (size_t)i * n;
    long long dv = (long long)(v - c[p]) * pw[p];
    const OFFT *off = offtab + (size_t)i * n * q;
    const uint8_t *cb = combs + (size_t)p * NCOMB * CSTRIDE;
    int clast = c[n - 1];
    int mi = (clast < 64) ? clast : 0;
    const void *mk = use8 ? (const void *)maskv8[mi] : (const void *)maskv16[mi];
    if (EVFN) return EVFN(cb, NCOMB, cidx[i], off, dv, n - 1, clast, mk, thresh);
    return gen_walk(cb, NCOMB, cidx[i], off, dv, 0);
}

static void commit_move(int i, int p, int v)
{
    uint8_t *c = code + (size_t)i * n;
    long long dv = (long long)(v - c[p]) * pw[p];
    OFFT *off = offtab + (size_t)i * n * q;
    const uint8_t *cb = combs + (size_t)p * NCOMB * CSTRIDE;
    long long dunc = CMFN ? CMFN(cb, NCOMB, cidx[i], off, dv)
                          : gen_walk(cb, NCOMB, cidx[i], off, dv, 1);
    cidx[i] += dv;
    c[p] = (uint8_t)v;
    make_offs_row(c, off, p);
    uncovered += dunc;
}

/* ------------------------------------------------------------------ */
/* code I/O                                                            */
/* ------------------------------------------------------------------ */

static const char *DIG = "0123456789abcdefghijklmnopqrstuvwxyz";

static long long widx(const uint8_t *w);
static void word_of(long long idx, uint8_t *w);

/* Write the code, guaranteeing exactly `m` DISTINCT codewords.  The judge
 * counts distinct words and demands at least M of them, so a cover that
 * happened to contain a repeated codeword would score 0 instead of 1000.
 * Padding with arbitrary unused words is always safe: extra codewords can only
 * cover more.  Deduplication is by sorting the M word indices, not by a q^n
 * scratch array, so this is cheap enough to call repeatedly during a run. */
static int cmp_ll(const void *a, const void *b)
{
    long long x = *(const long long *)a, y = *(const long long *)b;
    return (x > y) - (x < y);
}

static void write_code(const char *path, const uint8_t *cw, int m)
{
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s.part", path);
    FILE *f = fopen(tmp, "w");
    if (!f) { perror(tmp); return; }

    long long *ix = malloc(sizeof(long long) * (size_t)m);
    long long *sorted = malloc(sizeof(long long) * (size_t)m);
    uint8_t *out = malloc((size_t)m * n);
    int k = 0;
    for (int i = 0; i < m; i++) ix[i] = widx(cw + (size_t)i * n);
    memcpy(sorted, ix, sizeof(long long) * (size_t)m);
    qsort(sorted, (size_t)m, sizeof(long long), cmp_ll);
    /* keep the first occurrence of each distinct word */
    for (int i = 0; i < m; i++) {
        int dup = 0;
        for (int j = 0; j < i; j++) if (ix[j] == ix[i]) { dup = 1; break; }
        if (dup) continue;
        memcpy(out + (size_t)k * n, cw + (size_t)i * n, n);
        k++;
        if (k == m) break;
    }
    if (k < m) {   /* pad with words not already present (sorted list is small) */
        int si = 0;
        for (long long w = 0; k < m && w < NTOT; w++) {
            while (si < m && sorted[si] < w) si++;
            if (si < m && sorted[si] == w) continue;
            word_of(w, out + (size_t)k * n);
            k++;
        }
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            if (q <= 36) fputc(DIG[out[(size_t)i * n + j]], f);
            else fprintf(f, "%d%s", out[(size_t)i * n + j], j + 1 < n ? " " : "");
        }
        fputc('\n', f);
    }
    fclose(f);
    rename(tmp, path);            /* atomic: a kill never leaves a torn file */
    free(ix); free(sorted); free(out);
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

static long long widx(const uint8_t *w)
{
    long long x = 0;
    for (int j = 0; j < n; j++) x += (long long)w[j] * pw[j];
    return x;
}
static void word_of(long long idx, uint8_t *w)
{
    for (int j = 0; j < n; j++) w[j] = (uint8_t)((idx / pw[j]) % q);
}

/* ------------------------------------------------------------------ */
/* rebuild / verification                                              */
/* ------------------------------------------------------------------ */

static void rebuild(void)
{
    if (use8) memset(cnt8, 0, (size_t)NTOT + 32);
    else      memset(cnt16, 0, sizeof(uint16_t) * ((size_t)NTOT + 32));
    uncovered = NTOT;
    for (int i = 0; i < M; i++) {
        cidx[i] = widx(code + (size_t)i * n);
        ball_apply(code + (size_t)i * n, cidx[i], +1);
        if (satflag) promote16();
    }
}

static long long count_uncovered_scan(void)
{
    long long u = 0;
    if (use8) { for (long long w = 0; w < NTOT; w++) u += (cnt8[w] == 0); }
    else      { for (long long w = 0; w < NTOT; w++) u += (cnt16[w] == 0); }
    return u;
}

/* ------------------------------------------------------------------ */
/* uncovered-word sampling                                             */
/* ------------------------------------------------------------------ */

static long long pick_uncovered(rng_t *r)
{
    if (uncovered <= 0) return -1;
    if (ulist_on) {
        if (ulen <= 0) return -1;
        return ulist[rmod(r, (uint64_t)ulen)];
    }
    for (int t = 0; t < 64; t++) {
        long long w = (long long)rmod(r, (uint64_t)NTOT);
        if (is_unc(w)) return w;
    }
    long long start = (long long)rmod(r, (uint64_t)NTOT);
    for (long long d = 0; d < NTOT; d++) {
        long long w = start + d; if (w >= NTOT) w -= NTOT;
        if (is_unc(w)) return w;
    }
    return -1;
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    double tlimit = 60.0;
    uint64_t seed = 12345;
    const char *inpath = NULL, *outpath = NULL;
    int cand = 100000, tabu_len = 0, kick = 2, noise = 0, init_greedy = 1;
    long long maxiter = -1, stall_limit = 5000;
    int verbose = 1, selftest = 0, force16 = 0, nolist = 0, noearly = 0, nosimd = 0;
    int init_aborted = 0;

    /* The clock starts HERE, not after initialisation: on K_8(9,4) building
     * the initial 940 balls is seconds of work and the arena contract is a
     * wall-clock budget for the whole process. */
    struct timespec t0, tn;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    T0 = t0;

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
        else if (!strcmp(a, "--selftest")) selftest = 1;
        else if (!strcmp(a, "--force16")) force16 = 1;
        else if (!strcmp(a, "--nolist")) nolist = 1;
        else if (!strcmp(a, "--noearly")) noearly = 1;
        else if (!strcmp(a, "--nosimd")) nosimd = 1;
        else if (!strcmp(a, "--modrng")) modrng = 1;
        else if (ARG("--satat")) sat_thresh = (unsigned)atoi(argv[++i]);
        else if (!strcmp(a, "--quiet")) verbose = 0;
        else { fprintf(stderr, "unknown/incomplete option %s\n", a); return 2; }
        #undef ARG
    }
    if (q < 2 || n < 1 || R < 0 || M < 1) {
        fprintf(stderr, "usage: covfast -q Q -n N -R R -M M [-t sec] [-s seed] [--out f]\n"
                        "   [--in f] [--cand K] [--tabu T] [--kick K] [--noise P] [--iters I]\n"
                        "   [--stall S] [--threads T] [--random-init] [--selftest] [--quiet]\n"
                        "   [--force16] [--nolist] [--noearly] [--nosimd]\n");
        return 2;
    }
#ifdef _OPENMP
    if (nthreads) omp_set_num_threads(nthreads);
    nthreads = omp_get_max_threads();
#endif

    if (n > 32 || R > 12 || R >= n + 1 || (long long)n * q > 2048) {
        fprintf(stderr, "unsupported shape (need n<=32, R<=12, n*q<=2048)\n"); return 2;
    }
    q1 = q - 1;
    NTOT = 1;
    for (int i = 0; i < n; i++) {
        if (NTOT > (long long)2e9 / q) { fprintf(stderr, "q^n too large\n"); return 2; }
        NTOT *= q;
    }
    for (int p = 0; p < n; p++) pw[p] = ipow(q, n - 1 - p);
    if (M > 65535) { fprintf(stderr, "M > 65535 not supported\n"); return 2; }

    /* Start on uint8 always; promote to uint16 if a counter ever gets close to
     * overflowing (which for M <= 255 is impossible). */
    use8 = !force16;

    if (use8) { cnt8 = calloc((size_t)NTOT + 32, 1); if (!cnt8) { fprintf(stderr, "oom\n"); return 2; } }
    else { cnt16 = calloc((size_t)NTOT + 32, 2); if (!cnt16) { fprintf(stderr, "oom\n"); return 2; } }

    code = malloc((size_t)M * n);
    bestcode = malloc((size_t)M * n);
    cidx = malloc(sizeof(long long) * M);
    offtab = malloc(sizeof(OFFT) * (size_t)M * n * q);
    build_combs();
    rseed(&grng, seed);

    /* NEON lane masks */
    for (int cl = 0; cl < q && cl < 64; cl++) {
        for (int v = 0; v < 8; v++) maskv16[cl][v] = (v < q && v != cl) ? 0xFFFFu : 0;
        for (int v = 0; v < 16; v++) maskv8[cl][v] = (v < q && v != cl) ? 0xFFu : 0;
    }
    nosimd_opt = nosimd;
    set_simd();
    select_kernels();

    /* uncovered-word list: 8 bytes/word, allocated lazily on first use */
    ulist_avail = (!nolist && NTOT < (1LL << 31));
    ulist_on = 0;

    long long S = binom(n - 1, R) * ipow(q1, R);
    long long V = 0;
    for (int i = 0; i <= R && i <= n; i++) V += binom(n, i) * ipow(q1, i);
    if (verbose) {
        printf("# covfast q=%d n=%d R=%d M=%d q^n=%lld ball=%lld sphere=%lld "
               "threads=%u cnt=%s simd=%d list=%d\n",
               q, n, R, M, NTOT, V, S, nthreads, use8 ? "u8" : "u16", use_simd, ulist_avail);
        fflush(stdout);
    }

    /* ---- initial code ---- */
    int have = 0;
    ulist_on = 0;                          /* list off while bulk-filling */
    if (inpath) have = read_code(inpath, code, M);
    if (use8) memset(cnt8, 0, (size_t)NTOT + 32);
    else      memset(cnt16, 0, sizeof(uint16_t) * ((size_t)NTOT + 32));
    uncovered = NTOT;
    for (int i = 0; i < have; i++) {
        cidx[i] = widx(code + (size_t)i * n);
        ball_apply(code + (size_t)i * n, cidx[i], +1);
        if (satflag) promote16();
    }
    if (init_greedy) {
        uint8_t *tw = malloc(n);
        par_init = (nthreads > 1);
        for (int i = have; i < M; i++) {
            /* On the biggest cells even the initial ball marking can outrun the
             * budget.  If that happens, place the remaining codewords on
             * uncovered words WITHOUT marking their balls (the bookkeeping is
             * about to be thrown away anyway) and get out with a complete,
             * valid code rather than being killed with nothing. */
            if (elapsed() > tlimit) {
                for (; i < M; i++) {
                    long long w2 = pick_uncovered(&grng);
                    if (w2 < 0) w2 = (long long)rmod(&grng, (uint64_t)NTOT);
                    word_of(w2, code + (size_t)i * n);
                    cidx[i] = w2;
                }
                init_aborted = 1;
                break;
            }
            long long w = pick_uncovered(&grng);
            if (w < 0) w = (long long)rmod(&grng, (uint64_t)NTOT);
            word_of(w, tw);
            memcpy(code + (size_t)i * n, tw, n);
            cidx[i] = w;
            ball_apply(code + (size_t)i * n, cidx[i], +1);
            if (satflag) promote16();
        }
        par_init = 0;
        free(tw);
    } else {
        for (int i = have; i < M; i++)
            for (int j = 0; j < n; j++) code[(size_t)i * n + j] = (uint8_t)rmod(&grng, q);
        rebuild();
    }
    u_track();
    for (int i = 0; i < M; i++) make_offs(code + (size_t)i * n, offtab + (size_t)i * n * q);

    memcpy(bestcode, code, (size_t)M * n);
    best_uncovered = uncovered;
    if (verbose) { printf("# init uncovered=%lld\n", uncovered); fflush(stdout); }
    /* Publish immediately: from here on a SIGKILL always finds a valid code on
     * disk, and the file is refreshed (atomically) whenever the incumbent
     * improves, at most once a second. */
    if (outpath) write_code(outpath, bestcode, M);
    double last_write = elapsed();
    if (init_aborted) {
        printf("RESULT q=%d n=%d R=%d M=%d uncovered=-1 iters=0 kicks=0 cands=0 "
               "promote=0 time=%.2f rate=0  (initialisation ran out of budget)\n",
               q, n, R, M, elapsed());
        return 1;
    }

    int *tabu = calloc((size_t)M * n * q, sizeof(int));

    uint8_t *u = malloc(n), *tmpw = malloc(n);
    /* dtar is R+1 when some codeword is that close and dmin (which can be as
     * large as n) otherwise, so the candidate count is bounded by M*n. */
    size_t candcap = (size_t)M * n + 8;
    int *cand_i = malloc(sizeof(int) * candcap);
    int *cand_p = malloc(sizeof(int) * candcap);
    int *dist = malloc(sizeof(int) * M);
    long long *cand_d = malloc(sizeof(long long) * candcap);

    long long it = 0, since_improve = 0, nkicks = 0, nevals = 0;
    double last_report = 0;

    while (uncovered > 0) {
        /* checked every iteration: a single sphere walk can take a second on
         * the big cells, so a coarser check would overshoot -t badly */
        clock_gettime(CLOCK_MONOTONIC, &tn);
        double el = (tn.tv_sec - t0.tv_sec) + 1e-9 * (tn.tv_nsec - t0.tv_nsec);
        if (el > tlimit) break;
        if (verbose && el - last_report > 10.0) {
            last_report = el;
            printf("# t=%6.1fs it=%-10lld unc=%-10lld best=%lld (%.0f it/s)\n",
                   el, it, uncovered, best_uncovered, it / el);
            fflush(stdout);
        }
        if (maxiter >= 0 && it >= maxiter) break;
        it++;
        u_track();

        /* Self-calibration of par_mode (see the comment on CAL_N).  The
         * PARALLEL block is probed first, because if one iteration turns out to
         * cost more than 5 ms the answer is already known -- an OpenMP
         * fork/join is at worst ~1 ms even on a contended box -- and the serial
         * probe, which is the expensive one on exactly those cells, is skipped.
         * A probe block ends after 0.05 s or CAL_N iterations, whichever comes
         * first, and the next re-measure is scheduled at 20x the probe cost. */
        if (nthreads > 1) {
            if (cal_phase < 2) {
                if (cal_n == 0) { cal_t0 = el; if (cal_phase == 0) cal_begin = el;
                                  par_mode = (cal_phase == 0); cal_n = 1; }
                else {
                    double dt = el - cal_t0;
                    if ((cal_n >= 3 && dt >= 0.05) || cal_n >= CAL_N) {
                        double per = dt / cal_n;
                        if (cal_phase == 0) {
                            cal_par = per;
                            if (per >= 0.005) { cal_serial = -1.0; cal_phase = 2; par_mode = 1; }
                            else { cal_phase = 1; cal_n = 0; }
                        } else { cal_serial = per; cal_phase = 2; par_mode = (cal_par < cal_serial); }
                        if (cal_phase == 2) {
                            double back = 20.0 * (el - cal_begin);
                            cal_until = el + (back > 10.0 ? back : 10.0);
                            if (verbose) {
                                if (cal_serial < 0)
                                    printf("# calib: parallel %.1f us/it, serial probe skipped"
                                           " (iteration >= 5ms) -> parallel"
                                           " (recheck at t=%.1fs)\n", cal_par * 1e6, cal_until);
                                else
                                    printf("# calib: parallel %.1f us/it  serial %.1f us/it"
                                           "  -> %s (recheck at t=%.1fs)\n",
                                           cal_par * 1e6, cal_serial * 1e6,
                                           par_mode ? "parallel" : "serial", cal_until);
                                fflush(stdout);
                            }
                        }
                    } else cal_n++;
                }
            } else if (el > cal_until) { cal_phase = 0; cal_n = 0; }
        }

        long long uw = pick_uncovered(&grng);
        if (uw < 0) break;
        word_of(uw, u);

        int dmin = n + 1;
        for (int i = 0; i < M; i++) {
            const uint8_t *c = code + (size_t)i * n;
            int d = 0;
            for (int j = 0; j < n; j++) d += (c[j] != u[j]);
            dist[i] = d;
            if (d < dmin) dmin = d;
        }
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
        if (ncand == 0) {
            int i = (int)rmod(&grng, M), p = (int)rmod(&grng, n);
            cand_i[0] = i; cand_p[0] = p; ncand = 1;
        }

        int take = ncand < cand ? ncand : cand;
        nevals += take;
        for (int s = 0; s < take; s++) {
            int j = s + (int)rmod(&grng, (uint64_t)(ncand - s));
            int ti = cand_i[s], tp = cand_p[s];
            cand_i[s] = cand_i[j]; cand_p[s] = cand_p[j];
            cand_i[j] = ti; cand_p[j] = tp;
        }

        int bi = -1, bp = -1, bv = -1;
        if (noise && (int)rmod(&grng, 1000) < noise) {
            int s = (int)rmod(&grng, (uint64_t)take);
            bi = cand_i[s]; bp = cand_p[s]; bv = u[bp];
        } else {
            long long bd_shared = LLONG_MAX;
            long long U = noearly ? LLONG_MAX / 4 : uncovered;
#pragma omp parallel for schedule(dynamic, 1) if (take > 3 && par_mode)
            for (int s = 0; s < take; s++) {
                long long b = __atomic_load_n(&bd_shared, __ATOMIC_RELAXED);
                long long th = (b == LLONG_MAX) ? LLONG_MAX : b + U;
                long long d = eval_move(cand_i[s], cand_p[s], u[cand_p[s]], th);
                cand_d[s] = d;
                long long cur = __atomic_load_n(&bd_shared, __ATOMIC_RELAXED);
                while (d < cur &&
                       !__atomic_compare_exchange_n(&bd_shared, &cur, d, 1,
                                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                    ;
            }
            long long bestd = 1LL << 60;
            int nties = 0;
            for (int s = 0; s < take; s++) {
                int i = cand_i[s], p = cand_p[s], v = u[p];
                long long d = cand_d[s];
                int is_tabu = tabu[((size_t)i * n + p) * q + v] > it;
                if (is_tabu && uncovered + d >= best_uncovered) continue;
                if (d < bestd) { bestd = d; bi = i; bp = p; bv = v; nties = 1; }
                else if (d == bestd) { nties++; if (rmod(&grng, (uint64_t)nties) == 0) { bi = i; bp = p; bv = v; } }
            }
            if (bi < 0) {
                int s = (int)rmod(&grng, (uint64_t)take);
                bi = cand_i[s]; bp = cand_p[s]; bv = u[bp];
            }
        }

        int oldv = code[(size_t)bi * n + bp];
        commit_move(bi, bp, bv);
        if (__builtin_expect(satflag, 0)) promote16();
        tabu[((size_t)bi * n + bp) * q + oldv] = (int)(it + tabu_len + rmod(&grng, 4));

        if (selftest) {
            long long real = count_uncovered_scan();
            if (real != uncovered) {
                fprintf(stderr, "SELFTEST FAIL at it=%lld: incremental=%lld real=%lld\n",
                        it, uncovered, real);
                return 3;
            }
            if (ulist_on && ulen != uncovered) {
                fprintf(stderr, "SELFTEST FAIL at it=%lld: ulen=%lld uncovered=%lld\n",
                        it, ulen, uncovered);
                return 3;
            }
        }

        if (uncovered < best_uncovered) {
            best_uncovered = uncovered;
            memcpy(bestcode, code, (size_t)M * n);
            since_improve = 0;
            if (verbose) {
                printf("# improved: unc=%lld at it=%lld t=%.1fs\n", best_uncovered, it, el);
                fflush(stdout);
            }
            if (outpath && (el - last_write > 1.0 || best_uncovered == 0)) {
                write_code(outpath, bestcode, M);
                last_write = elapsed();
            }
            if (best_uncovered == 0) break;
        } else if (++since_improve > stall_limit) {
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
                if (satflag) promote16();
                make_offs(code + (size_t)i * n, offtab + (size_t)i * n * q);
            }
            nkicks++;
            memset(tabu, 0, sizeof(int) * (size_t)M * n * q);
            since_improve = 0;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &tn);
    double el = (tn.tv_sec - t0.tv_sec) + 1e-9 * (tn.tv_nsec - t0.tv_nsec);
    { struct timespec cp; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cp);
      printf("CPU %.3f\n", cp.tv_sec + 1e-9*cp.tv_nsec); }
    printf("RESULT q=%d n=%d R=%d M=%d uncovered=%lld iters=%lld kicks=%lld "
           "cands=%lld promote=%lld time=%.2f rate=%.0f\n",
           q, n, R, M, best_uncovered, it, nkicks, nevals, npromote, el, it / (el > 0 ? el : 1));
    if (outpath) write_code(outpath, bestcode, M);
    return best_uncovered == 0 ? 0 : 1;
}
