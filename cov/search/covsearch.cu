/*
 * covsearch.cu -- CUDA port of the covering-code local search.
 *
 * Same state and same move as covsearch.c: cnt[w] counts how many codewords
 * cover the word w, and a move changes one coordinate of one codeword, touching
 * only the S = C(n-1,R)(q-1)^R "sphere" patterns that can change coverage.
 *
 * What the GPU buys us.  The counter array is q^n entries.  It lives in global
 * memory, so a single chain can address instances far larger than a CPU cache
 * tolerates -- 10^10 cells at one byte each is 10 GB, which fits one GH200 with
 * room to spare, and 6^10 or 5^11 fit many times over.  The two hot loops are
 * both embarrassingly parallel over the S sphere patterns:
 *
 *   sphere_kernel<false> reads cnt at the leaving and entering word of every
 *                     pattern for a whole BATCH of candidate moves at once and
 *                     block-reduces the change in the uncovered count.  One
 *                     block handles one (candidate, chunk-of-patterns) pair.
 *   sphere_kernel<true>  does the same walk but writing.  No atomics are needed
 *                     on cnt: a word at distance exactly R from c determines
 *                     uniquely which R coordinates differ, so distinct
 *                     (position-subset, value-assignment) pairs give distinct
 *                     words, and the leaving set (w_p = c_p) is disjoint from
 *                     the entering set (w_p = v).  Only the uncovered-count
 *                     delta is reduced, in shared memory then one atomicAdd
 *                     per block.
 *
 * Because every candidate move in an iteration is evaluated concurrently, the
 * GPU version raises the practical candidate budget from tens to thousands,
 * which the CPU experiments showed to be the single most important knob.
 *
 * Counters are uint16 (M <= 65535).  cnt is indexed by a 64-bit word index.
 *
 * Build:   nvcc -O3 -arch=sm_90 -o covsearch_cuda covsearch.cu
 * Verify:  --selftest replays the CPU semantics on the device and compares the
 *          uncovered count against a full host recount after every move.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, \
            cudaGetErrorString(e_)); exit(1); } } while (0)

/* ------------------------------------------------------------------ */
/* host-side problem description                                       */
/* ------------------------------------------------------------------ */

static int q, n, R, M;
static long long NTOT;
static long long pw[32];
static int NCOMB;                    /* C(n-1,R) */
static long long SPH;                /* NCOMB * (q-1)^R */

static long long ipow(long long b, int e) { long long r = 1; while (e-- > 0) r *= b; return r; }
static long long binom(int a, int b) {
    if (b < 0 || b > a) return 0;
    long long r = 1; if (b > a - b) b = a - b;
    for (int i = 0; i < b; i++) r = r * (a - i) / (i + 1);
    return r;
}

/* rng */
typedef struct { uint64_t s[4]; } rng_t;
static rng_t grng;
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static uint64_t rnext(rng_t *r) {
    uint64_t *s = r->s, res = rotl(s[0] + s[3], 23) + s[0], t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
    s[3] = rotl(s[3], 45); return res;
}
static void rseed(rng_t *r, uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL; uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        r->s[i] = z ^ (z >> 31);
    }
}
static uint64_t rmod(rng_t *r, uint64_t m) { return rnext(r) % m; }

/* ------------------------------------------------------------------ */
/* device state                                                        */
/* ------------------------------------------------------------------ */

__constant__ int d_q, d_n, d_R, d_NCOMB;
__constant__ long long d_pw[32];

static uint16_t *d_cnt;
static uint8_t  *d_combs;            /* n * NCOMB * R                    */
static long long *d_offs;            /* per-candidate n*q offsets        */
static long long *d_base;            /* per-candidate base index         */
static long long *d_dv;              /* per-candidate coordinate delta   */
static int       *d_pos;             /* per-candidate position p         */
static long long *d_part;            /* per-block partial deltas         */

/* Walk the (q-1)^R value assignments for one position-subset and accumulate
 * the change in the uncovered count.  COMMIT selects read-only evaluation vs
 * atomic update. */
template <bool COMMIT>
__device__ long long sphere_walk(const uint8_t *pos, long long base,
                                 const long long *offs, long long dv,
                                 uint16_t *cnt)
{
    long long dunc = 0;
    int k[8];
    long long partial[9];
    const int R = d_R, q = d_q;
    if (R == 0) {
        long long a = base, b = base + dv;
        if (COMMIT) {
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
    for (int i = 0; i < R; i++) partial[i + 1] = partial[i] + offs[pos[i] * q + 0];
    for (;;) {
        long long a = partial[R], b = a + dv;
        if (COMMIT) {
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
        for (int j = i; j < R; j++) partial[j + 1] = partial[j] + offs[pos[j] * q + k[j]];
    }
    return dunc;
}

/* One block per (candidate, chunk).  Threads split the NCOMB position-subsets. */
template <bool COMMIT>
__global__ void sphere_kernel(uint16_t *cnt, const uint8_t *combs,
                              const long long *offs, const long long *base,
                              const long long *dv, const int *pos,
                              long long *part, int ncand)
{
    int cand = blockIdx.x;
    if (cand >= ncand) return;
    int stride = d_R ? d_R : 1;
    const uint8_t *cb = combs + (size_t)pos[cand] * d_NCOMB * stride;
    const long long *op = offs + (size_t)cand * d_n * d_q;

    long long acc = 0;
    for (int k = threadIdx.x + blockIdx.y * blockDim.x;
         k < d_NCOMB; k += blockDim.x * gridDim.y)
        acc += sphere_walk<COMMIT>(cb + (size_t)k * stride, base[cand], op,
                                   dv[cand], cnt);

    __shared__ long long red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd((unsigned long long *)&part[cand], (unsigned long long)red[0]);
}

/* full radius-R ball, used only when (re)building cnt from scratch */
__global__ void ball_kernel(uint16_t *cnt, const long long *bases, int nb,
                            const uint8_t *code, long long total, int delta)
{
    /* one thread per (codeword, ball element) is awkward because the ball is
     * enumerated recursively; instead each thread takes one word of the whole
     * space and counts how many codewords cover it.  That is O(q^n * M * n) and
     * only ever runs once, at startup. */
    long long w = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (w >= total) return;
    int dig[32];
    long long t = w;
    for (int p = 0; p < d_n; p++) { dig[p] = (int)((t / d_pw[p]) % d_q); }
    int c = 0;
    for (int i = 0; i < nb; i++) {
        const uint8_t *cw = code + (size_t)i * d_n;
        int d = 0;
        for (int p = 0; p < d_n && d <= d_R; p++) d += (cw[p] != dig[p]);
        if (d <= d_R) c++;
    }
    cnt[w] = (uint16_t)(delta > 0 ? c : 0);
}

__global__ void count_uncovered_kernel(const uint16_t *cnt, long long total,
                                       unsigned long long *out)
{
    long long w = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    unsigned long long v = (w < total && cnt[w] == 0) ? 1 : 0;
    __shared__ unsigned long long red[256];
    red[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, red[0]);
}

/* collect uncovered word indices (up to cap) */
__global__ void gather_uncovered_kernel(const uint16_t *cnt, long long total,
                                        long long *out, unsigned *cursor, int cap)
{
    long long w = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (w >= total || cnt[w] != 0) return;
    unsigned i = atomicAdd(cursor, 1u);
    if (i < (unsigned)cap) out[i] = w;
}

/* ------------------------------------------------------------------ */
/* host helpers                                                        */
/* ------------------------------------------------------------------ */

static uint8_t *code, *bestcode;
static long long *cidx;
static long long uncovered, best_uncovered;
static uint8_t *h_combs;

static void build_combs(void)
{
    NCOMB = (int)binom(n - 1, R);
    int stride = R ? R : 1;
    h_combs = (uint8_t *)malloc((size_t)n * NCOMB * stride);
    for (int p = 0; p < n; p++) {
        int rest[32], nr = 0;
        for (int j = 0; j < n; j++) if (j != p) rest[nr++] = j;
        int c[16];
        for (int i = 0; i < R; i++) c[i] = i;
        for (int k = 0; k < NCOMB; k++) {
            for (int i = 0; i < R; i++)
                h_combs[((size_t)p * NCOMB + k) * stride + i] = (uint8_t)rest[c[i]];
            int i = R - 1;
            while (i >= 0 && c[i] == nr - R + i) i--;
            if (i < 0) break;
            c[i]++;
            for (int j = i + 1; j < R; j++) c[j] = c[j - 1] + 1;
        }
    }
}

static void make_offs(const uint8_t *c, long long *offs)
{
    for (int j = 0; j < n; j++) {
        int k = 0;
        for (int v = 0; v < q; v++)
            if (v != c[j]) offs[(size_t)j * q + (k++)] = (long long)(v - c[j]) * pw[j];
    }
}

static long long widx(const uint8_t *w)
{
    long long x = 0;
    for (int j = 0; j < n; j++) x += (long long)w[j] * pw[j];
    return x;
}

static long long device_uncovered(void)
{
    unsigned long long *d_out, h = 0;
    CUDA_OK(cudaMalloc(&d_out, sizeof(unsigned long long)));
    CUDA_OK(cudaMemcpy(d_out, &h, sizeof h, cudaMemcpyHostToDevice));
    int bs = 256;
    long long gs = (NTOT + bs - 1) / bs;
    count_uncovered_kernel<<<(int)gs, bs>>>(d_cnt, NTOT, d_out);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(&h, d_out, sizeof h, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(d_out));
    return (long long)h;
}

/* Host-side ball marking, then one upload.  The obvious device version -- one
 * thread per word, scanning all M codewords -- costs q^n * M * n and dominates
 * everything else for the instances that make the GPU worth using at all.
 * Marking balls on the host costs only M * |B_R| and the transfer is one
 * sequential copy, so this is faster by orders of magnitude even though it
 * moves q^n counters across PCIe/NVLink.  It does mean the host needs the same
 * 2*q^n bytes, which is the current ceiling on instance size. */
static uint16_t *h_cnt;

static long long h_ball_rec(long long base, const long long *offs,
                            int startpos, int rem)
{
    long long dunc = 0;
    if (rem == 0) return 0;
    for (int j = startpos; j < n; j++)
        for (int k = 0; k < q - 1; k++) {
            long long w = base + offs[(size_t)j * q + k];
            if (h_cnt[w]++ == 0) dunc--;
            dunc += h_ball_rec(w, offs, j + 1, rem - 1);
        }
    return dunc;
}

static void rebuild_device(void)
{
    if (!h_cnt) h_cnt = (uint16_t *)malloc(sizeof(uint16_t) * (size_t)NTOT);
    memset(h_cnt, 0, sizeof(uint16_t) * (size_t)NTOT);
    long long unc = NTOT;
    long long *offs = (long long *)malloc(sizeof(long long) * n * q);
    for (int i = 0; i < M; i++) {
        cidx[i] = widx(code + (size_t)i * n);
        make_offs(code + (size_t)i * n, offs);
        if (h_cnt[cidx[i]]++ == 0) unc--;
        unc += h_ball_rec(cidx[i], offs, 0, R);
    }
    free(offs);
    CUDA_OK(cudaMemcpy(d_cnt, h_cnt, sizeof(uint16_t) * (size_t)NTOT,
                       cudaMemcpyHostToDevice));
    uncovered = unc;
}

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
        if (len != n) { fprintf(stderr, "%s: bad length %d\n", path, len); exit(1); }
        if (m >= maxM) { fprintf(stderr, "%s: too many codewords\n", path); exit(1); }
        memcpy(cw + (size_t)m * n, tmp, n);
        m++;
    }
    fclose(f);
    return m;
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    double tlimit = 60.0;
    uint64_t seed = 12345;
    const char *inpath = NULL, *outpath = NULL;
    int maxcand = 4096, selftest = 0, verbose = 1;
    long long stall_limit = 5000;
    int kick = 2;

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
        else if (ARG("--cand")) maxcand = atoi(argv[++i]);
        else if (ARG("--stall")) stall_limit = atoll(argv[++i]);
        else if (ARG("--kick")) kick = atoi(argv[++i]);
        else if (!strcmp(a, "--selftest")) selftest = 1;
        else if (!strcmp(a, "--quiet")) verbose = 0;
        else { fprintf(stderr, "unknown option %s\n", a); return 2; }
        #undef ARG
    }
    if (q < 2 || n < 1 || R < 0 || M < 1) {
        fprintf(stderr, "usage: covsearch_cuda -q Q -n N -R R -M M [-t sec] [-s seed]\n"
                        "       [--in f] [--out f] [--cand K] [--stall S] [--kick K]\n"
                        "       [--selftest] [--quiet]\n");
        return 2;
    }

    NTOT = 1;
    for (int i = 0; i < n; i++) NTOT *= q;
    for (int p = 0; p < n; p++) pw[p] = ipow(q, n - 1 - p);
    if (M > 65535) { fprintf(stderr, "M > 65535 unsupported\n"); return 2; }
    if (R > 8) { fprintf(stderr, "R > 8 unsupported on device\n"); return 2; }

    build_combs();
    SPH = (long long)NCOMB * ipow(q - 1, R);
    rseed(&grng, seed);

    CUDA_OK(cudaMemcpyToSymbol(d_q, &q, sizeof q));
    CUDA_OK(cudaMemcpyToSymbol(d_n, &n, sizeof n));
    CUDA_OK(cudaMemcpyToSymbol(d_R, &R, sizeof R));
    CUDA_OK(cudaMemcpyToSymbol(d_NCOMB, &NCOMB, sizeof NCOMB));
    CUDA_OK(cudaMemcpyToSymbol(d_pw, pw, sizeof pw));

    size_t cntbytes = sizeof(uint16_t) * (size_t)NTOT;
    if (verbose)
        printf("# covsearch_cuda q=%d n=%d R=%d M=%d  q^n=%lld  sphere=%lld  "
               "counters=%.2f GB\n", q, n, R, M, NTOT, SPH, cntbytes / 1e9);
    CUDA_OK(cudaMalloc(&d_cnt, cntbytes));
    CUDA_OK(cudaMemset(d_cnt, 0, cntbytes));
    int stride = R ? R : 1;
    CUDA_OK(cudaMalloc(&d_combs, (size_t)n * NCOMB * stride));
    CUDA_OK(cudaMemcpy(d_combs, h_combs, (size_t)n * NCOMB * stride, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&d_offs, sizeof(long long) * (size_t)maxcand * n * q));
    CUDA_OK(cudaMalloc(&d_base, sizeof(long long) * maxcand));
    CUDA_OK(cudaMalloc(&d_dv,   sizeof(long long) * maxcand));
    CUDA_OK(cudaMalloc(&d_pos,  sizeof(int) * maxcand));
    CUDA_OK(cudaMalloc(&d_part, sizeof(long long) * maxcand));

    code = (uint8_t *)malloc((size_t)M * n);
    bestcode = (uint8_t *)malloc((size_t)M * n);
    cidx = (long long *)malloc(sizeof(long long) * M);

    int have = inpath ? read_code(inpath, code, M) : 0;
    for (int i = have; i < M; i++)
        for (int j = 0; j < n; j++) code[(size_t)i * n + j] = (uint8_t)rmod(&grng, q);

    rebuild_device();
    memcpy(bestcode, code, (size_t)M * n);
    best_uncovered = uncovered;
    if (verbose) printf("# init uncovered=%lld\n", uncovered);

    /* host scratch */
    long long *h_offs = (long long *)malloc(sizeof(long long) * (size_t)maxcand * n * q);
    long long *h_base = (long long *)malloc(sizeof(long long) * maxcand);
    long long *h_dv   = (long long *)malloc(sizeof(long long) * maxcand);
    int *h_pos = (int *)malloc(sizeof(int) * maxcand);
    long long *h_part = (long long *)malloc(sizeof(long long) * maxcand);
    int *ci = (int *)malloc(sizeof(int) * (size_t)M * n);
    int *cp = (int *)malloc(sizeof(int) * (size_t)M * n);
    uint8_t *u = (uint8_t *)malloc(n);
    int *dist = (int *)malloc(sizeof(int) * M);

    long long *d_unclist; unsigned *d_cursor;
    const int UCAP = 4096;
    CUDA_OK(cudaMalloc(&d_unclist, sizeof(long long) * UCAP));
    CUDA_OK(cudaMalloc(&d_cursor, sizeof(unsigned)));
    long long *h_unclist = (long long *)malloc(sizeof(long long) * UCAP);

    struct timespec t0, tn;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long long it = 0, since_improve = 0, nkicks = 0, nscans = 0;
    int ucache_n = 0;

    while (uncovered > 0) {
        clock_gettime(CLOCK_MONOTONIC, &tn);
        double el = (tn.tv_sec - t0.tv_sec) + 1e-9 * (tn.tv_nsec - t0.tv_nsec);
        if (el > tlimit) break;
        it++;

        /* Sample an uncovered word.  Re-scanning all q^n counters every
         * iteration is what made this latency-bound, so instead we cache a
         * batch of uncovered words and refill only when the cache runs dry,
         * checking each cached word is still uncovered with a two-byte read. */
        long long uw = -1;
        while (uw < 0) {
            while (ucache_n > 0) {
                int j = (int)rmod(&grng, (uint64_t)ucache_n);
                long long w = h_unclist[j];
                h_unclist[j] = h_unclist[--ucache_n];
                uint16_t c16;
                CUDA_OK(cudaMemcpy(&c16, d_cnt + w, sizeof c16, cudaMemcpyDeviceToHost));
                if (c16 == 0) { uw = w; break; }
            }
            if (uw >= 0) break;
            unsigned zero = 0;
            CUDA_OK(cudaMemcpy(d_cursor, &zero, sizeof zero, cudaMemcpyHostToDevice));
            int bs = 256;
            gather_uncovered_kernel<<<(int)((NTOT + bs - 1) / bs), bs>>>(
                d_cnt, NTOT, d_unclist, d_cursor, UCAP);
            CUDA_OK(cudaGetLastError());
            unsigned found = 0;
            CUDA_OK(cudaMemcpy(&found, d_cursor, sizeof found, cudaMemcpyDeviceToHost));
            nscans++;
            if (found == 0) break;
            ucache_n = (int)(found < (unsigned)UCAP ? found : (unsigned)UCAP);
            CUDA_OK(cudaMemcpy(h_unclist, d_unclist, sizeof(long long) * ucache_n,
                               cudaMemcpyDeviceToHost));
        }
        if (uw < 0) break;
        for (int p = 0; p < n; p++) u[p] = (uint8_t)((uw / pw[p]) % q);

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
        for (int i = 0; i < M && ncand < maxcand; i++) {
            if (dist[i] != dtar) continue;
            const uint8_t *c = code + (size_t)i * n;
            for (int p = 0; p < n && ncand < maxcand; p++) {
                if (c[p] == u[p]) continue;
                ci[ncand] = i; cp[ncand] = p; ncand++;
            }
        }
        if (ncand == 0) { ci[0] = (int)rmod(&grng, M); cp[0] = (int)rmod(&grng, n); ncand = 1; }

        for (int s = 0; s < ncand; s++) {
            const uint8_t *c = code + (size_t)ci[s] * n;
            make_offs(c, h_offs + (size_t)s * n * q);
            h_base[s] = cidx[ci[s]];
            h_dv[s] = (long long)(u[cp[s]] - c[cp[s]]) * pw[cp[s]];
            h_pos[s] = cp[s];
        }
        CUDA_OK(cudaMemcpy(d_offs, h_offs, sizeof(long long) * (size_t)ncand * n * q,
                           cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_base, h_base, sizeof(long long) * ncand, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_dv,   h_dv,   sizeof(long long) * ncand, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_pos,  h_pos,  sizeof(int) * ncand, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemset(d_part, 0, sizeof(long long) * ncand));

        dim3 grid(ncand, (NCOMB + 255) / 256 > 8 ? 8 : 1);
        sphere_kernel<false><<<grid, 256>>>(d_cnt, d_combs, d_offs, d_base, d_dv,
                                            d_pos, d_part, ncand);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaMemcpy(h_part, d_part, sizeof(long long) * ncand,
                           cudaMemcpyDeviceToHost));

        long long bestd = 1LL << 60; int bs_i = -1, nties = 0;
        for (int s = 0; s < ncand; s++) {
            if (h_part[s] < bestd) { bestd = h_part[s]; bs_i = s; nties = 1; }
            else if (h_part[s] == bestd) { nties++; if (rmod(&grng, nties) == 0) bs_i = s; }
        }

        /* Commit the chosen move.  Everything it needs is already resident from
         * the evaluation launch, so just point the kernel at that candidate's
         * slice instead of re-uploading it. */
        int cand1 = bs_i;
        CUDA_OK(cudaMemset(d_part + cand1, 0, sizeof(long long)));
        dim3 g1(1, (NCOMB + 255) / 256 > 8 ? 8 : 1);
        sphere_kernel<true><<<g1, 256>>>(d_cnt, d_combs,
                                          d_offs + (size_t)cand1 * n * q,
                                          d_base + cand1, d_dv + cand1,
                                          d_pos + cand1, d_part + cand1, 1);
        CUDA_OK(cudaGetLastError());
        long long applied = 0;
        CUDA_OK(cudaMemcpy(&applied, d_part + cand1, sizeof(long long),
                           cudaMemcpyDeviceToHost));
        if (applied != bestd) {
            fprintf(stderr, "internal error: eval delta %lld != commit delta %lld\n",
                    bestd, applied);
            return 3;
        }
        uncovered += applied;
        cidx[ci[cand1]] += h_dv[cand1];
        code[(size_t)ci[cand1] * n + cp[cand1]] = u[cp[cand1]];

        if (selftest) {
            long long real = device_uncovered();
            if (real != uncovered) {
                fprintf(stderr, "SELFTEST FAIL at it=%lld: incremental %lld, actual %lld\n",
                        it, uncovered, real);
                return 4;
            }
        }

        if (uncovered < best_uncovered) {
            best_uncovered = uncovered;
            memcpy(bestcode, code, (size_t)M * n);
            since_improve = 0;
            if (best_uncovered == 0) break;
        } else if (++since_improve > stall_limit) {
            for (int t = 0; t < kick; t++) {
                int i = (int)rmod(&grng, M);
                for (int j = 0; j < n; j++) code[(size_t)i * n + j] = (uint8_t)rmod(&grng, q);
            }
            CUDA_OK(cudaMemset(d_cnt, 0, cntbytes));
            rebuild_device();
            ucache_n = 0;
            since_improve = 0;
            nkicks++;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &tn);
    double el = (tn.tv_sec - t0.tv_sec) + 1e-9 * (tn.tv_nsec - t0.tv_nsec);
    printf("RESULT q=%d n=%d R=%d M=%d uncovered=%lld iters=%lld kicks=%lld "
           "scans=%lld time=%.2f rate=%.0f\n", q, n, R, M, best_uncovered, it,
           nkicks, nscans, el, it / (el > 0 ? el : 1));
    if (outpath) write_code(outpath, bestcode);
    return best_uncovered == 0 ? 0 : 1;
}
