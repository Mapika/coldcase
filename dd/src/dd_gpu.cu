// dd_gpu.cu -- CUDA engine for the degree/diameter problem.
//
// Same construction space and objective as dd_search.cpp (Cayley graphs of
// metacyclic groups Z_m rtimes_a Z_n, minimise f(S) = N - |B_D(S)|), but the whole
// iterated local search runs on the device: one CUDA block owns one search chain,
// keeps its connection set and its ball bitset in shared memory, and never talks to
// the host except to report a hit.
//
// Parallel decomposition of the ball step.  A subset of G is R = n rows of M = m
// bits, W = ceil(M/64) words per row.  Right multiplication by s = (i,j) maps row r
// to row (r+j) mod n rotated by a^r*i mod m, so, indexing by DESTINATION row,
//
//     nxt[r'] = cur[r']  |  OR over s in S of  rot( cur[src_s(r')], sh_s(r') )
//     src_s(r') = (r' - j + n) mod n,     sh_s(r') = a^{src} * i mod m
//
// Every destination word (r', w) is therefore an independent reduction over |S|
// generators of at most four source words -- perfectly parallel, no atomics, and
// only one __syncthreads() per BFS level.  Threads are laid out over the flat
// R*W destination words.
//
// Correctness of the kernel is checked against the CPU engine (`dd_search --eval`)
// by crosscheck_gpu.py, and every emitted graph still goes through emit_graph.py +
// verify_dd.py like any other result.
//
// Build: nvcc -O3 -arch=sm_90 -o dd_gpu dd_gpu.cu

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>

typedef unsigned long long u64;

#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(1);} } while (0)

#define MAXDEG 32

// ---------------------------------------------------------------- host helpers
static long long gcdll(long long a, long long b) { while (b) { long long t = a % b; a = b; b = t; } return a; }
static int powmod_h(int b, long long e, int m) {
    if (m == 1) return 0;
    long long r = 1, bb = b % m;
    while (e > 0) { if (e & 1) r = r * bb % m; bb = bb * bb % m; e >>= 1; }
    return (int)r;
}
static bool is_prime_h(int x) { if (x < 2) return false; for (int d = 2; (long long)d * d <= x; d++) if (x % d == 0) return false; return true; }

struct Spec { int m, n, a; };

static std::vector<Spec> enum_specs(int N, int max_a, std::mt19937_64 &rng,
                                    int min_n, int max_n, bool faithful) {
    std::vector<Spec> out;
    for (int n = 1; n <= N; n++) {
        if (N % n) continue;
        if (n < min_n || n > max_n) continue;
        int m = N / n;
        if (m < 2) continue;
        std::vector<int> c;
        for (int a = 2; a < m; a++) {
            if (gcdll(a, m) != 1) continue;
            if (powmod_h(a, n, m) != 1 % m) continue;
            if (faithful) {
                bool full = true;
                for (int q = 2; q <= n; q++) {
                    if (n % q || !is_prime_h(q)) continue;
                    if (powmod_h(a, n / q, m) == 1 % m) { full = false; break; }
                }
                if (!full) continue;
            }
            c.push_back(a);
        }
        if (c.empty()) continue;
        if ((int)c.size() > max_a) { std::shuffle(c.begin(), c.end(), rng); c.resize(max_a); }
        for (int a : c) out.push_back({m, n, a});
    }
    return out;
}

// ------------------------------------------------------------------ device RNG
__device__ __forceinline__ u64 rnd(u64 &s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
}

// group parameters, constant for one kernel launch
struct GP { int m, n, a, N, R, W; u64 topmask; };

// ---------------------------------------------------------- the ball evaluation
// Returns |B_D| (block-wide, valid in every thread). Uses shared cur/nxt/apow.
__device__ int ball_size(const GP g, const int *__restrict gi, const int *__restrict gj,
                         int deg, int D, u64 *cur, u64 *nxt, const int *apow,
                         int *sh_count, int *shf) {
    const int RW = g.R * g.W;
    for (int t = threadIdx.x; t < RW; t += blockDim.x) cur[t] = 0ULL;
    // shf[e*R + r] = rotation applied to the source row feeding destination row r
    // under generator e.  Hoisting it out of the word loop removes one modmul per
    // (word x generator); with W words per row that is a W-fold saving.
    if (shf) {
        const int tot = deg * g.R;
        for (int idx = threadIdx.x; idx < tot; idx += blockDim.x) {
            const int e = idx / g.R, r = idx - e * g.R;
            int src = r - gj[e]; if (src < 0) src += g.n;
            shf[idx] = (int)(((long long)apow[src] * gi[e]) % g.m);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) cur[0] = 1ULL;
    __syncthreads();

    int cnt = 1;
    for (int lvl = 1; lvl <= D; lvl++) {
        for (int t = threadIdx.x; t < RW; t += blockDim.x) {
            const int r = t / g.W;             // destination row
            const int w = t - r * g.W;         // word inside the row
            u64 acc = cur[t];                  // B_{k-1} subset of B_k
            for (int e = 0; e < deg; e++) {
                int src = r - gj[e]; if (src < 0) src += g.n;
                const int sft = shf ? shf[e * g.R + r]
                                    : (int)(((long long)apow[src] * gi[e]) % g.m);
                const u64 *row = cur + (size_t)src * g.W;
                u64 v = 0;
                if (sft == 0) {
                    v = row[w];
                } else {
                    const int q = sft >> 6, b = sft & 63;
                    const int s2 = g.m - sft, q2 = s2 >> 6, b2 = s2 & 63;
                    int k = w - q;
                    if (k >= 0) {
                        v = row[k] << b;
                        if (b && k - 1 >= 0) v |= row[k - 1] >> (64 - b);
                    }
                    int k2 = w + q2;
                    if (k2 < g.W) {
                        u64 z = row[k2] >> b2;
                        if (b2 && k2 + 1 < g.W) z |= row[k2 + 1] << (64 - b2);
                        v |= z;
                    }
                    if (w == g.W - 1) v &= g.topmask;
                }
                acc |= v;
            }
            nxt[t] = acc;
        }
        __syncthreads();
        // count and swap
        if (threadIdx.x == 0) *sh_count = 0;
        __syncthreads();
        int local = 0;
        for (int t = threadIdx.x; t < RW; t += blockDim.x) { cur[t] = nxt[t]; local += __popcll(nxt[t]); }
        atomicAdd(sh_count, local);
        __syncthreads();
        cnt = *sh_count;
        __syncthreads();
        if (cnt == g.N) return g.N;
    }
    return cnt;
}

// -------------------------------------------------------------------- ILS kernel
// One block = one independent search chain on one group spec.
__global__ void search_kernel(GP g, int deg, int D, int use_shf, int iters, int stall_lim, int kick_lim,
                              const int *__restrict invol, int n_invol,
                              u64 seed, int *found_flag, int *out_S, int *best_f_out,
                              unsigned long long *eval_ctr) {
    extern __shared__ u64 smem[];
    const int RW = g.R * g.W;
    u64 *cur = smem;
    u64 *nxt = smem + RW;
    int *apow = (int *)(smem + 2 * RW);
    int *shf  = use_shf ? apow + g.n : nullptr;
    int *sh   = apow + g.n + (use_shf ? deg * g.R : 0);   // scratch: [0]=count, [1..]=misc
    __shared__ int gi[MAXDEG], gj[MAXDEG];      // generator (i,j) components
    __shared__ int bgi[MAXDEG], bgj[MAXDEG];    // incumbent
    __shared__ int accept_flag;

    // a^r table
    if (threadIdx.x == 0) {
        apow[0] = 1 % g.m;
        for (int r = 1; r < g.n; r++) apow[r] = (int)(((long long)apow[r - 1] * g.a) % g.m);
    }
    __syncthreads();

    u64 rs = seed * 6364136223846793005ULL + (u64)(blockIdx.x + 1) * 1442695040888963407ULL;
    rnd(rs); rnd(rs);

    // ---- random inverse-closed start: floor(deg/2) pairs, plus one involution
    //      in the last slot when deg is odd
    const int npair = deg & ~1;                 // number of slots used by pairs
    if (threadIdx.x == 0) {
        if (deg & 1) {
            int e = invol[rnd(rs) % (u64)n_invol];
            gi[deg - 1] = e % g.m; gj[deg - 1] = e / g.m;
        }
        for (int p = 0; p < npair; p += 2) {
            int e, i, j;
            while (true) {
                e = (int)(rnd(rs) % (u64)g.N);
                if (e == 0) continue;
                i = e % g.m; j = e / g.m;
                // inverse of (i,j) is (-a^{-j} i, -j)
                int jn = (g.n - j) % g.n;
                int ii = (int)(((long long)((g.m - i % g.m) % g.m) * apow[jn]) % g.m);
                if (ii == i && jn == j) continue;               // involution: skip (needs odd slot)
                gi[p] = i; gj[p] = j; gi[p + 1] = ii; gj[p + 1] = jn;
                break;
            }
        }
        for (int e = 0; e < deg; e++) { bgi[e] = gi[e]; bgj[e] = gj[e]; }
    }
    __syncthreads();

    int f = g.N - ball_size(g, gi, gj, deg, D, cur, nxt, apow, sh, shf);
    int bf = f, stall = 0, kicks = 0, nev = 1;

    for (int it = 0; it < iters && f > 0; it++) {
        // propose: replace one pair
        if (threadIdx.x == 0) {
            // with probability 1/4 (when available) move the involution instead
            bool use_inv = (deg & 1) && n_invol > 1 && ((rnd(rs) & 3) == 0);
            if (use_inv) {
                sh[1] = -1; sh[2] = gi[deg - 1]; sh[3] = gj[deg - 1];
                for (int tr = 0; tr < 64; tr++) {
                    int e = invol[rnd(rs) % (u64)n_invol];
                    int i = e % g.m, j = e / g.m;
                    if (i == sh[2] && j == sh[3]) continue;
                    gi[deg - 1] = i; gj[deg - 1] = j; break;
                }
            } else {
                int p = (int)(rnd(rs) % (u64)(npair / 2)) * 2;
                sh[1] = p; sh[2] = gi[p]; sh[3] = gj[p]; sh[4] = gi[p + 1]; sh[5] = gj[p + 1];
                int e, i, j;
                while (true) {
                    e = (int)(rnd(rs) % (u64)g.N);
                    if (e == 0) continue;
                    i = e % g.m; j = e / g.m;
                    int jn = (g.n - j) % g.n;
                    int ii = (int)(((long long)((g.m - i % g.m) % g.m) * apow[jn]) % g.m);
                    if (ii == i && jn == j) continue;
                    bool dup = false;
                    for (int q = 0; q < deg; q++) if (gi[q] == i && gj[q] == j) dup = true;
                    if (dup) continue;
                    gi[p] = i; gj[p] = j; gi[p + 1] = ii; gj[p + 1] = jn;
                    break;
                }
            }
        }
        __syncthreads();

        int fn = g.N - ball_size(g, gi, gj, deg, D, cur, nxt, apow, sh, shf); nev++;
        if (threadIdx.x == 0) {
            int p = sh[1];
            bool acc = (fn < f) || (fn == f && (rnd(rs) & 1));
            if (acc) { accept_flag = 1; }
            else if (p < 0) { accept_flag = 0; gi[deg - 1] = sh[2]; gj[deg - 1] = sh[3]; }
            else { accept_flag = 0; gi[p] = sh[2]; gj[p] = sh[3]; gi[p + 1] = sh[4]; gj[p + 1] = sh[5]; }
        }
        __syncthreads();
        if (accept_flag) f = fn;

        if (f < bf) {
            bf = f; stall = 0;
            if (threadIdx.x == 0) for (int e = 0; e < deg; e++) { bgi[e] = gi[e]; bgj[e] = gj[e]; }
            __syncthreads();
        } else if (++stall > stall_lim) {
            if (++kicks > kick_lim) break;
            if (threadIdx.x == 0) {
                for (int e = 0; e < deg; e++) { gi[e] = bgi[e]; gj[e] = bgj[e]; }
                int nk = 1 + (int)(rnd(rs) % 3);
                if ((deg & 1) && n_invol > 1 && (rnd(rs) & 1)) {
                    int e = invol[rnd(rs) % (u64)n_invol];
                    gi[deg - 1] = e % g.m; gj[deg - 1] = e / g.m;
                }
                for (int z = 0; z < nk; z++) {
                    int p = (int)(rnd(rs) % (u64)(npair / 2)) * 2;
                    int e, i, j;
                    while (true) {
                        e = (int)(rnd(rs) % (u64)g.N);
                        if (e == 0) continue;
                        i = e % g.m; j = e / g.m;
                        int jn = (g.n - j) % g.n;
                        int ii = (int)(((long long)((g.m - i % g.m) % g.m) * apow[jn]) % g.m);
                        if (ii == i && jn == j) continue;
                        gi[p] = i; gj[p] = j; gi[p + 1] = ii; gj[p + 1] = jn;
                        break;
                    }
                }
            }
            __syncthreads();
            f = g.N - ball_size(g, gi, gj, deg, D, cur, nxt, apow, sh, shf); nev++;
            stall = 0;
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(eval_ctr, (unsigned long long)nev);
        atomicMin(best_f_out, bf);
        if (bf == 0 && atomicCAS(found_flag, 0, 1) == 0)
            for (int e = 0; e < deg; e++) out_S[e] = bgj[e] * g.m + bgi[e];
    }
}

// -------------------------------------------------------------- evaluation mode
__global__ void eval_kernel(GP g, int deg, int D, int use_shf, const int *S, int *out) {
    extern __shared__ u64 smem[];
    const int RW = g.R * g.W;
    u64 *cur = smem, *nxt = smem + RW;
    int *apow = (int *)(smem + 2 * RW);
    int *shf  = use_shf ? apow + g.n : nullptr;
    int *sh   = apow + g.n + (use_shf ? deg * g.R : 0);
    __shared__ int gi[MAXDEG], gj[MAXDEG];
    if (threadIdx.x == 0) {
        apow[0] = 1 % g.m;
        for (int r = 1; r < g.n; r++) apow[r] = (int)(((long long)apow[r - 1] * g.a) % g.m);
        for (int e = 0; e < deg; e++) { gi[e] = S[e] % g.m; gj[e] = S[e] / g.m; }
    }
    __syncthreads();
    for (int d = 1; d <= D; d++) {
        int b = ball_size(g, gi, gj, deg, d, cur, nxt, apow, sh, shf);
        if (threadIdx.x == 0) out[d - 1] = b;
        __syncthreads();
    }
}

// --------------------------------------------------------------------- driver
int main(int argc, char **argv) {
    int delta = 0, D = 0, Nmin = 0, Nmax = 0, iters = 200000, max_a = 40;
    int min_n = 3, max_n = 1 << 30, blocks = 4096, threads = 128;
    int stall_lim = 300, kick_lim = 60;
    double seconds = 60;
    u64 seed = 12345;
    std::string out_path, evalspec;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto nx = [&]() { return atoi(argv[++i]); };
        if (k == "--delta") delta = nx();
        else if (k == "--diam" || k == "--D") D = nx();
        else if (k == "--Nmin") Nmin = nx();
        else if (k == "--Nmax") Nmax = nx();
        else if (k == "--time") seconds = atof(argv[++i]);
        else if (k == "--seed") seed = strtoull(argv[++i], 0, 10);
        else if (k == "--iters") iters = nx();
        else if (k == "--maxa") max_a = nx();
        else if (k == "--minn") min_n = nx();
        else if (k == "--maxn") max_n = nx();
        else if (k == "--blocks") blocks = nx();
        else if (k == "--threads") threads = nx();
        else if (k == "--stall") stall_lim = nx();
        else if (k == "--kicks") kick_lim = nx();
        else if (k == "--out") out_path = argv[++i];
        else if (k == "--eval") evalspec = argv[++i];
        else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 1; }
    }

    if (!evalspec.empty()) {                 // "m,n,a:s1,s2,..."
        char *p = (char *)evalspec.c_str();
        int m = strtol(p, &p, 10); p++;
        int n = strtol(p, &p, 10); p++;
        int a = strtol(p, &p, 10);
        std::vector<int> S;
        while (*p) { if (*p == ':' || *p == ',') p++; else S.push_back(strtol(p, &p, 10)); }
        GP g; g.m = m; g.n = n; g.a = a; g.N = m * n; g.R = n; g.W = (m + 63) / 64;
        int rem = m - 64 * (g.W - 1);
        g.topmask = (rem == 64) ? ~0ULL : ((1ULL << rem) - 1ULL);
        int DD = D ? D : 8;
        int *dS, *dout;
        CUDA_OK(cudaMalloc(&dS, S.size() * sizeof(int)));
        CUDA_OK(cudaMalloc(&dout, DD * sizeof(int)));
        CUDA_OK(cudaMemcpy(dS, S.data(), S.size() * sizeof(int), cudaMemcpyHostToDevice));
        int use_shf = ((int)S.size() * g.R <= 6144);
        size_t sm = 2ULL * g.R * g.W * sizeof(u64)
                  + (g.n + 8 + (use_shf ? (int)S.size() * g.R : 0)) * sizeof(int);
        CUDA_OK(cudaFuncSetAttribute(eval_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
        eval_kernel<<<1, threads, sm>>>(g, (int)S.size(), DD, use_shf, dS, dout);
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<int> h(DD);
        CUDA_OK(cudaMemcpy(h.data(), dout, DD * sizeof(int), cudaMemcpyDeviceToHost));
        printf("N=%d deg=%zu", g.N, S.size());
        for (int d = 0; d < DD; d++) printf(" |B%d|=%d", d + 1, h[d]);
        printf("\n");
        return 0;
    }

    if (!delta || !D || !Nmin) {
        fprintf(stderr, "usage: dd_gpu --delta d --diam D --Nmin n [--Nmax n] [--time s]\n"
                        "              [--blocks B] [--threads T] [--iters I] [--out FILE]\n"
                        "       dd_gpu --eval \"m,n,a:s1,s2,...\" --D d\n");
        return 1;
    }
    if (delta > MAXDEG) { fprintf(stderr, "delta > %d\n", MAXDEG); return 1; }
    if (!Nmax) Nmax = Nmin;

    std::mt19937_64 rng(seed);
    struct Task { Spec s; int N; };
    std::vector<Task> tasks;
    for (int N = Nmax; N >= Nmin; N--) {
        if (N < delta + 1) continue;
        for (auto &s : enum_specs(N, max_a, rng, min_n, max_n, true)) tasks.push_back({s, N});
    }
    if (tasks.empty()) { fprintf(stderr, "no groups in range\n"); return 2; }

    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "[dd_gpu] %s, %d SMs, %zu KB shared/block max\n",
            prop.name, prop.multiProcessorCount, prop.sharedMemPerBlockOptin / 1024);
    fprintf(stderr, "[dd_gpu] delta=%d D=%d N=%d..%d  %zu specs, %.0fs, %d blocks x %d threads\n",
            delta, D, Nmin, Nmax, tasks.size(), seconds, blocks, threads);

    int *d_found, *d_S, *d_bf, *d_inv = nullptr; unsigned long long *d_ev;
    size_t d_inv_cap = 0;
    CUDA_OK(cudaMalloc(&d_found, sizeof(int)));
    CUDA_OK(cudaMalloc(&d_S, MAXDEG * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_bf, sizeof(int)));
    CUDA_OK(cudaMalloc(&d_ev, sizeof(unsigned long long)));
    { unsigned long long z = 0; CUDA_OK(cudaMemcpy(d_ev, &z, sizeof(z), cudaMemcpyHostToDevice)); }

    FILE *fo = out_path.empty() ? stdout : fopen(out_path.c_str(), "a");
    if (!fo) { perror("out"); return 3; }

    cudaEvent_t t0e, t1e;
    CUDA_OK(cudaEventCreate(&t0e)); CUDA_OK(cudaEventCreate(&t1e));
    double spent = 0;
    long long launches = 0, evals_est = 0;
    int global_best = 1 << 30, gb_N = 0; Spec gb{0, 0, 0};
    int smem_attr = 0;
    const int NSPAN = Nmax - Nmin + 1;
    std::vector<int> per_f((size_t)NSPAN, 1 << 30);
    std::vector<Spec> per_spec((size_t)NSPAN, Spec{0, 0, 0});
    size_t cursor = rng() % tasks.size();
    int hits = 0;

    while (spent < seconds) {
        const Task &T = tasks[cursor % tasks.size()];
        cursor += 1 + (rng() % 7);
        GP g;
        g.m = T.s.m; g.n = T.s.n; g.a = T.s.a; g.N = T.N;
        g.R = g.n; g.W = (g.m + 63) / 64;
        int rem = g.m - 64 * (g.W - 1);
        g.topmask = (rem == 64) ? ~0ULL : ((1ULL << rem) - 1ULL);
        // involutions of Z_m : Z_n -- (i,j) with 2j = 0 mod n and i(1+a^j) = 0 mod m
        std::vector<int> invol;
        if (delta & 1) {
            int js[2] = {0, (g.n % 2 == 0) ? g.n / 2 : -1};
            for (int z = 0; z < 2; z++) {
                int j = js[z];
                if (j < 0) continue;
                int aj = powmod_h(g.a, j, g.m);
                long long c = (1 + aj) % g.m;
                for (int i = 0; i < g.m; i++) {
                    if (i == 0 && j == 0) continue;
                    if ((c * i) % g.m == 0) invol.push_back(j * g.m + i);
                }
            }
            if (invol.empty()) continue;                 // no odd-degree Cayley graph here
            if (invol.size() > d_inv_cap) {
                if (d_inv) CUDA_OK(cudaFree(d_inv));
                d_inv_cap = invol.size();
                CUDA_OK(cudaMalloc(&d_inv, d_inv_cap * sizeof(int)));
            }
            CUDA_OK(cudaMemcpy(d_inv, invol.data(), invol.size() * sizeof(int), cudaMemcpyHostToDevice));
        }
        int use_shf = (delta * g.R <= 6144);
        size_t sm = 2ULL * g.R * g.W * sizeof(u64)
                  + (g.n + 8 + (use_shf ? delta * g.R : 0)) * sizeof(int);
        if (sm + 4096 > prop.sharedMemPerBlockOptin) { use_shf = 0;
            sm = 2ULL * g.R * g.W * sizeof(u64) + (g.n + 8) * sizeof(int); }
        if (sm + 4096 > prop.sharedMemPerBlockOptin) continue;   // too big for shared memory
        if ((int)sm > smem_attr) {                               // raise the opt-in limit as needed
            smem_attr = (int)sm;
            CUDA_OK(cudaFuncSetAttribute(search_kernel,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_attr));
        }

        int zero = 0, big = 1 << 30;
        CUDA_OK(cudaMemcpy(d_found, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_bf, &big, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_OK(cudaEventRecord(t0e));
        search_kernel<<<blocks, threads, sm>>>(g, delta, D, use_shf, iters, stall_lim, kick_lim,
                                               d_inv, (int)invol.size(),
                                               rng(), d_found, d_S, d_bf, d_ev);
        CUDA_OK(cudaEventRecord(t1e));
        CUDA_OK(cudaEventSynchronize(t1e));
        float ms = 0; CUDA_OK(cudaEventElapsedTime(&ms, t0e, t1e));
        spent += ms / 1000.0;
        launches++;


        int hf = 0, bf = 0;
        CUDA_OK(cudaMemcpy(&hf, d_found, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(&bf, d_bf, sizeof(int), cudaMemcpyDeviceToHost));
        if (bf < global_best) { global_best = bf; gb_N = T.N; gb = T.s; }
        { size_t bi = (size_t)(T.N - Nmin);
          if (bf < per_f[bi]) { per_f[bi] = bf; per_spec[bi] = T.s; } }
        if (hf) {
            std::vector<int> S(delta);
            CUDA_OK(cudaMemcpy(S.data(), d_S, delta * sizeof(int), cudaMemcpyDeviceToHost));
            fprintf(fo, "{\"model\":\"metacyclic\",\"delta\":%d,\"D\":%d,\"N\":%d,\"m\":%d,\"n\":%d,\"a\":%d,\"S\":[",
                    delta, D, T.N, T.s.m, T.s.n, T.s.a);
            for (int z = 0; z < delta; z++) fprintf(fo, "%s%d", z ? "," : "", S[z]);
            fprintf(fo, "]}\n");
            fflush(fo);
            hits++;
        }
    }
    unsigned long long tot_ev = 0;
    CUDA_OK(cudaMemcpy(&tot_ev, d_ev, sizeof(tot_ev), cudaMemcpyDeviceToHost));
    (void)evals_est;
    fprintf(stderr, "[dd_gpu] %.2fs  %lld launches  %llu evals  %.2f Mevals/s  "
                    "best f=%d at N=%d (m=%d n=%d a=%d)  hits=%d\n",
            spent, launches, tot_ev, tot_ev / spent / 1e6, global_best, gb_N,
            gb.m, gb.n, gb.a, hits);
    for (int i = NSPAN - 1; i >= 0; i--) {
        if (per_f[i] == (1 << 30)) continue;
        fprintf(stderr, "  N=%-9d best_f=%-8d (m=%d n=%d a=%d)\n",
                Nmin + i, per_f[i], per_spec[i].m, per_spec[i].n, per_spec[i].a);
    }
    if (fo != stdout) fclose(fo);
    return hits ? 0 : 10;
}
