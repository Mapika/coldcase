// dct.cu — exact full-grid distance-count transform engine for covering codes.
//
// PRIMITIVE: given an indicator/multiset array S over Z_q^n, compute for EVERY
// x the counts N_d(x) = #{u in S : d_H(x,u) = d}, d = 0..R, by a sequential
// axis DP.  Invariant: after processing axes 0..j, A_d(x) = #{u in S :
// u matches x on axes > j, and d_H(x,u) restricted to axes <= j equals d}.
// Axis step along a fiber (q values a = 0..q-1 at stride q^j):
//     newA_d[a] = A_d[a] + T_{d-1} - A_{d-1}[a],   T_d = sum_a A_d[a]
// (split on u_j == a vs u_j != a; A_{-1} = 0, so A_0 never changes).
// After all n axes A_d = N_d exactly.  All arithmetic int32; intermediates are
// bounded by C(j,d)(q-1)^d <= 1.5e7 for our cells, so no overflow.
//
// Uses:
//   S = code multiset      -> cnt(x) = sum_d N_d = coverage multiplicity (u16)
//   S = [cnt == 0]         -> gain(x) = sum_d N_d = uncovered words a codeword
//                             at x would newly cover (global exact gain map)
//   S = [cnt == 1]         -> loss(c) = sum_d N_d at codeword c = its private
//                             coverage (exact ruin scores)
//
// Everything else here is support: exact ball +/- updates of cnt, batched
// ball gathers (exact gain/loss of given positions), top-value extraction,
// reductions.  No search claim ever rests on these counters: covers are
// re-verified from the written file by cov/verify_cov.py.
//
// Build: nvcc -O3 -arch=sm_90 --shared -Xcompiler -fPIC -o libdct.so dct.cu

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define MAXN 16
#define MAXQ 10
#define MAXR 5

// ---------------------------------------------------------------- context
static int g_q, g_n, g_R;
static long long g_space;
static long long g_powq[MAXN + 1];
static int32_t *g_A[MAXR + 1];      // layer arrays, q^n int32 each
static uint16_t *g_cnt;             // exact coverage multiplicity
static uint64_t *g_ballpat;         // packed ball patterns (r=0..R concat)
static long long g_balllen;
static long long *g_codeidx;        // device copy of code word indices
static int g_codeM, g_codecap;
static long long *g_scalar;         // device scalar scratch (counters)
static int32_t *g_hist;             // histogram buffer
static uint32_t *g_extract_idx;     // extraction buffers
static int32_t *g_extract_val;
static int g_extract_cap;

__constant__ long long c_powq[MAXN + 1];
__constant__ int c_q, c_n, c_R;
__constant__ long long c_space;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
  return -(100 + (int)e); } } while (0)

// ---------------------------------------------------------------- axis pass
struct Ptrs { int32_t *a[MAXR + 1]; };

// OUT: 0 = write updated layers; 1 = write sum_d N_d into A0 (gain/loss map);
//      2 = write min(sum, 65535) into cnt (coverage map).
template <int Q, int R1, int OUT>
__global__ void axis_kernel(Ptrs P, long long stride, long long nfib,
                            uint16_t *cnt) {
  long long gs = (long long)gridDim.x * blockDim.x;
  for (long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       f < nfib; f += gs) {
    long long lo = f % stride, hi = f / stride;
    long long base = hi * stride * Q + lo;
    int32_t v[R1][Q];
    int32_t T[R1];
#pragma unroll
    for (int d = 0; d < R1; d++) T[d] = 0;
#pragma unroll
    for (int a = 0; a < Q; a++) {
      long long x = base + (long long)a * stride;
#pragma unroll
      for (int d = 0; d < R1; d++) { v[d][a] = P.a[d][x]; T[d] += v[d][a]; }
    }
#pragma unroll
    for (int a = 0; a < Q; a++) {
      long long x = base + (long long)a * stride;
      if (OUT == 0) {
#pragma unroll
        for (int d = R1 - 1; d >= 1; d--)
          P.a[d][x] = v[d][a] + T[d - 1] - v[d - 1][a];
      } else {
        int32_t s = v[0][a];
#pragma unroll
        for (int d = 1; d < R1; d++) s += v[d][a] + T[d - 1] - v[d - 1][a];
        if (OUT == 1) P.a[0][x] = s;
        else cnt[x] = (uint16_t)(s > 65535 ? 65535 : s);
      }
    }
  }
}

// generic (runtime q, R) fallback for correctness tests on small grids
template <int OUT>
__global__ void axis_kernel_gen(Ptrs P, long long stride, long long nfib,
                                uint16_t *cnt) {
  long long gs = (long long)gridDim.x * blockDim.x;
  int Q = c_q, R1 = c_R + 1;
  for (long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       f < nfib; f += gs) {
    long long lo = f % stride, hi = f / stride;
    long long base = hi * stride * Q + lo;
    int32_t v[(MAXR + 1) * MAXQ];
    int32_t T[MAXR + 1];
    for (int d = 0; d < R1; d++) T[d] = 0;
    for (int a = 0; a < Q; a++) {
      long long x = base + (long long)a * stride;
      for (int d = 0; d < R1; d++) {
        int32_t y = P.a[d][x]; v[d * MAXQ + a] = y; T[d] += y;
      }
    }
    for (int a = 0; a < Q; a++) {
      long long x = base + (long long)a * stride;
      if (OUT == 0) {
        for (int d = R1 - 1; d >= 1; d--)
          P.a[d][x] = v[d * MAXQ + a] + T[d - 1] - v[(d - 1) * MAXQ + a];
      } else {
        int32_t s = v[0 * MAXQ + a];
        for (int d = 1; d < R1; d++)
          s += v[d * MAXQ + a] + T[d - 1] - v[(d - 1) * MAXQ + a];
        if (OUT == 1) P.a[0][x] = s;
        else cnt[x] = (uint16_t)(s > 65535 ? 65535 : s);
      }
    }
  }
}

// ---------------------------------------------------------------- init kernels
__global__ void k_init_from_cnt(int32_t *A0, const uint16_t *cnt, int target,
                                long long N) {
  long long gs = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gs)
    A0[i] = (cnt[i] == target) ? 1 : 0;
}

__global__ void k_scatter_code(int32_t *A0, const long long *idx, int M) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < M) atomicAdd(&A0[idx[i]], 1);
}

// ------------------------------------------------------- ball walk kernels
// One block per word.  Shared: per-(pos,delta) index offsets, so each pattern
// costs one table load plus <=R shared adds (no div/mod in the hot loop).
__device__ __forceinline__ void build_contrib(long long w, long long *contrib,
                                              uint8_t *dig) {
  int tid = threadIdx.x;
  __shared__ uint8_t sdig[MAXN];
  if (tid == 0) {
    long long x = w;
    for (int i = 0; i < c_n; i++) { sdig[i] = (uint8_t)(x % c_q); x /= c_q; }
  }
  __syncthreads();
  for (int t = tid; t < c_n * (c_q - 1); t += blockDim.x) {
    int p = t / (c_q - 1), delta = 1 + t % (c_q - 1);
    int nd = sdig[p] + delta;
    contrib[t] = ((nd < c_q) ? (long long)delta : (long long)(delta - c_q)) *
                 c_powq[p];
  }
  if (dig && tid < c_n) dig[tid] = sdig[tid];
  __syncthreads();
}

__device__ __forceinline__ long long pat_word(long long w, uint64_t pat,
                                              const long long *contrib) {
  long long x = w;
#pragma unroll
  for (int k = 0; k < MAXR; k++) {
    unsigned b = (unsigned)(pat >> (8 * k)) & 255u;
    if (!b) break;
    x += contrib[(b >> 4) * (c_q - 1) + (b & 15u) - 1];
  }
  return x;
}

// delta = +1 / -1 on cnt over the ball of each word (exact; cnt never wraps
// because it only decrements words that are genuinely covered).
__global__ void k_ball_update(const long long *words, int nwords, int delta,
                              uint16_t *cnt, const uint64_t *pat,
                              long long balllen) {
  __shared__ long long contrib[MAXN * (MAXQ - 1)];
  long long w = words[blockIdx.x];
  build_contrib(w, contrib, nullptr);
  for (long long t = threadIdx.x; t < balllen; t += blockDim.x) {
    long long x = pat_word(w, pat[t], contrib);
    uint32_t *cell = (uint32_t *)((uintptr_t)(&cnt[x]) & ~(uintptr_t)3);
    uint32_t add = ((x & 1) ? (1u << 16) : 1u);
    if (delta > 0) atomicAdd(cell, add);
    else atomicSub(cell, add);
  }
}

// count words in the ball with cnt == target (0 -> exact placement gain,
// 1 -> exact removal loss).  out[i] for word i.
__global__ void k_ball_gather(const long long *words, int nwords, int target,
                              const uint16_t *cnt, const uint64_t *pat,
                              long long balllen, int32_t *out) {
  __shared__ long long contrib[MAXN * (MAXQ - 1)];
  __shared__ int32_t red[256];
  long long w = words[blockIdx.x];
  build_contrib(w, contrib, nullptr);
  int32_t acc = 0;
  for (long long t = threadIdx.x; t < balllen; t += blockDim.x) {
    long long x = pat_word(w, pat[t], contrib);
    acc += (cnt[x] == target);
  }
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = red[0];
}

// ---------------------------------------------------------------- reductions
__global__ void k_count_eq(const uint16_t *cnt, int target, long long N,
                           long long *out) {
  __shared__ long long red[256];
  long long gs = (long long)gridDim.x * blockDim.x;
  long long acc = 0;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gs)
    acc += (cnt[i] == target);
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd((unsigned long long *)out,
                                  (unsigned long long)red[0]);
}

__global__ void k_max_i32(const int32_t *A, long long N, long long *out) {
  __shared__ int32_t red[256];
  long long gs = (long long)gridDim.x * blockDim.x;
  int32_t acc = 0;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gs) { int32_t v = A[i]; if (v > acc) acc = v; }
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s && red[threadIdx.x + s] > red[threadIdx.x])
      red[threadIdx.x] = red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicMax((int *)out, red[0]);
}

__global__ void k_hist(const int32_t *A, long long N, int nbins, int32_t vmax,
                       int32_t *hist) {
  extern __shared__ int32_t sh[];
  for (int i = threadIdx.x; i < nbins; i += blockDim.x) sh[i] = 0;
  __syncthreads();
  long long gs = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gs) {
    int32_t v = A[i];
    if (v > 0) {
      int b = (int)((long long)(v - 1) * nbins / vmax);
      if (b >= nbins) b = nbins - 1;
      atomicAdd(&sh[b], 1);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < nbins; i += blockDim.x)
    atomicAdd(&hist[i], sh[i]);
}

__global__ void k_extract(const int32_t *A, long long N, int32_t thr,
                          uint32_t *oidx, int32_t *oval, int cap,
                          long long *ocount) {
  long long gs = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gs) {
    int32_t v = A[i];
    if (v >= thr) {
      long long slot = atomicAdd((unsigned long long *)ocount, 1ULL);
      if (slot < cap) { oidx[slot] = (uint32_t)i; oval[slot] = v; }
    }
  }
}

__global__ void k_read_at(const int32_t *A, const long long *idx, int k,
                          int32_t *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k) out[i] = A[idx[i]];
}

// ---------------------------------------------------------------- host API
extern "C" {

// returns 0 on success; bytes allocated written to *bytes
int dct_init(int q, int n, int R, const uint64_t *ballpat, long long balllen,
             int extract_cap, long long *bytes) {
  if (q < 2 || q > MAXQ || n < 2 || n > MAXN || R < 1 || R > MAXR) return -1;
  g_q = q; g_n = n; g_R = R;
  g_space = 1;
  for (int i = 0; i < n; i++) g_space *= q;
  g_powq[0] = 1;
  for (int i = 1; i <= n; i++) g_powq[i] = g_powq[i - 1] * q;
  if (g_space > 4200000000LL) return -2;  // uint32 extraction indices
  CK(cudaMemcpyToSymbol(c_powq, g_powq, sizeof(g_powq)));
  CK(cudaMemcpyToSymbol(c_q, &q, 4));
  CK(cudaMemcpyToSymbol(c_n, &n, 4));
  CK(cudaMemcpyToSymbol(c_R, &R, 4));
  CK(cudaMemcpyToSymbol(c_space, &g_space, 8));
  long long tot = 0;
  for (int d = 0; d <= R; d++) {
    CK(cudaMalloc(&g_A[d], g_space * 4)); tot += g_space * 4;
  }
  CK(cudaMalloc(&g_cnt, g_space * 2)); tot += g_space * 2;
  CK(cudaMemset(g_cnt, 0, g_space * 2));
  g_balllen = balllen;
  CK(cudaMalloc(&g_ballpat, balllen * 8)); tot += balllen * 8;
  CK(cudaMemcpy(g_ballpat, ballpat, balllen * 8, cudaMemcpyHostToDevice));
  g_codecap = 1 << 16;
  CK(cudaMalloc(&g_codeidx, (size_t)g_codecap * 8)); tot += g_codecap * 8;
  g_codeM = 0;
  CK(cudaMalloc(&g_scalar, 64 * 8));
  CK(cudaMalloc(&g_hist, 4096 * 4));
  g_extract_cap = extract_cap;
  CK(cudaMalloc(&g_extract_idx, (size_t)extract_cap * 4));
  CK(cudaMalloc(&g_extract_val, (size_t)extract_cap * 4));
  tot += (size_t)extract_cap * 8 + 64 * 8 + 4096 * 4;
  if (bytes) *bytes = tot;
  return 0;
}

int dct_free(void) {
  for (int d = 0; d <= g_R; d++) if (g_A[d]) { cudaFree(g_A[d]); g_A[d] = 0; }
  if (g_cnt) { cudaFree(g_cnt); g_cnt = 0; }
  if (g_ballpat) { cudaFree(g_ballpat); g_ballpat = 0; }
  if (g_codeidx) { cudaFree(g_codeidx); g_codeidx = 0; }
  if (g_scalar) { cudaFree(g_scalar); g_scalar = 0; }
  if (g_hist) { cudaFree(g_hist); g_hist = 0; }
  if (g_extract_idx) { cudaFree(g_extract_idx); g_extract_idx = 0; }
  if (g_extract_val) { cudaFree(g_extract_val); g_extract_val = 0; }
  cudaDeviceSynchronize();
  return 0;
}

int dct_set_code(const long long *idx, int M) {
  if (M > g_codecap) return -1;
  CK(cudaMemcpy(g_codeidx, idx, (size_t)M * 8, cudaMemcpyHostToDevice));
  g_codeM = M;
  return 0;
}

// init_mode: 0 = code multiset, 1 = [cnt==0], 2 = [cnt==1]
// out_mode:  1 = sum -> A0 (gain/loss map), 2 = clamped sum -> cnt
int dct_transform(int init_mode, int out_mode) {
  long long N = g_space;
  int blocks = 4096, threads = 256;
  for (int d = 1; d <= g_R; d++) CK(cudaMemsetAsync(g_A[d], 0, N * 4));
  if (init_mode == 0) {
    CK(cudaMemsetAsync(g_A[0], 0, N * 4));
    if (g_codeM > 0)
      k_scatter_code<<<(g_codeM + 255) / 256, 256>>>(g_A[0], g_codeidx,
                                                     g_codeM);
  } else {
    k_init_from_cnt<<<blocks, threads>>>(g_A[0], g_cnt,
                                         init_mode == 1 ? 0 : 1, N);
  }
  Ptrs P;
  for (int d = 0; d <= g_R; d++) P.a[d] = g_A[d];
  long long nfib = N / g_q;
  for (int j = 0; j < g_n; j++) {
    long long stride = g_powq[j];
    int last = (j == g_n - 1);
    int out = last ? out_mode : 0;
#define DISPATCH(Q, R1)                                                        \
  do {                                                                         \
    if (out == 0)                                                              \
      axis_kernel<Q, R1, 0><<<blocks, threads>>>(P, stride, nfib, g_cnt);      \
    else if (out == 1)                                                         \
      axis_kernel<Q, R1, 1><<<blocks, threads>>>(P, stride, nfib, g_cnt);      \
    else                                                                       \
      axis_kernel<Q, R1, 2><<<blocks, threads>>>(P, stride, nfib, g_cnt);      \
  } while (0)
    if (g_q == 7 && g_R == 5) DISPATCH(7, 6);
    else if (g_q == 8 && g_R == 4) DISPATCH(8, 5);
    else if (g_q == 9 && g_R == 4) DISPATCH(9, 5);
    else if (g_q == 10 && g_R == 5) DISPATCH(10, 6);
    else if (g_q == 10 && g_R == 4) DISPATCH(10, 5);
    else if (g_q == 7 && g_R == 4) DISPATCH(7, 5);
    else if (g_q == 8 && g_R == 5) DISPATCH(8, 6);
    else if (g_q == 9 && g_R == 5) DISPATCH(9, 6);
    else if (g_q == 6 && g_R == 5) DISPATCH(6, 6);
    else if (g_q == 6 && g_R == 4) DISPATCH(6, 5);
    else {
      if (out == 0)
        axis_kernel_gen<0><<<blocks, threads>>>(P, stride, nfib, g_cnt);
      else if (out == 1)
        axis_kernel_gen<1><<<blocks, threads>>>(P, stride, nfib, g_cnt);
      else
        axis_kernel_gen<2><<<blocks, threads>>>(P, stride, nfib, g_cnt);
    }
#undef DISPATCH
  }
  CK(cudaDeviceSynchronize());
  return 0;
}

int dct_ball_update(const long long *words, int nwords, int delta) {
  if (nwords <= 0) return 0;
  long long *d_words;
  CK(cudaMalloc(&d_words, (size_t)nwords * 8));
  CK(cudaMemcpy(d_words, words, (size_t)nwords * 8, cudaMemcpyHostToDevice));
  k_ball_update<<<nwords, 256>>>(d_words, nwords, delta, g_cnt, g_ballpat,
                                 g_balllen);
  CK(cudaDeviceSynchronize());
  cudaFree(d_words);
  return 0;
}

int dct_ball_gather(const long long *words, int nwords, int target,
                    int32_t *out) {
  if (nwords <= 0) return 0;
  long long *d_words; int32_t *d_out;
  CK(cudaMalloc(&d_words, (size_t)nwords * 8));
  CK(cudaMalloc(&d_out, (size_t)nwords * 4));
  CK(cudaMemcpy(d_words, words, (size_t)nwords * 8, cudaMemcpyHostToDevice));
  k_ball_gather<<<nwords, 256>>>(d_words, nwords, target, g_cnt, g_ballpat,
                                 g_balllen, d_out);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out, d_out, (size_t)nwords * 4, cudaMemcpyDeviceToHost));
  cudaFree(d_words); cudaFree(d_out);
  return 0;
}

long long dct_count_eq(int target) {
  CK(cudaMemset(g_scalar, 0, 8));
  k_count_eq<<<4096, 256>>>(g_cnt, target, g_space, g_scalar);
  CK(cudaDeviceSynchronize());
  long long r;
  CK(cudaMemcpy(&r, g_scalar, 8, cudaMemcpyDeviceToHost));
  return r;
}

int dct_map_max(int32_t *out_max) {
  CK(cudaMemset(g_scalar, 0, 8));
  k_max_i32<<<4096, 256>>>(g_A[0], g_space, g_scalar);
  CK(cudaDeviceSynchronize());
  long long r;
  CK(cudaMemcpy(&r, g_scalar, 8, cudaMemcpyDeviceToHost));
  *out_max = (int32_t)r;
  return 0;
}

int dct_map_hist(int nbins, int32_t vmax, int32_t *out_hist) {
  if (nbins > 4096) return -1;
  CK(cudaMemset(g_hist, 0, (size_t)nbins * 4));
  k_hist<<<1024, 256, nbins * 4>>>(g_A[0], g_space, nbins, vmax, g_hist);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out_hist, g_hist, (size_t)nbins * 4, cudaMemcpyDeviceToHost));
  return 0;
}

// collect positions with map value >= thr; returns number found (may exceed
// cap; only cap entries are stored)
long long dct_map_extract(int32_t thr, uint32_t *out_idx, int32_t *out_val,
                          int cap) {
  if (cap > g_extract_cap) cap = g_extract_cap;
  CK(cudaMemset(g_scalar, 0, 8));
  k_extract<<<4096, 256>>>(g_A[0], g_space, thr, g_extract_idx, g_extract_val,
                           cap, g_scalar);
  CK(cudaDeviceSynchronize());
  long long found;
  CK(cudaMemcpy(&found, g_scalar, 8, cudaMemcpyDeviceToHost));
  long long m = found < cap ? found : cap;
  CK(cudaMemcpy(out_idx, g_extract_idx, (size_t)m * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(out_val, g_extract_val, (size_t)m * 4, cudaMemcpyDeviceToHost));
  return found;
}

int dct_map_read_at(const long long *idx, int k, int32_t *out) {
  if (k <= 0) return 0;
  long long *d_idx; int32_t *d_out;
  CK(cudaMalloc(&d_idx, (size_t)k * 8));
  CK(cudaMalloc(&d_out, (size_t)k * 4));
  CK(cudaMemcpy(d_idx, idx, (size_t)k * 8, cudaMemcpyHostToDevice));
  k_read_at<<<(k + 255) / 256, 256>>>(g_A[0], d_idx, k, d_out);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out, d_out, (size_t)k * 4, cudaMemcpyDeviceToHost));
  cudaFree(d_idx); cudaFree(d_out);
  return 0;
}

// download a range of cnt (for CPU spot checks)
int dct_read_cnt(long long off, long long len, uint16_t *out) {
  CK(cudaMemcpy(out, g_cnt + off, (size_t)len * 2, cudaMemcpyDeviceToHost));
  return 0;
}

}  // extern "C"
