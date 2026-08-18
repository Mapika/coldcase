// cwc_tabu.cu — massively parallel tabu search for binary constant-weight codes.
//
// Problem: find M codewords in {0,1}^n, each of Hamming weight w, with all
// pairwise Hamming distances >= d.  Success proves A(n,d,w) >= M.
//
// One thread block runs one independent tabu-search chain.  Chain state:
//   words:  M x uint64 (n <= 64)
//   dist:   M x M uint8 pairwise distance matrix (incrementally maintained)
//   tabu:   M x 64 int32 iteration stamps (move (i,bit) forbidden until stamp)
// A move swaps one set bit p and one unset bit q of a violating word i.
// Delta of dist(i,j) for the swap: (bit p of c_j ? +1 : -1) + (bit q of c_j ? -1 : +1).
// Cost = number of violating pairs (dist < d).  Cost 0 => valid code, report.
//
// Build: nvcc -O3 -arch=sm_90 --shared -Xcompiler -fPIC -o libcwc.so cwc_tabu.cu

#include <cstdint>
#include <cstdio>

#define MAX_M 2048

struct ChainResult {
  int32_t found;        // 1 if cost 0 reached
  int32_t best_cost;    // best cost seen
  int64_t iters_done;
  int32_t pad;
};

__device__ __forceinline__ uint32_t xorshift32(uint32_t &s) {
  s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}

// mode 0: constant-weight (swap moves).  mode 1: unrestricted (single-bit flips).
extern "C" __global__ void cwc_tabu_kernel(
    int n, int d, int w, int M, int mode,
    long long iters, int tenure_lo, int tenure_span,
    unsigned long long seed,
    const uint64_t *__restrict__ init_words,   // nchains x M (each chain's start)
    uint64_t *__restrict__ words_g,            // nchains x M working/output
    uint64_t *__restrict__ best_words_g,       // nchains x M snapshot at best cost
    uint8_t  *__restrict__ dist_g,             // nchains x M x M
    int32_t  *__restrict__ tabu_g,             // nchains x M x 64
    ChainResult *__restrict__ res_g)
{
  const int chain = blockIdx.x;
  const int tid   = threadIdx.x;
  const int nthr  = blockDim.x;

  uint64_t *words = words_g + (size_t)chain * M;
  uint64_t *bwords = best_words_g + (size_t)chain * M;
  uint8_t  *dist  = dist_g  + (size_t)chain * M * M;
  int32_t  *tabu  = tabu_g  + (size_t)chain * M * 64;
  ChainResult *res = res_g + chain;

  __shared__ uint64_t sw[MAX_M];          // codewords
  __shared__ int32_t viol[MAX_M];         // per-word violation count
  __shared__ int32_t total_viol;
  __shared__ int32_t sel_word;
  __shared__ int32_t best_cost_seen;
  // per-thread best-move reduction
  __shared__ int32_t red_delta[256];
  __shared__ int32_t red_pq[256];
  __shared__ uint32_t rng_state_sh;

  uint32_t rng = (uint32_t)(seed ^ (0x9e3779b9u * (chain + 1)) ^ (0x85ebca6bu * (tid + 1)));
  for (int k = 0; k < 4; k++) xorshift32(rng);

  // ---- init: load words, compute dist matrix and violations ----
  for (int i = tid; i < M; i += nthr) sw[i] = init_words[(size_t)chain * M + i];
  __syncthreads();

  for (int i = tid; i < M; i += nthr) {
    int v = 0;
    for (int j = 0; j < M; j++) {
      int dij = (j == i) ? 255 : __popcll(sw[i] ^ sw[j]);
      dist[(size_t)i * M + j] = (uint8_t)(dij > 255 ? 255 : dij);
      if (j != i && dij < d) v++;
    }
    viol[i] = v;
  }
  for (int i = tid; i < M * 64; i += nthr) tabu[i] = -1;
  if (tid == 0) {
    int tv = 0;
    for (int i = 0; i < M; i++) tv += viol[i];
    total_viol = tv / 2;
    best_cost_seen = total_viol;
    rng_state_sh = (uint32_t)(seed ^ (0xc2b2ae35u * (chain + 7)));
  }
  __syncthreads();
  for (int k = tid; k < M; k += nthr) bwords[k] = sw[k];
  __syncthreads();

  const uint64_t full_mask = (n == 64) ? ~0ULL : ((1ULL << n) - 1);

  long long it = 0;
  for (; it < iters; it++) {
    if (total_viol == 0) break;

    // ---- select a violating word (thread 0) ----
    if (tid == 0) {
      uint32_t r = xorshift32(rng_state_sh);
      // reservoir-ish: pick k-th violating word, k random
      int cnt = 0;
      for (int i = 0; i < M; i++) if (viol[i] > 0) cnt++;
      int k = (cnt > 0) ? (int)(r % (uint32_t)cnt) : 0;
      int pick = 0;
      for (int i = 0; i < M; i++) {
        if (viol[i] > 0) { if (k == 0) { pick = i; break; } k--; }
      }
      sel_word = pick;
    }
    __syncthreads();

    const int i = sel_word;
    const uint64_t ci = sw[i];
    const uint64_t zeros = full_mask & ~ci;

    // ---- enumerate candidate moves ----
    // mode 0 (swap): c = a * (n - w) + b -> p = a-th set bit, q = b-th unset bit
    // mode 1 (flip): c = bit index to flip (p == q == c)
    const int n_unset = n - w;
    const int ncand = (mode == 0) ? w * n_unset : n;

    int my_best_delta = INT32_MAX;
    int my_best_pq = -1;

    const uint8_t *drow = dist + (size_t)i * M;

    for (int c = tid; c < ncand; c += nthr) {
      int p, q;
      if (mode == 0) {
        int a = c / n_unset, b = c % n_unset;
        // a-th set bit of ci
        uint64_t t = ci;
        for (int s = 0; s < a; s++) t &= t - 1;
        p = __ffsll((long long)t) - 1;
        // b-th set bit of zeros
        t = zeros;
        for (int s = 0; s < b; s++) t &= t - 1;
        q = __ffsll((long long)t) - 1;
      } else {
        p = q = c;
      }

      const uint64_t pm = 1ULL << p, qm = 1ULL << q;
      // delta cost over all j
      int delta = 0;
      if (mode == 0) {
        for (int j = 0; j < M; j++) {
          if (j == i) continue;
          const uint64_t cj = sw[j];
          int dd = (int)drow[j] + ((cj & pm) ? 1 : -1) + ((cj & qm) ? -1 : 1);
          delta += (dd < d) - ((int)drow[j] < d);
        }
      } else {
        // flipping bit p of c_i: dist changes by +1 if c_j agrees at p, else -1
        const uint64_t ci_bit = ci & pm;
        for (int j = 0; j < M; j++) {
          if (j == i) continue;
          const uint64_t cj = sw[j];
          int dd = (int)drow[j] + (((cj & pm) == ci_bit) ? 1 : -1);
          delta += (dd < d) - ((int)drow[j] < d);
        }
      }

      // tabu check: forbid re-setting bit p or re-clearing bit q if recently moved
      bool is_tabu = (tabu[i * 64 + p] > (int)it) || (tabu[i * 64 + q] > (int)it);
      // aspiration: allow if it reaches a new global best
      int new_cost = total_viol + delta;
      if (is_tabu && new_cost >= best_cost_seen) continue;

      // tie-break with random jitter in low bits
      int keyed = delta * 4096 + (int)(xorshift32(rng) & 4095);
      if (keyed < my_best_delta) { my_best_delta = keyed; my_best_pq = p * 64 + q; }
    }

    red_delta[tid] = my_best_delta;
    red_pq[tid] = my_best_pq;
    __syncthreads();

    // ---- reduce best move ----
    for (int s = nthr / 2; s > 0; s >>= 1) {
      if (tid < s) {
        if (red_delta[tid + s] < red_delta[tid]) {
          red_delta[tid] = red_delta[tid + s];
          red_pq[tid] = red_pq[tid + s];
        }
      }
      __syncthreads();
    }

    // ---- apply move (thread 0 picks; all threads update dist) ----
    __shared__ int mv_p, mv_q, mv_ok;
    if (tid == 0) {
      if (red_pq[0] < 0) { // all moves tabu: random kick
        uint32_t r1 = xorshift32(rng_state_sh), r2 = xorshift32(rng_state_sh);
        if (mode == 0) {
          uint64_t t = ci; int a = (int)(r1 % (uint32_t)w);
          for (int s = 0; s < a; s++) t &= t - 1;
          mv_p = __ffsll((long long)t) - 1;
          t = zeros; int b = (int)(r2 % (uint32_t)n_unset);
          for (int s = 0; s < b; s++) t &= t - 1;
          mv_q = __ffsll((long long)t) - 1;
        } else {
          mv_p = mv_q = (int)(r1 % (uint32_t)n);
        }
      } else {
        mv_p = red_pq[0] / 64; mv_q = red_pq[0] % 64;
      }
      mv_ok = 1;
    }
    __syncthreads();

    const int p = mv_p, q = mv_q;
    const uint64_t pm = 1ULL << p, qm = 1ULL << q;
    const uint64_t ci_new = (p == q) ? (ci ^ pm) : ((ci & ~pm) | qm);
    const uint64_t ci_bit_p = ci & pm;

    // update dist row/col, viol counts
    int dv_i = 0;
    for (int j = tid; j < M; j += nthr) {
      if (j == i) continue;
      const uint64_t cj = sw[j];
      int old_d = (int)drow[j];
      int new_d = (p == q)
          ? old_d + (((cj & pm) == ci_bit_p) ? 1 : -1)
          : old_d + ((cj & pm) ? 1 : -1) + ((cj & qm) ? -1 : 1);
      dist[(size_t)i * M + j] = (uint8_t)new_d;
      dist[(size_t)j * M + i] = (uint8_t)new_d;
      int was = old_d < d, now = new_d < d;
      if (was != now) {
        atomicAdd(&viol[j], now - was);
        dv_i += now - was;
      }
    }
    atomicAdd(&viol[i], dv_i);
    __syncthreads();

    __shared__ int improved;
    if (tid == 0) {
      sw[i] = ci_new;
      words[i] = ci_new;
      int tv = 0;
      for (int jj = 0; jj < M; jj++) tv += viol[jj];
      total_viol = tv / 2;
      improved = 0;
      if (total_viol < best_cost_seen) { best_cost_seen = total_viol; improved = 1; }
      // set tabu: forbid touching bits p (just cleared) and q (just set) of word i
      uint32_t r = xorshift32(rng_state_sh);
      int tenure = tenure_lo + (int)(r % (uint32_t)tenure_span);
      tabu[i * 64 + p] = (int)it + tenure;
      tabu[i * 64 + q] = (int)it + tenure;
    }
    __syncthreads();
    if (improved) {
      for (int k = tid; k < M; k += nthr) bwords[k] = sw[k];
      __syncthreads();
    }
  }

  // ---- write out ----
  if (tid == 0) {
    res->found = (total_viol == 0) ? 1 : 0;
    res->best_cost = best_cost_seen;
    res->iters_done = it;
  }
  for (int k = tid; k < M; k += nthr) words[k] = sw[k];
}

// ---------------- host-side launcher ----------------
#include <cuda_runtime.h>

extern "C" int cwc_run(
    int n, int d, int w, int M, int mode, int nchains,
    long long iters, int tenure_lo, int tenure_span,
    unsigned long long seed, int threads,
    const uint64_t *init_words_h,   // nchains*M
    uint64_t *words_out_h,          // nchains*M
    uint64_t *best_words_out_h,     // nchains*M
    int32_t *found_out_h,           // nchains
    int32_t *best_cost_out_h,       // nchains
    long long *iters_out_h)         // nchains
{
  uint64_t *init_d = nullptr, *words_d = nullptr, *bwords_d = nullptr;
  uint8_t *dist_d = nullptr; int32_t *tabu_d = nullptr;
  ChainResult *res_d = nullptr;
  size_t wsz = (size_t)nchains * M * sizeof(uint64_t);
  cudaError_t err;
  if ((err = cudaMalloc(&init_d, wsz))) return -1;
  if ((err = cudaMalloc(&words_d, wsz))) return -2;
  if ((err = cudaMalloc(&bwords_d, wsz))) return -21;
  if ((err = cudaMalloc(&dist_d, (size_t)nchains * M * M))) return -3;
  if ((err = cudaMalloc(&tabu_d, (size_t)nchains * M * 64 * sizeof(int32_t)))) return -4;
  if ((err = cudaMalloc(&res_d, (size_t)nchains * sizeof(ChainResult)))) return -5;
  cudaMemcpy(init_d, init_words_h, wsz, cudaMemcpyHostToDevice);

  cwc_tabu_kernel<<<nchains, threads>>>(n, d, w, M, mode, iters, tenure_lo, tenure_span,
                                        seed, init_d, words_d, bwords_d, dist_d, tabu_d, res_d);
  err = cudaDeviceSynchronize();
  int rc = 0;
  if (err != cudaSuccess) { rc = -100 - (int)err; }
  else {
    cudaMemcpy(words_out_h, words_d, wsz, cudaMemcpyDeviceToHost);
    cudaMemcpy(best_words_out_h, bwords_d, wsz, cudaMemcpyDeviceToHost);
    ChainResult *res_h = new ChainResult[nchains];
    cudaMemcpy(res_h, res_d, (size_t)nchains * sizeof(ChainResult), cudaMemcpyDeviceToHost);
    for (int c = 0; c < nchains; c++) {
      found_out_h[c] = res_h[c].found;
      best_cost_out_h[c] = res_h[c].best_cost;
      iters_out_h[c] = res_h[c].iters_done;
    }
    delete[] res_h;
  }
  cudaFree(init_d); cudaFree(words_d); cudaFree(bwords_d); cudaFree(dist_d); cudaFree(tabu_d); cudaFree(res_d);
  return rc;
}
