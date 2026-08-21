// chainsolve.cu — chain-batched GPU covering-code solver (direction A).
//
// One thread block = one independent focused-local-search chain running the
// p5b algorithm (shared-sphere leave-side + uncovered-list enter-side,
// cf. cov/opt/METHODS.md §B/§C). No host synchronization inside the loop:
// each chain iterates autonomously until its budget or a full cover.
//
// Per-chain state (global): cnt[q^n] uint16 (exact multiplicities),
// U-list (lazy-deleted uncovered words), codewords. Codewords mirrored in
// dynamic shared memory. All sphere/ball walks enumerate distinct words, so
// within a walk plain RMW is race-free; the two commit walks are separated
// by __syncthreads(). The exact uncovered count is maintained incrementally
// from transition counts measured inside the commit walks themselves.
//
// Build: nvcc -O3 -arch=sm_90 --shared -Xcompiler -fPIC -o libchain.so chainsolve.cu

#include <cstdint>
#include <cuda_runtime.h>

#define MAXN 16
#define MAXQ 10
#define MAXCAND 24
#define UCAP (1 << 20)

struct ChainOut {
  int32_t best_uncovered;
  int32_t solved;
  int64_t iters_done;
  int64_t commits;
};

__device__ __forceinline__ uint32_t xs32(uint32_t &s) {
  s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}

// binomial table (host-filled)
__constant__ int d_choose[MAXN + 1][MAXN + 1];
__constant__ int64_t d_pow_q[MAXN + 1];   // q^i for index math
__constant__ int d_q, d_n, d_R, d_M;
__constant__ int64_t d_space;             // q^n
__constant__ int d_sphere_cnt;            // C(n,R)*(q-1)^R
__constant__ int d_qm1_pow[MAXN + 1];     // (q-1)^i

// unrank the c-th R-subset of {0..n-1} (colex order) into idx[]
__device__ __forceinline__ void unrank_subset(int c, int R, int n, int *idx) {
  // colex unranking: largest element first
  for (int i = R; i >= 1; i--) {
    int v = i - 1;
    while (v + 1 <= n - 1 && d_choose[v + 1][i] <= c) v++;
    idx[i - 1] = v;
    c -= d_choose[v][i];
  }
}

// word index of codeword digits
__device__ __forceinline__ int64_t widx(const uint8_t *dig) {
  int64_t x = 0;
  for (int i = d_n - 1; i >= 0; i--) x = x * d_q + dig[i];
  return x;
}

extern "C" __global__ void chain_kernel(
    long long iters,
    unsigned long long seed,
    uint8_t  *__restrict__ code_g,     // nchains * M * MAXN (working)
    uint8_t  *__restrict__ best_g,     // nchains * M * MAXN (best snapshot)
    uint16_t *__restrict__ cnt_g,      // nchains * q^n
    uint32_t *__restrict__ ulist_g,    // nchains * UCAP
    ChainOut *__restrict__ out_g)
{
  const int chain = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthr = blockDim.x;

  uint16_t *cnt = cnt_g + (size_t)chain * d_space;
  uint32_t *U = ulist_g + (size_t)chain * UCAP;
  uint8_t *code_glob = code_g + (size_t)chain * d_M * MAXN;
  uint8_t *best_glob = best_g + (size_t)chain * d_M * MAXN;

  extern __shared__ uint8_t sh[];
  uint8_t *scode = sh;                             // M*MAXN
  int32_t *sH = (int32_t *)(scode + d_M * MAXN);   // MAXN
  int32_t *sgain = sH + MAXN;                      // MAXN*MAXQ
  int32_t *smisc = sgain + MAXN * MAXQ;            // misc scalars
  // smisc: [0]=Ulen [1]=ucount [2]=T [3]=best_u [4]=u_word_lo [5]=u_word_hi
  //        [6]=ncand [9]=stop [10]=delta_dec [11]=delta_inc
  int32_t *scand = smisc + 16;                     // MAXCAND
  __shared__ unsigned long long sbest;             // (key<<32)|move, atomicMin

  uint32_t rng = (uint32_t)(seed ^ (0x9e3779b9u * (chain + 1)) ^ (0x85ebca6bu * (tid + 1)));
  for (int k = 0; k < 4; k++) xs32(rng);
  __shared__ uint32_t srng;
  if (tid == 0) srng = (uint32_t)(seed ^ (0xc2b2ae35u * (chain + 7)));

  // ---- init: load code, build cnt + U + ucount ----
  for (int i = tid; i < d_M * MAXN; i += nthr) scode[i] = code_glob[i];
  __syncthreads();

  // zero cnt
  for (int64_t w = tid; w < d_space; w += nthr) cnt[w] = 0;
  __syncthreads();
  // mark balls: for each codeword, enumerate ball radius 0..R
  // (words within one codeword's ball are distinct; across codewords may
  //  collide -> use atomicAdd via 32-bit CAS on 16-bit lanes)
  for (int m = 0; m < d_M; m++) {
    const uint8_t *c = scode + m * MAXN;
    // radius 0
    if (tid == 0) {
      int64_t w0 = widx(c);
      atomicAdd((uint32_t *)&cnt[w0 & ~1LL], (w0 & 1) ? (1u << 16) : 1u);
    }
    for (int r = 1; r <= d_R; r++) {
      int nsub = d_choose[d_n][r];
      int nval = d_qm1_pow[r];
      int tot = nsub * nval;
      for (int t = tid; t < tot; t += nthr) {
        int sub = t / nval, val = t % nval;
        int idx[MAXN];
        unrank_subset(sub, r, d_n, idx);
        int64_t w = 0;
        uint8_t dig[MAXN];
        for (int i = 0; i < d_n; i++) dig[i] = c[i];
        int vv = val;
        for (int i = 0; i < r; i++) {
          int delta = 1 + vv % (d_q - 1);
          vv /= (d_q - 1);
          dig[idx[i]] = (uint8_t)((dig[idx[i]] + delta) % d_q);
        }
        for (int i = d_n - 1; i >= 0; i--) w = w * d_q + dig[i];
        atomicAdd((uint32_t *)&cnt[w & ~1LL], (w & 1) ? (1u << 16) : 1u);
      }
    }
    __syncthreads();
  }
  // build U list + exact ucount
  if (tid == 0) { smisc[0] = 0; smisc[1] = 0; }
  __syncthreads();
  for (int64_t w = tid; w < d_space; w += nthr) {
    if (cnt[w] == 0) {
      int slot = atomicAdd(&smisc[0], 1);
      atomicAdd(&smisc[1], 1);
      if (slot < UCAP) U[slot] = (uint32_t)w;
    }
  }
  __syncthreads();
  if (tid == 0) {
    if (smisc[0] > UCAP) smisc[0] = UCAP;  // overflow: fall back handled below
    smisc[3] = smisc[1];                   // best_u
    smisc[9] = 0;
  }
  __syncthreads();

  // ---- main loop ----
  long long it = 0;
  for (; it < iters; it++) {
    if (smisc[1] == 0) { break; }          // solved

    // --- pick uncovered word u (thread 0, lazy-delete stale entries) ---
    if (tid == 0) {
      int64_t u = -1;
      for (int tries = 0; tries < 64; tries++) {
        int len = smisc[0];
        if (len == 0) break;
        int i = (int)(xs32(srng) % (uint32_t)len);
        uint32_t w = U[i];
        if (cnt[w] == 0) { u = w; break; }
        // stale: swap-delete
        U[i] = U[len - 1];
        smisc[0] = len - 1;
      }
      if (u < 0) {
        // list unusable (stale-heavy or overflowed): linear rescan rebuild
        smisc[0] = 0;
        for (int64_t w = 0; w < d_space && smisc[0] < UCAP; w++)
          if (cnt[w] == 0) U[smisc[0]++] = (uint32_t)w;
        u = (smisc[0] > 0) ? U[xs32(srng) % (uint32_t)smisc[0]] : -1;
      }
      smisc[4] = (int32_t)(u & 0xffffffff);
      smisc[5] = (int32_t)(u >> 32);
    }
    __syncthreads();
    int64_t u = ((int64_t)smisc[5] << 32) | (uint32_t)smisc[4];
    if (u < 0) break;
    uint8_t ud[MAXN];
    {
      int64_t x = u;
      for (int i = 0; i < d_n; i++) { ud[i] = (uint8_t)(x % d_q); x /= d_q; }
    }

    // --- candidate codewords: distance R+1 from u (fallback: min distance) ---
    if (tid == 0) smisc[6] = 0;
    __syncthreads();
    for (int m = tid; m < d_M; m += nthr) {
      const uint8_t *c = scode + m * MAXN;
      int dd = 0;
      for (int i = 0; i < d_n; i++) dd += (c[i] != ud[i]);
      if (dd == d_R + 1) {
        int s = atomicAdd(&smisc[6], 1);
        if (s < MAXCAND) scand[s] = m;
      }
    }
    __syncthreads();
    if (smisc[6] == 0) {
      // fallback: pick a random codeword to relocate toward u
      if (tid == 0) { scand[0] = (int)(xs32(srng) % (uint32_t)d_M); smisc[6] = 1; }
      __syncthreads();
    }
    int ncand = min(smisc[6], MAXCAND);

    // --- evaluate candidates; track global best (key = delta*4096 + jitter) ---
    if (tid == 0) sbest = 0xffffffffffffffffULL;
    __syncthreads();

    for (int ci = 0; ci < ncand; ci++) {
      int m = scand[ci];
      const uint8_t *c = scode + m * MAXN;

      // zero H and gains
      for (int i = tid; i < MAXN; i += nthr) sH[i] = 0;
      for (int i = tid; i < MAXN * MAXQ; i += nthr) sgain[i] = 0;
      if (tid == 0) smisc[2] = 0;
      __syncthreads();

      // enter side: scan U for words at distance R+1 from c
      int len = smisc[0];
      for (int i = tid; i < len; i += nthr) {
        uint32_t w = U[i];
        if (cnt[w] != 0) continue;
        int64_t x = w;
        int dd = 0; int dpos[MAXN]; uint8_t dval[MAXN];
        for (int j = 0; j < d_n; j++) {
          uint8_t dj = (uint8_t)(x % d_q); x /= d_q;
          if (dj != c[j]) { if (dd < MAXN) { dpos[dd] = j; dval[dd] = dj; } dd++; }
        }
        if (dd == d_R + 1) {
          for (int k = 0; k < d_R + 1; k++)
            atomicAdd(&sgain[dpos[k] * MAXQ + dval[k]], 1);
        }
      }

      // leave side: sphere |D|=R of c; count T and H_j over cnt==1 words
      int myT = 0;
      int myH[MAXN];
      for (int i = 0; i < d_n; i++) myH[i] = 0;
      int nsub = d_choose[d_n][d_R];
      int nval = d_qm1_pow[d_R];
      int tot = nsub * nval;
      for (int t = tid; t < tot; t += nthr) {
        int sub = t / nval, val = t % nval;
        int idx[MAXN];
        unrank_subset(sub, d_R, d_n, idx);
        uint8_t dig[MAXN];
        for (int i = 0; i < d_n; i++) dig[i] = c[i];
        int vv = val;
        for (int i = 0; i < d_R; i++) {
          int delta = 1 + vv % (d_q - 1);
          vv /= (d_q - 1);
          dig[idx[i]] = (uint8_t)((dig[idx[i]] + delta) % d_q);
        }
        int64_t w = 0;
        for (int i = d_n - 1; i >= 0; i--) w = w * d_q + dig[i];
        if (cnt[w] == 1) {
          myT++;
          for (int i = 0; i < d_R; i++) myH[idx[i]]++;
        }
      }
      atomicAdd(&smisc[2], myT);
      for (int i = 0; i < d_n; i++) if (myH[i]) atomicAdd(&sH[i], myH[i]);
      __syncthreads();

      // pick best move of this candidate: coords p, values v != c_p
      int T = smisc[2];
      for (int pv = tid; pv < d_n * (d_q - 1); pv += nthr) {
        int p = pv / (d_q - 1);
        int v = pv % (d_q - 1);
        int val = (c[p] + 1 + v) % d_q;
        int delta = (T - sH[p]) - sgain[p * MAXQ + val];
        // key: shift delta to non-negative, add jitter for random tie-break
        unsigned long long key =
            ((unsigned long long)(unsigned)(delta + (1 << 20)) << 11)
            + (xs32(rng) & 2047);
        unsigned long long packed = (key << 32)
            | ((unsigned long long)ci << 20) | ((unsigned long long)p << 8)
            | (unsigned long long)val;
        atomicMin(&sbest, packed);
      }
      __syncthreads();
    }

    if (sbest == 0xffffffffffffffffULL) continue;
    unsigned int mv = (unsigned int)(sbest & 0xffffffff);
    int ci = mv >> 20, p = (mv >> 8) & 0xff, vv = mv & 0xff;
    int m = scand[ci];
    uint8_t *c = scode + m * MAXN;
    uint8_t oldv = c[p];

    // --- commit: walk ball(c) decrement, then ball(c') increment ---
    if (tid == 0) { smisc[10] = 0; smisc[11] = 0; }
    __syncthreads();
    for (int phase = 0; phase < 2; phase++) {
      // phase 0: old position (decrement); phase 1: new position (increment)
      uint8_t base[MAXN];
      for (int i = 0; i < d_n; i++) base[i] = c[i];
      if (phase == 1) base[p] = (uint8_t)vv;
      // radius 0..R
      for (int r = 0; r <= d_R; r++) {
        int nsub = d_choose[d_n][r];
        int nval = d_qm1_pow[r];
        int tot = nsub * nval;
        for (int t = tid; t < tot; t += nthr) {
          int sub = t / nval, val = t % nval;
          int idx[MAXN];
          if (r > 0) unrank_subset(sub, r, d_n, idx);
          uint8_t dig[MAXN];
          for (int i = 0; i < d_n; i++) dig[i] = base[i];
          int vx = val;
          for (int i = 0; i < r; i++) {
            int delta = 1 + vx % (d_q - 1);
            vx /= (d_q - 1);
            dig[idx[i]] = (uint8_t)((dig[idx[i]] + delta) % d_q);
          }
          int64_t w = 0;
          for (int i = d_n - 1; i >= 0; i--) w = w * d_q + dig[i];
          if (phase == 0) {
            uint16_t nc = --cnt[w];
            if (nc == 0) {
              atomicAdd(&smisc[10], 1);   // newly uncovered
              int slot = atomicAdd(&smisc[0], 1);
              if (slot < UCAP) U[slot] = (uint32_t)w;
              else atomicSub(&smisc[0], 1);
            }
          } else {
            uint16_t nc = cnt[w]++;
            if (nc == 0) atomicAdd(&smisc[11], 1);  // newly covered
          }
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      c[p] = (uint8_t)vv;
      code_glob[m * MAXN + p] = (uint8_t)vv;
      smisc[1] += smisc[10] - smisc[11];
      if (smisc[1] < smisc[3]) {
        smisc[3] = smisc[1];
      }
    }
    __syncthreads();
    // snapshot best code (all threads) when improved
    if (smisc[1] == smisc[3]) {
      for (int i = tid; i < d_M * MAXN; i += nthr) best_glob[i] = scode[i];
    }
    __syncthreads();
    (void)oldv;
  }

  if (tid == 0) {
    out_g[chain].best_uncovered = smisc[3];
    out_g[chain].solved = (smisc[1] == 0) ? 1 : 0;
    out_g[chain].iters_done = it;
    out_g[chain].commits = it;
  }
  // final: if current is best (or solved), snapshot
  if (smisc[1] <= smisc[3]) {
    for (int i = tid; i < d_M * MAXN; i += nthr) best_glob[i] = scode[i];
  }
}

// ---------------- host launcher ----------------
extern "C" int chain_run(
    int q, int n, int R, int M, int nchains,
    long long iters, unsigned long long seed, int threads,
    const uint8_t *code_h,        // nchains*M*MAXN
    uint8_t *best_h,              // nchains*M*MAXN
    int32_t *best_u_h, int32_t *solved_h, long long *iters_h)
{
  if (n > MAXN || q > MAXQ || M < 1) return -9;
  long long space = 1;
  for (int i = 0; i < n; i++) space *= q;

  int choose[MAXN + 1][MAXN + 1] = {};
  for (int i = 0; i <= MAXN; i++) { choose[i][0] = 1;
    for (int j = 1; j <= i; j++)
      choose[i][j] = choose[i - 1][j - 1] + ((i - 1 >= j) ? choose[i - 1][j] : 0);
  }
  int qm1p[MAXN + 1]; qm1p[0] = 1;
  for (int i = 1; i <= MAXN; i++) qm1p[i] = qm1p[i - 1] * (q - 1);
  int64_t powq[MAXN + 1]; powq[0] = 1;
  for (int i = 1; i <= MAXN; i++) powq[i] = powq[i - 1] * q;
  int sphere = choose[n][R] * qm1p[R];

  cudaMemcpyToSymbol(d_choose, choose, sizeof(choose));
  cudaMemcpyToSymbol(d_q, &q, 4);
  cudaMemcpyToSymbol(d_n, &n, 4);
  cudaMemcpyToSymbol(d_R, &R, 4);
  cudaMemcpyToSymbol(d_M, &M, 4);
  cudaMemcpyToSymbol(d_space, &space, 8);
  cudaMemcpyToSymbol(d_sphere_cnt, &sphere, 4);
  cudaMemcpyToSymbol(d_qm1_pow, qm1p, sizeof(qm1p));
  cudaMemcpyToSymbol(d_pow_q, powq, sizeof(powq));

  size_t codesz = (size_t)nchains * M * MAXN;
  uint8_t *code_d, *best_d; uint16_t *cnt_d; uint32_t *ul_d; ChainOut *out_d;
  if (cudaMalloc(&code_d, codesz)) return -1;
  if (cudaMalloc(&best_d, codesz)) return -2;
  if (cudaMalloc(&cnt_d, (size_t)nchains * space * 2)) return -3;
  if (cudaMalloc(&ul_d, (size_t)nchains * UCAP * 4)) return -4;
  if (cudaMalloc(&out_d, (size_t)nchains * sizeof(ChainOut))) return -5;
  cudaMemcpy(code_d, code_h, codesz, cudaMemcpyHostToDevice);
  cudaMemcpy(best_d, code_h, codesz, cudaMemcpyHostToDevice);

  size_t shmem = (size_t)M * MAXN + (MAXN + MAXN * MAXQ + 16 + MAXCAND) * 4;
  cudaFuncSetAttribute(chain_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)shmem + 1024);
  chain_kernel<<<nchains, threads, shmem>>>(iters, seed, code_d, best_d,
                                            cnt_d, ul_d, out_d);
  cudaError_t err = cudaDeviceSynchronize();
  int rc = 0;
  if (err != cudaSuccess) rc = -100 - (int)err;
  else {
    cudaMemcpy(best_h, best_d, codesz, cudaMemcpyDeviceToHost);
    ChainOut *o = new ChainOut[nchains];
    cudaMemcpy(o, out_d, (size_t)nchains * sizeof(ChainOut), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nchains; i++) {
      best_u_h[i] = o[i].best_uncovered;
      solved_h[i] = o[i].solved;
      iters_h[i] = o[i].iters_done;
    }
    delete[] o;
  }
  cudaFree(code_d); cudaFree(best_d); cudaFree(cnt_d); cudaFree(ul_d); cudaFree(out_d);
  return rc;
}
