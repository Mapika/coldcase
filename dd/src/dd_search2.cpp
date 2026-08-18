// dd_search.cpp -- degree/diameter record hunting via Cayley graphs of metacyclic groups.
//
// Group family:  G = Z_m  rtimes_a  Z_n,  |G| = N = m*n
//   elements (i,j), i in [0,m), j in [0,n); index = j*m + i
//   (i1,j1)*(i2,j2) = (i1 + a^{j1} i2 mod m,  j1 + j2 mod n),   a^n == 1 (mod m)
//   n = 1            -> cyclic Z_m       (circulant graphs)
//   a = 1            -> abelian Z_m x Z_n
//
// Cayley graph Cay(G,S) with S = S^{-1}, e not in S:
//   * undirected, |S|-regular, vertex-transitive
//   * VERTEX-TRANSITIVITY => diameter == eccentricity of the identity, so a single
//     ball-growth from e decides the diameter.  (left translation x -> gx is an
//     automorphism carrying e to g, and automorphisms preserve eccentricity.)
//
// Fast kernel: a subset of G is a bit-matrix of n rows x m bits (W 64-bit words/row).
// Right multiplication by s=(i2,j2) maps row j to row (j+j2 mod n) cyclically rotated
// by (a^j * i2 mod m).  So one generator application costs n*O(W) word ops instead of
// N*O(1) scalar ops -- roughly a 16-64x speedup over pointer-chasing BFS.
//
//   B_0 = {e};   B_k = B_{k-1}  U  B_{k-1}.S
//   diameter <= D  <=>  |B_D| == N
//
// Search: minimise f(S) = N - |B_D(S)| by hill-climbing/annealing on S
// (swap one inverse-closed generator item at a time).  f == 0 is a (Delta,D) graph
// of order N.
//
// Build: g++ -O3 -march=native -fopenmp -o dd_search dd_search.cpp

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <omp.h>

typedef uint64_t u64;

// ----------------------------------------------------------------- utilities
static long long gcdll(long long a, long long b) { while (b) { long long t = a % b; a = b; b = t; } return a; }

static int mulmod(int x, int y, int m) { return (int)(((long long)x * y) % m); }

static int powmod(int b, long long e, int m) {
    if (m == 1) return 0;
    long long r = 1, bb = b % m;
    while (e > 0) { if (e & 1) r = (r * bb) % m; bb = (bb * bb) % m; e >>= 1; }
    return (int)r;
}

static int inv_mod(int a, int m) {
    if (m == 1) return 0;
    long long g = m, x = 0, x1 = 1, a1 = a % m;
    long long old_r = a1, r = m, old_s = 1, s = 0;
    while (r != 0) { long long q = old_r / r; long long t = old_r - q * r; old_r = r; r = t; t = old_s - q * s; old_s = s; s = t; }
    if (old_r != 1) return -1;
    (void)g; (void)x; (void)x1;
    long long res = old_s % m; if (res < 0) res += m;
    return (int)res;
}

// ------------------------------------------------------------------- rotate
// dst |= cyclic-left-rotate(src, r) inside an m-bit vector held in W words.
static inline void rot_or(u64 *__restrict dst, const u64 *__restrict src,
                          int W, int m, u64 topmask, int r) {
    if (r == 0) { for (int i = 0; i < W; i++) dst[i] |= src[i]; return; }
    const int q = r >> 6, b = r & 63;
    const int s = m - r, q2 = s >> 6, b2 = s & 63;
    for (int i = W - 1; i >= 0; --i) {
        u64 v = 0;
        int k = i - q;
        if (k >= 0) {
            v = src[k] << b;
            if (b && k - 1 >= 0) v |= src[k - 1] >> (64 - b);
        }
        int k2 = i + q2;
        if (k2 < W) {
            u64 w = src[k2] >> b2;
            if (b2 && k2 + 1 < W) w |= src[k2 + 1] << (64 - b2);
            v |= w;
        }
        if (i == W - 1) v &= topmask;
        dst[i] |= v;
    }
}

// -------------------------------------------------------------------- group
// Generalised row model.  A group element is a pair (row, col):
//     index = row * M + col,   row in [0,R),  col in [0,M)
// and right multiplication by a fixed group element s is required to act as
//     (row, col)  ->  (destrow_s[row],  col + shift_s[row]  mod M)
// i.e. a permutation of the R rows combined with a cyclic rotation of each row.
// Two concrete models are provided:
//
//  MODEL_METACYCLIC   G = Z_m rtimes_a Z_n
//      R = n, M = m, element (i,j) -> row j col i
//      destrow_s[j] = (j + j2) mod n,   shift_s[j] = a^j * i2 mod m
//
//  MODEL_AFFINE2      G = (Z_p x Z_p) rtimes_A Z_n,  A in GL_2(Z_p) of order n
//      R = n*p, M = p,  element ((x,y), j) -> row j*p+x, col y
//      right-mult by ((x2,y2), j2):  (x,y,j) -> (x + (A^j v2)_0, y + (A^j v2)_1, j+j2)
//      destrow[j*p+x] = ((j+j2)%n)*p + (x + (A^j v2)_0) % p
//      shift[j*p+x]   = (A^j v2)_1
//
// Both keep the bitset kernel identical: one generator application costs R*O(W)
// word ops where W = ceil(M/64).
enum { MODEL_METACYCLIC = 0, MODEL_AFFINE2 = 1 };

struct Spec2 {
    int model;
    int m, n, a;          // metacyclic
    int p;                // affine2 prime
    int A[4];             // affine2 matrix, row-major
    int N;
};

struct Group {
    int model;
    int R, M, N, W;
    u64 topmask;
    // metacyclic
    int m, n, a;
    std::vector<int> apow;
    // affine2
    int p;
    std::vector<int> Apow;   // Apow[4*j .. 4*j+3] = A^j

    void init(const Spec2 &sp) {
        model = sp.model;
        if (model == MODEL_METACYCLIC) {
            m = sp.m; n = sp.n; a = sp.a;
            R = n; M = m; N = m * n;
            apow.resize(n);
            apow[0] = 1 % m;
            for (int j = 1; j < n; j++) apow[j] = mulmod(apow[j - 1], a, m);
        } else {
            p = sp.p; n = sp.n; a = 0; m = p;
            R = n * p; M = p; N = p * p * n;
            Apow.assign(4 * n, 0);
            Apow[0] = 1 % p; Apow[1] = 0; Apow[2] = 0; Apow[3] = 1 % p;
            for (int j = 1; j < n; j++) {
                const int *X = &Apow[4 * (j - 1)];
                int *Y = &Apow[4 * j];
                Y[0] = (int)(((long long)X[0] * sp.A[0] + (long long)X[1] * sp.A[2]) % p);
                Y[1] = (int)(((long long)X[0] * sp.A[1] + (long long)X[1] * sp.A[3]) % p);
                Y[2] = (int)(((long long)X[2] * sp.A[0] + (long long)X[3] * sp.A[2]) % p);
                Y[3] = (int)(((long long)X[2] * sp.A[1] + (long long)X[3] * sp.A[3]) % p);
            }
        }
        W = (M + 63) / 64;
        int rem = M - 64 * (W - 1);
        topmask = (rem == 64) ? ~0ULL : ((1ULL << rem) - 1ULL);
    }

    // multiply e1*e2 (used only for building the item pool / inverses)
    int mul(int e1, int e2) const {
        if (model == MODEL_METACYCLIC) {
            int i1 = e1 % m, j1 = e1 / m, i2 = e2 % m, j2 = e2 / m;
            return (int)((i1 + (long long)apow[j1] * i2) % m) + m * ((j1 + j2) % n);
        } else {
            int y1 = e1 % p, r1 = e1 / p, x1 = r1 % p, j1 = r1 / p;
            int y2 = e2 % p, r2 = e2 / p, x2 = r2 % p, j2 = r2 / p;
            const int *X = &Apow[4 * j1];
            int nx = (int)(((long long)X[0] * x2 + (long long)X[1] * y2 + x1) % p);
            int ny = (int)(((long long)X[2] * x2 + (long long)X[3] * y2 + y1) % p);
            int nj = (j1 + j2) % n;
            return (nj * p + nx) * p + ny;
        }
    }
    int inverse(int e) const {                     // closed form, O(1)
        if (model == MODEL_METACYCLIC) {
            int i = e % m, j = e / m;
            int jn = (n - j) % n;                  // a^{-j} = a^{n-j}
            int ii = mulmod((m - i % m) % m, apow[jn], m);
            return ii + m * jn;
        } else {
            int y = e % p, rr = e / p, x = rr % p, j = rr / p;
            int jn = (n - j) % n;
            const int *X = &Apow[4 * jn];          // A^{-j} = A^{n-j}
            int nx = (int)((p - ((long long)X[0] * x + (long long)X[1] * y) % p) % p);
            int ny = (int)((p - ((long long)X[2] * x + (long long)X[3] * y) % p) % p);
            return (jn * p + nx) * p + ny;
        }
    }
};

struct Gen {                       // precomputed right-multiplication descriptor
    int elem;
    std::vector<int> destrow;      // size R
    std::vector<int> shift;        // size R
};

static Gen make_gen(const Group &G, int e) {
    Gen g; g.elem = e;
    g.destrow.resize(G.R); g.shift.resize(G.R);
    if (G.model == MODEL_METACYCLIC) {
        int i = e % G.m, j = e / G.m;
        for (int r = 0; r < G.R; r++) {
            int dr = r + j; if (dr >= G.n) dr -= G.n;
            g.destrow[r] = dr;
            g.shift[r] = mulmod(G.apow[r], i, G.m);
        }
    } else {
        const int p = G.p, n = G.n;
        int y2 = e % p, rr = e / p, x2 = rr % p, j2 = rr / p;
        for (int j = 0; j < n; j++) {
            const int *X = &G.Apow[4 * j];
            int dx = (int)(((long long)X[0] * x2 + (long long)X[1] * y2) % p);
            int dy = (int)(((long long)X[2] * x2 + (long long)X[3] * y2) % p);
            int nj = (j + j2) % n;
            for (int x = 0; x < p; x++) {
                g.destrow[j * p + x] = nj * p + (x + dx) % p;
                g.shift[j * p + x] = dy;
            }
        }
    }
    return g;
}

// ---------------------------------------------------------------- ball eval
struct Workspace {
    std::vector<u64> cur, nxt;
    void resize(int R, int W) { cur.assign((size_t)R * W, 0); nxt.assign((size_t)R * W, 0); }
};

static long long ball_size(const Group &G, const std::vector<Gen> &S, int D,
                           Workspace &ws, long long *prev_out) {
    const int R = G.R, W = G.W;
    const size_t sz = (size_t)R * W;
    std::fill(ws.cur.begin(), ws.cur.end(), 0ULL);
    ws.cur[0] = 1ULL;                       // identity is index 0 in both models
    long long prev = 1, cnt = 1;
    for (int lvl = 1; lvl <= D; lvl++) {
        prev = cnt;
        memcpy(ws.nxt.data(), ws.cur.data(), sz * sizeof(u64));
        for (size_t gi = 0; gi < S.size(); gi++) {
            const Gen &g = S[gi];
            const int *dr = g.destrow.data();
            const int *sh = g.shift.data();
            for (int r = 0; r < R; r++)
                rot_or(&ws.nxt[(size_t)dr[r] * W], &ws.cur[(size_t)r * W], W, G.M, G.topmask, sh[r]);
        }
        ws.cur.swap(ws.nxt);
        cnt = 0;
        for (size_t i = 0; i < sz; i++) cnt += __builtin_popcountll(ws.cur[i]);
        if (cnt == G.N && lvl < D) { if (prev_out) *prev_out = (lvl == D - 1) ? cnt : G.N; return G.N; }
    }
    if (prev_out) *prev_out = prev;
    return cnt;
}

// --------------------------------------------------------------- group list
static bool is_prime(int x) { if (x < 2) return false; for (int d = 2; (long long)d * d <= x; d++) if (x % d == 0) return false; return true; }

// order of a 2x2 matrix over Z_p (returns 0 if singular or order > bound)
static int mat_order(const int *A, int p, int bound) {
    int C[4] = {A[0], A[1], A[2], A[3]};
    for (int k = 1; k <= bound; k++) {
        if (C[0] == 1 % p && C[3] == 1 % p && C[1] == 0 && C[2] == 0) return k;
        int T[4];
        T[0] = (int)(((long long)C[0] * A[0] + (long long)C[1] * A[2]) % p);
        T[1] = (int)(((long long)C[0] * A[1] + (long long)C[1] * A[3]) % p);
        T[2] = (int)(((long long)C[2] * A[0] + (long long)C[3] * A[2]) % p);
        T[3] = (int)(((long long)C[2] * A[1] + (long long)C[3] * A[3]) % p);
        memcpy(C, T, sizeof(T));
    }
    return 0;
}

static std::vector<Spec2> enum_specs(int N, int max_a_per_mn, std::mt19937_64 &rng,
                                     int min_m, int max_n, int min_n, int nonab, int faithful,
                                     int models) {
    std::vector<Spec2> out;
    if (models & 1) {                                   // --- metacyclic ---
        for (int n = 1; n <= N; n++) {
            if (N % n) continue;
            if (n > max_n || n < min_n) continue;
            int m = N / n;
            if (m < min_m) continue;
            std::vector<int> cands;
            for (int a = 1; a < m; a++) {
                if (gcdll(a, m) != 1) continue;
                if (powmod(a, n, m) != 1 % m) continue;
                cands.push_back(a);
            }
            if (m == 1) cands.assign(1, 0);
            if (nonab) {
                std::vector<int> f;
                for (int a : cands) {
                    if (a == 1 % m) continue;
                    if (faithful) {
                        bool full = true;
                        for (int q = 2; q <= n; q++) {
                            if (n % q) continue;
                            if (!is_prime(q)) continue;
                            if (powmod(a, n / q, m) == 1 % m) { full = false; break; }
                        }
                        if (!full) continue;
                    }
                    f.push_back(a);
                }
                cands.swap(f);
            }
            if (cands.empty()) continue;
            if ((int)cands.size() > max_a_per_mn) {
                std::shuffle(cands.begin(), cands.end(), rng);
                bool has1 = false; for (int k = 0; k < max_a_per_mn; k++) if (cands[k] == 1) has1 = true;
                cands.resize(max_a_per_mn);
                if (!has1 && !nonab) cands[0] = 1;
            }
            for (int a : cands) { Spec2 sp{}; sp.model = MODEL_METACYCLIC; sp.m = m; sp.n = n; sp.a = a; sp.N = N; out.push_back(sp); }
        }
    }
    if (models & 2) {                                   // --- (Z_q x Z_q) : Z_n ---
        for (int p = 2; (long long)p * p <= N; p++) {   // q need not be prime; A must lie in GL_2(Z_q)
            if (N % (p * p)) continue;
            int n = N / (p * p);
            if (n > max_n || n < min_n) continue;
            if (n < 2 || n > 2000) continue;        // keep spec enumeration cheap
            if ((long long)n > 4LL * p * p * p * p) continue;   // |GL_2(Z_p)| < p^4
            // sample matrices of order exactly n
            int kept = 0;
            for (int tries = 0; tries < 3000 && kept < max_a_per_mn; tries++) {
                Spec2 sp{}; sp.model = MODEL_AFFINE2; sp.p = p; sp.n = n; sp.N = N;
                for (int z = 0; z < 4; z++) sp.A[z] = (int)(rng() % (u64)p);
                long long det = ((long long)sp.A[0] * sp.A[3] - (long long)sp.A[1] * sp.A[2]) % p;
                if (det < 0) det += p;
                if (gcdll(det, p) != 1) continue;        // A invertible over Z_q
                if (mat_order(sp.A, p, n) != n) continue;
                out.push_back(sp); kept++;
            }
        }
    }
    return out;
}

// --------------------------------------------------------------- item lists
// An "item" is an inverse-closed chunk of the connection set:
//   pair       : {g, g^-1}, contributes 2 to the degree
//   involution : {g} with g^2 = e, contributes 1
struct ItemPool {
    std::vector<int> pair_rep;   // representative g (g < g^{-1})
    std::vector<int> invol;
};

static ItemPool build_items(const Group &G) {
    ItemPool P;
    for (int e = 1; e < G.N; e++) {
        int ei = G.inverse(e);
        if (ei == e) P.invol.push_back(e);
        else if (e < ei) P.pair_rep.push_back(e);
    }
    return P;
}

// ------------------------------------------------------------------- result
struct Hit {
    int model, m, n, a, p, A0, A1, A2, A3;
    int N, delta, D;
    std::vector<int> S;
};

// ---------------------------------------------------------------------- main
struct Args {
    int delta = 0, D = 0;
    int Nmin = 0, Nmax = 0;
    double seconds = 60;
    int threads = 0;
    u64 seed = 12345;
    int iters = 4000;          // local-search iterations per restart
    int max_a = 12;            // max a-values per (m,n)
    int min_m = 1;
    int max_n = 1 << 30;
    int min_n = 1;
    int nonab = 0;        // require a != 1 (skip abelian groups)
    int faithful = 0;     // require ord(a) == n
    int models = 3;       // bit 0 = metacyclic, bit 1 = affine2
    int stop_on_hit = 1;
    int stall = 300;           // non-improving moves before a kick
    int kicks = 60;            // kicks before abandoning a restart
    int verbose = 0;
    int max_invol = -1;   // -1 -> allow up to delta involutions
    std::string out;
    std::string evalspec;   // "m,n,a:s1,s2,..." -> print |B_k| and exit
    std::string evalspec2;  // affine2: "p,n,A0,A1,A2,A3:s1,s2,..."
};

static void usage() {
    fprintf(stderr,
        "usage: dd_search --delta D --diam D --Nmin N [--Nmax N] [--time S] [--threads T]\n"
        "                 [--seed S] [--iters I] [--maxa A] [--minm M] [--maxn N]\n"
        "                 [--out FILE] [--nostop] [--verbose]\n");
}

int main(int argc, char **argv) {
    Args A;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto nx = [&]() { return atoi(argv[++i]); };
        if (k == "--delta") A.delta = nx();
        else if (k == "--diam" || k == "--D") A.D = nx();
        else if (k == "--Nmin") A.Nmin = nx();
        else if (k == "--Nmax") A.Nmax = nx();
        else if (k == "--time") A.seconds = atof(argv[++i]);
        else if (k == "--threads") A.threads = nx();
        else if (k == "--seed") A.seed = strtoull(argv[++i], 0, 10);
        else if (k == "--iters") A.iters = nx();
        else if (k == "--stall") A.stall = nx();
        else if (k == "--kicks") A.kicks = nx();
        else if (k == "--minn") A.min_n = nx();
        else if (k == "--nonab") A.nonab = 1;
        else if (k == "--faithful") { A.faithful = 1; A.nonab = 1; }
        else if (k == "--models") A.models = nx();
        else if (k == "--maxa") A.max_a = nx();
        else if (k == "--minm") A.min_m = nx();
        else if (k == "--maxn") A.max_n = nx();
        else if (k == "--maxinvol") A.max_invol = nx();
        else if (k == "--out") A.out = argv[++i];
        else if (k == "--eval") A.evalspec = argv[++i];
        else if (k == "--eval2") A.evalspec2 = argv[++i];
        else if (k == "--nostop") A.stop_on_hit = 0;
        else if (k == "--verbose") A.verbose = 1;
        else { usage(); return 1; }
    }
    if (!A.evalspec2.empty()) {                    // affine2 single-set evaluation
        char *p = (char *)A.evalspec2.c_str();
        Spec2 sp{}; sp.model = MODEL_AFFINE2;
        sp.p = strtol(p, &p, 10); p++;
        sp.n = strtol(p, &p, 10); p++;
        for (int z = 0; z < 4; z++) { sp.A[z] = strtol(p, &p, 10); p++; }
        sp.N = sp.p * sp.p * sp.n;
        std::vector<int> ss;
        while (*p) { if (*p == ':' || *p == ',') p++; else ss.push_back(strtol(p, &p, 10)); }
        Group G; G.init(sp);
        std::vector<Gen> S; for (int e : ss) S.push_back(make_gen(G, e));
        Workspace ws; ws.resize(G.R, G.W);
        printf("N=%d deg=%zu", G.N, S.size());
        int DD = A.D ? A.D : 8;
        for (int d = 1; d <= DD; d++) { long long pr = 0; printf(" |B%d|=%lld", d, ball_size(G, S, d, ws, &pr)); }
        printf("\n");
        return 0;
    }
    if (!A.evalspec.empty()) {                     // exact single-set evaluation
        char *p = (char *)A.evalspec.c_str();
        int m = strtol(p, &p, 10); p++;
        int n = strtol(p, &p, 10); p++;
        int a = strtol(p, &p, 10);
        std::vector<int> ss;
        while (*p) { if (*p == ':' || *p == ',') p++; else ss.push_back(strtol(p, &p, 10)); }
        Spec2 sp{}; sp.model = MODEL_METACYCLIC; sp.m = m; sp.n = n; sp.a = a; sp.N = m * n;
        Group G; G.init(sp);
        std::vector<Gen> S; for (int e : ss) S.push_back(make_gen(G, e));
        Workspace ws; ws.resize(G.R, G.W);
        printf("N=%d deg=%zu", G.N, S.size());
        int DD = A.D ? A.D : 8;
        for (int d = 1; d <= DD; d++) { long long pr = 0; printf(" |B%d|=%lld", d, ball_size(G, S, d, ws, &pr)); }
        printf("\n");
        return 0;
    }
    if (!A.delta || !A.D || !A.Nmin) { usage(); return 1; }
    if (!A.Nmax) A.Nmax = A.Nmin;
    if (A.max_invol < 0) A.max_invol = A.delta;
    if (A.threads) omp_set_num_threads(A.threads);

    std::mt19937_64 grng(A.seed);

    // Build the full task list: (N, spec) pairs.
    struct Task { Spec2 s; int N; };
    std::vector<Task> tasks;
    for (int N = A.Nmax; N >= A.Nmin; N--) {
        if (N < A.delta + 1) continue;              // need at least delta+1 vertices
        if ((A.delta & 1) && (N & 1)) continue;     // handshake: odd degree needs even order
        auto sp = enum_specs(N, A.max_a, grng, A.min_m, A.max_n, A.min_n, A.nonab, A.faithful, A.models);
        for (auto &s : sp) tasks.push_back({s, N});
    }
    if (tasks.empty()) { fprintf(stderr, "no groups in range\n"); return 2; }
    fprintf(stderr, "[dd_search] delta=%d D=%d N=%d..%d  %zu group specs, %.1fs, %d threads\n",
            A.delta, A.D, A.Nmin, A.Nmax, tasks.size(), A.seconds, omp_get_max_threads());

    double t0 = omp_get_wtime();
    volatile int found = 0;
    long long total_evals = 0;
    std::vector<Hit> hits;
    long long best_f = 1LL << 60; int best_N = 0; Spec2 best_spec{};
    const int NSPAN = A.Nmax - A.Nmin + 1;
    std::vector<long long> per_f((size_t)NSPAN, 1LL << 60);
    std::vector<Spec2> per_spec((size_t)NSPAN, Spec2{});
    std::vector<long long> per_evals((size_t)NSPAN, 0);

#pragma omp parallel reduction(+ : total_evals)
    {
        std::mt19937_64 rng(A.seed * 0x9E3779B97F4A7C15ULL + 1315423911ULL * (omp_get_thread_num() + 1));
        Workspace ws;
        Group G;
        std::vector<Gen> S;
        long long my_evals = 0;
        long long my_best_f = 1LL << 60; int my_best_N = 0; Spec2 my_best_spec{};
        size_t cursor = (size_t)(rng() % tasks.size());

        while (!(found && A.stop_on_hit)) {
            if (omp_get_wtime() - t0 > A.seconds) break;
            const Task &T = tasks[cursor % tasks.size()];
            cursor += 1 + (rng() % 7);
            G.init(T.s);
            if (G.N != T.N) continue;
            ws.resize(G.R, G.W);
            ItemPool P = build_items(G);

            // choose the involution count t: 2k + t = delta
            int tmax = std::min({A.max_invol, A.delta, (int)P.invol.size()});
            std::vector<int> tchoices;
            for (int t = A.delta & 1; t <= tmax; t += 2) tchoices.push_back(t);
            if (tchoices.empty()) continue;
            int t = tchoices[rng() % tchoices.size()];
            int k = (A.delta - t) / 2;
            if (k > (int)P.pair_rep.size()) continue;
            if (k == 0 && t == 0) continue;

            // random initial item selection (distinct)
            std::vector<int> pidx, iidx;
            {
                std::vector<int> perm(P.pair_rep.size());
                for (size_t z = 0; z < perm.size(); z++) perm[z] = (int)z;
                for (int z = 0; z < k; z++) { int r = z + (int)(rng() % (perm.size() - z)); std::swap(perm[z], perm[r]); pidx.push_back(perm[z]); }
                std::vector<int> permi(P.invol.size());
                for (size_t z = 0; z < permi.size(); z++) permi[z] = (int)z;
                for (int z = 0; z < t; z++) { int r = z + (int)(rng() % (permi.size() - z)); std::swap(permi[z], permi[r]); iidx.push_back(permi[z]); }
            }

            auto build_S = [&](std::vector<Gen> &out) {
                out.clear();
                for (int z : pidx) { out.push_back(make_gen(G, P.pair_rep[z])); out.push_back(make_gen(G, G.inverse(P.pair_rep[z]))); }
                for (int z : iidx) out.push_back(make_gen(G, P.invol[z]));
            };

            auto score = [&](long long &f_out) -> long long {
                long long prev = 0;
                long long b = ball_size(G, S, A.D, ws, &prev);
                my_evals++;
                long long f = G.N - b;
                f_out = f;
                return f * (long long)(G.N + 1) + (G.N - prev);
            };

            build_S(S);
            long long f_cur = 0;
            long long sc = score(f_cur);

            const bool pair_movable = (k > 0 && (int)P.pair_rep.size() > k);
            const bool inv_movable  = (t > 0 && (int)P.invol.size() > t);
            if (!pair_movable && !inv_movable) { if (f_cur) continue; }

            // ---- iterated local search: hill-climb, then kick the incumbent ----
            std::vector<int> bp = pidx, bi = iidx;
            long long bsc = sc, bf = f_cur;
            int stall = 0, kicks = 0;

            auto replace_random = [&](int q) {
                for (int z = 0; z < q; z++) {
                    bool up = pair_movable && (!inv_movable || (rng() & 3));
                    if (up) {
                        int sl = (int)(rng() % k);
                        for (int tr = 0; tr < 64; tr++) {
                            int nw = (int)(rng() % P.pair_rep.size());
                            if (std::find(pidx.begin(), pidx.end(), nw) == pidx.end()) { pidx[sl] = nw; break; }
                        }
                    } else if (inv_movable) {
                        int sl = (int)(rng() % t);
                        for (int tr = 0; tr < 64; tr++) {
                            int nw = (int)(rng() % P.invol.size());
                            if (std::find(iidx.begin(), iidx.end(), nw) == iidx.end()) { iidx[sl] = nw; break; }
                        }
                    }
                }
            };

            for (int it = 0; it < A.iters && f_cur > 0; it++) {
                if ((it & 127) == 0 && omp_get_wtime() - t0 > A.seconds) break;
                if (!pair_movable && !inv_movable) break;
                bool use_pair = pair_movable && (!inv_movable || (rng() & 3));
                int slot = -1, old = -1, nw = -1;
                bool ok = false;
                if (use_pair) {
                    slot = (int)(rng() % k);
                    old = pidx[slot];
                    for (int tries = 0; tries < 64; tries++) {
                        nw = (int)(rng() % P.pair_rep.size());
                        if (std::find(pidx.begin(), pidx.end(), nw) == pidx.end()) { ok = true; break; }
                    }
                    if (!ok) continue;
                    pidx[slot] = nw;
                } else {
                    slot = (int)(rng() % t);
                    old = iidx[slot];
                    for (int tries = 0; tries < 64; tries++) {
                        nw = (int)(rng() % P.invol.size());
                        if (std::find(iidx.begin(), iidx.end(), nw) == iidx.end()) { ok = true; break; }
                    }
                    if (!ok) continue;
                    iidx[slot] = nw;
                }

                build_S(S);
                long long f_new = 0;
                long long sc_new = score(f_new);
                bool accept = (sc_new < sc) || (sc_new == sc && (rng() & 1));
                if (accept) { sc = sc_new; f_cur = f_new; }
                else if (use_pair) pidx[slot] = old; else iidx[slot] = old;

                if (sc < bsc) { bsc = sc; bf = f_cur; bp = pidx; bi = iidx; stall = 0; }
                else if (++stall > A.stall) {
                    if (++kicks > A.kicks) break;          // abandon this restart
                    pidx = bp; iidx = bi;                  // return to incumbent ...
                    replace_random(1 + (int)(rng() % 3));  // ... and kick it
                    build_S(S);
                    sc = score(f_cur);
                    stall = 0;
                }
            }
            if (bf < f_cur) { pidx = bp; iidx = bi; f_cur = bf; }
            build_S(S);

            if (f_cur == 0) {
#pragma omp critical
                {
                    Hit h; h.model = G.model; h.m = G.m; h.n = G.n; h.a = G.a; h.p = G.p;
                    h.A0 = T.s.A[0]; h.A1 = T.s.A[1]; h.A2 = T.s.A[2]; h.A3 = T.s.A[3];
                    h.N = G.N; h.delta = A.delta; h.D = A.D;
                    for (auto &g : S) h.S.push_back(g.elem);
                    hits.push_back(h);
                    found = 1;
                }
            }
            if (f_cur < my_best_f) { my_best_f = f_cur; my_best_N = T.N; my_best_spec = T.s; }
            {
                size_t bi2 = (size_t)(T.N - A.Nmin);
                if (f_cur < per_f[bi2]) {
#pragma omp critical(pern)
                    if (f_cur < per_f[bi2]) { per_f[bi2] = f_cur; per_spec[bi2] = T.s; }
                }
            }
        }
#pragma omp critical
        {
            if (my_best_f < best_f) { best_f = my_best_f; best_N = my_best_N; best_spec = my_best_spec; }
        }
        total_evals += my_evals;
    }

    double dt = omp_get_wtime() - t0;
    fprintf(stderr, "[dd_search] %.2fs  %lld evals  %.3f Mevals/s  best f=%lld at N=%d (m=%d n=%d a=%d)\n",
            dt, total_evals, total_evals / dt / 1e6, best_f, best_N,
            best_spec.model ? best_spec.p : best_spec.m, best_spec.n,
            best_spec.model ? -1 : best_spec.a);

    if (A.verbose) {
        for (int i = NSPAN - 1; i >= 0; i--) {
            if (per_f[i] == (1LL << 60)) continue;
            fprintf(stderr, "  N=%-8d best_f=%-8lld  (m=%d n=%d a=%d)\n",
                    A.Nmin + i, per_f[i],
                    per_spec[i].model ? per_spec[i].p : per_spec[i].m, per_spec[i].n,
                    per_spec[i].model ? -1 : per_spec[i].a);
        }
    }
    FILE *out = stdout;
    if (!A.out.empty()) { out = fopen(A.out.c_str(), "a"); if (!out) { perror("out"); return 3; } }
    for (auto &h : hits) {
        if (h.model == MODEL_METACYCLIC)
            fprintf(out, "{\"model\":\"metacyclic\",\"delta\":%d,\"D\":%d,\"N\":%d,\"m\":%d,\"n\":%d,\"a\":%d,\"S\":[",
                    h.delta, h.D, h.N, h.m, h.n, h.a);
        else
            fprintf(out, "{\"model\":\"affine2\",\"delta\":%d,\"D\":%d,\"N\":%d,\"p\":%d,\"n\":%d,\"A\":[%d,%d,%d,%d],\"S\":[",
                    h.delta, h.D, h.N, h.p, h.n, h.A0, h.A1, h.A2, h.A3);
        for (size_t z = 0; z < h.S.size(); z++) fprintf(out, "%s%d", z ? "," : "", h.S[z]);
        fprintf(out, "]}\n");
    }
    if (out != stdout) fclose(out);
    return hits.empty() ? 10 : 0;
}
