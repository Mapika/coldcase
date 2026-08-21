# gemm2 — the "fat" GPU formulation: chain-batched SIMT, no tensor cores

## 1. Analysis first: why direction A, and why NOT tensor cores

Starting facts from cov/gemm/NOTES.md (not re-derived): (i) one-hot GEMM is
stuck at K = n·q ≈ 50 where tensor cores deliver 6% of peak — the 2000-TOPS
premise is off ~30× by construction; (ii) the GEMM route spends ~7,000
elementary ops per exact move delta vs ~1,100 for the CPU shared-sphere
evaluator; (iii) neighbourhood breadth pays logarithmically (1514× wider
bought 1.2–3.4×); (iv) host-sync latency dominated the previous GPU loop.

Consequences, in order:
- (ii)+(iii) kill every tensor-core formulation *regardless of K*: even a
  free GEMM evaluator wins at most a small constant, and it starts 6× behind
  on op count. Direction B (fat-K aggregation stages) optimizes stages of a
  pipeline that loses upstream. Direction C (bit-planes) raises GEMM
  throughput at most ~4× on paper — less than the 6× op deficit — and sm_90
  1-bit MMA is not reachable from this torch stack without a CUTLASS build.
  Both rejected on arithmetic, before implementation.
- The GPU's *actual* comparative advantages for this workload are massive
  latency hiding on scattered 1-2 byte reads (the CPU is latency-bound: one
  core sustains only ~18.6 it/s on K8(9,4) against a 500 GB/s socket
  ceiling) and 8× the CPU's DRAM bandwidth (4 TB/s HBM3 vs 500 GB/s LPDDR).
- Therefore direction A, taken literally: run the *winning CPU algorithm*
  (p5b: focused search, shared-sphere leave-side, uncovered-list
  enter-side) as hundreds of resident chains, one per thread block, with
  ZERO host synchronization — the entire search loop lives in one kernel.
  This multiplies the thing the arena proved valuable (independent chains)
  instead of widening neighbourhoods (proved near-worthless).

## 2. What was built

`chainsolve.cu` (+ `chain.py` ctypes driver): one block = one chain.
Per-chain state in global memory: exact uint16 multiplicity array cnt[q^n],
lazy-deleted uncovered list (cap 2^20), codewords mirrored in dynamic shared
memory. Iteration: thread 0 samples an uncovered word u (stale entries
swap-deleted; O(q^n) rebuild fallback); block collects codewords at distance
R+1 from u (≤24 candidates); for each candidate the block evaluates ALL
n(q−1) moves via the shared-sphere identity (T − H_p, per-thread partial
H[16] in registers, one shared reduction) plus enter-side gains from the
uncovered list; global best move selected by a single 64-bit atomicMin on
(key‖move) — race-free; commit walks the two balls (distinct-word
enumerations, plain RMW, one __syncthreads between phases) and maintains the
exact uncovered count from transition counts measured inside the walks.
Best-code snapshots on every improvement. No host contact until the budget
expires or the chain solves.

Correctness: every returned best code is recounted on CPU (independent
dilation) — exact match on all chains in every run performed (8+64+64+
64+256+512 chains); solved codes pass ../verify_cov.py. The uncovered count
never drifts because it is derived from the same walks that mutate cnt.

## 3. Measured (ALL numbers on a GPU 100% occupied by another tenant's job)

Cold random starts, METHODS.md-style targets:

| probe | result |
|---|---|
| K6(6,3)@41, warm start (incumbent −3 words), 8 chains | 5/8 SOLVED in 11–515 iterations, 5.7 s wall incl. init |
| K6(6,3)@41 cold, 64 chains × 30k iters | 33k agg it/s; median best_u 28 (target 30 reached), min 2 |
| K6(8,4)@169 cold, 64 chains × 4k iters | 1.6k agg it/s; median best_u 11 (target 20 beaten), min 1 |
| chain scaling K6(6,3), 6k iters | 64ch: 616 it/s/chain; 256ch: 645 (LINEAR, no loss); 512ch: 320 (saturated) |

Contended plateau: ~165k agg it/s at 256 chains on K6(6,3).

CPU reference (cov/opt tables, p5b, one Grace core): K6(6,3) 15.3k it/s,
K6(8,4) 2.0k it/s, K8(9,4) 18.6 it/s; socket ≈ 64×.

## 4. Honest verdict and idle projection

- Contended, small/mid cells: the GPU aggregate is ~1/6 of the CPU socket
  (165k vs ~980k it/s on K6(6,3); ~6k-at-256-chains vs 128k on K6(8,4)).
  Per-iteration work is not identical (this kernel evaluates up to 24
  candidate codewords per iteration), so it/s comparisons are indicative,
  not exact.
- Idle projection (co-tenant cost measured at 2–8× by the prior agent, and
  our blocks currently share SMs with a 100%-util job): ×2–8 → rough parity
  with the socket on small/mid cells. Parity on cells the CPU already owns
  is not a win.
- The undecided and most promising case is the big-cell regime (K8(9,4),
  q^n=134M): the CPU core is latency-crushed there (18.6 it/s) while this
  design hides latency across ~256×256 threads and has 8× the bandwidth.
  Memory per chain (uint16: 268 MB) capped chains at ~37 under the
  contended 10 GB budget, so this was NOT measured; idle (97 GB) allows
  256+ chains. Bandwidth model: commit+eval traffic ≈ 1 MB/iter → 4 TB/s
  supports ~4M scattered-read-limited iter-equivalents/s aggregate vs the
  socket's ~1.2k measured — even at 1% efficiency the GPU should win this
  regime by an order of magnitude. That is a projection, clearly labeled.
- Known 2–8× headroom in the kernel itself, unimplemented: the CPU's own
  p5b lesson (narrow read path — a u8/2-bit read shadow instead of uint16
  reads), plus precomputed pattern-offset tables instead of per-pattern
  unranking (divergence + integer div/mod in the hot loop).

Bottom line: tensor cores are the wrong unit for this problem (confirmed,
now for the fat formulations too — rejected on arithmetic, not vibes);
chain-batched SIMT is the right GPU shape, achieves contended ~1/6-socket
today with linear chain scaling and exactness intact, projects to parity on
small cells and a likely order-of-magnitude win on the big-cell regime where
the CPU is weakest — exactly the monster cells at the end of the sweep
queue. Run the idle-GPU benchmark (esp. K8(9,4) at 256 chains) when the
co-tenant's job ends before deciding production use.
