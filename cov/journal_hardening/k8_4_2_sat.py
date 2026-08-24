#!/usr/bin/env python3
"""K8(4,2) at M<=22 as a SAT decision problem (pysat + CaDiCaL).

Encoding:
  - one Boolean per word of Z_8^4 (x_w = "w is a codeword"), 4096 vars;
  - covering clause OR_{w in B_2(v)} x_w for every word v (4096 clauses,
    323 literals each);
  - cardinality sum x <= 22 (pysat CardEnc, seqcounter);
  - x_0 = 1: sound because Hamming distance is translation-invariant, so any
    cover translates to an equal-size cover containing the zero word;
  - OPTIONAL lex-leader symmetry breaking (--sym): for a set of generators g
    of the stabilizer of the zero word inside Aut(Z_8^4, Hamming) =
    (S_8 wr S_4), add X <=lex g(X).  Soundness for BOTH answers: the set of
    covers of size <=22 containing 0 is closed under the stabilizer, so if it
    is nonempty its lex-min member satisfies every added constraint; if the
    strengthened formula is UNSAT, the original (with x_0=1) is UNSAT too.
    Generators used (all fix the zero word):
      * coordinate transposition (0 1) and 4-cycle (0 1 2 3);
      * in each coordinate: symbol transposition (1 2) and 7-cycle
        (1 2 3 4 5 6 7)  (symbol 0 fixed).

UNSAT  => K8(4,2) = 23 (new exact value).
SAT    => a 22-word cover; verified in-process, written to --out.
"""
import argparse
import sys
import time

import numpy as np

Q, N, R = 8, 4, 2
V = Q ** N


def all_words():
    w = np.arange(V, dtype=np.int64)
    dig = np.empty((V, N), dtype=np.int8)
    for i in range(N):
        dig[:, N - 1 - i] = (w // Q ** i) % Q
    return dig


def ball_matrix():
    dig = all_words()
    d = np.zeros((V, V), dtype=np.int8)
    for i in range(N):
        d += (dig[:, None, i] != dig[None, :, i])
    return d <= R


def word_perm_from_coord_perm(perm):
    """word -> word with coordinates permuted: new[i] = old[perm[i]]."""
    dig = all_words()
    nd = dig[:, perm]
    w = np.zeros(V, dtype=np.int64)
    for i in range(N):
        w = w * Q + nd[:, i]
    return w


def word_perm_from_symbol_perm(coord, sperm):
    """word -> word with symbol permutation applied in one coordinate."""
    dig = all_words().copy()
    sp = np.array(sperm, dtype=np.int8)
    dig[:, coord] = sp[dig[:, coord]]
    w = np.zeros(V, dtype=np.int64)
    for i in range(N):
        w = w * Q + dig[:, i]
    return w


def stabilizer_generators():
    gens = []
    # coordinate permutations
    gens.append(("coord (0 1)", word_perm_from_coord_perm([1, 0, 2, 3])))
    gens.append(("coord (0123)", word_perm_from_coord_perm([1, 2, 3, 0])))
    # symbol permutations fixing 0, per coordinate
    swap12 = [0, 2, 1, 3, 4, 5, 6, 7]
    cyc7 = [0, 2, 3, 4, 5, 6, 7, 1]
    for c in range(N):
        gens.append(("sym c%d (1 2)" % c, word_perm_from_symbol_perm(c, swap12)))
        gens.append(("sym c%d (12..7)" % c, word_perm_from_symbol_perm(c, cyc7)))
    return gens


def check_gen_is_isometry(A, perm):
    """A[v,w] must equal A[perm[v],perm[w]] (spot-check rows)."""
    idx = np.random.default_rng(1).integers(0, V, 40)
    for v in idx:
        # isometry <=> A[perm[v], perm[w]] == A[v, w] for all w
        if not np.array_equal(A[perm[v]][perm], A[v]):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tl", type=float, default=28800.0)
    ap.add_argument("--sym", action="store_true")
    ap.add_argument("--card", type=int, default=22)
    ap.add_argument("--out", default=None)
    ap.add_argument("--solver", default="cd19")
    a = ap.parse_args()

    from pysat.formula import CNF, IDPool
    from pysat.card import CardEnc, EncType
    from pysat.solvers import Solver

    A = ball_matrix()
    assert A[0].sum() == 323
    pool = IDPool()
    xv = [pool.id(("x", w)) for w in range(V)]

    cnf = CNF()
    for v in range(V):
        cnf.append([xv[w] for w in np.flatnonzero(A[v])])
    cnf.append([xv[0]])  # x_0 = 1 (translation argument)
    card = CardEnc.atmost(lits=xv, bound=a.card, vpool=pool,
                          encoding=EncType.seqcounter)
    cnf.extend(card.clauses)

    nsym = 0
    if a.sym:
        for name, perm in stabilizer_generators():
            assert perm[0] == 0, name  # must stabilize the zero word
            assert check_gen_is_isometry(A, perm), name
            # X <=lex X∘perm on positions 0..V-1, skipping fixed points
            prev = None  # aux var: prefix equal so far
            for t in range(V):
                u, w = xv[t], xv[int(perm[t])]
                if u == w:
                    continue
                if prev is None:
                    cnf.append([-u, w])            # x_t <= x_perm(t)
                    nxt = pool.id(("e", name, t))
                    cnf.append([-u, -w, nxt])      # equal -> nxt
                    cnf.append([u, w, nxt])
                else:
                    cnf.append([-prev, -u, w])
                    nxt = pool.id(("e", name, t))
                    cnf.append([-prev, -u, -w, nxt])
                    cnf.append([-prev, u, w, nxt])
                prev = nxt
            nsym += 1
    print("[sat] vars=%d clauses=%d (sym gens=%d) card<=%d solver=%s tl=%.0fs"
          % (pool.top, len(cnf.clauses), nsym, a.card, a.solver, a.tl),
          flush=True)

    t0 = time.time()
    with Solver(name=a.solver, bootstrap_with=cnf.clauses, use_timer=True) as s:
        # cooperative time limit via interrupt
        import threading
        timer = threading.Timer(a.tl, s.interrupt)
        timer.start()
        res = s.solve_limited(expect_interrupt=True)
        timer.cancel()
        dt = time.time() - t0
        if res is True:
            model = set(l for l in s.get_model() if l > 0)
            chosen = [w for w in range(V) if xv[w] in model]
            ok = A[np.array(chosen)].any(axis=0).all()
            print("[sat] SAT in %.0fs: %d words, cover verified in-process: %s"
                  % (dt, len(chosen), ok), flush=True)
            if a.out and ok:
                dig = all_words()[np.array(chosen)]
                with open(a.out, "w") as f:
                    for row in dig:
                        f.write("".join(str(d) for d in row) + "\n")
                print("[sat] wrote %s" % a.out, flush=True)
            return 10
        elif res is False:
            print("[sat] UNSAT in %.0fs  => K8(4,2) = 23 "
                  "(no cover of size <= %d)" % (dt, a.card), flush=True)
            return 20
        else:
            print("[sat] UNDECIDED (interrupted) after %.0fs" % dt, flush=True)
            return 30


if __name__ == "__main__":
    sys.exit(main())
