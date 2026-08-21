#!/usr/bin/env bash
# Arena entry "strategy":  run_entry.sh Q N R M SEED TIME_S OUTFILE
#
# Portfolio driver.  Launches P independent search chains of T threads each with
# P*T <= 6, picks the chain with the fewest uncovered words, and copies its code
# to OUTFILE.  P and T are chosen from the instance shape: parallelising one
# chain over candidate moves only pays when a single sphere walk is large enough
# to hide the OpenMP fork/join; below that, six independent single-threaded
# chains are both ~6x faster in moves/s and give a best-of-6.  See NOTES.md.
set -u
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
Q=$1; N=$2; R=$3; M=$4; SEED=$5; TS=$6; OUT=$7

BIN=$D/strategy
if [ ! -x "$BIN" ] || [ "$D/strategy.c" -nt "$BIN" ]; then
    ( cd "$D" && ( gcc -O3 -march=native -fopenmp -o strategy strategy.c -lm 2>/dev/null \
                   || gcc -O2 -fopenmp -o strategy strategy.c -lm ) ) >/dev/null 2>&1
fi

# ---------------------------------------------------------------- fallback
# The judge scores an unparseable or empty OUTFILE at -10^6, so put a valid file
# there before anything else can go wrong.
python3 - "$Q" "$N" "$M" "$SEED" "$OUT" <<'PYEOF'
import sys, random
q, n, M, seed, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
DIG = "0123456789abcdefghijklmnopqrstuvwxyz"
M = min(M, q ** n)
rnd = random.Random(seed)
seen, ws = set(), []
while len(ws) < M:
    w = tuple(rnd.randrange(q) for _ in range(n))
    if w in seen:
        continue
    seen.add(w); ws.append(w)
with open(out, "w") as f:
    if q <= 36:
        f.write("".join("".join(DIG[d] for d in w) + "\n" for w in ws))
    else:
        f.write("".join(" ".join(str(d) for d in w) + "\n" for w in ws))
PYEOF

# ---------------------------------------------------------------- policy
read -r P T < <(python3 - "$Q" "$N" "$R" "$M" <<'PYEOF'
import sys
from math import comb
q, n, R, M = (int(x) for x in sys.argv[1:5])
NTOT = q ** n
V = sum(comb(n, i) * (q - 1) ** i for i in range(0, R + 1))   # ball volume
# One construction pass costs M*V memory touches and every chain pays it in
# full.  When that dominates, six chains would each run six times slower for a
# best-of-six worth much less than the lost progress; when it is cheap, six
# independent single-threaded chains beat one six-threaded chain outright,
# because the per-iteration OpenMP fork/join dominates a small sphere walk
# (measured: 870 vs 111 moves/s on K_6(6,3)).
P, T = (1, 6) if M * V >= 1e8 else (6, 1)
maxP = max(1, int(20e9 / max(1, 2 * NTOT)))   # 25 GB cap, 2 bytes per word
P = min(P, maxP)
T = min(6, max(1, 6 // P))
print(P, T)
PYEOF
)
[ -z "${P:-}" ] && { P=1; T=6; }
# development overrides
if [ -n "${STRATEGY_PT:-}" ]; then read -r P T <<<"$STRATEGY_PT"; fi
STRATEGY_ARGS=${STRATEGY_ARGS:-}

# ---------------------------------------------------------------- run
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
BUDGET=$(python3 -c "print(max(1.0, $TS - max(2.0, 0.04*$TS)))")
HARD=$(python3 -c "print(max(2, int($TS - 0.3)))")

pids=()
for i in $(seq 0 $((P - 1))); do
    timeout -s KILL "${HARD}s" nice -n 15 "$BIN" -q "$Q" -n "$N" -R "$R" -M "$M" \
        -t "$BUDGET" -s $((SEED * 977 + i)) --threads "$T" --quiet $STRATEGY_ARGS \
        --out "$TMP/c$i.txt" >/dev/null 2>&1 &
    pids+=($!)
done

# Stop the whole portfolio the moment any chain reports a full cover: the judge
# breaks ties on median wall time, so there is nothing to gain by letting the
# other five chains burn the rest of the budget.
deadline=$(( $(date +%s) + HARD + 3 ))
while :; do
    alive=0
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ "$alive" = 0 ] && break
    solved=0
    for i in $(seq 0 $((P - 1))); do
        [ -s "$TMP/c$i.txt.cnt" ] || continue
        [ "$(head -1 "$TMP/c$i.txt.cnt")" = "0" ] && solved=1 && break
    done
    if [ "$solved" = 1 ]; then
        for p in "${pids[@]}"; do kill -TERM "$p" 2>/dev/null; done
        break
    fi
    [ "$(date +%s)" -ge "$deadline" ] && break
    sleep 0.2
done
wait 2>/dev/null

best=""; bestv=""
for i in $(seq 0 $((P - 1))); do
    f=$TMP/c$i.txt
    [ -s "$f" ] || continue
    [ -s "$f.cnt" ] || continue
    v=$(head -1 "$f.cnt")
    case "$v" in ''|*[!0-9]*) continue;; esac
    if [ -z "$bestv" ] || [ "$v" -lt "$bestv" ]; then bestv=$v; best=$f; fi
done
[ -n "$best" ] && cp "$best" "$OUT"
exit 0
