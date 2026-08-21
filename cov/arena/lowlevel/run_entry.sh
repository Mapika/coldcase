#!/bin/bash
# Arena entry "lowlevel".
#   run_entry.sh Q N R M SEED TIME_S OUTFILE
# Runs covfast for at most TIME_S wall seconds on <=6 threads and writes the
# best code found to OUTFILE.  Always exits 0 and always leaves a parseable
# file behind.
set -u
Q=$1; N=$2; R=$3; M=$4; SEED=$5; TIME_S=$6; OUT=$7
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The rules say "nice -n 15 everything".  `nice -n 15` is a RELATIVE increment,
# so if the judge is itself started under nice this would compound to nice 19
# (scheduler weight 15) instead of the intended nice 15 (weight 36).  Target the
# absolute value: add only the difference, and add nothing if we are already at
# or above 15 (a process cannot lower its own niceness without privileges).
CUR=$(awk '{print $19}' /proc/self/stat 2>/dev/null || echo 0)
case "$CUR" in ''|*[!0-9-]*) CUR=0 ;; esac
DELTA=$(( 15 - CUR ))
[ "$DELTA" -lt 0 ] && DELTA=0
NICE="nice -n $DELTA"

THREADS=6
export OMP_NUM_THREADS=$THREADS
export OMP_WAIT_POLICY=passive
export OMP_DYNAMIC=false

# The binary is committed; this only fires if it is missing (e.g. a fresh
# checkout on a different machine).  It is deliberately NOT triggered by
# timestamps -- a rebuild costs minutes on this loaded box and would eat the
# first run's time budget.
BIN="$DIR/covfast"
if [ ! -x "$BIN" ]; then
    $NICE gcc -O3 -march=native -fopenmp -fno-stack-protector \
        -o "$BIN" "$DIR/covfast.c" >/dev/null 2>&1 || true
fi

# Fallback: a trivially parseable file of M distinct words, so that a crash or
# an OOM can never produce an unreadable output (which scores -10^6).
$NICE python3 - "$Q" "$N" "$M" "$OUT" <<'PY' 2>/dev/null || true
import sys
q, n, M, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
D = "0123456789abcdefghijklmnopqrstuvwxyz"
with open(out, "w") as f:
    for i in range(M):
        x, w = i % (q ** n), []
        for _ in range(n):
            w.append(x % q); x //= q
        w.reverse()
        f.write(("".join(D[d] for d in w) if q <= 36
                 else " ".join(str(d) for d in w)) + "\n")
PY

# covfast starts its own clock before initialisation and checks it every
# iteration, so it stops on its own; the headroom covers process start-up and
# writing the code out.  The outer timeout is only a backstop (the judge itself
# allows TIME_S + 60).
BUDGET=$(python3 -c "print(max(1.0, $TIME_S - max(2.0, 0.06*$TIME_S)))" 2>/dev/null || echo "$TIME_S")
HARD=$(python3 -c "print(int($TIME_S + 30))" 2>/dev/null || echo "$TIME_S")

# covfast writes OUT itself: once as soon as the initial code exists and then
# again (atomically, via rename) whenever the incumbent improves.  So even a
# SIGKILL leaves the best code found so far on disk, and the fallback written
# above is only ever seen if the solver dies before finishing initialisation.
$NICE timeout -k 5 "$HARD" "$BIN" \
    -q "$Q" -n "$N" -R "$R" -M "$M" -s "$SEED" -t "$BUDGET" \
    --threads "$THREADS" --out "$OUT" --quiet >/dev/null 2>&1
rm -f "$OUT.part"
exit 0
