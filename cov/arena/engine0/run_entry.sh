#!/bin/bash
# Arena wrapper for the merged production engine.
#   run_entry.sh Q N R M SEED TIME_S OUTFILE
#
# All the work is in cov/engine/ (see cov/engine/NOTES.md).  This file exists
# only so that scripts/arena_judge.py can score the engine against the four
# entries it was built from, on the same cells, under the same contract:
# <= 6 threads, <= 25 GB, nice 15, at most TIME_S wall seconds, and a valid
# parseable file in OUTFILE on every path.
#
# TWO THINGS ABOUT THIS COMPARISON, STATED UP FRONT.
#
# 1. COVENGINE_NO_RESULTS=1 is set, which switches OFF everything the engine
#    would otherwise take from cov/results/: no descent seed from an incumbent
#    code, and no direct-sum factor read out of a code we already own.  Two of
#    the four public cells -- K_6(6,3) at M=41 and K_6(8,4) at M=169 -- are
#    cells whose answer is already recorded in cov/results/, so without this the
#    engine would be "solving" them by copying its own stored answer and the
#    comparison against structure / lowlevel / strategy would mean nothing.
#    What remains available to phase A is exactly what was available to every
#    other entry: the algebra (lincov, symsearch) and constructions.py's
#    primitive recipes (whole space, single word, perfect Hamming codes).
#
# 2. Six cores, because that is the arena contract.  In production the engine
#    is given up to 40 and the portfolio is ~7x wider.
set -u
if [ $# -lt 7 ]; then
  echo "usage: run_entry.sh Q N R M SEED TIME_S OUTFILE" >&2
  exit 2
fi
Q=$1; N=$2; R=$3; M=$4; SEED=$5; TIME_S=$6; OUT=$7
ENGINE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../engine" && pwd)

# build only if a binary is missing (a rebuild costs minutes on this box and
# would eat the first run's budget); never triggered by timestamps
for b in covsearch2e covfast lincov symsearch covcount; do
  [ -x "$ENGINE/$b" ] || (cd "$ENGINE" && make -s "$b") >/dev/null 2>&1
done

# `nice -n 15` is a RELATIVE increment, so if the judge is itself niced this
# would compound.  Target the absolute value 15 (credit: lowlevel/run_entry.sh).
CUR=$(awk '{print $19}' /proc/self/stat 2>/dev/null || echo 0)
case "$CUR" in ''|*[!0-9-]*) CUR=0 ;; esac
DELTA=$(( 15 - CUR )); [ "$DELTA" -lt 0 ] && DELTA=0

export COVENGINE_CORES=6
export COVENGINE_NICE=15
export COVENGINE_RAM=20e9
export COVENGINE_NO_RESULTS=1
export COVENGINE_QUIET=1
export OMP_WAIT_POLICY=passive
export OMP_DYNAMIC=false

# The engine writes a legal M-word file into OUT before any solver starts and
# republishes atomically on every improvement, so the outer timeout is only a
# backstop: a kill at any moment leaves the best code found so far in place.
nice -n "$DELTA" timeout -k 5 "$(( TIME_S + 25 ))" \
    python3 "$ENGINE/covengine" "$Q" "$N" "$R" "$M" "$SEED" "$TIME_S" "$OUT" \
    >/dev/null 2>&1
rm -f "$OUT.part" "$OUT.pub"
exit 0
