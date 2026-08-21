#!/bin/bash
# Baseline reference, scored exactly the way scripts/arena_judge.py scores:
#   * the solver is given TIME_S and killed at TIME_S+60 (the judge's subprocess
#     timeout), and whatever is on disk at that point is what counts;
#   * an unparseable / empty OUTFILE is -10^6, a full cover is +1000, otherwise
#     the score is -(uncovered).
# This matters: cov/search/covsearch tests its deadline only every 256th
# iteration, and on K_8(9,4) one iteration costs ~0.7 s.
#
# usage: base_scored.sh [TIME_S] [NSEEDS]
CS=/lambda/nfs/new-fs/longshots/cov/search/covsearch
D=$(cd "$(dirname "$0")/.." && pwd)
T=${1:-60}
NS=${2:-5}
for cell in "6 6 3 41" "6 8 4 169" "8 9 4 940" "3 11 4 81"; do
  set -- $cell
  Q=$1; N=$2; R=$3; M=$4
  for i in $(seq 0 $((NS-1))); do
    s=$((1000+i))
    out=$(mktemp)
    st=$(date +%s.%N)
    timeout -s KILL $((T+60)) nice -n 15 "$CS" -q $Q -n $N -R $R -M $M -t $T \
        -s $s --threads 6 --quiet --out "$out" >/dev/null 2>&1
    en=$(date +%s.%N)
    wall=$(awk -v a="$st" -v b="$en" 'BEGIN{printf "%.1f", b-a}')
    if [ -s "$out" ]; then
      u=$("$D/covcount" -q $Q -n $N -R $R -M $M --in "$out" |
          awk '{for(i=1;i<=NF;i++) if($i ~ /^uncovered=/){split($i,a,"=");print a[2]}}')
      if [ "$u" = 0 ]; then sc=1000; else sc=-$u; fi
    else
      sc=-1000000; u=INVALID
    fi
    echo "baseline K_${Q}($N,$R)@$M seed=$s wall=${wall}s uncovered=$u score=$sc"
    rm -f "$out"
  done
done
