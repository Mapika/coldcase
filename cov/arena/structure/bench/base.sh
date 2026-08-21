#!/bin/bash
# Baseline reference: cov/search/covsearch as shipped, 6 threads, same cells and
# seeds as the judge.  Reported as-is; note that on K_8(9,4) the baseline checks
# its deadline only every 256 iterations, so it overruns the budget badly.
CS=/lambda/nfs/new-fs/longshots/cov/search/covsearch
O=$(dirname "$0")
T=${1:-60}
SEEDS=${2:-5}
for cell in "6 6 3 41" "6 8 4 169" "8 9 4 940" "3 11 4 81"; do
  set -- $cell
  for i in $(seq 0 $((SEEDS-1))); do
    s=$((1000+i))
    /usr/bin/time -f "WALL %e" nice -n 15 $CS -q $1 -n $2 -R $3 -M $4 -t $T -s $s \
      --threads 6 --quiet --out $O/base_$1_$2_$3_$4_$s.txt 2>&1 | grep -E 'RESULT|WALL' | tr '\n' ' '
    echo
  done
done
