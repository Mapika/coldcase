#!/usr/bin/env bash
# Decisive paired test on the big cell: how much of the budget should go to the
# greedy construction rather than to the local search?  All variants launched
# together per seed so they share whatever load the machine has.
#   ./ab_bigcell.sh [TIME] [SEEDS...]
set -u
cd "$(dirname "$(readlink -f "$0")")"
T=${1:-60}; shift || true
SEEDS=${*:-"1000 1001"}
for s in $SEEDS; do
  nice -n 15 ./strategy -q 8 -n 9 -R 4 -M 940 -t "$T" -s "$s" --threads 6 --quiet \
      --kinit 12 --cfrac 0.55 | sed "s/^/[k12/c.55 s=$s] /" &
  nice -n 15 ./strategy -q 8 -n 9 -R 4 -M 940 -t "$T" -s "$s" --threads 6 --quiet \
      --kinit 4  --cfrac 0.25 | sed "s/^/[k4/c.25  s=$s] /" &
  nice -n 15 ./strategy -q 8 -n 9 -R 4 -M 940 -t "$T" -s "$s" --threads 6 --quiet \
      --kinit 0 | sed "s/^/[k0      s=$s] /" &
  nice -n 15 ./baseline -q 8 -n 9 -R 4 -M 940 -t "$T" -s "$s" --threads 6 --quiet \
      | sed "s/^/[baseline s=$s] /" &
  wait
done
