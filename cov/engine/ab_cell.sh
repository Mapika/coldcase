#!/bin/bash
# A/B the engine's portfolio policy on one cell.  Every configuration and every
# seed is launched at the same moment, because the box is shared and sequential
# wall-clock numbers on it are worthless.
#   ab_cell.sh Q N R M T SEEDS "NAME:ENV=V,ENV=V" ...
set -u
E=$(cd "$(dirname "$0")" && pwd)
Q=$1; N=$2; R=$3; M=$4; T=$5; NS=$6; shift 6
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
pids=""
for cfg in "$@"; do
  name=${cfg%%:*}; envs=${cfg#*:}
  for s in $(seq 1000 $((1000+NS-1))); do
    ( IFS=','; for kv in $envs; do [ -n "$kv" ] && export "$kv"; done
      export COVENGINE_CORES=6 COVENGINE_NO_RESULTS=1 COVENGINE_QUIET=1 COVENGINE_NICE=12
      python3 "$E/covengine" $Q $N $R $M $s $T "$W/$name.$s.txt" >/dev/null 2>&1 ) &
    pids="$pids $!"
  done
done
wait $pids 2>/dev/null
echo "cell K$Q($N,$R)@$M  t=${T}s  $NS seeds  (6 cores per run)"
for cfg in "$@"; do
  name=${cfg%%:*}
  line=""; solved=0
  for s in $(seq 1000 $((1000+NS-1))); do
    u=$(cd "$E/.." && python3 verify_cov.py "$W/$name.$s.txt" -q $Q -n $N -R $R --method numpy 2>/dev/null \
          | awk -F'uncovered=' '/method numpy/{print $2+0}')
    [ -z "$u" ] && u="?"
    [ "$u" = "0" ] && solved=$((solved+1))
    line="$line $u"
  done
  printf "  %-14s solved %d/%d   uncovered:%s\n" "$name" "$solved" "$NS" "$line"
done
