#!/bin/bash
# Head-to-head on the real objective: best uncovered reached within a wall-clock
# budget.  Runs the three configurations round-robin so all of them see the same
# machine load.  The baseline is run at BOTH 6 threads (its shipped setting:
# par_cand is true on every benchmark cell) and 1 thread, and both are reported,
# so it is compared at its best.
# Usage: quality.sh Q N R M SECONDS SEEDS
cd "$(dirname "$0")"
Q=$1; N=$2; R=$3; M=$4; T=$5; SEEDS=$6
KEYS=(base6 base1 fast)
CMDS=("./baseline|6" "./baseline|1" "./covfast|6")
declare -A RES
for s in $(seq 1000 $((1000+SEEDS-1))); do
  for i in "${!KEYS[@]}"; do
    b="${CMDS[$i]%|*}"; t="${CMDS[$i]#*|}"
    out=$(nice -n 15 $b -q $Q -n $N -R $R -M $M -s $s -t $T --threads $t --quiet 2>&1 | tail -1)
    RES[${KEYS[$i]},$s]=$(echo "$out" | sed -n 's/.*uncovered=\([0-9]*\).*/\1/p')
  done
done
echo "K$Q($N,$R)@$M  t=${T}s  seeds=$SEEDS   (best uncovered reached; 0 = solved)"
for k in "${KEYS[@]}"; do
  line=""; solved=0; sum=0
  for s in $(seq 1000 $((1000+SEEDS-1))); do
    v=${RES[$k,$s]}; line="$line $v"
    [ "$v" = "0" ] && solved=$((solved+1)); sum=$((sum+v))
  done
  printf "  %-6s solved=%d/%d  sum=%-9d %s\n" "$k" "$solved" "$SEEDS" "$sum" "$line"
done
