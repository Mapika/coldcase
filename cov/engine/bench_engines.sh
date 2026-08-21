#!/bin/bash
# Paired head-to-head of the three free-search inner loops available to the
# engine, plus the quotient search, on one cell.  All variants for a cell are
# launched at the SAME MOMENT with the same seeds so they see the same machine
# (the box is shared with a production sweep; sequential timings on it are not
# comparable).  Reported number is best uncovered reached inside T wall seconds,
# single-threaded, which is exactly what the portfolio cares about.
#
#   bench_engines.sh Q N R M T SEEDS
set -u
HERE=$(cd "$(dirname "$0")" && pwd); COV=$(cd "$HERE/.." && pwd)
Q=$1; N=$2; R=$3; M=$4; T=$5; NS=${6:-3}; TH=${7:-1}
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
export OMP_NUM_THREADS=1
run() { # tag cmd...
  local tag=$1; shift
  nice -n 12 timeout -k 5 $((T+40)) "$@" > "$W/$tag.log" 2>&1
}
pids=""
for s in $(seq 1000 $((1000+NS-1))); do
  run cf.$s "$COV/arena/lowlevel/covfast" -q $Q -n $N -R $R -M $M -s $s -t $T \
      --threads 1 --out "$W/cf.$s.txt" & pids="$pids $!"
  run p5b.$s "$HERE/covsearch2e" -q $Q -n $N -R $R -M $M -s $s -t $T \
      --preset p5b --threads 1 --out "$W/p5b.$s.txt" & pids="$pids $!"
  run wide.$s "$HERE/covsearch2e" -q $Q -n $N -R $R -M $M -s $s -t $T \
      --preset p5b --wide --threads 1 --out "$W/wide.$s.txt" & pids="$pids $!"
  ORB=$(( M / Q ))
  if [ "$ORB" -ge 2 ]; then
    run sym.$s "$COV/arena/structure/symsearch" -q $Q -n $N -R $R -M $M -s $s -t $T \
        --threads $TH --out "$W/sym.$s.txt" & pids="$pids $!"
  fi
done
wait $pids 2>/dev/null
echo "cell K$Q($N,$R)@$M   t=${T}s  $NS seeds  (free chains 1 thread, symsearch $TH)"
for v in cf p5b wide sym; do
  line=""
  for s in $(seq 1000 $((1000+NS-1))); do
    [ -f "$W/$v.$s.log" ] || continue
    u=$(awk '{for(i=1;i<=NF;i++) if($i ~ /^(word_uncovered|uncovered_words|uncovered)=/){split($i,a,"=");u=a[2]}} END{print (u==""?"-":u)}' "$W/$v.$s.log")
    it=$(awk '/^RESULT/{for(i=1;i<=NF;i++) if($i ~ /^iters=/){split($i,a,"=");print a[2]}}' "$W/$v.$s.log")
    line="$line ${u}(${it:-?}it)"
  done
  [ -n "$line" ] && printf "  %-5s %s\n" "$v" "$line"
done
