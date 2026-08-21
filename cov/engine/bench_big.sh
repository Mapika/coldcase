#!/bin/bash
# Paired comparison of the four ways to spend 6 cores on ONE big cell, all
# launched at the same moment (the box is shared; sequential numbers on it are
# not comparable).  Reported number is best uncovered inside T seconds.
#   bench_big.sh Q N R M T SEED [SEED2 ...]
set -u
HERE=$(cd "$(dirname "$0")" && pwd); COV=$(cd "$HERE/.." && pwd)
Q=$1; N=$2; R=$3; M=$4; T=$5; shift 5
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
unc() { awk '{for(i=1;i<=NF;i++) if($i ~ /^(word_uncovered|uncovered_words|uncovered|unc)=/){split($i,a,"=");if(a[2]>=0)u=a[2]}} END{print (u==""?"-":u)}' "$1"; }
pids=""
for s in "$@"; do
  OMP_NUM_THREADS=6 nice -n 12 timeout -k 5 $((T+45)) "$COV/arena/structure/symsearch" \
      -q $Q -n $N -R $R -M $M -s $s -t $T --threads 6 --out "$W/sym.$s.txt" \
      > "$W/sym.$s.log" 2>&1 & pids="$pids $!"
  OMP_NUM_THREADS=6 nice -n 12 timeout -k 5 $((T+45)) "$HERE/covsearch2e" \
      -q $Q -n $N -R $R -M $M -s $s -t $T --preset p5b --wide --threads 6 --out "$W/w6.$s.txt" \
      > "$W/w6.$s.log" 2>&1 & pids="$pids $!"
  OMP_NUM_THREADS=6 nice -n 12 timeout -k 5 $((T+45)) "$COV/arena/lowlevel/covfast" \
      -q $Q -n $N -R $R -M $M -s $s -t $T --threads 6 --out "$W/cf6.$s.txt" \
      > "$W/cf6.$s.log" 2>&1 & pids="$pids $!"
  for k in 0 1 2 3 4 5; do
    OMP_NUM_THREADS=1 nice -n 12 timeout -k 5 $((T+45)) "$HERE/covsearch2e" \
        -q $Q -n $N -R $R -M $M -s $((s*10+k)) -t $T --preset p5b --wide --threads 1 \
        --out "$W/p$k.$s.txt" > "$W/p$k.$s.log" 2>&1 & pids="$pids $!"
  done
done
wait $pids 2>/dev/null
echo "cell K$Q($N,$R)@$M   t=${T}s   6 cores per variant"
for s in "$@"; do
  best=""
  for k in 0 1 2 3 4 5; do
    u=$(unc "$W/p$k.$s.log")
    [ "$u" = "-" ] && continue
    if [ -z "$best" ] || [ "$u" -lt "$best" ]; then best=$u; fi
  done
  printf "  seed %-6s symsearch6=%-10s covsearch2e-wide6=%-10s covfast6=%-10s portfolio6x1=%s\n" \
     "$s" "$(unc "$W/sym.$s.log")" "$(unc "$W/w6.$s.log")" "$(unc "$W/cf6.$s.log")" "${best:--}"
done
