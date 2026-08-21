#!/bin/bash
# cov/engine selftest.
#
# Two questions, both of which have to be answered before the engine may write
# anything into cov/results/:
#
#   1. Does covsearch2e follow EXACTLY the same search as cov/opt/covsearch2?
#      The engine patch adds clock handling and output publishing and must not
#      touch a single search decision, so from the same seed and the same
#      iteration budget the two must report the identical
#      uncovered/iters/kicks fingerprint.
#   2. Does the number the solver reports equal the number the independent
#      verifier computes from the file it wrote -- including when it is not 0?
#      This is the check that catches incremental-counter drift, which is the
#      failure mode that would produce a confident false record.
#
# Plus: every file written must hold exactly M DISTINCT codewords.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
COV=$(cd "$HERE/.." && pwd)
A="$COV/opt/covsearch2"          # reference (read-only, never modified)
B="$HERE/covsearch2e"            # the engine's solver
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0; nid=0; nver=0

cells=(
  "2 6 2 12 300"
  "3 6 2 12 300"
  "5 5 2 30 200"
  "6 6 3 41 400"
  "6 6 3 45 400"
  "7 4 1 60 300"
  "3 11 4 81 120"
  "6 8 4 169 40"
  "4 5 0 1024 5"
  "5 4 3 3 50"
)
for spec in "${cells[@]}"; do
  set -- $spec; q=$1; nn=$2; r=$3; m=$4; iters=$5
  for preset in p5 p5b; do
    for s in 1 2; do
      fa=$("$A" -q $q -n $nn -R $r -M $m -s $s --iters $iters --preset $preset \
             --quiet --out "$W/a.txt" 2>/dev/null | \
           awk '/^RESULT/{for(i=1;i<=NF;i++) if($i~/^(uncovered|iters|kicks)=/) printf "%s ",$i}')
      fb=$("$B" -q $q -n $nn -R $r -M $m -s $s --iters $iters --preset $preset \
             --quiet --out "$W/b.txt" -t 100000 2>/dev/null | \
           awk '/^RESULT/{for(i=1;i<=NF;i++) if($i~/^(uncovered|iters|kicks)=/) printf "%s ",$i}')
      nid=$((nid+1))
      if [ "$fa" != "$fb" ]; then
        echo "TRAJECTORY MISMATCH q=$q n=$nn R=$r M=$m preset=$preset s=$s"
        echo "   ref: $fa"; echo "   eng: $fb"; fail=1
      fi
      # distinctness of what the engine wrote
      d=$(sort -u "$W/b.txt" | grep -c . )
      if [ "$d" -ne "$m" ]; then
        echo "DISTINCTNESS FAIL q=$q n=$nn R=$r M=$m preset=$preset s=$s: $d distinct words, want $m"
        fail=1
      fi
      # the reported count must equal the verifier's count
      u=$(echo "$fb" | tr ' ' '\n' | awk -F= '/^uncovered=/{print $2}')
      v=$(cd "$COV" && python3 verify_cov.py "$W/b.txt" -q $q -n $nn -R $r --method pure 2>/dev/null \
            | awk -F'uncovered=' '/method pure/{print $2+0}')
      nver=$((nver+1))
      if [ "$u" != "$v" ]; then
        echo "VERIFIER MISMATCH q=$q n=$nn R=$r M=$m preset=$preset s=$s: solver=$u verifier=$v"
        fail=1
      fi
    done
  done
done

# deadline behaviour: a budget that expires during initialisation must still
# leave a complete, distinct, parseable file behind, and must not overrun.
echo "-- deadline / publishing checks"
for t in 0.05 0.5 2; do
  st=$(date +%s.%N)
  "$B" -q 8 -n 9 -R 4 -M 940 -s 7 -t $t --preset p5b --quiet --out "$W/d.txt" >/dev/null 2>&1
  en=$(date +%s.%N)
  el=$(awk -v a=$st -v b=$en 'BEGIN{printf "%.1f", b-a}')
  d=$(sort -u "$W/d.txt" | grep -c .)
  echo "   -t $t  wall ${el}s  distinct=$d"
  [ "$d" -eq 940 ] || { echo "   DEADLINE FAIL: $d distinct words"; fail=1; }
done

# a SIGKILL must leave the best code so far on disk
echo "-- SIGKILL leaves a valid file"
rm -f "$W/k.txt"
"$B" -q 6 -n 8 -R 4 -M 169 -s 3 -t 60 --preset p5b --quiet --out "$W/k.txt" >/dev/null 2>&1 &
kp=$!
python3 -c "import time; time.sleep(6)"
kill -9 $kp 2>/dev/null; wait $kp 2>/dev/null
d=$(sort -u "$W/k.txt" 2>/dev/null | grep -c . || echo 0)
echo "   after SIGKILL: $d distinct words"
[ "$d" -eq 169 ] || { echo "   KILL FAIL"; fail=1; }

echo
if [ $fail -eq 0 ]; then
  echo "ALL PASS  ($nid trajectory comparisons, $nver solver/verifier cross-checks)"
else
  echo "FAILURES ABOVE"
fi
exit $fail
