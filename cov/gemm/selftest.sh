#!/bin/bash
# Correctness gate for the GEMM solver.
#
#  1. test_gemm.py   -- every GEMM count against an integer CPU reference.
#  2. --selftest     -- the incrementally maintained cnt array is compared,
#                       after EVERY move, against a from-scratch full-space
#                       GEMM recount; any drift aborts.
#  3. verify_cov.py  -- the emitted code file is re-read from disk and its
#                       uncovered count recomputed by two independent
#                       exhaustive methods; it must equal what the solver
#                       reported, including when that number is not zero.
set -u
cd "$(dirname "$0")"
OUT=${1:-/tmp/gemm_selftest}
mkdir -p "$OUT"
fail=0

echo "=== 1. GEMM primitives vs integer CPU reference ==="
python3 test_gemm.py > "$OUT/test_gemm.log" 2>&1 || fail=1
tail -1 "$OUT/test_gemm.log"

echo
echo "=== 2/3. solver selftest + verifier cross-check ==="
# q n R M iters   -- deliberately includes R=0, R=n-1, q=2..7, M above and
# below the optimum, and sizes that must end with uncovered > 0.
CFGS=(
  "6 4 2 20 25"
  "6 4 2 40 25"
  "2 8 3 20 25"
  "3 5 2 12 25"
  "7 3 1 25 25"
  "5 4 0 620 8"
  "4 4 3 2 8"
  "6 6 3 41 40"
  "6 6 3 30 25"
  "3 6 2 73 25"
)
for cfg in "${CFGS[@]}"; do
  set -- $cfg; q=$1; n=$2; R=$3; M=$4; it=$5
  f="$OUT/c_q${q}_n${n}_R${R}_M${M}.txt"
  line=$(python3 gemmsolve.py -q $q -n $n -R $R -M $M --iters $it -s 7 \
           --selftest --no-compile --quiet --out "$f" 2>&1 | tail -1)
  rc=$?
  unc=$(echo "$line" | sed -n 's/.*uncovered=\([0-9]*\).*/\1/p')
  if [ $rc -ne 0 ] || [ -z "$unc" ]; then
    echo "  FAIL solver q=$q n=$n R=$R M=$M : $line"; fail=1; continue
  fi
  v=$(python3 ../verify_cov.py "$f" -q $q -n $n -R $R --method both 2>&1)
  # both exhaustive methods must report the same number, and it must be the
  # number the solver printed
  vu=$(echo "$v" | sed -n 's/^method [a-z]* *: uncovered=\([0-9]*\).*/\1/p' \
        | sort -u | tr '\n' ',' | sed 's/,$//')
  if [ "$vu" = "$unc" ]; then
    echo "  ok   q=$q n=$n R=$R M=$M  uncovered=$unc  (verifier agrees)"
  else
    echo "  FAIL q=$q n=$n R=$R M=$M  solver=$unc verifier=$vu"
    echo "$v" | head -5
    fail=1
  fi
done

echo
[ $fail -eq 0 ] && echo "ALL CHECKS PASSED" || echo "FAILURES PRESENT"
exit $fail
