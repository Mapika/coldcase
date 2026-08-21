#!/bin/bash
# Correctness gate for covsearch2.
#
# 1. TRAJECTORY IDENTITY.  Every variant that is a pure implementation change
#    (--walk/--hoist/--st/--eval/--huge/--pf) must reproduce the ORIGINAL
#    solver's search bit-for-bit at a fixed iteration budget.  Since the
#    trajectory is determined by the exact delta of every candidate move, an
#    identical final uncovered count over many configurations is a strong
#    check that the fast evaluators compute exactly the same deltas.
# 2. VERIFIER AGREEMENT.  The number the solver reports must equal the number
#    cov/verify_cov.py computes from the written code file -- including when
#    it is not zero.  This is the check that matters: the solver's counters
#    are incremental and are never the evidence.
#
# usage: bash selftest.sh
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
RUN="nice -n 15"
VER=../verify_cov.py
fail=0

echo "== 1. trajectory identity vs the original solver =="
CFGS=(
  "-q 6 -n 6 -R 3 -M 41  --iters 2000"
  "-q 6 -n 8 -R 4 -M 169 --iters 150"
  "-q 7 -n 5 -R 2 -M 120 --iters 400"
  "-q 3 -n 7 -R 1 -M 34  --iters 1500"
  "-q 5 -n 4 -R 0 -M 600 --iters 300"
  "-q 4 -n 5 -R 4 -M 3   --iters 100"
  "-q 2 -n 8 -R 2 -M 12  --iters 800"
  "-q 6 -n 5 -R 2 -M 66  --iters 900"
)
VARS=("--preset base" "--preset p1" "--preset p2" "--preset p2b" \
      "--preset p3" "--preset p4" "--preset p5" "--preset p5b" "--preset p5 --pf")
for cfg in "${CFGS[@]}"; do
  for s in 1 2 3; do
    b=$($RUN ./covsearch_base $cfg -t 9999 -s $s --threads 1 --quiet | grep -o 'uncovered=[0-9]*')
    line="  [$cfg s=$s] $b"
    for v in "${VARS[@]}"; do
      r=$($RUN ./covsearch2 $cfg -t 9999 -s $s --threads 1 --quiet $v | grep -o 'uncovered=[0-9]*')
      if [ "$r" != "$b" ]; then line="$line  MISMATCH[$v]=$r"; fail=1; fi
    done
    echo "$line ok"
  done
done

echo "== 2. solver count == verifier count, all variants, including nonzero =="
mkdir -p /tmp/covopt_selftest
VARS2=("--preset base" "--preset p5" "--preset p5b" "--preset p5b --wide" \
       "--preset p5 --pf" "--preset p5 --early" \
       "--preset p5 --fix0" "--preset p5 --upick" "--preset p5 --ucache" \
       "--preset p4 --sa")
CFGS2=(
  "6 4 2 15 4000"
  "7 3 1 25 4000"
  "2 8 2 12 4000"
  "6 6 3 41 3000"
  "3 4 1 9  4000"
  "5 4 1 125 2000"
  "6 5 2 60 1500"
)
for cfg in "${CFGS2[@]}"; do
  set -- $cfg
  for v in "${VARS2[@]}"; do
    f=/tmp/covopt_selftest/c.txt
    out=$($RUN ./covsearch2 -q $1 -n $2 -R $3 -M $4 --iters $5 -t 9999 -s 7 \
            --threads 1 --quiet --out $f $v)
    u=$(echo "$out" | grep -o 'uncovered=[0-9]*' | cut -d= -f2)
    vout=$(python3 $VER $f -q $1 -n $2 -R $3 --method both 2>&1)
    vu=$(echo "$vout" | grep -oE 'method pure *: uncovered=[0-9]+' | grep -oE '[0-9]+$')
    if [ "$u" = "$vu" ]; then
      echo "  K$1($2,$3) M=$4 $v -> solver=$u verifier=$vu OK"
    else
      echo "  K$1($2,$3) M=$4 $v -> solver=$u verifier=$vu  ** MISMATCH **"
      echo "$vout" | tail -4
      fail=1
    fi
  done
done

echo
if [ $fail -eq 0 ]; then echo "SELFTEST: ALL PASS"; else echo "SELFTEST: FAILURES"; fi
exit $fail
