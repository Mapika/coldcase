#!/bin/bash
# Correctness gate.
#
#  (a) --selftest recomputes the uncovered count (and the uncovered-word list
#      length) from scratch after EVERY committed move and aborts on any
#      mismatch, so the incremental bookkeeping is checked move by move.
#  (b) every emitted code is handed to the independent verifier
#      cov/verify_cov.py, which must report exactly the uncovered count the
#      solver claimed -- not only when that count is zero.
#  (c) the specialised kernels are cross-checked against the generic ones:
#      --nosimd/--force16/--noearly/--nolist must reach the identical state.
cd "$(dirname "$0")"
VER=../../verify_cov.py
fail=0

check() {   # q n R M seed iters extra...
  local Q=$1 N=$2 R=$3 M=$4 S=$5 IT=$6; shift 6
  local out rc claim real
  out=$(nice -n 15 ./covfast -q $Q -n $N -R $R -M $M -s $S -t 3600 --iters $IT \
        --selftest --threads 2 --quiet --out /tmp/st_$$.txt "$@" 2>&1)
  rc=$?
  if [ $rc -ge 2 ]; then echo "FAIL(selftest) q=$Q n=$N R=$R M=$M $*: $out"; fail=1; return; fi
  claim=$(echo "$out" | sed -n 's/.*uncovered=\([0-9]*\).*/\1/p')
  real=$(nice -n 15 python3 $VER -q $Q -n $N -R $R /tmp/st_$$.txt 2>&1 |
         sed -n 's/^method [a-z]* *: uncovered=\([0-9]*\).*/\1/p' | head -1)
  if [ "$claim" != "$real" ]; then
     echo "FAIL(verify) q=$Q n=$N R=$R M=$M $*: solver=$claim verifier=$real"; fail=1
  else
     printf "  ok  q=%-2s n=%-2s R=%-2s M=%-4s %-30s uncovered=%s\n" $Q $N $R $M "$*" "$claim"
  fi
  rm -f /tmp/st_$$.txt
}

# kernel agreement: specialised vs generic must reach the same state
agree() {   # q n R M seed iters
  local Q=$1 N=$2 R=$3 M=$4 S=$5 IT=$6
  local a b c d
  a=$(nice -n 15 ./covfast -q $Q -n $N -R $R -M $M -s $S -t 3600 --iters $IT --threads 1 --quiet --modrng --nolist --noearly | tail -1)
  b=$(nice -n 15 ./covfast -q $Q -n $N -R $R -M $M -s $S -t 3600 --iters $IT --threads 1 --quiet --modrng --nolist --noearly --nosimd | tail -1)
  c=$(nice -n 15 ./covfast -q $Q -n $N -R $R -M $M -s $S -t 3600 --iters $IT --threads 1 --quiet --modrng --nolist --noearly --force16 --nosimd | tail -1)
  d=$(nice -n 15 ./baseline -q $Q -n $N -R $R -M $M -s $S -t 3600 --iters $IT --threads 1 --quiet | tail -1)
  strip() { echo "$1" | sed 's/ promote=[0-9]*//; s/ time=.*//'; }
  if [ "$(strip "$a")" = "$(strip "$b")" ] && [ "$(strip "$a")" = "$(strip "$c")" ] \
     && [ "$(strip "$a")" = "$(strip "$d")" ]; then
     printf "  ok  trajectory identical to baseline: %s\n" "$(strip "$a")"
  else
     echo "FAIL(agree) q=$Q n=$N R=$R M=$M"; echo "   simd:$a"; echo " nosimd:$b"
     echo "    u16:$c"; echo "   base:$d"; fail=1
  fi
}

echo "== move-by-move selftest + independent verification =="
check 6 6 3 41  1 400
check 6 8 4 169 1 60
check 3 11 4 81 1 200
check 8 9 4 940 1 3
check 2 8 2 25  1 200
check 3 5 1 30  1 300
check 4 4 1 30  1 300
check 5 5 4 8   1 200      # R = n-1
check 6 4 0 900 1 300      # R = 0
check 7 3 1 25  1 300
check 6 6 3 41  2 400 --force16
check 6 6 3 41  3 400 --nosimd
check 6 6 3 41  4 400 --nolist
check 6 6 3 41  5 400 --noearly
check 10 4 1 50 1 300      # q > 8: NEON path must be off for uint16
check 9 4 2 40  1 300
check 6 6 3 41  6 400 --satat 2      # force the uint8 -> uint16 promotion path
check 6 8 4 169 6 60  --satat 3
check 3 11 4 81 6 200 --satat 4
check 8 9 4 940 6 3   --satat 2

echo "== specialised kernels agree with the generic path and the baseline =="
agree 6 6 3 41  1 2000
agree 3 11 4 81 1 500
agree 6 8 4 169 1 100
agree 5 5 2 30  1 2000
agree 7 4 2 30  1 2000

echo "== uint8 with forced promotion reproduces the uint16 trajectory =="
for c in "6 6 3 41 1 2000" "6 8 4 169 1 100" "3 11 4 81 1 500"; do
  set -- $c
  a=$(nice -n 15 ./covfast -q $1 -n $2 -R $3 -M $4 -s $5 -t 3600 --iters $6 --threads 1 --quiet --modrng --nolist --noearly --force16 | tail -1 | sed 's/ promote=[0-9]*//; s/ time=.*//')
  for sa in 2 3 7 250; do
    b=$(nice -n 15 ./covfast -q $1 -n $2 -R $3 -M $4 -s $5 -t 3600 --iters $6 --threads 1 --quiet --modrng --nolist --noearly --satat $sa | tail -1 | sed 's/ promote=[0-9]*//; s/ time=.*//')
    if [ "$a" != "$b" ]; then echo "FAIL(promote) q=$1 n=$2 R=$3 satat=$sa"; echo "  u16:$a"; echo "  u8 :$b"; fail=1; fi
  done
  echo "  ok  q=$1 n=$2 R=$3 M=$4: $a"
done

[ $fail = 0 ] && echo "ALL PASS" || echo "FAILURES"
exit $fail
