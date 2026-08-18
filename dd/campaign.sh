#!/usr/bin/env bash
# Sequential attack campaign: for each soft cell, sweep orders above the record.
# Each line of results/raw/*.jsonl is a Cayley graph that the engine believes has
# the target degree/diameter; every one is re-derived and re-verified afterwards
# by harvest.py (emit_graph.py + verify_dd.py).
set -u
cd "$(dirname "$0")"
mkdir -p results/raw
BIN=./src/dd_search
COMMON="--threads 64 --nonab --faithful --maxa 80 --iters 200000 --nostop --verbose"

run() {   # delta D Nmin Nmax secs tag
  local d=$1 D=$2 lo=$3 hi=$4 t=$5 tag=$6
  echo "===== ATTACK (${d},${D})  N=${lo}..${hi}  ${t}s  [$tag] $(date -u +%H:%M:%S) ====="
  $BIN --delta $d --diam $D --Nmin $lo --Nmax $hi --time $t $COMMON \
       --out results/raw/${tag}.jsonl 2>&1 | tail -60
  local nh=0
  [ -s results/raw/${tag}.jsonl ] && nh=$(wc -l < results/raw/${tag}.jsonl)
  echo "HITS[$tag]=$nh"
  if [ "$nh" -gt 0 ]; then
    echo "CANDIDATE-ORDERS[$tag]: $(python3 -c "
import json,sys
o=sorted({json.loads(l)['N'] for l in open('results/raw/${tag}.jsonl')})
print(o)")"
  fi
}

# cell   delta D  Nmin  Nmax  secs  tag        (record in comment)
run 14 3  1027  1250  2400  a14_3b   # rec 979; VERIFIED 1026 already, push higher
run 10 4  2486  2600  1800  a10_4    # rec 2485  -- engine matches 2485 in 26s
run 15 3  1225  1360  1200  a15_3    # rec 1224  -- engine reached f=10
run  9 4  1641  1780  1200  a09_4    # rec 1640
run 16 3  1611  1760  1200  a16_3    # rec 1610
run 11 4  3221  3400  1500  a11_4    # rec 3220
run 12 3   791   900  1000  a12_3    # rec 790
run 13 3   857   980  1000  a13_3    # rec 856
echo "===== CAMPAIGN DONE $(date -u) ====="
