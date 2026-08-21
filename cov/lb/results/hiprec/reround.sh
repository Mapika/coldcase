#!/bin/bash
cd /lambda/nfs/new-fs/longshots/cov/lb
# default-lstar cells
for c in "7 10 2" "21 8 3" "17 8 3" "9 10 3" "7 9 2" "9 9 3" "7 9 3" "19 7 3" "20 7 3" "20 8 3" "9 8 3" "14 8 3" "7 8 2" "8 8 2" "6 8 2" "7 7 2" "10 7 2" "16 7 3" "17 7 3" "15 7 3" "18 8 4" "19 8 4" "20 8 4" "21 8 4" "17 8 4" "4 9 1"; do
  set -- $c
  nice -n 10 python3 solve_hp.py $1 $2 $3 --reuse --out certs_hp/cert_q${1}_n${2}_R${3}.json >> results/hiprec/log_q${1}_n${2}_R${3}.txt 2>&1
  echo "REROUND $1 $2 $3 rc=$? $(grep 'certified SDP value' results/hiprec/log_q${1}_n${2}_R${3}.txt | tail -1)" >> results/hiprec/sweep.log
done
# lstar=1e2 cells
for c in "18 7 3" "12 8 3" "13 8 3" "15 8 3" "16 8 3" "9 8 2"; do
  set -- $c
  nice -n 10 python3 solve_hp.py $1 $2 $3 --reuse --lstar 1.0E2 --out certs_hp/cert_q${1}_n${2}_R${3}.json >> results/hiprec/log_q${1}_n${2}_R${3}.txt 2>&1
  echo "REROUND $1 $2 $3 rc=$? $(grep 'certified SDP value' results/hiprec/log_q${1}_n${2}_R${3}.txt | tail -1)" >> results/hiprec/sweep.log
done
nice -n 10 python3 solve_hp.py 21 7 3 --reuse --lstar 1.0E2 --workdir /tmp/claude-1000/-lambda-nfs-new-fs-longshots/3ae6d111-07fa-4880-8ce4-cca841595d21/scratchpad/dbg --out certs_hp/cert_q21_n7_R3.json >> results/hiprec/log_q21_n7_R3.txt 2>&1
echo "REROUND 21 7 3 rc=$?" >> results/hiprec/sweep.log
echo "REROUND-ALL-DONE" >> results/hiprec/sweep.log
