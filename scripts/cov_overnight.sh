#!/bin/bash
cd /lambda/nfs/new-fs/longshots
LOG=results/cov_overnight.log
run() { echo "=== $(date +%H:%M:%S) $*" >> $LOG; timeout "$1" python3 cov/campaign.py "${@:2}" >> $LOG 2>&1; }
# The near-miss levers first, with sieges
run 10800 -q 6 -n 6 -R 3 --descend-from 41 --workers 60 -t 3000 --rounds 3
run 7200 -q 8 -n 8 -R 4 --descend-from 512 --workers 60 -t 1800 --rounds 3
run 7200 -q 5 -n 11 -R 5 --descend-from 625 --workers 60 -t 1800 --rounds 3
# Deepen every record cell with long budgets
for cell in "6 8 4 167" "6 9 5 125" "7 8 4 331" "6 7 3 233" "7 9 4 1814" "6 9 4 710" "6 8 3 1045" "6 10 5 610" "6 10 4 2951"; do
  set -- $cell
  run 5400 -q $1 -n $2 -R $3 --descend-from $4 --workers 60 -t 1500 --rounds 2
done
echo "=== OVERNIGHT DONE $(date +%H:%M:%S)" >> $LOG
