#!/bin/bash
Q=$1; N=$2; R=$3; M=$4; SEED=$5; T=$6; OUT=$7
exec nice -n 15 timeout -k 5 "$T" /lambda/nfs/new-fs/longshots/cov/search/covsearch \
  -q "$Q" -n "$N" -R "$R" -M "$M" -s "$SEED" -t "$((T>5 ? T-3 : T))" --threads 6 --out "$OUT" --quiet
