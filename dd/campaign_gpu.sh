#!/usr/bin/env bash
# GPU campaign.  Runs concurrently with the CPU campaign: dd_gpu uses the GH200
# and one host thread, dd_search2 uses the 64 ARM cores.
#
# dd_gpu handles even degrees only (its connection sets are built from inverse
# pairs, no involutions), and it pays off in the large-N regime where the CPU's
# bitset spills out of L1:
#
#   delta=14 D=3 N~1000    CPU 30 Mevals/s   GPU  40 Mevals/s   (parity)
#   delta=14 D=5 N~79000   CPU 0.29 Mevals/s GPU 1.72 Mevals/s  (~6x)
#
# so the GPU takes the big even-degree D=5 cells and the CPU keeps the rest.
set -u
cd "$(dirname "$0")"
mkdir -p results/raw
BIN=./src/dd_gpu

go() {  # delta D Nmin Nmax secs blocks threads tag
  echo "##### GPU (${1},${2}) N=${3}..${4} ${5}s -- $(date -u +%H:%M:%S) #####"
  $BIN --delta $1 --diam $2 --Nmin $3 --Nmax $4 --time $5 \
       --blocks $6 --threads $7 --iters 30000 --maxa 24 --minn 3 \
       --out results/raw/${8}.jsonl 2>&1 | tail -2
}

#   cell    record   (softness)
go 14 5  60391  60700 900 1024 256 gpu14_5    # 60390 Badaoui, Aug 16 2026
go 10 5  13735  14050 900 2048 128 gpu10_5    # 13734 Yugeswardeenoo, Aug 2026
go  8 5   5254   5560 900 2048 128 gpu08_5    # 5253  Yugeswardeenoo, Aug 2026
go 12 5  34993  35300 900 1024 256 gpu12_5    # 34992 Mizuno, Jul 2026
go 10 4   2486   2760 900 2048 128 gpu10_4    # 2485  Yugeswardeenoo, Aug 2026
go  6 5   1405   1700 600 2048 128 gpu06_5    # 1404  Loz 2006 (semidirect)
go 16 5 147457 147800 900 1024 256 gpu16_5    # 147456 Mizuno, Jul 2026
go  8 4   1101   1400 600 2048 128 gpu08_4    # 1100  Loz 2006 (semidirect)
go 14 3   1027   1500 900 2048 128 gpu14_3    # 979 published; 1026 already found here
go  6 4    391    560 600 2048 128 gpu06_4    # 390   Loz 2006 (semidirect)
echo "##### GPU CAMPAIGN DONE $(date -u) #####"
