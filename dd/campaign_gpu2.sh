#!/usr/bin/env bash
# GPU campaign, part 2: the odd-degree cells (the engine now supports one
# involution in the connection set, so odd degrees work).  These are the softest
# targets on the board -- (11,5), (13,5), (15,5) were all set in August 2026 as
# Cayley graphs of semidirect products, and (9,4)/(7,5) are older ones in the same
# family (the engine re-finds (9,4)=1640 in Z_41 : Z_40, the record's own group,
# in well under a minute).
set -u
cd "$(dirname "$0")"
mkdir -p results/raw
BIN=./src/dd_gpu
go() {
  echo "##### GPU (${1},${2}) N=${3}..${4} ${5}s -- $(date -u +%H:%M:%S) #####"
  $BIN --delta $1 --diam $2 --Nmin $3 --Nmax $4 --time $5 \
       --blocks $6 --threads $7 --iters 30000 --maxa 24 --minn 3 \
       --out results/raw/${8}.jsonl 2>&1 | tail -2
}
go 15 5  79153  79500 900 1024 256 gpu15_5   # 79152 Wupperfeld, Aug 6 2026
go 13 5  42681  43000 900 1024 256 gpu13_5   # 42680 Yugeswardeenoo, Aug 2026
go 11 5  20647  20980 900 2048 128 gpu11_5   # 20646 Yugeswardeenoo, Aug 2026
go  9 5   8761   9100 900 2048 128 gpu09_5   # 8760  Yugeswardeenoo, Aug 2026
go  9 4   1641   1760 900 2048 128 gpu09_4   # 1640  Comellas 2024 (Z_41 : Z_40); CPU found f=2 at N=1650
go  7 5   2757   3080 600 2048 128 gpu07_5   # 2756  Loz 2006 (Z_53 : Z_52)
go  7 4    673    900 600 2048 128 gpu07_4   # 672   Loz 2006
go  5 4    213    400 600 2048 128 gpu05_4   # 212   Exoo 2010
go 14 3   1027   1120 900 2048 128 gpu14_3   # 1026 found here; CPU left f=1 at N=1032
go 10 5  13735  14060 900 2048 128 gpu10_5   # 13734 Yugeswardeenoo, Aug 2026
go  8 5   5254   5600 900 2048 128 gpu08_5   # 5253  Yugeswardeenoo, Aug 2026
go 12 5  34993  35320 900 1024 256 gpu12_5   # 34992 Mizuno, Jul 2026
go 10 4   2486   2760 900 2048 128 gpu10_4   # 2485  Yugeswardeenoo, Aug 2026
go  6 5   1405   1700 600 2048 128 gpu06_5   # 1404  Loz 2006
go  8 4   1101   1400 600 2048 128 gpu08_4   # 1100  Loz 2006
echo "##### GPU CAMPAIGN 2 DONE $(date -u) #####"
