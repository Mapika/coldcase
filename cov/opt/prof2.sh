#!/bin/bash
# Post-optimisation profile + direct measurement of the early-exit pruning rate.
# All runs start from the fixed code files so the state is the benchmark state.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
R="nice -n 15 ./covsearch2_prof"
mkdir -p prof
$R -q 6 -n 6 -R 3 -M 41   --in seeds_K6_6_3_M41.txt   --iters 20000 -t 99999 -s 1 --threads 1 --quiet --preset p5 > prof/small_p5.txt 2>&1
$R -q 6 -n 8 -R 4 -M 169  --in seeds_K6_8_4_M169.txt  --iters 3000  -t 99999 -s 1 --threads 1 --quiet --preset p5 > prof/med_p5.txt 2>&1
$R -q 8 -n 9 -R 4 -M 2944 --in seeds_K8_9_4_M2944.txt --iters 150   -t 99999 -s 1 --threads 1 --quiet --preset p5 > prof/large_p5.txt 2>&1
# baseline profile from the same fixed start, for a like-for-like comparison
$R -q 6 -n 6 -R 3 -M 41   --in seeds_K6_6_3_M41.txt   --iters 20000 -t 99999 -s 1 --threads 1 --quiet --preset base > prof/small_base2.txt 2>&1
$R -q 6 -n 8 -R 4 -M 169  --in seeds_K6_8_4_M169.txt  --iters 600   -t 99999 -s 1 --threads 1 --quiet --preset base > prof/med_base2.txt 2>&1
$R -q 8 -n 9 -R 4 -M 2944 --in seeds_K8_9_4_M2944.txt --iters 40    -t 99999 -s 1 --threads 1 --quiet --preset base > prof/large_base2.txt 2>&1
# early-exit pruning rate: --early only applies to the per-candidate evaluator (p2)
$R -q 6 -n 6 -R 3 -M 41   --in seeds_K6_6_3_M41.txt   --iters 8000 -t 99999 -s 1 --threads 1 --quiet --preset p2 --early > prof/small_early.txt 2>&1
$R -q 6 -n 8 -R 4 -M 169  --in seeds_K6_8_4_M169.txt  --iters 400  -t 99999 -s 1 --threads 1 --quiet --preset p2 --early > prof/med_early.txt 2>&1
$R -q 8 -n 9 -R 4 -M 2944 --in seeds_K8_9_4_M2944.txt --iters 25   -t 99999 -s 1 --threads 1 --quiet --preset p2 --early > prof/large_early.txt 2>&1
echo PROF2_DONE
grep -h "early_saved" prof/*_early.txt
