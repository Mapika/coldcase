#!/bin/bash
# Ablation on a fixed iteration budget, single thread.
#
# The box is shared with a nice-10 production campaign that keeps it 2-4x
# oversubscribed, so WALL time is meaningless here; every run reports its own
# CLOCK_PROCESS_CPUTIME_ID, and we take the best of REPS runs.
#
# Every configuration except the last uses --modrng, which makes covfast
# consume the RNG exactly like the baseline and therefore follow a
# BIT-IDENTICAL search trajectory.  The printed fingerprint
# "uncovered/kicks/candidates-evaluated" proves it: identical fingerprint =
# identical work, so the CPU-time ratio is a pure low-level speedup.
#
# Usage: bench.sh Q N R M ITERS [REPS]
cd "$(dirname "$0")"
Q=$1; N=$2; R=$3; M=$4; IT=$5; REPS=${6:-3}

TAGS=("baseline"
      "specialised kernels"
      "  + uint8 counters"
      "  + NEON"
      "  + early exit"
      "  + uncovered list (traj differs)")
CMDS=("baseline"
      "covfast --modrng --nosimd --force16 --noearly --nolist"
      "covfast --modrng --nosimd --noearly --nolist"
      "covfast --modrng --noearly --nolist"
      "covfast --modrng --nolist"
      "covfast --modrng")

declare -a BEST FP
for i in "${!CMDS[@]}"; do BEST[$i]=999999; done

for ((r=0;r<REPS;r++)); do
  for i in "${!CMDS[@]}"; do
    out=$(nice -n 15 ./${CMDS[$i]} -q $Q -n $N -R $R -M $M -s 1000 -t 100000 \
          --iters $IT --threads 1 --quiet 2>&1 | tail -2)
    t=$(echo "$out" | sed -n 's/^CPU \([0-9.]*\)/\1/p')
    FP[$i]=$(echo "$out" | sed -n 's/.*uncovered=\([0-9]*\) iters=[0-9]* kicks=\([0-9]*\) cands=\([0-9]*\).*/\1\/\2\/\3/p')
    BEST[$i]=$(python3 -c "print(min(${BEST[$i]},$t))")
  done
done

echo "K$Q($N,$R)@$M  iters=$IT reps=$REPS  (best-of CPU seconds, 1 thread)"
b=${BEST[0]}
for i in "${!CMDS[@]}"; do
  printf "  %-32s %8ss cpu  x%-6s traj=%s\n" "${TAGS[$i]}" "${BEST[$i]}" \
     "$(python3 -c "print(round($b/${BEST[$i]},2))")" "${FP[$i]}"
done
