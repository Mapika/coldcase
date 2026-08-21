#!/bin/bash
# Drive the record hunt with the merged engine, inside the resource contract:
# <= 40 cores total, nice 12, no GPU.  Each campaign2 run is given CORES cores
# and they are run one after another, so the total never exceeds CORES.
#
#   run_attacks.sh CORES SECONDS_PER_M
set -u
E=$(cd "$(dirname "$0")" && pwd)
CORES=${1:-40}
T=${2:-1800}
mkdir -p "$E/logs"
run() {  # name  args...
  local name=$1; shift
  echo "=== $(date +%H:%M:%S)  $name" >&2
  nice -n 12 python3 "$E/campaign2.py" --cores "$CORES" -t "$T" "$@" \
      > "$E/logs/$name.log" 2>&1
  grep -E 'BEATS KERI|VERIFIED and stored|=== recorded' "$E/logs/$name.log" >&2 || true
}

# (c) the failed sieges first: two of the three are algebraic at the incumbent,
#     so phase A gives the descent a real starting point in under a second.
run sieges   --attack "$E/attack_sieges.json" --keep-going
# (b) every record cell, one notch below our current best
run records  --attack "$E/attack_records.json"
echo "=== $(date +%H:%M:%S)  done" >&2
