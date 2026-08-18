#!/usr/bin/env bash
# Second campaign: adaptive two-phase (scan then focus) over more cells,
# including the cheap D=2 row and the larger D=4/D=5 cells.
set -u
cd "$(dirname "$0")"
F="python3 focus.py"

go() { echo "##### $* #####"; $F "$@" 2>&1 | tail -30; }

# --- D=2 row (tiny orders, very high throughput; Moore-ratio is high so these are hard)
go --delta 16 --D 2 --span 58  --scan-secs 90  --focus-secs 120 --topk 10 --minn 2 --tag g16_2
go --delta 15 --D 2 --span 38  --scan-secs 90  --focus-secs 120 --topk 10 --minn 2 --tag g15_2
go --delta 13 --D 2 --span 8   --scan-secs 60  --focus-secs 90  --topk 8  --minn 2 --tag g13_2
go --delta 12 --D 2 --span 12  --scan-secs 60  --focus-secs 90  --topk 8  --minn 2 --tag g12_2

# --- D=3 row, deeper than campaign 1
go --delta 14 --D 3 --lo 1027 --hi 1400 --scan-secs 300 --focus-secs 240 --topk 14 --tag g14_3
go --delta 16 --D 3 --span 300 --scan-secs 300 --focus-secs 240 --topk 14 --tag g16_3
go --delta 15 --D 3 --span 300 --scan-secs 300 --focus-secs 240 --topk 14 --tag g15_3
go --delta 12 --D 3 --span 250 --scan-secs 240 --focus-secs 180 --topk 12 --tag g12_3
go --delta 13 --D 3 --span 250 --scan-secs 240 --focus-secs 180 --topk 12 --tag g13_3
go --delta 11 --D 3 --span 200 --scan-secs 180 --focus-secs 150 --topk 10 --tag g11_3
go --delta 10 --D 3 --span 200 --scan-secs 180 --focus-secs 150 --topk 10 --tag g10_3
go --delta  9 --D 3 --span 200 --scan-secs 180 --focus-secs 150 --topk 10 --tag g09_3

# --- D=4
go --delta 10 --D 4 --span 300 --scan-secs 420 --focus-secs 300 --topk 12 --tag g10_4
go --delta  9 --D 4 --span 250 --scan-secs 360 --focus-secs 300 --topk 12 --tag g09_4
go --delta 11 --D 4 --span 300 --scan-secs 420 --focus-secs 300 --topk 12 --tag g11_4
go --delta  8 --D 4 --span 200 --scan-secs 300 --focus-secs 240 --topk 10 --tag g08_4

# --- D=5
go --delta  8 --D 5 --span 400 --scan-secs 600 --focus-secs 420 --topk 10 --tag g08_5
go --delta  9 --D 5 --span 400 --scan-secs 600 --focus-secs 420 --topk 10 --tag g09_5
echo "##### CAMPAIGN2 DONE $(date -u) #####"
