#!/usr/bin/env bash
# Targeted campaign, ordered by (softness of the incumbent) x (reachability).
#
# classify_cells.py scrapes the per-cell description pages and records which
# construction each standing record uses.  Cells whose record is itself a Cayley
# graph of a semidirect product are the ones this engine provably reaches, so a
# systematic sweep there should beat the casual search that set them.  Cells whose
# record is a compound graph or a generalized-quadrangle polarity quotient with
# vertex additions -- (11,4), (12,3), (12,4), (13,3), (13,4), (14,4), (15,3),
# (15,4), (16,3), (16,4) -- are structurally out of reach for a pure Cayley search
# and are not attacked here.
#
#  tier 1  recent (2024-2026) records that are semidirect-product Cayley graphs
#          (9,4)=1640 Comellas'24   (10,4)=2485 Yugeswardeenoo'26
#          (11,5)=20646 (13,5)=42680 (14,5)=60390 (15,5)=79152  Aug'26
#          (14,3): already beaten at 1026 -- push it further
#  tier 2  older semidirect-product records (Loz'06, Exoo, Comellas-Mitjana'94)
#          (7,5)=2756 (6,5)=1404 (8,4)=1100 (7,4)=672 (6,4)=390 (5,4)=212 (8,3)=253
#  tier 3  other Cayley records  (16,2)=200 (6,3)=111 (7,3)=168
set -u
cd "$(dirname "$0")"
F="python3 focus.py"
go() { echo "##### focus $* -- $(date -u +%H:%M:%S) #####"; $F "$@" 2>&1 | tail -35; }

# ---- tier 1 -----------------------------------------------------------------
go --delta  9 --D 4 --span 300 --scan-secs 240 --focus-secs 150 --topk 6 --tag h09_4
go --delta 10 --D 4 --span 260 --scan-secs 240 --focus-secs 150 --topk 6 --tag h10_4
go --delta 14 --D 3 --lo 1027 --hi 1450 --scan-secs 240 --focus-secs 150 --topk 6 --tag h14_3
go --delta 11 --D 5 --span 300 --scan-secs 240 --focus-secs 150 --topk 6 --maxa 24 --tag h11_5
go --delta 13 --D 5 --span 300 --scan-secs 240 --focus-secs 150 --topk 6 --maxa 24 --tag h13_5
go --delta 14 --D 5 --span 300 --scan-secs 240 --focus-secs 150 --topk 6 --maxa 24 --tag h14_5
go --delta 15 --D 5 --span 300 --scan-secs 240 --focus-secs 150 --topk 6 --maxa 24 --tag h15_5
go --delta 10 --D 5 --span 300 --scan-secs 240 --focus-secs 150 --topk 6 --maxa 24 --tag h10_5
go --delta  9 --D 5 --span 300 --scan-secs 240 --focus-secs 150 --topk 6 --maxa 32 --tag h09_5
go --delta  8 --D 5 --span 250 --scan-secs 240 --focus-secs 150 --topk 6 --maxa 32 --tag h08_5
# ---- tier 2 -----------------------------------------------------------------
go --delta  7 --D 5 --span 300 --scan-secs 240 --focus-secs 150 --topk 6 --tag h07_5
go --delta  6 --D 5 --span 250 --scan-secs 240 --focus-secs 150 --topk 6 --tag h06_5
go --delta  8 --D 4 --span 250 --scan-secs 240 --focus-secs 150 --topk 6 --tag h08_4
go --delta  7 --D 4 --span 200 --scan-secs 240 --focus-secs 150 --topk 6 --tag h07_4
go --delta  6 --D 4 --span 140 --scan-secs 240 --focus-secs 150 --topk 6 --tag h06_4
go --delta  5 --D 4 --span  90 --scan-secs 240 --focus-secs 150 --topk  8 --tag h05_4
go --delta  8 --D 3 --span  90 --scan-secs 240 --focus-secs 150 --topk 6 --minn 2 --tag h08_3
# ---- tier 3 -----------------------------------------------------------------
go --delta 16 --D 2 --span  58 --scan-secs 240 --focus-secs 150 --topk 6 --minn 2 --tag h16_2
go --delta  6 --D 3 --span  60 --scan-secs 240 --focus-secs  90 --topk  8 --minn 2 --tag h06_3
go --delta  7 --D 3 --span  60 --scan-secs 240 --focus-secs  90 --topk  8 --minn 2 --tag h07_3
echo "##### CAMPAIGN3 DONE $(date -u) #####"
