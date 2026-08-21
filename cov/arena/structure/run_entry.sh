#!/bin/bash
# arena entry "structure" -- run_entry.sh Q N R M SEED TIME_S OUTFILE
#
# Angle: exploit the invariances of the problem instead of raw search speed.
# Three engines, ordered by how much structure they impose, plus a portfolio so
# that over-constraining can never cost us the answer:
#
#   lincov      the code is a union of cosets of a linear [n,k]_q code.  The
#               covering condition collapses to  U_i (sigma_i + S) = GF(q)^m,
#               i.e. from q^n words to q^(n-k) syndromes.  Applies only when q
#               is a prime power and q^k | M -- but where it applies it is
#               decisive (K_3(11,4) at M=81 falls in about 10 ms).
#   symsearch   the code is invariant under x -> x + 1^n.  The search runs in
#               the quotient Z_q^(n-1): q times less memory, q codewords moved
#               per move, M/q free variables instead of M.
#   freesearch  the baseline focused local search, symmetry broken -- seeded
#               either by symsearch's invariant code or by nothing at all.
#
# Schedule: symmetry first, then free.  Each worker runs symsearch to fix most
# of the code, then hands it to freesearch, which tops it up to M words and
# relaxes the rest.  How much of the code stays invariant is the knob, and on
# small cells the six workers spread along it (see the table in phase 2) rather
# than betting on one setting -- two of them stay fully free throughout, so an
# over-constrained symmetry costs at most some cores, never the answer.
# covcount arbitrates by recomputing coverage from the files themselves.
#
# nice -n 15 throughout, 6 threads total, and the counter arrays are the only
# memory of note (<= 300 MB on these cells, far under the 25 GB cap).

set -u
if [ $# -lt 7 ]; then
  echo "usage: run_entry.sh Q N R M SEED TIME_S OUTFILE" >&2
  exit 2
fi
Q=$1; N=$2; R=$3; M=$4; SEED=$5; TLIM=$6; OUT=$7
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

for b in lincov symsearch freesearch covcount; do
  if [ ! -x "$DIR/$b" ]; then (cd "$DIR" && make -s "$b") >/dev/null 2>&1; fi
done

WORK=$(mktemp -d "${TMPDIR:-/tmp}/structure.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
: > "$OUT"

now() { date +%s.%N; }
T0=$(now)
elapsed() { awk -v a="$T0" -v b="$(now)" 'BEGIN{printf "%.3f", b-a}'; }

# ---- cell arithmetic: ball volume V, space size q^n, cost estimates ---------
read -r QN BV <<EOF
$(awk -v q="$Q" -v n="$N" -v R="$R" 'BEGIN{
    qn=1; for(i=0;i<n;i++) qn*=q;
    v=0; cb=1;
    for(i=0;i<=R && i<=n;i++){ cb=1; for(j=0;j<i;j++) cb=cb*(n-j)/(j+1);
                               t=cb; for(j=0;j<i;j++) t*=(q-1); v+=t; }
    printf "%.6g %.6g", qn, v }')
EOF

# Publishing costs: the finisher is `covcount --fast` (de-duplicate, top up to M
# distinct words, write -- no coverage array, instant at any size) whenever the
# space is big, and the full recount only where that is genuinely cheap.
if awk -v qn="$QN" 'BEGIN{exit !(qn > 5e6)}'; then FINISH=--fast; else FINISH=""; fi
RES=$(awk -v qn="$QN" -v v="$BV" -v m="$M" -v t="$TLIM" -v f="$FINISH" 'BEGIN{
    c = (f=="--fast") ? 0 : (qn/1.2e8 + m*v/1.2e8);
    r = 2.2*c + 1.5; if(r>0.25*t) r=0.25*t; printf "%.3f", r }')
BUD=$(awk -v t="$TLIM" -v r="$RES" 'BEGIN{b=t-r-1.0; if(b<0.5)b=0.5; printf "%.3f", b}')

# one freesearch initialisation: M*V ball marks into a 2*q^n byte array.  The
# per-mark cost is ~1e-8 s while that array is cache resident and ~9e-8 s once
# it is not (measured: 27.5 s for 3.13e8 marks into 262 MB on K_8(9,4), and
# nearly twice that when the machine is loaded).
INITF=$(awk -v qn="$QN" -v v="$BV" -v m="$M" 'BEGIN{
    f = qn/2e7; if (f>1) f=1;
    printf "%.3f", m*v*(1e-8 + 8e-8*f) }')

# Publish a legal answer straight away.  It covers almost nothing, but from this
# line on there is no execution path -- overrun, kill, OOM -- that leaves the
# judge an unparseable OUTFILE.
nice -n 15 "$DIR/covcount" $FINISH -q "$Q" -n "$N" -R "$R" -M "$M" \
    --in /dev/null --out "$OUT" >/dev/null 2>&1

BEST=""; BESTU=""

keep() {  # remember $1 if its uncovered count $2 is the smallest so far
  [ -s "$1" ] || return 0
  case "$2" in ''|*[!0-9]*) return 0;; esac
  if [ -z "$BESTU" ] || [ "$2" -lt "$BESTU" ]; then BEST=$1; BESTU=$2; fi
}

emit() {  # de-duplicate, top up to M distinct words, publish, done
  local src="$WORK/empty"
  : > "$WORK/empty"
  [ -n "$BEST" ] && [ -s "$BEST" ] && src="$BEST"
  timeout -s KILL 25 nice -n 15 "$DIR/covcount" $FINISH -q "$Q" -n "$N" -R "$R" \
      -M "$M" --in "$src" --out "$OUT" >/dev/null 2>&1
  # if the finisher was cut short, publish the raw best rather than nothing
  [ -s "$OUT" ] || { [ -s "$src" ] && cp "$src" "$OUT"; }
  exit 0
}

unc_of() {  # pull the uncovered-word count out of a solver's RESULT line
  awk '{for(i=1;i<=NF;i++){ if($i ~ /^uncovered=/){split($i,a,"=");u=a[2]}
                            if($i ~ /^uncovered_words=/){split($i,a,"=");u=a[2]}
                            if($i ~ /^word_uncovered=/){split($i,a,"=");u=a[2]} }}
       END{ if(u=="") print -1; else print u }' "$1" 2>/dev/null
}

hard() {  # hard wall-clock backstop for a child, in whole seconds
  awk -v x="$1" 'BEGIN{ h=int(x)+2; if(h<2) h=2; print h }'
}

# ---------------------------------------------------------------- phase 1 ---
# Coset-of-a-linear-code search.  k walks downwards: large k means few cosets
# and the most structure, small k means many cosets and a search closer to free.
# lincov exits 3 instantly when the arithmetic does not fit, so a cell it cannot
# handle costs nothing at all.
TLIN=$(awk -v b="$BUD" 'BEGIN{x=0.12*b; if(x>15)x=15; if(x<0.4)x=0.4; printf "%.3f", x}')
SLICE=$(awk -v x="$TLIN" 'BEGIN{printf "%.3f", x/3}')
tried=0
for k in 12 11 10 9 8 7 6 5 4 3 2 1; do
  [ "$tried" -ge 3 ] && break
  timeout -s TERM -k 3 "$(hard "$SLICE")" \
      nice -n 15 "$DIR/lincov" -q "$Q" -n "$N" -R "$R" -M "$M" -k "$k" \
      -t "$SLICE" -s "$SEED" --out "$WORK/lin$k.txt" > "$WORK/lin$k.log" 2>&1
  rc=$?
  [ "$rc" -eq 3 ] && continue
  tried=$((tried+1))
  keep "$WORK/lin$k.txt" "$(unc_of "$WORK/lin$k.log")"
  [ "$rc" -eq 0 ] && emit
done

REM=$(awk -v b="$BUD" -v e="$(elapsed)" 'BEGIN{r=b-e; if(r<0.3)r=0.3; printf "%.3f", r}')

# ---------------------------------------------------------------- phase 2 ---
# The measured shape of the symmetry-then-free trade-off, on K_6(8,4) at M=169
# (55 s symmetric + 55 s free, 3 seeds each, uncovered words at the end):
#
#     orbits    28      27      26      24      20      0 (pure free)
#     words    168     162     156     144     120      -
#     free       1       7      13      25      49      169
#     result   9,11,6  0,0,6   0,0,0   0,0,0   3,3,2   4,3,8
#
# Neither extreme solves it.  A fully invariant code plateaus (with one spare
# word you cannot repair an orbit: sum_j |{i : c_i = w_i + j}| = n, so a single
# codeword can be within R of at most two members of a q-orbit).  A fully free
# search plateaus too.  Structure for most of the code plus a genuinely free
# remainder solves it every time.  So on cells small enough to restart cheaply
# we run a portfolio across that axis, one thread per worker, rather than
# betting on one point of it.
#
# On a big cell the axis collapses: one freesearch initialisation alone eats
# 28 s on K_8(9,4) (vs 6 s for symsearch's q-times-smaller quotient), so there
# is no budget for a free remainder and the invariant search takes the cores.
# (The q^n cap is also a memory guard: small mode runs six solvers at once, each
# holding 2*q^n bytes of counters, so it is only used where that is small.)
NBORB=$(( M / Q ))
if awk -v i="$INITF" -v r="$REM" -v qn="$QN" 'BEGIN{exit !(i > 0.2*r || qn > 5e7)}' \
   || [ "$NBORB" -lt 2 ]; then
  MODE=big
else
  MODE=small
fi

# worker TAG ORBITS SEED THREADS OUTFILE  -- symmetric phase then free phase
worker() {
  local tag=$1 orb=$2 sd=$3 thr=$4 dst=$5
  local ts=0 tf="$REM" t1
  if [ "$orb" -gt 0 ]; then
    ts=$(awk -v r="$REM" 'BEGIN{printf "%.3f", 0.45*r}')
    t1=$(now)
    timeout -s TERM -k 3 "$(hard "$ts")" \
        nice -n 15 "$DIR/symsearch" -q "$Q" -n "$N" -R "$R" -M "$M" --orbits "$orb" \
        -t "$ts" -s "$sd" --threads "$thr" --out "$WORK/$tag.sym" \
        > "$WORK/$tag.symlog" 2>&1
    tf=$(awk -v r="$REM" -v a="$t1" -v b="$(now)" 'BEGIN{printf "%.3f", r-(b-a)}')
  fi
  if awk -v l="$tf" 'BEGIN{exit !(l > 0.5)}'; then
    if [ "$orb" -gt 0 ] && [ -s "$WORK/$tag.sym" ]; then
      timeout -s TERM -k 3 "$(hard "$tf")" \
          nice -n 15 "$DIR/freesearch" -q "$Q" -n "$N" -R "$R" -M "$M" -t "$tf" \
          -s "$sd" --threads "$thr" --in "$WORK/$tag.sym" --out "$dst" \
          > "$WORK/$tag.log" 2>&1
    else
      timeout -s TERM -k 3 "$(hard "$tf")" \
          nice -n 15 "$DIR/freesearch" -q "$Q" -n "$N" -R "$R" -M "$M" -t "$tf" \
          -s "$sd" --threads "$thr" --out "$dst" > "$WORK/$tag.log" 2>&1
    fi
  fi
}

TAGS=""
if [ "$MODE" = big ]; then
  # Measured on K_8(9,4) at M=940: symsearch reaches ~6.2 M uncovered where
  # freesearch reaches ~8.9 M, and running the two together is worse than
  # running symsearch alone -- freesearch's 262 MB counter array saturates
  # memory bandwidth and, on a loaded machine, its initialisation alone can
  # outlast the whole budget without producing a single line.  So the invariant
  # search gets all six threads, and the free solver is kept only as the
  # fallback for a cell symsearch cannot represent at all.
  timeout -s TERM -k 2 "$(hard "$REM")" \
      nice -n 15 "$DIR/symsearch" -q "$Q" -n "$N" -R "$R" -M "$M" -t "$REM" \
      -s "$SEED" --threads 6 --out "$WORK/w1.sym" > "$WORK/w1.symlog" 2>&1
  TAGS="w1"
  if [ ! -s "$WORK/w1.sym" ]; then
    LEFT=$(awk -v b="$BUD" -v e="$(elapsed)" 'BEGIN{r=b-e; printf "%.3f", r}')
    if awk -v l="$LEFT" 'BEGIN{exit !(l > 1.0)}'; then
      timeout -s TERM -k 2 "$(hard "$LEFT")" \
          nice -n 15 "$DIR/freesearch" -q "$Q" -n "$N" -R "$R" -M "$M" -t "$LEFT" \
          -s "$SEED" --threads 6 --out "$OUT" > "$WORK/w0.log" 2>&1
      TAGS="w0 w1"
    fi
  fi
else
  # spread the workers along the symmetry axis: all orbits, then progressively
  # fewer, plus two fully free runs
  STEP=$(awk -v b="$NBORB" 'BEGIN{s=int(b*0.07+0.5); if(s<1)s=1; print s}')
  worker w0 0 "$SEED" 1 "$OUT" &
  worker w1 "$NBORB" "$SEED" 1 "$WORK/w1.txt" &
  worker w2 $(( NBORB - STEP > 0 ? NBORB - STEP : 1 ))     $((SEED+1)) 1 "$WORK/w2.txt" &
  worker w3 $(( NBORB - 2*STEP > 0 ? NBORB - 2*STEP : 1 )) $((SEED+2)) 1 "$WORK/w3.txt" &
  worker w4 $(( NBORB - 3*STEP > 0 ? NBORB - 3*STEP : 1 )) $((SEED+3)) 1 "$WORK/w4.txt" &
  worker w5 0 $((SEED+5)) 1 "$WORK/w5.txt" &
  wait
  TAGS="w0 w1 w2 w3 w4 w5"
fi

for t in $TAGS; do
  dst="$WORK/$t.txt"; [ "$t" = w0 ] && dst="$OUT"
  [ -s "$WORK/$t.log" ]    && keep "$dst"          "$(unc_of "$WORK/$t.log")"
  # the invariant code itself is a candidate too -- it has q*orbits words and
  # covcount tops it up -- in case the free phase never got going
  [ -s "$WORK/$t.symlog" ] && keep "$WORK/$t.sym"  "$(unc_of "$WORK/$t.symlog")"
done

emit
