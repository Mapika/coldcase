#!/usr/bin/env bash
# Self-test for the degree/diameter track.  Everything here is exact; a failure
# means a result from this directory cannot be trusted.
set -u
cd "$(dirname "$0")"
fail=0
ok()   { echo "  PASS  $1"; }
bad()  { echo "  FAIL  $1"; fail=1; }

echo "== build =="
g++ -O3 -march=native -fopenmp -o src/dd_search  src/dd_search.cpp  && ok "dd_search builds"  || bad "dd_search build"
g++ -O3 -march=native -fopenmp -o src/dd_search2 src/dd_search2.cpp && ok "dd_search2 builds" || bad "dd_search2 build"

command -v nvcc >/dev/null && { nvcc -O3 -arch=sm_90 -o src/dd_gpu src/dd_gpu.cu 2>/dev/null && ok "dd_gpu builds" || bad "dd_gpu build"; }

echo "== kernel vs pure Python =="
python3 crosscheck.py  11 120 | tail -1 | grep -q " 0 mismatches" && ok "metacyclic kernel exact" || bad "metacyclic kernel"
python3 crosscheck2.py 11 120 | tail -1 | grep -q " 0 mismatches" && ok "affine2 kernel exact"    || bad "affine2 kernel"
[ -x src/dd_gpu ] && { python3 crosscheck_gpu.py 11 60 | tail -1 | grep -q " 0 mismatches" \
  && ok "CUDA kernel == CPU == Python" || bad "CUDA kernel"; }

echo "== known-value regressions =="
# the 11-cycle is the largest degree-2 diameter-5 graph; N=12 must be impossible
./src/dd_search --delta 2 --diam 5 --Nmin 11 --time 3 --threads 8 2>/dev/null | grep -q '"N":11' \
  && ok "(2,5): finds N=11" || bad "(2,5): should find N=11"
./src/dd_search --delta 2 --diam 5 --Nmin 12 --time 3 --threads 8 --nostop 2>&1 | grep -q "best f=1" \
  && ok "(2,5): N=12 correctly impossible" || bad "(2,5): N=12 should be impossible"
# re-find the published (14,3) record inside its own family
./src/dd_search --delta 14 --diam 3 --Nmin 979 --time 60 --threads 64 --nonab --faithful \
                --maxa 60 --iters 200000 2>/dev/null | grep -q '"N":979' \
  && ok "(14,3): re-finds the published record 979" || bad "(14,3): failed to re-find 979"

echo "== verifier =="
python3 verify_dd.py results/d14_D3_N1026_m342_n3_a49_c1.edges --quiet \
  && ok "verify_dd accepts the (14,3)=1026 certificate" || bad "verify_dd rejects a known-good graph"
python3 verify_vt.py results/d14_D3_N1026_m342_n3_a49_c1.edges >/dev/null \
  && ok "verify_vt agrees on (14,3)=1026" || bad "verify_vt disagrees"
python3 verify_big.py results/d14_D3_N1026_m342_n3_a49_c1.edges >/dev/null \
  && ok "verify_big agrees on (14,3)=1026" || bad "verify_big disagrees"
python3 verify_big.py results/d14_D3_N1026_m342_n3_a49_c1.edges --D 2 >/dev/null \
  && bad "verify_big wrongly accepted diameter 3 as D=2" || ok "verify_big rejects D=2"
if [ -f results/d14_D5_N60452_m2159_n28_a123_gpu.edges ]; then
  python3 verify_vt.py  results/d14_D5_N60452_m2159_n28_a123_gpu.edges >/dev/null \
    && ok "verify_vt accepts the (14,5)=60452 certificate"  || bad "verify_vt rejects (14,5)"
  python3 verify_big.py results/d14_D5_N60452_m2159_n28_a123_gpu.edges >/dev/null \
    && ok "verify_big accepts the (14,5)=60452 certificate" || bad "verify_big rejects (14,5)"
fi
# negative controls
printf 'N 4\n0 1\n1 2\n2 3\n3 0\n' > /tmp/_sq.edges
python3 verify_dd.py /tmp/_sq.edges --delta 2 --D 2 --quiet \
  && ok "verifier accepts C4 as (2,2)" || bad "verifier rejects C4"
python3 verify_dd.py /tmp/_sq.edges --delta 2 --D 1 --quiet >/dev/null \
  && bad "verifier wrongly accepted diameter 2 as D=1" || ok "verifier rejects C4 as (2,1)"
printf 'N 4\n0 1\n2 3\n' > /tmp/_dis.edges
python3 verify_dd.py /tmp/_dis.edges --delta 2 --D 3 --quiet >/dev/null \
  && bad "verifier wrongly accepted a disconnected graph" || ok "verifier rejects disconnected graph"
printf 'N 3\n0 1\n0 1\n1 2\n' > /tmp/_dup.edges
python3 verify_dd.py /tmp/_dup.edges --delta 2 --D 2 --quiet >/dev/null \
  && bad "verifier wrongly accepted a repeated edge" || ok "verifier rejects repeated edge"
printf 'N 3\n0 0\n0 1\n1 2\n' > /tmp/_loop.edges
python3 verify_dd.py /tmp/_loop.edges --delta 2 --D 2 --quiet >/dev/null \
  && bad "verifier wrongly accepted a self-loop" || ok "verifier rejects self-loop"

echo "== emitter guards =="
echo '{"delta":4,"D":2,"N":13,"m":13,"n":1,"a":1,"S":[1,12,5,7]}' > /tmp/_bad.json
python3 emit_graph.py /tmp/_bad.json --outdir /tmp/_e 2>/dev/null \
  && bad "emitter accepted a non-inverse-closed S" || ok "emitter rejects non-inverse-closed S"

rm -rf /tmp/_e /tmp/_sq.edges /tmp/_dis.edges /tmp/_dup.edges /tmp/_loop.edges /tmp/_bad.json
echo
[ $fail -eq 0 ] && echo "ALL SELF-TESTS PASSED" || echo "SELF-TESTS FAILED"
exit $fail
