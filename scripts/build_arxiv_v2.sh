#!/bin/bash
# Assemble the arXiv v2 submission from the CURRENT verified state.
# Usage: bash scripts/build_arxiv_v2.sh
# Steps: tally -> regenerate tables -> rebuild PDF -> refresh anc/ -> tarball.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 cov/engine/tally.py
python3 cov/lb/report.py --certs cov/lb/certs_all --out cov/lb/results/lb_master.json \
    --md cov/lb/results/lb_master.md --jobs 40 | tail -1
python3 scripts/paper_tables.py

# ancillary files: every record code, the 7 record certificates, both verifiers
ANC=paper/covering/anc
rm -rf "$ANC" && mkdir -p "$ANC"
python3 - <<'EOF'
import json, shutil
import numpy as np

recs = json.load(open('cov/results/final_records.json'))
# independent dilation verification of every code that goes into the package
for r in recs:
    q, n, R, M = r['q'], r['n'], r['R'], r['ours']
    path = f"cov/results/{r['code_file']}"
    words = [[int(c, 36) for c in l.strip()] for l in open(path) if l.strip()]
    assert len(set(map(tuple, words))) == M, f"{path}: dup/count"
    assert all(len(w) == n and all(0 <= c < q for c in w) for w in words), f"{path}: alphabet"
    arr = np.zeros((q,) * n, dtype=bool)
    for w in words:
        arr[tuple(w)] = True
    for _ in range(R):
        base = arr
        new = base.copy()
        for ax in range(n):
            new |= base.any(axis=ax, keepdims=True)
        arr = new
    assert int(q ** n - arr.sum()) == 0, f"{path}: NOT A COVER"
    print(f"verified K{q}({n},{R}) <= {M}")
for r in recs:
    shutil.copy(f"cov/results/{r['code_file']}", 'paper/covering/anc/')
lbs = [r for r in json.load(open('cov/lb/results/lb_master.json')) if r['improves_best_known']]
for r in lbs:
    shutil.copy(f"cov/lb/certs_all/{r['cert']}", 'paper/covering/anc/')
print(f"anc/: {len(recs)} codes, {len(lbs)} certificates")
EOF
cp cov/verify_cov.py cov/verify_independent.py cov/lb/certify.py "$ANC/"
cat > "$ANC/README.txt" <<'EOF'
Ancillary files for "New upper and lower bounds on covering codes K_q(n,R)
for alphabets of size 5 <= q <= 21".

K{q}_{n}_{R}_M{M}.txt   -- explicit code achieving K_q(n,R) <= M, one codeword
                           per line, digits 0-9 then a-z for q > 10.
  Verify:   python3 verify_cov.py -q 6 -n 8 -R 4 K6_8_4_M166.txt
            (dependency-free; numpy accelerates if present)
  A second, code-disjoint verifier is included: verify_independent.py
  (dilation + min-distance scan).
  Resource needs for the largest records (measured, CPU only), using
  verify_independent.py --method dilate:
    K_9(10,5)  (q^n = 3.5e9): ~3.5 GB RAM, ~5 minutes
    K_10(10,5) (q^n = 1e10):  ~10 GB RAM,  ~20 minutes
  All other codes verify in seconds. Above q^n ~ 1e9 prefer the dilation
  method; verify_cov.py --method numpy also completes but needs more
  memory at the largest sizes.

cert_q{q}_n{n}_R{R}.json -- exact rational dual certificate proving the
                            lower bound of Table 2 for that cell.
  Verify:   python3 certify.py cert_q6_n10_R4.json
            (Python standard library only; rebuilds the SDP exactly and
             checks the certificate in exact arithmetic)
EOF

cd paper/covering
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
tar czf arxiv_v2_submission.tar.gz main.tex refs.bib main.bbl tables_generated.tex appendix_code.tex fig_slack.pdf fig_family82.pdf anc/
echo "== $(pdfinfo main.pdf | grep Pages) =="
echo "wrote paper/covering/arxiv_v2_submission.tar.gz"
tar tzf arxiv_v2_submission.tar.gz | head -30
