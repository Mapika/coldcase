# Submission packet — covering-code records (DRAFTS, nothing sent)

## 1. arXiv submission

- **File**: `paper/covering/arxiv_submission.tar.gz` (main.tex + `anc/` ancillary files).
- **Primary category**: cs.IT (Information Theory). **Cross-list**: math.CO.
  (cs.IT avoids the Dec-2025 math.* endorsement tightening; this matches how
  Echols/Lysenstøen submitted comparable notes in 2026.)
- **Title**: New upper bounds on covering codes K_q(n,R) for alphabets of size six and seven
- **Abstract** (plain text, for the form):
  We present improved upper bounds for eight entries of the standard tables of
  bounds on K_q(n,R), the minimum cardinality of a q-ary code of length n with
  covering radius R, for q in {6,7}: K_6(7,3)<=235, K_6(8,3)<=1054,
  K_6(8,4)<=169, K_6(9,4)<=719, K_6(9,5)<=126, K_6(10,4)<=2951,
  K_6(10,5)<=610, and K_7(8,4)<=333. The previous best bounds, recorded in
  Keri's tables (last updated 2011), all arose from general constructions
  (direct sums and related product rules) rather than from explicit search; to
  our knowledge these are the first improvements to any upper bound on
  K_q(n,R) with q>=5 since 2011. The new bounds were found by focused local
  search seeded with the construction-based incumbents. All eight codes are
  given explicitly in the ancillary files, together with a standalone
  verifier; each code was checked by four independent exhaustive verification
  methods.
- **Comments field**: 6 pages. Ancillary files contain the eight codes and a
  standalone verifier.
- **License**: CC BY 4.0 (matches community practice; allows table reuse).

### Before submitting (user actions / decisions)
- [ ] Confirm authorship & affiliation (currently: Mark Marosi, with an
      acknowledgment that Claude did the work autonomously — adjust to taste;
      some 2026 precedents credit the AI system in the acknowledgments, a few
      in the author list).
- [ ] arXiv account + (possibly) endorsement for cs.IT.
- [ ] Decide the public code repository URL (placeholder in Data Availability)
      — publishing the `longshots` repo to GitHub is recommended but is an
      outward-facing step awaiting approval.
- [ ] Final read of main.pdf.

## 2. Community notification drafts (send AFTER arXiv posting, with the arXiv ID)

### 2a. To Sven Polak + Dion Gijswijt (active researchers, SDP lower bounds)
Subject: New upper bounds for K_6(n,R) and K_7(8,4) — first q>=5 upper-bound
improvements since Kéri's 2011 tables

Dear Dr. Gijswijt, Dr. Polak,

Your recent SDP lower bounds paper (arXiv:2504.01932) prompted us to look at
the other side of the covering-code tables. We found that for q>=6 nearly all
of Kéri's upper bounds are inherited from direct-sum-type constructions and
were never the object of search — and that focused local search improves them
substantially. We have posted eight new upper bounds (largest single
improvement: K_6(8,4) 216 -> 169) with explicit codes and a standalone
verifier: [ARXIV LINK].

Since Kéri's tables appear to be frozen since 2011, we are also assembling a
merged machine-readable table (Kéri 2011 + post-2011 lower bounds incl. yours
+ these upper bounds). We'd welcome any corrections, and are happy to
coordinate if you maintain or plan anything similar.

Best regards, [SIGNATURE]

### 2b. To Andreas Florath (covering-codes-lean, arXiv:2606.09600)
Subject: Eight new K_q(n,R) upper bounds (q=6,7) — candidate entries for your
certified database

Dear Dr. Florath,

We enjoyed your Lean-certified covering-code bounds database. We have posted
eight new upper bounds for q in {6,7} ([ARXIV LINK]); each comes with an
explicit code (ancillary files) and verifies in seconds by exhaustive ball
marking, so they should be mechanically importable into your
LowerTrace/UpperTrace format. Happy to help with the import if useful.

Best regards, [SIGNATURE]

### 2c. To Gerzson Kéri (courtesy; table maintainer, retired)
Subject: Improvements to eight entries of your covering-code tables

Dear Dr. Kéri,

Your tables of bounds on K_q(n,R) remain the reference for this problem. We
write to report eight improved upper bounds for q=6,7 (details and explicit
codes: [ARXIV LINK]), all in cells whose previous bound came from direct-sum
constructions. We also noticed one internal gap in the tables: K_17(7,2) <=
17*K_17(6,2) = 245208 < 252735 by your rule (e). With gratitude for the years
of table maintenance —

Best regards, [SIGNATURE]

### 2d. coveringrepository.com (Acerbi) — note
The live successor repository covers C(v,k,t) covering DESIGNS, not covering
CODES — no submission channel there for K_q(n,R). No action.

### 2e. OEIS
The affected K_q(n,R) cells are not currently tracked in OEIS sequences
(A004044/A000983/A060438-A060440 cover q<=4 and K(n,1) families only) — no
OEIS action needed for these eight bounds.

## 3. Timing / competition note
Nothing in the literature audit suggests anyone else is working these cells;
still, arXiv posting establishes priority and should precede the emails.
