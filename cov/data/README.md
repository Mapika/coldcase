# cov/data

`keri_third_party.csv` — an independent transcription of Kéri's covering-code
tables, taken from <https://github.com/florath/covering-codes-lean>
(`reference-data/keri/non_mixed_covering_codes.csv`), where it was produced for
a Lean formalisation of covering-code bounds. It is included here so that the
cross-check in `tables.py --check` is reproducible without network access.

It is *not* our source of truth: `bounds.json` is parsed from the PDFs in
`../../data/raw/keri_*.pdf`. This file exists purely so that a second pair of
eyes on the same PDFs can be diffed against ours. As of this commit the two
agree on all 1145 cells, both bounds, with no cell present in one and not the
other.
