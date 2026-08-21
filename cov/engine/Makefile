# cov/engine -- the merged production engine.
#
# The four solvers below are VERBATIM COPIES of code that was measured
# elsewhere and works; they are copied rather than referenced so that the
# engine is self-contained and cannot be broken by a change in another
# directory.  Provenance:
#
#   lincov.c, symsearch.c, covcount.c   cov/arena/structure/  (unmodified)
#   covfast.c                           cov/arena/lowlevel/   (unmodified)
#   covsearch2e.c                       cov/opt/covsearch2.c  + the deadline
#                                       and publishing hardening of `covfast`
#                                       (see the block comment in the file and
#                                       NOTES.md; the search is bit-identical,
#                                       checked by selftest.sh)

CC      ?= gcc
CFLAGS  ?= -O3 -march=native -fopenmp -fno-stack-protector
LDLIBS  ?= -lm

BINS = covsearch2e covfast lincov symsearch covcount

all: $(BINS)

covsearch2e: covsearch2e.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
covfast: covfast.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
lincov: lincov.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
symsearch: symsearch.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
covcount: covcount.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

test: $(BINS)
	bash selftest.sh

clean:
	rm -f $(BINS)

.PHONY: all test clean
