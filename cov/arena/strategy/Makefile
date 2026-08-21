CC ?= gcc
CFLAGS ?= -O3 -march=native -fopenmp -Wall

all: strategy baseline

strategy: strategy.c
	$(CC) $(CFLAGS) -o $@ $< -lm

# a copy of cov/search/covsearch.c, built here only as the A/B control
baseline: baseline.c
	$(CC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -f strategy baseline
