CC ?= gcc
CFLAGS ?= -O3 -march=native -fopenmp -Wall
all: lincov symsearch freesearch covcount
lincov: lincov.c
	$(CC) -O3 -march=native -Wall -o $@ $< -lm
symsearch: symsearch.c
	$(CC) $(CFLAGS) -o $@ $< -lm
freesearch: freesearch.c
	$(CC) $(CFLAGS) -o $@ $< -lm
covcount: covcount.c
	$(CC) -O3 -march=native -Wall -o $@ $<
clean:
	rm -f lincov symsearch freesearch covcount
