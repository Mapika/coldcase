CC ?= gcc
CFLAGS ?= -O3 -march=native -fopenmp -Wall
NVCC ?= nvcc
NVFLAGS ?= -O3 -arch=sm_90

all: covsearch

covsearch: covsearch.c
	$(CC) $(CFLAGS) -o $@ $< -lm

covsearch_cuda: covsearch.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

clean:
	rm -f covsearch covsearch_cuda
