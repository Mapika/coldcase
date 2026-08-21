/*
 * covcount.c -- arena entry "structure": the referee and the finisher.
 *
 * Reads a code file, and
 *   * de-duplicates it (the judge counts DISTINCT words: a file with repeats
 *     claims fewer words than M and scores 0 even when it covers),
 *   * tops it up to exactly M distinct words, each new word placed on a word
 *     that is still uncovered (padding is free, so it may as well help),
 *   * reports the exact number of uncovered words of Z_q^n by direct ball
 *     marking -- recomputed from the file, never taken from a solver's own
 *     incremental counters.
 *
 * The driver uses the reported number to choose between the portfolio's
 * candidate answers, so this is the one place where the arithmetic has to be
 * independent of whichever search produced the file.
 *
 * --fast skips the coverage part entirely (no q^n array, no ball marking) and
 * only fixes the file up: de-duplicate, top up to M distinct words, write.  It
 * is instant at any size, which is what makes it safe to call on the far side
 * of a wall-clock deadline.  Coverage costs M*|B_R| marks into a q^n byte
 * array and on K_8(9,4) that is seconds on an idle machine and tens of seconds
 * on a loaded one -- not something to have between you and the judge's timer.
 *
 * Build: gcc -O3 -march=native -o covcount covcount.c
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int q, n, R, M;
static long long NTOT, pw[32];
static uint8_t *cov;

static const char *DIG = "0123456789abcdefghijklmnopqrstuvwxyz";

static long long ipow(long long b, int e){ long long r=1; while(e-->0) r*=b; return r; }

static void mark_rec(long long base, const long long *offs, int startpos, int rem)
{
    if (rem == 0) return;
    for (int j = startpos; j < n; j++)
        for (int k = 0; k < q-1; k++) {
            long long w = base + offs[(size_t)j*q + k];
            cov[w] = 1;
            mark_rec(w, offs, j+1, rem-1);
        }
}

static void mark_ball(const uint8_t *c)
{
    long long offs[32*64];
    long long base = 0;
    for (int j = 0; j < n; j++) {
        base += (long long)c[j] * pw[j];
        int k = 0;
        for (int v = 0; v < q; v++)
            if (v != c[j]) offs[(size_t)j*q + (k++)] = (long long)(v - c[j]) * pw[j];
    }
    cov[base] = 1;
    mark_rec(base, offs, 0, R);
}

int main(int argc, char **argv)
{
    const char *inpath = NULL, *outpath = NULL;
    int fast = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        #define ARG(x) (!strcmp(a,x) && i+1<argc)
        if      (ARG("-q")) q = atoi(argv[++i]);
        else if (ARG("-n")) n = atoi(argv[++i]);
        else if (ARG("-R")) R = atoi(argv[++i]);
        else if (ARG("-M")) M = atoi(argv[++i]);
        else if (ARG("--in")) inpath = argv[++i];
        else if (ARG("--out")) outpath = argv[++i];
        else if (!strcmp(a, "--fast")) fast = 1;
        else { fprintf(stderr, "bad option %s\n", a); return 2; }
        #undef ARG
    }
    if (q < 2 || n < 1 || R < 0 || M < 1 || !inpath) {
        fprintf(stderr, "usage: covcount -q Q -n N -R R -M M --in f [--out g]\n");
        return 2;
    }
    NTOT = 1;
    for (int i = 0; i < n; i++) {
        NTOT *= q;
        if (NTOT > (long long)1.2e9) { if (!fast) return 2; NTOT = (long long)1.2e9; break; }
    }
    for (int p = 0; p < n; p++) pw[p] = ipow(q, n-1-p);

    uint8_t *code = malloc((size_t)(M + 8) * n);
    int m = 0;
    FILE *f = fopen(inpath, "r");
    if (f) {
        char line[8192];
        while (fgets(line, sizeof line, f) && m < M) {
            char *h = strchr(line, '#'); if (h) *h = 0;
            uint8_t tmp[64]; int len = 0, bad = 0;
            if (strchr(line, ' ') || strchr(line, ',') || strchr(line, '\t')) {
                char *tok = strtok(line, " ,\t\r\n");
                while (tok && len < n) { tmp[len++] = (uint8_t)atoi(tok); tok = strtok(NULL, " ,\t\r\n"); }
            } else {
                for (char *s = line; *s && *s != '\n' && *s != '\r'; s++) {
                    const char *d = strchr(DIG, *s);
                    if (!d) continue;
                    if (len < n) tmp[len++] = (uint8_t)(d - DIG);
                }
            }
            if (len != n) continue;
            for (int j = 0; j < n; j++) if (tmp[j] >= q) bad = 1;
            if (bad) continue;
            memcpy(code + (size_t)m * n, tmp, n);
            m++;
        }
        fclose(f);
    }

    /* de-duplicate */
    {
        int keep = 0;
        for (int i = 0; i < m; i++) {
            int dup = 0;
            for (int j = 0; j < keep && !dup; j++)
                if (!memcmp(code + (size_t)i*n, code + (size_t)j*n, n)) dup = 1;
            if (!dup) { if (keep != i) memcpy(code + (size_t)keep*n, code + (size_t)i*n, n); keep++; }
        }
        m = keep;
    }

    if (fast) {
        /* no coverage array: just make the file legal.  Words are taken in
         * base-q counting order and skipped if already present. */
        long long nxt = 0;
        uint8_t tw[64];
        while (m < M) {
            long long x = nxt++;
            for (int j = n - 1; j >= 0; j--) { tw[j] = (uint8_t)(x % q); x /= q; }
            int dup = 0;
            for (int i = 0; i < m && !dup; i++)
                if (!memcmp(code + (size_t)i*n, tw, n)) dup = 1;
            if (dup) continue;
            memcpy(code + (size_t)m*n, tw, n);
            m++;
        }
        printf("COUNT q=%d n=%d R=%d M=%d words=%d uncovered=-1\n", q, n, R, M, m);
        if (outpath) {
            char tmp[4096];
            snprintf(tmp, sizeof tmp, "%s.part", outpath);
            FILE *g = fopen(tmp, "w");
            if (!g) { perror(tmp); return 2; }
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) fputc(DIG[code[(size_t)i*n + j]], g);
                fputc('\n', g);
            }
            fclose(g);
            rename(tmp, outpath);
        }
        return 1;
    }

    cov = calloc((size_t)NTOT, 1);
    if (!cov) { fprintf(stderr, "oom\n"); return 2; }
    for (int i = 0; i < m; i++) mark_ball(code + (size_t)i * n);

    /* top up to M distinct words, preferring still-uncovered positions */
    long long scan = 0;
    uint64_t rs = 88172645463325252ULL ^ (uint64_t)m;
    while (m < M) {
        long long w = -1;
        for (int t = 0; t < 64 && w < 0; t++) {
            rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
            long long c2 = (long long)(rs % (uint64_t)NTOT);
            if (!cov[c2]) w = c2;
        }
        if (w < 0) for (; scan < NTOT; scan++) if (!cov[scan]) { w = scan; break; }
        if (w < 0) {
            /* everything covered: any unused word will do */
            for (;;) {
                rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
                long long cand = (long long)(rs % (uint64_t)NTOT);
                int dup = 0;
                uint8_t tw[64];
                for (int j = 0; j < n; j++) tw[j] = (uint8_t)((cand / pw[j]) % q);
                for (int i = 0; i < m && !dup; i++)
                    if (!memcmp(code + (size_t)i*n, tw, n)) dup = 1;
                if (!dup) { memcpy(code + (size_t)m*n, tw, n); break; }
            }
        } else {
            uint8_t *tw = code + (size_t)m * n;
            for (int j = 0; j < n; j++) tw[j] = (uint8_t)((w / pw[j]) % q);
        }
        mark_ball(code + (size_t)m * n);
        m++;
    }

    long long unc = 0;
    for (long long i = 0; i < NTOT; i++) if (!cov[i]) unc++;
    printf("COUNT q=%d n=%d R=%d M=%d words=%d uncovered=%lld\n", q, n, R, M, m, unc);

    if (outpath) {
        char tmp[4096];
        snprintf(tmp, sizeof tmp, "%s.part", outpath);
        FILE *g = fopen(tmp, "w");
        if (!g) { perror(tmp); return 2; }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) fputc(DIG[code[(size_t)i*n + j]], g);
            fputc('\n', g);
        }
        fclose(g);
        rename(tmp, outpath);
    }
    return unc == 0 ? 0 : 1;
}
