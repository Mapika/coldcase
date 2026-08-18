#!/usr/bin/env python3
"""Independent verifier for binary constant-weight codes.

Deliberately simple, pure Python, no dependencies. Reads a code file (one
codeword per line, either as 0/1 strings of length n, or as a hex/int uint64
with '#'-comments) and checks: word count, length, constant weight w, and
minimum pairwise Hamming distance >= d.

Usage: verify_cwc.py FILE n d w [M]
Exit 0 iff the file proves A(n,d,w) >= M (M defaults to the word count).
"""
import sys


def parse_code_file(path, n):
    words = []
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            if set(line) <= {"0", "1"} and len(line) >= n:
                words.append(int(line[:n][::-1], 2))  # bit i of string -> bit i
            else:
                words.append(int(line, 0))
    return words


def verify(words, n, d, w):
    errors = []
    if len(set(words)) != len(words):
        errors.append("duplicate codewords")
    for i, x in enumerate(words):
        if x >> n:
            errors.append(f"word {i} exceeds length {n}")
        if bin(x).count("1") != w:
            errors.append(f"word {i} has weight {bin(x).count('1')} != {w}")
    mind = None
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            dist = bin(words[i] ^ words[j]).count("1")
            if mind is None or dist < mind:
                mind = dist
    if mind is not None and mind < d:
        errors.append(f"minimum distance {mind} < {d}")
    return errors, mind


def main():
    path, n, d, w = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    words = parse_code_file(path, n)
    M = int(sys.argv[5]) if len(sys.argv) > 5 else len(words)
    if len(words) < M:
        print(f"FAIL: only {len(words)} words, need {M}")
        sys.exit(1)
    errors, mind = verify(words[:M] if len(words) > M else words, n, d, w)
    if errors:
        print("FAIL:", "; ".join(errors))
        sys.exit(1)
    print(f"OK: {M} words, length {n}, constant weight {w}, min distance {mind} >= {d}"
          f" => A({n},{d},{w}) >= {M}")
    sys.exit(0)


if __name__ == "__main__":
    main()
