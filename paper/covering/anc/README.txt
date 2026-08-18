Ancillary files for:
  "New upper bounds on covering codes K_q(n,R) for alphabets of size six and seven"

Contents
--------
Eight q-ary covering codes, one codeword per line, each codeword written as n
digits over {0,...,q-1}:

  file                  q  n  R  |C|   claim
  K6_7_3_M235.txt       6  7  3  235   K_6(7,3)  <= 235   (was 246)
  K6_8_3_M1054.txt      6  8  3  1054  K_6(8,3)  <= 1054  (was 1080)
  K6_8_4_M169.txt       6  8  4  169   K_6(8,4)  <= 169   (was 216)
  K6_9_4_M719.txt       6  9  4  719   K_6(9,4)  <= 719   (was 738)
  K6_9_5_M126.txt       6  9  5  126   K_6(9,5)  <= 126   (was 144)
  K6_10_4_M2951.txt     6 10  4  2951  K_6(10,4) <= 2951  (was 2952)
  K6_10_5_M610.txt      6 10  5  610   K_6(10,5) <= 610   (was 615)
  K7_8_4_M333.txt       7  8  4  333   K_7(8,4)  <= 333   (was 343)

"was" = the upper bound in G. Keri's tables (https://old.sztaki.hu/~keri/codes/,
last updated November 2011).

Verification
------------
verify_cov.py is standalone (Python 3, no third-party dependencies required;
numpy is used automatically if present, making the largest checks faster).

  python3 verify_cov.py -q 6 -n 8  -R 4 K6_8_4_M169.txt
  python3 verify_cov.py -q 6 -n 10 -R 4 K6_10_4_M2951.txt   # ~6.0e7 words
  ...

The verifier exhaustively confirms that every word of Z_q^n lies within
Hamming distance R of some codeword, and that the file contains exactly the
claimed number of distinct valid codewords. Every check runs in at most a few
minutes; add --radius to report the exact covering radius.
