Ancillary files for:
  "New upper bounds on covering codes K_q(n,R) for alphabets of size six and seven"

Contents
--------
Nine q-ary covering codes, one codeword per line, each codeword written as n
digits over {0,...,q-1}:

  file                  q   n  R   |C|   claim
  K6_7_3_M232.txt       6   7  3   232  K_6(7,3) <=   232  (was 246)
  K6_8_3_M1045.txt      6   8  3  1045  K_6(8,3) <=  1045  (was 1080)
  K6_8_4_M167.txt       6   8  4   167  K_6(8,4) <=   167  (was 216)
  K6_9_4_M703.txt       6   9  4   703  K_6(9,4) <=   703  (was 738)
  K6_9_5_M123.txt       6   9  5   123  K_6(9,5) <=   123  (was 144)
  K6_10_4_M2951.txt     6  10  4  2951  K_6(10,4) <=  2951  (was 2952)
  K6_10_5_M610.txt      6  10  5   610  K_6(10,5) <=   610  (was 615)
  K7_8_4_M329.txt       7   8  4   329  K_7(8,4) <=   329  (was 343)
  K7_9_4_M1743.txt      7   9  4  1743  K_7(9,4) <=  1743  (was 1843)

"was" = the upper bound in G. Keri's tables (https://old.sztaki.hu/~keri/codes/,
last updated November 2011).

Verification
------------
verify_cov.py is standalone (Python 3, no third-party dependencies required;
numpy is used automatically if present, making the largest checks faster).

  python3 verify_cov.py -q 6 -n 8  -R 4 K6_8_4_M167.txt
  python3 verify_cov.py -q 6 -n 10 -R 4 K6_10_4_M2951.txt   # ~6.0e7 words
  ...

The verifier exhaustively confirms that every word of Z_q^n lies within
Hamming distance R of some codeword, and that the file contains exactly the
claimed number of distinct valid codewords. Every check runs in at most a few
minutes; add --radius to report the exact covering radius.
