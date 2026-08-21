Ancillary files for "New upper and lower bounds on covering codes K_q(n,R)
for alphabets of size five to fifteen".

K{q}_{n}_{R}_M{M}.txt   -- explicit code achieving K_q(n,R) <= M, one codeword
                           per line, digits 0-9 then a-z for q > 10.
  Verify:   python3 verify_cov.py K6_8_4_M166.txt 6 8 4
            (dependency-free; numpy accelerates if present)

cert_q{q}_n{n}_R{R}.json -- exact rational dual certificate proving the
                            lower bound of Table 2 for that cell.
  Verify:   python3 certify.py cert_q6_n10_R4.json
            (Python standard library only; rebuilds the SDP exactly and
             checks the certificate in exact arithmetic)
