#!/usr/bin/env python3
"""Regression: Engine.load must accept (a) a seed .txt file's digit rows,
(b) 1-D word indices, (c) 2-D digit arrays — all equivalent; and the
descent restore path (load of index_word output) must work."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpuchain import Engine, read_code, write_code_atomic  # noqa: E402

q, n, R = 6, 6, 3
rng = np.random.default_rng(3)
code = rng.integers(0, q, size=(41, n)).astype(np.uint8)

e = Engine(q, n, R, seed=1, out=None, log=lambda *a: None)
try:
    # path (c): 2-D digits
    e.load(code)
    u_digits = e.uncov
    idx = e.code.copy()
    # path (b): 1-D indices
    e.load(idx)
    assert e.uncov == u_digits, (e.uncov, u_digits)
    # path (a): seed file round trip
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        p = f.name
    write_code_atomic(p, e.eng.index_word(e.code))
    e.load(e.eng.word_index(read_code(p, q, n)))
    assert e.uncov == u_digits
    e.load(read_code(p, q, n))               # digits direct
    assert e.uncov == u_digits
    os.unlink(p)
    # descent restore path: load(index_word(code)) == load(code indices)
    snap = e.eng.index_word(e.code)          # digit rows, as siege stores
    e.ruin(5, "random")
    e.load(snap)
    assert e.uncov == u_digits
    print("test_load: ALL PASS")
finally:
    e.close()
