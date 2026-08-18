#!/usr/bin/env python3
"""Expand Brouwer's cwc/ code files (plain, hex, orbit/cycle-compressed) into
validated plain integer codes under data/seeds/.

Each output file a{n}.{d}.{w}.{M}.txt holds one integer (decimal) per line.
A file is only written if the expanded code has exactly M distinct words of
weight w and min distance >= d. Failures are logged to data/seeds/failures.log.
"""
import os, sys, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "cwc")
OUT = os.path.join(ROOT, "data", "seeds")
os.makedirs(OUT, exist_ok=True)


def popcount(x):
    return bin(x).count("1")


def parse_file(path, n):
    base = 2
    n_override = None
    perms = []          # list of permutation image-lists
    cycles = None       # list of cycle lengths
    mode = None         # None | 'orbit' | 'cycle' | 'orbitx'
    words_raw = []
    in_perm_section = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("$"):
                tok = line[1:].split()
                if tok[0].startswith("BASE"):
                    base = int(line.split("=")[1])
                elif tok[0].startswith("N="):
                    n_override = int(line.split("=")[1])
                elif tok[0] == "EXEC":
                    mode = tok[1]
                    if mode == "cycle":
                        cycles = [int(t) for t in tok[2:]] or None
                    if mode in ("orbit", "orbitx"):
                        in_perm_section = True
                continue
            if line == "..":
                in_perm_section = False
                continue
            if in_perm_section:
                if line.startswith("("):
                    # cycle notation: (a,b,c)(d,e)... -> image list
                    cyc_groups = [g for g in line.replace(")", ")|").split("|") if g.strip()]
                    entries = []
                    for g in cyc_groups:
                        g = g.strip().lstrip("(").rstrip(")")
                        if g:
                            entries.append([int(t) for t in g.replace(";", ",").split(",")])
                    perms.append(("cycles", entries))
                else:
                    perms.append(("images", [int(t) for t in line.replace(";", ",").split(",")
                                             if t.strip() != ""]))
                continue
            words_raw.append(line)
    nn = n_override or n
    words = []
    hexdigits = set("0123456789abcdefABCDEF")
    for wr in words_raw:
        if set(wr) <= {"0", "1"} and len(wr) >= 8:
            words.append(int(wr, 2))       # convention fixed; action tried both ways
        elif base == 16 and set(wr) <= hexdigits:
            words.append(int(wr, 16))
        else:
            pass  # comment/annotation line (e.g. "n=36 size=245") — skip
    return {"mode": mode, "perms": perms, "cycles": cycles, "words": words, "n": nn}


def cycles_to_perm(cycles, n):
    """Cycle lengths on consecutive coordinate blocks; remaining coords fixed."""
    perm = list(range(n))
    pos = 0
    for L in cycles:
        for i in range(L):
            perm[pos + i] = pos + (i + 1) % L
        pos += L
    return perm  # perm[i] = image of coordinate i


def apply_perm_bits(x, perm, n, inverse=False):
    y = 0
    for i in range(n):
        if inverse:
            if (x >> i) & 1:
                y |= 1 << perm[i]
        else:
            if (x >> perm[i]) & 1:
                y |= 1 << i
    return y


def orbit_closure(words, perms, n, inverse, cap=200000):
    seen = set(words)
    frontier = list(words)
    while frontier:
        nxt = []
        for x in frontier:
            for p in perms:
                y = apply_perm_bits(x, p, n, inverse)
                if y not in seen:
                    seen.add(y)
                    nxt.append(y)
                    if len(seen) > cap:
                        return None
        frontier = nxt
    return seen


def check(words, n, d, w, M):
    if len(words) != M:
        return f"count {len(words)} != {M}"
    for x in words:
        if x >> n:
            return "word exceeds n bits"
        if popcount(x) != w:
            return f"bad weight {popcount(x)}"
    ws = sorted(words)
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            if popcount(ws[i] ^ ws[j]) < d:
                return f"dist {popcount(ws[i]^ws[j])} < {d}"
    return None


def normalize_perm(perm, nn, based):
    """perm is ('images', list) or ('cycles', [[...],[...]]). Return image list
    of length nn or None if invalid under this basing."""
    kind, data = perm
    if kind == "images":
        p = [e - based for e in data]
        if any(e < 0 or e >= nn for e in p):
            return None
        if len(p) < nn:
            p = p + list(range(len(p), nn))
    else:
        p = list(range(nn))
        for cyc in data:
            c = [e - based for e in cyc]
            if any(e < 0 or e >= nn for e in c):
                return None
            for i in range(len(c)):
                p[c[i]] = c[(i + 1) % len(c)]
    if len(p) != nn or sorted(p) != list(range(nn)):
        return None
    return p


def expand_one(path, n, d, w, M):
    info = parse_file(path, n)
    nn = info["n"]
    cand_sets = []
    if info["mode"] is None:
        cand_sets.append(set(info["words"]))
    elif info["mode"] in ("orbit", "orbitx"):
        perms = info["perms"]
        # permutation entries might be 0- or 1-based; try both, and both actions
        for based in (0, 1):
            ps = [normalize_perm(p, nn, based) for p in perms]
            if any(p is None for p in ps):
                continue
            for inv in (False, True):
                # also try reversed string convention for binary words
                for wl in (info["words"],
                           [int(format(x, f"0{nn}b")[::-1], 2) for x in info["words"]]):
                    o = orbit_closure(wl, ps, nn, inv)
                    if o is not None:
                        cand_sets.append(o)
    elif info["mode"] == "cycle":
        cyc = info["cycles"]
        perm = cycles_to_perm(cyc, nn) if cyc else cycles_to_perm([nn], nn)
        for inv in (False, True):
            for wl in (info["words"],
                       [int(format(x, f"0{nn}b")[::-1], 2) for x in info["words"]]):
                o = orbit_closure(wl, [perm], nn, inv)
                if o is not None:
                    cand_sets.append(o)
    for cs in cand_sets:
        err = check(list(cs), nn, d, w, M)
        if err is None:
            return sorted(cs), nn, None
    return None, nn, (f"no valid expansion (mode={info['mode']}, "
                      f"sizes tried={sorted(set(len(c) for c in cand_sets))})" if cand_sets
                      else "no candidate expansions")


def expand_i_file(path, n, d, w):
    """i-files: no size in name; accept the largest valid expansion."""
    info = parse_file(path, n)
    nn = info["n"]
    cand_sets = []
    perms = info["perms"]
    if info["mode"] in ("orbit", "orbitx") and perms:
        for based in (0, 1):
            ps = [normalize_perm(p, nn, based) for p in perms]
            if any(p is None for p in ps):
                continue
            for inv in (False, True):
                for wl in (info["words"],
                           [int(format(x, f"0{nn}b")[::-1], 2) for x in info["words"]]):
                    o = orbit_closure(wl, ps, nn, inv)
                    if o is not None:
                        cand_sets.append(o)
    else:
        cand_sets.append(set(info["words"]))
    best = None
    for cs in cand_sets:
        lst = list(cs)
        if len(lst) < 8:
            continue
        if check(lst, nn, d, w, len(lst)) is None:
            if best is None or len(lst) > len(best):
                best = lst
    return (sorted(best), nn, None) if best else (None, nn, "no valid expansion")


def main():
    ok = fail = skipped = 0
    failures = []
    for dsub in sorted(os.listdir(RAW)):
        ddir = os.path.join(RAW, dsub)
        if not os.path.isdir(ddir):
            continue
        d_dir = int(dsub.lstrip("d"))
        for fn in sorted(os.listdir(ddir)):
            fpath = os.path.join(ddir, fn)
            if fn.startswith("a"):
                p = fn.lstrip("a").split(".")
                try:
                    n, d, w = int(p[0]), int(p[1]), int(p[2])
                    M = int("".join(c for c in p[3] if c.isdigit()))
                except (ValueError, IndexError):
                    continue
                outfn = os.path.join(OUT, f"a{n}.{d}.{w}.{M}.txt")
                if os.path.exists(outfn):
                    skipped += 1
                    continue
                words, nn, err = expand_one(fpath, n, d, w, M)
                outfn = os.path.join(OUT, f"a{nn}.{d}.{w}.{M}.txt")
            elif fn.startswith("i"):
                p = fn.lstrip("i").split(".")
                try:
                    n = int(p[0])
                    w = int("".join(c for c in p[1] if c.isdigit()))
                except (ValueError, IndexError):
                    continue
                d = d_dir
                words, nn, err = expand_i_file(fpath, n, d, w)
                if words:
                    outfn = os.path.join(OUT, f"a{nn}.{d}.{w}.{len(words)}.txt")
                    if os.path.exists(outfn):
                        skipped += 1
                        continue
            else:
                continue
            if words:
                with open(outfn, "w") as f:
                    for x in words:
                        f.write(f"{x}\n")
                ok += 1
            else:
                fail += 1
                failures.append(f"{fn}: {err}")
    with open(os.path.join(OUT, "failures.log"), "w") as f:
        f.write("\n".join(failures))
    print(f"expanded OK: {ok}, failed: {fail}")
    for x in failures[:15]:
        print(" ", x)


if __name__ == "__main__":
    main()
