import os,sys,json,time,argparse,multiprocessing as mp
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
def one(c):
    q,n,R,outdir,scaling,bits=c
    os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
    sys.path.insert(0,HERE); import solve_ipm
    out=os.path.join(outdir,"cert_q%d_n%d_R%d.json"%(q,n,R)); t=time.time()
    try: r=solve_ipm.run(q,n,R,out=out,verbose=False,scaling=scaling,bits=bits,tag="geo")
    except Exception as e: return {"q":q,"n":n,"R":R,"error":repr(e),"sec":time.time()-t}
    if r is None: return {"q":q,"n":n,"R":R,"error":"none","sec":time.time()-t}
    return {"q":q,"n":n,"R":R,"cube":r["cube_root_float"],"K":r["K_lower_bound"],"cert":out,"sec":time.time()-t}
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("cells"); ap.add_argument("--jobs",type=int,default=10)
    ap.add_argument("--outdir",default=os.path.join(HERE,"certs_geo")); ap.add_argument("--scaling",default="geo")
    ap.add_argument("--bits",type=int,default=46); ap.add_argument("--out",default=None)
    a=ap.parse_args(); os.makedirs(a.outdir,exist_ok=True)
    cells=[tuple(int(x) for x in c.split(","))+(a.outdir,a.scaling,a.bits) for c in a.cells.split(";")]
    with mp.Pool(a.jobs) as p: rows=p.map(one,cells)
    for r in rows: print(r)
    if a.out: json.dump(rows,open(a.out,"w"),indent=1)
