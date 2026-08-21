import os,sys,json,time,multiprocessing as mp
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
CELLS=[(6,7,3),(6,8,3),(6,8,4),(6,9,4),(6,9,5),(6,10,4),(6,10,5),(7,8,4),(7,9,4)]
def one(c):
    q,n,R=c
    os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
    sys.path.insert(0,HERE); import solve_ipm
    out=os.path.join(HERE,"certs","cert_q%d_n%d_R%d.json"%(q,n,R))
    t=time.time()
    try: r=solve_ipm.run(q,n,R,out=out,verbose=False,tag="records")
    except Exception as e: return {"q":q,"n":n,"R":R,"error":repr(e),"sec":time.time()-t}
    if r is None: return {"q":q,"n":n,"R":R,"error":"none","sec":time.time()-t}
    return {"q":q,"n":n,"R":R,"cube":r["cube_root_float"],"K":r["K_lower_bound"],
            "num":str(r["bound_num"]),"den":str(r["bound_den"]),"cert":out,"sec":time.time()-t}
if __name__=="__main__":
    with mp.Pool(9) as p: rows=p.map(one,CELLS)
    json.dump(rows,open(os.path.join(HERE,"results","records_lb.json"),"w"),indent=1)
    for r in rows: print(r)
