"""CPU-only geometry grouping audit for direct LSD bond typing."""
from collections import Counter, defaultdict
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig
from tools import audit_line_first_lsd_all_edge_ownership_r5 as lsd
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chem
DATA=chem.DATA; SHA='09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'; OUT=ROOT/'outputs/audits/chemvector-identity-aligned-line-first-lsd-direct-graph-type-geometry-r5/report.json'

def boxes(base):
    out={}
    for g in json.loads(Path(str(base)+'.text-records.json').read_text()).get('glyphs',[]):
        a=int(g['atom_id']); b=g.get('bbox')
        if b:
            b=list(map(float,b)); out[a]=b if a not in out else [min(out[a][0],b[0]),min(out[a][1],b[1]),max(out[a][2],b[2]),max(out[a][3],b[3])]
    return out

def merge_lines(items):
    groups=[]
    for q in sorted(items,key=lambda x:-x[4]):
        hit=None
        for j,g in enumerate(groups):
            da=abs(((q[0]-g[0]+np.pi/2)%np.pi)-np.pi/2); overlap=min(q[3],g[3])-max(q[2],g[2]); gap=max(max(q[2],g[2])-min(q[3],g[3]),0)
            if da<=np.deg2rad(7) and abs(q[1]-g[1])<=3 and (overlap>=-4 or gap<=10): hit=j; break
        if hit is None: groups.append(list(q))
        else:
            g=groups[hit]; g[2]=min(g[2],q[2]); g[3]=max(g[3],q[3]); g[4]=max(g[4],q[4]); g[0]=(g[0]+q[0])/2; g[1]=(g[1]+q[1])/2
    return groups

def evaluate(row):
    base=DATA/'samples'/row['sample_key']; image=Image.open(Path(str(base)+'.png')).convert('L'); seg=lsd._alt_detect(image,config=LineSegmentRescueConfig(maximum_candidates=512)); lab,c=cluster_endpoints(seg.reshape(-1,2),10.); no={i:i for i in range(len(c))}
    for aid,b in boxes(base).items():
        p=np.asarray([(b[0]+b[2])/2,(b[1]+b[3])/2]);
        for i in np.where(np.linalg.norm(c-p,axis=1)<=24)[0]: no[int(i)]=-1-aid
    edge_lines=defaultdict(list)
    for i,s in enumerate(seg):
        a,b=int(lab[2*i]),int(lab[2*i+1]); a,b=no.get(a,a),no.get(b,b)
        if a==b: continue
        v=s[1]-s[0]; L=float(np.linalg.norm(v));
        if L<1e-5: continue
        u=v/L; th=float(np.arctan2(u[1],u[0])); n=np.asarray([-u[1],u[0]]); mid=(s[0]+s[1])/2; off=float(np.dot(mid,n)); lo=float(np.dot(s[0],u)); hi=float(np.dot(s[1],u)); lo,hi=min(lo,hi),max(lo,hi)
        edge_lines[tuple(sorted((a,b)))].append((th,off,lo,hi,L))
    predicted={p:(3 if len(merge_lines(v))>=3 else 2 if len(merge_lines(v))==2 else 1) for p,v in edge_lines.items()}
    layers=json.loads(Path(str(base)+'.layers.json').read_text()); graph=json.loads(Path(str(base)+'.graph.json').read_text()); targets=np.asarray([a['center'] for a in layers['atoms']],np.float32); gt={tuple(sorted((int(b['begin_atom']),int(b['end_atom'])))):str(b.get('chemical_type','')).upper() for b in graph['bonds']}
    def target(n):
        if n<0: return -n-1
        if n>=len(c): return None
        ds=np.linalg.norm(targets-c[n],axis=1); return int(np.argmin(ds)) if len(ds) and ds.min()<=24 else None
    mapped={}
    for p,k in predicted.items():
        a,b=target(p[0]),target(p[1]);
        if a is not None and b is not None and a!=b: mapped[tuple(sorted((a,b)))]=k
    true=set(mapped)&set(gt); cor=corna=totna=0
    for e in true:
        wanted={'SINGLE':1,'DOUBLE':2,'TRIPLE':3}.get(gt[e],0); hit=int(bool(wanted) and mapped[e]==wanted); cor+=hit
        if gt[e] not in ('AROMATIC','AROM'): totna+=1; corna+=hit
    nodes=sorted({z for e in mapped for z in e}); idx={z:i for i,z in enumerate(nodes)}; edges={(idx[a],idx[b]):1 for a,b in mapped}; elems=[str(layers['atoms'][z].get('element') or 'C') for z in nodes]; arom=[bool(layers['atoms'][z].get('aromatic',False)) for z in nodes]; ok,_,_=chem._sanitize_graph([(z,elems[i],arom[i]) for i,z in enumerate(nodes)],edges)
    return Counter(images=1,mapped_edges=len(mapped),true_edges=len(true),gt_bonds=len(gt),type_correct=cor,type_total=len(true),type_correct_non_arom=corna,type_total_non_arom=totna,sanitize_ok=int(ok),raw_segments=len(seg),line_groups=len(predicted))

def finish(c):
    d=dict(c); d.update(edge_precision=c['true_edges']/max(c['mapped_edges'],1),edge_recall=c['true_edges']/max(c['gt_bonds'],1),type_accuracy_non_arom=c['type_correct_non_arom']/max(c['type_total_non_arom'],1),oracle_sanitize_rate=c['sanitize_ok']/max(c['images'],1)); return d

def main():
    if chem.file_digest(chem.INITIAL)!=SHA: raise RuntimeError('checkpoint changed')
    rows=[json.loads(x) for x in (DATA/'samples.jsonl').read_text().splitlines() if x.strip() and json.loads(x).get('render_status')=='accepted']; out={}
    for split in ('train','holdout'):
        rr=[r for r in rows if r['split']==split]; vals=[evaluate(r) for r in rr]; out[split]={'all':finish(sum(vals,Counter())),'by_profile':{p:finish(sum([v for v,r in zip(vals,rr) if r['profile_id']==p],Counter())) for p in sorted({r['profile_id'] for r in rr})}}
    h=out['holdout']['all']; hp=out['holdout']['by_profile']; gate={'checkpoint_unchanged':chem.file_digest(chem.INITIAL)==SHA,'sealed_test_read':False,'holdout_precision_ge_0.95':h['edge_precision']>=.95,'holdout_recall_ge_0.80':h['edge_recall']>=.80,'holdout_type_non_aromatic_ge_0.80':h['type_accuracy_non_arom']>=.80,'all_profiles_recall_ge_0.75':all(v['edge_recall']>=.75 for v in hp.values()),'all_profiles_sanitize_ge_0.90':all(v['oracle_sanitize_rate']>=.90 for v in hp.values()),'production_go':False,'gpu_go':False}
    rep={'schema_version':'chemvector-identity-aligned-line-first-lsd-direct-graph-type-geometry-r5-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'proposal_contract':'direct LSD; supporting-line merge before multiplicity type inference; GT post-hoc only','splits':out,'holdout':h,'holdout_profiles':hp,'gate':gate}; OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(rep,indent=2)+'\n'); print(json.dumps({'output':str(OUT),'holdout':h,'gate':gate}))
if __name__=='__main__': main()
