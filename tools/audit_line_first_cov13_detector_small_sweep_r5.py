"""Read-only CPU detector sweep for cov13 coverage."""
import json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_line_first_lsd_config_sweep_r5 import detect_segments,cluster_single_link
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chem
DATA=chem.DATA
SHA='09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'
OUT=ROOT/'outputs/audits/chemvector-identity-aligned-line-first-cov13-detector-small-sweep-r5/report.json'
# Keep this deliberately small: the objective is to locate scale/length
# sensitivity, not to launch another broad detector search.
CONFIGS=[{'minimum_length_px':l,'scale':s,'refine':'STD','cluster_radius_px':10.} for l in (2.,5.) for s in (.75,1.)]

def rows(split):
    return sorted([json.loads(x) for x in (DATA/'samples.jsonl').read_text().splitlines() if x.strip() and json.loads(x).get('split')==split and json.loads(x).get('render_status')=='accepted'],key=lambda x:x['sample_key'])

def ev(row,c):
    base=DATA/'samples'/row['sample_key'];im=Image.open(base.with_suffix('.png')).convert('L');seg=detect_segments(im,minimum_length=c['minimum_length_px'],refine=c['refine'],scale=c['scale'],border_margin=2.);pts=seg.reshape(-1,2) if len(seg) else np.empty((0,2));lab,cent=cluster_single_link(pts,c['cluster_radius_px']);edges={tuple(sorted((int(lab[2*i]),int(lab[2*i+1])))) for i in range(len(seg)) if int(lab[2*i])!=int(lab[2*i+1])};lay=json.loads(base.with_suffix('.layers.json').read_text());g=json.loads(base.with_suffix('.graph.json').read_text());t=np.asarray([a['center'] for a in lay['atoms']],float);gt={tuple(sorted((int(b['begin_atom']),int(b['end_atom'])))) for b in g['bonds']};d=np.linalg.norm(cent[:,None]-t[None,:],axis=-1) if len(cent) else np.empty((0,len(t)));mp={i:int(np.argmin(x)) for i,x in enumerate(d) if len(x) and float(np.min(x))<=24.};mapped={tuple(sorted((mp[a],mp[b]))) for a,b in edges if a in mp and b in mp and mp[a]!=mp[b]};true=mapped&gt
    return {'images':1,'segments':len(seg),'nodes':len(cent),'matched_nodes':len(set(mp.values())),'target_atoms':len(t),'mapped_edges':len(mapped),'true_edges':len(true),'gt_bonds':len(gt),'all_bonds':int(true==gt)}

def sm(v):
    k=('images','segments','nodes','matched_nodes','target_atoms','mapped_edges','true_edges','gt_bonds','all_bonds');d={x:sum(z[x] for z in v) for x in k};d.update(node_recall=d['matched_nodes']/max(d['target_atoms'],1),edge_precision=d['true_edges']/max(d['mapped_edges'],1),edge_recall=d['true_edges']/max(d['gt_bonds'],1),all_bonds_fraction=d['all_bonds']/max(d['images'],1));return d

def main():
    if chem.file_digest(chem.INITIAL)!=SHA: raise RuntimeError('checkpoint changed')
    tr,ho=rows('train'),rows('holdout'); runs=[]
    for i,c in enumerate(CONFIGS,1):
        tv=[ev(r,c) for r in tr]; tp={p:sm([v for v,r in zip(tv,tr) if r['profile_id']==p]) for p in sorted({r['profile_id'] for r in tr})}; runs.append({'config':c,'train':sm(tv),'train_profiles':tp});print(f"train {i}/{len(CONFIGS)} {c} R={runs[-1]['train']['edge_recall']:.4f} cov13={tp['cov13_base48']['node_recall']:.4f}",flush=True)
    selected=max(runs,key=lambda x:(x['train_profiles']['cov13_base48']['node_recall'],x['train']['edge_recall'],x['train']['edge_precision']))
    for z in runs:
        hv=[ev(r,z['config']) for r in ho];z['holdout']=sm(hv);z['holdout_profiles']={p:sm([v for v,r in zip(hv,ho) if r['profile_id']==p]) for p in sorted({r['profile_id'] for r in ho})}
    sh=next(z for z in runs if z['config']==selected['config']);h=sh['holdout'];hp=sh['holdout_profiles'];gate={'checkpoint_unchanged':chem.file_digest(chem.INITIAL)==SHA,'sealed_test_read':False,'production_go':False,'holdout_edge_precision_ge_0.95':h['edge_precision']>=.95,'holdout_edge_recall_ge_0.80':h['edge_recall']>=.80,'all_profiles_recall_ge_0.75':all(v['edge_recall']>=.75 for v in hp.values())};rep={'schema_version':'chemvector-identity-aligned-line-first-cov13-detector-small-sweep-r5-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'initial_sha256':SHA,'initial_checkpoint_sha256_after':chem.file_digest(chem.INITIAL),'proposal_contract':'temporary LSD detector/cluster; GT post-hoc only','configs':CONFIGS,'selection':'train-only cov13 node recall then edge recall','selected':selected,'selected_holdout':sh,'sweeps':runs,'gate':gate};OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps({'output':str(OUT),'selected':selected['config'],'holdout':h,'profiles':hp,'gate':gate}))
if __name__=='__main__': main()
