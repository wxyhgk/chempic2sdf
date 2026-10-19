"""Read-only full round-trip gate for score-regularized target-free ownership.

This is a CPU diagnostic continuation of ownership-assignment-r5. The only
inference mapping is Hungarian assignment between raster provisional nodes and
frozen scored candidates; GT is used only after proposal construction.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path
import sys
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from rdkit import Chem, rdBase
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as base
from tools.audit_line_first_candidate_alignment_r5 import read_jsonl,IMAGE
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig,detect_raster_line_segments
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target
DATA=base.DATA; CANDIDATES=base.CANDIDATES; STATE=base.STATE; INITIAL=base.INITIAL; INITIAL_SHA=base.INITIAL_SHA; RADIUS=24.; LAMBDA=8.

def hungarian(centers,cand):
 if len(centers)==0 or len(cand)==0:return {}
 d=np.linalg.norm(centers[:,None,:]-cand[None,:,:2],axis=-1); cost=np.where(d<=RADIUS,d-LAMBDA*cand[None,:,2],1e5+d);rr,cc=linear_sum_assignment(cost);return {int(i):int(j) for i,j in zip(rr,cc) if d[i,j]<=RADIUS}

def one(row,crows,model,mean,std,th):
 key=row['sample_key'];b=DATA/'samples'/key;im=__import__('PIL').Image.open(b.with_suffix('.png')).convert('L');seg=detect_raster_line_segments(im,config=LineSegmentRescueConfig(maximum_candidates=512));pts=seg.reshape(-1,2) if len(seg) else np.empty((0,2),np.float32);labels,cent=cluster_endpoints(pts,16.);cand=np.asarray([[v['x']*IMAGE,v['y']*IMAGE,float(v.get('score',0.))] for v in crows],np.float64);nm=hungarian(cent,cand);keys=base._candidate_keys(key); features,_,_=base._proposals(key)
 with torch.inference_mode():elog,tlog=model(torch.from_numpy((features-mean)/std))
 scores=torch.sigmoid(elog).numpy(); kinds=(torch.argmax(tlog,1)+1).numpy(); selected={};dang=0;selfloops=0
 for i,(a,b2) in enumerate(keys):
  if scores[i]<th:continue
  if a not in nm or b2 not in nm:dang+=1;continue
  ca,cb=nm[a],nm[b2]
  if ca==cb:selfloops+=1;continue
  pair=tuple(sorted((ca,cb)));old=selected.get(pair)
  if old is None or scores[i]>old[0]:selected[pair]=(float(scores[i]),int(kinds[i]))
 lay=json.loads(b.with_suffix('.layers.json').read_text());graph=json.loads(b.with_suffix('.graph.json').read_text());targets=np.asarray([x['center'] for x in lay['atoms']],np.float32);c2t=strict_candidate_to_target(cand[:,:2],targets);gt={tuple(sorted((int(x['begin_atom']),int(x['end_atom'])))) for x in graph['bonds']};gty={tuple(sorted((int(x['begin_atom']),int(x['end_atom'])))): (base.TYPE_NAMES.index(str(x.get('chemical_type','OTHER')).upper()) if str(x.get('chemical_type','OTHER')).upper() in base.TYPE_NAMES else base.TYPE_NAMES.index('OTHER')) for x in graph['bonds']};mapped={};
 for (ca,cb),(_,kind) in selected.items():
  if ca in c2t and cb in c2t and c2t[ca]!=c2t[cb]:mapped[tuple(sorted((c2t[ca],c2t[cb])))] = kind
 true=set(mapped)&gt;tc=sum(mapped[e]==gty[e] for e in true)
 # candidate topology and oracle-element topology; elements are never used for selection.
 active=sorted(set(nm.values()));ci={x:i for i,x in enumerate(active)}; cnodes=[(x,'C',False) for x in active];cedges={(ci[a],ci[b]):kind for (a,b),(_,kind) in selected.items() if a in ci and b in ci};cok,ce,_=base._sanitize_graph(cnodes,cedges)
 onodes=[(x,str(lay['atoms'][t]['element'] or 'C'),bool(lay['atoms'][t].get('aromatic',False))) for x,t in sorted(c2t.items()) if x in ci];oi={x:i for i,(x,_,_) in enumerate(onodes)};oedges={(oi[a],oi[b]):kind for (a,b),(_,kind) in selected.items() if a in oi and b in oi};ook,oe,_=base._sanitize_graph(onodes,oedges)
 return Counter(images=1,selected_edges=len(selected),dangling_edges=dang,self_loops=selfloops,mapped_edges=len(mapped),true_edges=len(true),gt_bonds=len(gt),type_correct=tc,type_total=len(true),candidate_sanitize_ok=int(cok),candidate_aromatic_errors=int(ce=='aromatic'),candidate_valence_errors=int(ce=='valence'),oracle_sanitize_ok=int(ook),oracle_aromatic_errors=int(oe=='aromatic'),oracle_valence_errors=int(oe=='valence'),all_bonds_recovered=int(true==gt),aligned_nodes=len(nm),unmapped_nodes=len(cent)-len(nm))
def finish(c):
 return {**dict(c),'edge_precision':c['true_edges']/max(c['mapped_edges'],1),'edge_recall':c['true_edges']/max(c['gt_bonds'],1),'bond_type_accuracy':c['type_correct']/max(c['type_total'],1),'dangling_rate':c['dangling_edges']/max(c['selected_edges'],1),'oracle_sanitize_rate':c['oracle_sanitize_ok']/max(c['images'],1),'candidate_sanitize_rate':c['candidate_sanitize_ok']/max(c['images'],1),'all_bonds_fraction':c['all_bonds_recovered']/max(c['images'],1)}
def main():
 rdBase.DisableLog('rdApp.error');
 if base.file_digest(INITIAL)!=INITIAL_SHA:raise RuntimeError('production checkpoint changed')
 cr={r['sample_key']:r['candidates'] for r in read_jsonl(CANDIDATES)};st=torch.load(STATE,map_location='cpu',weights_only=False);m=base.EdgeBondMLP(len(st['feature_names']));m.load_state_dict(st['model']);m.eval();mean=np.asarray(st['mean'],np.float32);std=np.asarray(st['std'],np.float32);th=float(st['threshold']);rows=read_jsonl(DATA/'samples.jsonl');by=defaultdict(Counter)
 for row in rows:
  by[row['split']].update(one(row,cr[row['sample_key']],m,mean,std,th))
 # profile stratification by split
 prof=defaultdict(Counter)
 for row in rows:prof[row['split']+':'+row['profile_id']].update(one(row,cr[row['sample_key']],m,mean,std,th))
 out={'schema_version':'chemvector-identity-aligned-line-first-ownership-roundtrip-r5-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'proposal_contract':'LSD -> provisional nodes -> Hungarian(score-distance) target-free candidate ownership -> frozen edge/type sidecar; GT only post-hoc','config':{'radius_px':RADIUS,'score_lambda_px':LAMBDA,'cluster_radius_px':16.},'splits':{k:finish(v) for k,v in sorted(by.items())},'by_split_profile':{k:finish(v) for k,v in sorted(prof.items())},'gate':{}}
 h=out['splits']['holdout'];g=out['gate'];g.update({'checkpoint_unchanged':base.file_digest(INITIAL)==INITIAL_SHA,'sealed_test_read':False,'holdout_edge_precision_ge_0.90':h['edge_precision']>=.90,'holdout_edge_recall_ge_0.75':h['edge_recall']>=.75,'holdout_type_accuracy_ge_0.90':h['bond_type_accuracy']>=.90,'holdout_dangling_le_0.05':h['dangling_rate']<=.05,'holdout_oracle_sanitize_ge_0.90':h['oracle_sanitize_rate']>=.90,'production_go':False,'gpu_training_admissible':False});g['roundtrip_shadow_worth_attempting']=all(v for k,v in g.items() if k not in {'sealed_test_read','production_go','gpu_training_admissible'})
 outp=ROOT/'outputs/audits/chemvector-identity-aligned-line-first-ownership-roundtrip-r5/report.json';outp.parent.mkdir(parents=True,exist_ok=True);outp.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'output':str(outp),'holdout':out['splits']['holdout'],'gate':g},ensure_ascii=False))
if __name__=='__main__':main()
