"""CPU-only target-free candidate ownership policy sweep for line-first route.

Compares greedy distance alignment with global one-to-one assignment that uses
frozen candidate score only as a tie-break/regularizer. Ground-truth centers,
atom identities and bonds are read only after proposal construction for metrics.
No training, checkpoint writes, sealed-test access, or production merge.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_line_first_candidate_alignment_r5 import align_nodes, read_jsonl, IMAGE
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig, detect_raster_line_segments
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as audit
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target

DATA=ROOT/"outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
CANDIDATES=ROOT/"outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5-scored/candidates.jsonl"
IMAGE=768.0

def greedy(centers, candidates, radius): return align_nodes(centers.astype(float),candidates[:,:2].astype(float),radius)

def hungarian(centers, candidates, radius, lam):
    if len(centers)==0 or len(candidates)==0:return {}
    d=np.linalg.norm(centers[:,None,:]-candidates[None,:,:2],axis=-1)
    # score regularizer is deliberately small and candidate-only; no target.
    # Invalid pairs are rejected after assignment. Add deterministic epsilon.
    cost=d-lam*candidates[None,:,2]
    cost=np.where(d<=radius,cost,1e5+d)
    rr,cc=linear_sum_assignment(cost)
    return {int(i):int(j) for i,j in zip(rr,cc) if d[i,j]<=radius}

def load_sample(row,crows,cluster_radius):
    key=row['sample_key']; b=DATA/'samples'/key
    image=Image.open(b.with_suffix('.png')).convert('L')
    seg=detect_raster_line_segments(image,config=LineSegmentRescueConfig(maximum_candidates=512))
    points=seg.reshape(-1,2) if len(seg) else np.empty((0,2),np.float32)
    labels,centers=cluster_endpoints(points,cluster_radius)
    provisional={tuple(sorted((int(labels[2*i]),int(labels[2*i+1])))) for i in range(len(seg)) if labels[2*i]!=labels[2*i+1]}
    cand=np.asarray([[float(v['x']*IMAGE),float(v['y']*IMAGE),float(v.get('score',0.0))] for v in crows],np.float64)
    layers=json.loads(b.with_suffix('.layers.json').read_text()); graph=json.loads(b.with_suffix('.graph.json').read_text())
    targets=np.asarray([x['center'] for x in layers['atoms']],np.float64)
    c2t=strict_candidate_to_target(cand[:,:2],targets)
    gt={tuple(sorted((int(x['begin_atom']),int(x['end_atom'])))) for x in graph['bonds']}
    return centers,cand,provisional,c2t,gt

def evaluate(sample,radius,lam,policy):
    centers,cand,provisional,c2t,gt=sample
    node_map=greedy(centers,cand,radius) if policy=='greedy' else hungarian(centers,cand,radius,lam)
    edges={tuple(sorted((node_map[a],node_map[b]))) for a,b in provisional if a in node_map and b in node_map and node_map[a]!=node_map[b]}
    mapped={tuple(sorted((c2t[a],c2t[b]))) for a,b in edges if a in c2t and b in c2t and c2t[a]!=c2t[b]}
    true=mapped & gt
    # Duplicate ownership / unmatched diagnostics are post-hoc only.
    return {'images':1,'provisional_nodes':len(centers),'aligned_nodes':len(node_map),'candidate_edges':len(edges),'mapped_edges':len(mapped),'true_edges':len(true),'gt_bonds':len(gt),'all_bonds':int(true==gt),'unmapped_centers':len(centers)-len(node_map)}

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/audits/chemvector-identity-aligned-line-first-ownership-assignment-r5/report.json');p.add_argument('--cluster-radius',type=float,default=16.0);args=p.parse_args()
 rows=read_jsonl(DATA/'samples.jsonl'); crows={r['sample_key']:r['candidates'] for r in read_jsonl(CANDIDATES)}
 samples={r['sample_key']:load_sample(r,crows[r['sample_key']],args.cluster_radius) for r in rows}
 configs=[]
 for radius in (12.,16.,20.,24.,28.):
  configs.append(('greedy_d'+str(int(radius)),radius,0.,'greedy'))
  for lam in (0.25,0.5,1.,2.,4.,8.): configs.append((f'hungarian_r{int(radius)}_l{str(lam).replace(".","p")}',radius,lam,'hungarian'))
 report={'schema_version':'chemvector-identity-aligned-line-first-ownership-assignment-r5-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'proposal_contract':'LSD endpoints -> target-free provisional edges -> frozen candidate ownership policy; targets post-hoc metrics only. Hungarian is inference ownership between predicted nodes and predicted candidates; never Hungarian candidate-to-GT identity matching.','configs':{}}
 for name,radius,lam,policy in configs:
  by=defaultdict(Counter)
  for row in rows:
   x=evaluate(samples[row['sample_key']],radius,lam,policy); by[row['split']].update(x); by[row['split']+':'+row['profile_id']].update(x)
  def finish(c): return {'images':c['images'],'edge_precision':c['true_edges']/max(c['mapped_edges'],1),'edge_recall':c['true_edges']/max(c['gt_bonds'],1),'all_bonds_fraction':c['all_bonds']/max(c['images'],1),'mean_aligned_nodes':c['aligned_nodes']/max(c['images'],1),'mean_unmapped_centers':c['unmapped_centers']/max(c['images'],1)}
  report['configs'][name]={'radius_px':radius,'score_lambda_px':lam,'policy':policy,'by_split':{k:finish(v) for k,v in sorted(by.items()) if k in {'train','holdout'}},'by_split_profile':{k:finish(v) for k,v in sorted(by.items()) if ':' in k}}
 admissible={name:v for name,v in report['configs'].items() if v['by_split']['train']['edge_precision']>=0.90}
 selected=max(admissible,key=lambda name:(admissible[name]['by_split']['train']['edge_recall'],admissible[name]['by_split']['train']['edge_precision'],-admissible[name]['score_lambda_px']))
 report['decision']={'production_authorized':False,'gpu_training_admissible':False,'selection_rule':'train-only maximum edge recall subject to train precision >= 0.90; holdout is report-only','train_selected_config':selected}
 report['output']=str(args.output); args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n'); print(json.dumps({'output':str(args.output),'configs':len(configs)},ensure_ascii=False))
if __name__=='__main__':main()
