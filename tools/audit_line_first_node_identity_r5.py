"""CPU-only node identity audit for the line-first candidate route."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from tools.audit_line_first_candidate_alignment_r5 import DATA, CANDIDATES, IMAGE, TARGET_RADIUS, AMBIGUITY_RADIUS, align_nodes, read_jsonl
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig, detect_raster_line_segments
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/audits/chemvector-identity-aligned-line-first-node-identity-r5'

def evaluate(row,cands,cluster_radius=10.0,candidate_radius=16.0):
    base=DATA/'samples'/row['sample_key']; image=Image.open(base.with_suffix('.png')).convert('L'); layers=json.loads(base.with_suffix('.layers.json').read_text())
    seg=detect_raster_line_segments(image,config=LineSegmentRescueConfig(maximum_candidates=512)); endpoints=seg.reshape(-1,2) if len(seg) else np.empty((0,2)); labels,centers=cluster_endpoints(endpoints,cluster_radius)
    points=np.asarray([[x['x']*IMAGE,x['y']*IMAGE] for x in cands],dtype=np.float64); node_map=align_nodes(centers,points,candidate_radius)
    targets=np.asarray([x['center'] for x in layers['atoms']],dtype=np.float64); matched=[]; ambiguous=0
    for node,ci in node_map.items():
        d=np.linalg.norm(targets-points[ci],axis=1); close=np.where(d<=TARGET_RADIUS)[0]; near=np.where(d<=AMBIGUITY_RADIUS)[0]
        if len(close)==1 and len(near)==1: matched.append(int(close[0]))
        elif len(close)>0: ambiguous+=1
    unique=len(set(matched)); return {'sample_key':row['sample_key'],'split':row['split'],'gt_atoms':len(targets),'provisional_nodes':len(centers),'aligned_nodes':len(node_map),'matched_nodes':unique,'ambiguous_nodes':ambiguous,'node_precision':unique/max(len(node_map),1),'node_recall':unique/max(len(targets),1)}

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--split',choices=('train','holdout','all'),default='holdout');p.add_argument('--cluster-radius',type=float,default=10.0);p.add_argument('--candidate-radius',type=float,default=16.0);args=p.parse_args()
    meta={x['sample_key']:x for x in read_jsonl(DATA/'samples.jsonl')}; cands={x['sample_key']:x['candidates'] for x in read_jsonl(CANDIDATES)}; rows=sorted([x for x in meta.values() if args.split=='all' or x['split']==args.split],key=lambda x:x['sample_key']); results=[evaluate(r,cands[r['sample_key']],args.cluster_radius,args.candidate_radius) for r in rows]
    mean=lambda k:float(np.mean([x[k] for x in results])) if results else 0.; report={'schema_version':'chemvector-identity-aligned-line-first-node-identity-r5-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'deployment_authorized':False,'production_checkpoint_written':False,'proposal_contract':'LSD endpoint clusters -> frozen scored vertex candidates; target labels post-hoc only','split':args.split,'rows':len(results),'config':{'cluster_radius_px':args.cluster_radius,'candidate_radius_px':args.candidate_radius,'target_radius_px':TARGET_RADIUS},'aggregate':{'mean_node_precision':mean('node_precision'),'mean_node_recall':mean('node_recall'),'mean_aligned_nodes':mean('aligned_nodes'),'mean_gt_atoms':mean('gt_atoms'),'images':len(results)},'by_profile':{},'samples':results,'gate':{'target_free_proposal':True,'training_performed':False,'sealed_test_read':False,'production_go':False}}
    for profile in sorted({meta[x['sample_key']]['profile_id'] for x in results}):
        g=[x for x in results if meta[x['sample_key']]['profile_id']==profile]; report['by_profile'][profile]={'images':len(g),'mean_node_precision':float(np.mean([x['node_precision'] for x in g])),'mean_node_recall':float(np.mean([x['node_recall'] for x in g]))}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'output':str(OUT/'report.json'),**report['aggregate']},sort_keys=True))

if __name__=='__main__':main()
