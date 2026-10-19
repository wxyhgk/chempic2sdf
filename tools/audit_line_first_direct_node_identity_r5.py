"""Read-only direct line-node identity ceiling (no frozen atom candidates)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig, detect_raster_line_segments
from tools.audit_line_first_graph_traversal_r5 import DATA, cluster_endpoints

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/audits/chemvector-identity-aligned-line-first-direct-node-r5'

def evaluate(row, cluster_radius=10., target_radius=24.):
    base=DATA/'samples'/row['sample_key']; image=Image.open(base.with_suffix('.png')).convert('L'); layers=json.loads(base.with_suffix('.layers.json').read_text())
    seg=detect_raster_line_segments(image,config=LineSegmentRescueConfig(maximum_candidates=512)); endpoints=seg.reshape(-1,2) if len(seg) else np.empty((0,2)); labels,centers=cluster_endpoints(endpoints,cluster_radius); targets=np.asarray([x['center'] for x in layers['atoms']],dtype=np.float64)
    d=np.linalg.norm(centers[:,None]-targets[None,:],axis=-1) if len(centers) and len(targets) else np.empty((len(centers),len(targets))); pairs=[(float(d[i,j]),i,j) for i in range(len(centers)) for j in range(len(targets)) if d[i,j]<=target_radius]; mapping={}; used=set()
    for _,i,j in sorted(pairs):
        if i not in mapping and j not in used: mapping[i]=j; used.add(j)
    return {'sample_key':row['sample_key'],'split':row['split'],'line_nodes':len(centers),'gt_atoms':len(targets),'matched_nodes':len(mapping),'node_precision':len(mapping)/max(len(centers),1),'node_recall':len(mapping)/max(len(targets),1)}

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--split',choices=('train','holdout','all'),default='holdout');p.add_argument('--cluster-radius',type=float,default=10.);p.add_argument('--target-radius',type=float,default=24.);args=p.parse_args(); rows=[json.loads(x) for x in (DATA/'samples.jsonl').read_text().splitlines() if x and json.loads(x).get('render_status','accepted')=='accepted' and (args.split=='all' or json.loads(x)['split']==args.split)]; rows.sort(key=lambda x:x['sample_key']); rs=[evaluate(r,args.cluster_radius,args.target_radius) for r in rows]; mean=lambda k:float(np.mean([x[k] for x in rs])) if rs else 0.; report={'schema_version':'chemvector-identity-aligned-line-first-direct-node-r5-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'proposal_contract':'LSD endpoint clusters -> direct provisional nodes; GT only post-hoc node metrics','split':args.split,'rows':len(rs),'config':{'cluster_radius_px':args.cluster_radius,'target_radius_px':args.target_radius},'aggregate':{'mean_node_precision':mean('node_precision'),'mean_node_recall':mean('node_recall'),'mean_line_nodes':mean('line_nodes'),'mean_gt_atoms':mean('gt_atoms')},'gate':{'proposal_target_independent':True,'training_performed':False,'sealed_test_read':False,'production_go':False},'samples':rs}; OUT.mkdir(parents=True,exist_ok=True);(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'output':str(OUT/'report.json'),**report['aggregate']}))
if __name__=='__main__':main()
