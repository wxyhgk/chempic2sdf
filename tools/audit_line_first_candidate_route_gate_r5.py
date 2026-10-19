"""Train-only radius gate for line-first edges aligned to frozen candidates."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from tools.audit_line_first_candidate_alignment_r5 import DATA, CANDIDATES, evaluate, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-candidate-route-gate-r5"

def rows(split):
    all_rows = [x for x in read_jsonl(DATA / "samples.jsonl") if x.get("split") == split and x.get("render_status", "accepted") == "accepted"]
    return sorted(all_rows, key=lambda x: x["sample_key"])

def summarize(items):
    return {"images":len(items), "mean_bond_recall":float(np.mean([x["bond_recall"] for x in items])), "mean_edge_precision":float(np.mean([x["edge_precision"] for x in items])), "images_all_bonds_recovered":int(sum(x["all_bonds_recovered"] for x in items)), "mean_aligned_nodes":float(np.mean([x["aligned_nodes"] for x in items])), "mean_gt_bonds":float(np.mean([x["gt_bonds"] for x in items]))}

def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--radii',default='8,10,12,16'); p.add_argument('--candidate-radius',type=float,default=16.0); p.add_argument('--max-segments',type=int,default=512); args=p.parse_args()
    cand={x['sample_key']:x['candidates'] for x in read_jsonl(CANDIDATES)}
    tr,ho=rows('train'),rows('holdout')
    sweeps=[]
    for radius in [float(x) for x in args.radii.split(',')]:
        result=[evaluate(r,cand[r['sample_key']],cluster_radius=radius,candidate_radius=args.candidate_radius,max_segments=args.max_segments) for r in tr]
        sweeps.append({'cluster_radius_px':radius,**summarize(result)})
    eligible=[x for x in sweeps if x['mean_edge_precision']>=.85]
    selected=max(eligible,key=lambda x:(x['mean_bond_recall'],x['mean_edge_precision'],-x['cluster_radius_px'])) if eligible else max(sweeps,key=lambda x:x['mean_edge_precision'])
    radius=selected['cluster_radius_px']
    hold=summarize([evaluate(r,cand[r['sample_key']],cluster_radius=radius,candidate_radius=args.candidate_radius,max_segments=args.max_segments) for r in ho])
    gate={'train_precision_ge_085':selected['mean_edge_precision']>=.85,'holdout_precision_ge_080':hold['mean_edge_precision']>=.80,'holdout_recall_ge_050':hold['mean_bond_recall']>=.50,'proposal_target_independent':True,'sealed_test_read':False,'training_performed':False,'production_checkpoint_written':False,'deployment_authorized':False,'cpu_gate':selected['mean_edge_precision']>=.85 and hold['mean_edge_precision']>=.80 and hold['mean_bond_recall']>=.50,'shadow_recommendation':'line/bond-type shadow' if hold['mean_bond_recall']>=.50 and hold['mean_edge_precision']>=.80 else 'NO-GO'}
    report={'schema_version':'chemvector-identity-aligned-line-first-candidate-route-gate-r5-0.1','complete':True,'read_only':True,'proposal_contract':'LSD endpoints -> provisional nodes/edges -> target-free frozen scored vertex candidate alignment; targets only post-hoc metrics','split':{'train_images':len(tr),'holdout_images':len(ho)},'config':{'selected_cluster_radius_px':radius,'candidate_radius_px':args.candidate_radius,'max_segments':args.max_segments},'selection':{'source':'train-only','candidates':sweeps,'selected':selected},'holdout':hold,'gate':gate}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps({'output':str(OUT/'report.json'),'selected':selected,'holdout':hold,'gate':gate},sort_keys=True))

if __name__=='__main__': main()
