"""Post-hoc chemistry/graph round-trip audit for the isolated line-first route.

All proposals and sidecar scores are generated before target data is read. GT
atom elements are used only after proposal construction to make a diagnostic
RDKit graph; this is not a production prediction or a training artifact.
"""
from __future__ import annotations
import hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from tools.shadow_line_first_edge_bond_type_r5 import DATA, STATE, EdgeBondMLP, _proposals, _candidate_keys, TYPE_NAMES
from tools.audit_line_first_candidate_alignment_r5 import align_nodes, read_jsonl, IMAGE, TARGET_RADIUS, AMBIGUITY_RADIUS
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest

ROOT=Path(__file__).resolve().parents[1]; CANDIDATES=ROOT/'outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5-scored/candidates.jsonl'; OUT=ROOT/'outputs/audits/chemvector-identity-aligned-line-first-roundtrip-r5'
INITIAL_SHA='09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'
BT={"SINGLE":Chem.BondType.SINGLE,"DOUBLE":Chem.BondType.DOUBLE,"TRIPLE":Chem.BondType.TRIPLE,"AROMATIC":Chem.BondType.AROMATIC,"DATIVE":Chem.BondType.DATIVE}

def strict_map(points, targets):
 out={}
 for i,p in enumerate(points):
  d=np.linalg.norm(targets-p,axis=1); close=np.where(d<=TARGET_RADIUS)[0]; near=np.where(d<=AMBIGUITY_RADIUS)[0]
  if len(close)==1 and len(near)==1: out[i]=int(close[0])
 return out

def evaluate(row,cands,model,mean,std,threshold):
 key=row['sample_key']; f,centers,_=_proposals(key); keys=_candidate_keys(key); points=np.asarray([[v['x']*IMAGE,v['y']*IMAGE] for v in cands],np.float32)
 node_map=align_nodes(centers.astype(np.float64),points.astype(np.float64),16.0)
 with torch.inference_mode(): edge_logits,type_logits=model(torch.from_numpy((f-mean)/std))
 scores=torch.sigmoid(edge_logits).numpy(); kinds=np.argmax(type_logits.numpy(),axis=1)+1
 pred={}
 for i,(a,b) in enumerate(keys):
  if scores[i]<threshold or a not in node_map or b not in node_map: continue
  ca,cb=node_map[a],node_map[b]
  if ca==cb: continue
  pair=tuple(sorted((ca,cb)))
  if pair not in pred or scores[i]>pred[pair][0]: pred[pair]=(float(scores[i]),int(kinds[i]))
 base=DATA/'samples'/key; layers=json.loads(base.with_suffix('.layers.json').read_text()); graph=json.loads(base.with_suffix('.graph.json').read_text()); targets=np.asarray([x['center'] for x in layers['atoms']],np.float32); c2t=strict_map(points,targets)
 target_atoms=sorted(graph['atoms'],key=lambda x:x['atom_id']); atom_by={int(x['atom_id']):x for x in target_atoms}; gt_edges={tuple(sorted((int(b['begin_atom']),int(b['end_atom'])))) for b in graph['bonds']}
 # Diagnostic graph uses only matched target atoms and C for unmatched candidate nodes.
 mol=Chem.RWMol(); node_order=[]; candidate_to_mol={}
 for ci in sorted(set(x for pair in pred for x in pair)):
  atom_id=c2t.get(ci); element=atom_by.get(atom_id,{}).get('element','C') if atom_id is not None else 'C'; a=Chem.Atom(str(element)); mi=mol.AddAtom(a); candidate_to_mol[ci]=mi; node_order.append(ci)
 duplicate_edges=0
 for (ca,cb),(_,kind) in pred.items():
  if ca not in candidate_to_mol or cb not in candidate_to_mol or ca==cb: continue
  try: mol.AddBond(candidate_to_mol[ca],candidate_to_mol[cb],BT.get(TYPE_NAMES[kind-1],Chem.BondType.SINGLE))
  except Exception: duplicate_edges+=1
 mol=mol.GetMol(); sanitize=True; error=''
 try: Chem.SanitizeMol(mol)
 except Exception as exc: sanitize=False; error=str(exc).split('\n')[0]
 mapped_edges={tuple(sorted((c2t[a],c2t[b]))) for a,b in pred if a in c2t and b in c2t and c2t[a]!=c2t[b]}; true=mapped_edges&gt_edges
 return {'sample_key':key,'split':row['split'],'pred_edges':len(pred),'mapped_edges':len(mapped_edges),'true_edges':len(true),'gt_bonds':len(gt_edges),'edge_precision':len(true)/max(len(mapped_edges),1),'edge_recall':len(true)/max(len(gt_edges),1),'sanitize':sanitize,'sanitize_error':error,'components':len(Chem.GetMolFrags(mol,asMols=False,sanitizeFrags=False)) if mol.GetNumAtoms() else 0,'duplicate_edge_errors':duplicate_edges,'nodes':len(node_order)}

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--split',choices=('train','holdout','all'),default='holdout');p.add_argument('--limit',type=int,default=0);args=p.parse_args()
 if file_digest(INITIAL)!=INITIAL_SHA: raise RuntimeError('initial checkpoint changed')
 state=torch.load(STATE,map_location='cpu',weights_only=False); model=EdgeBondMLP(len(state['feature_names']));model.load_state_dict(state['model']);model.eval();mean=np.asarray(state['mean'],np.float32);std=np.asarray(state['std'],np.float32);threshold=float(state['threshold']); cands={x['sample_key']:x['candidates'] for x in read_jsonl(CANDIDATES)}; rows=[json.loads(x) for x in (DATA/'samples.jsonl').read_text().splitlines() if x and json.loads(x).get('render_status')=='accepted' and (args.split=='all' or json.loads(x)['split']==args.split)];rows=sorted(rows,key=lambda x:x['sample_key']);rows=rows[:args.limit] if args.limit else rows; results=[evaluate(r,cands[r['sample_key']],model,mean,std,threshold) for r in rows]; n=len(results); s=lambda k:sum(x[k] for x in results); report={'schema_version':'chemvector-identity-aligned-line-first-roundtrip-r5-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'deployment_authorized':False,'production_checkpoint_written':False,'initial_sha256':INITIAL_SHA,'initial_checkpoint_sha256_after':file_digest(INITIAL),'sidecar_state_sha256':hashlib.sha256(STATE.read_bytes()).hexdigest(),'proposal_contract':'line-first proposal and sidecar score before target/element post-hoc chemistry audit','split':args.split,'rows':n,'aggregate':{'edge_precision':s('true_edges')/max(s('mapped_edges'),1),'edge_recall':s('true_edges')/max(s('gt_bonds'),1),'sanitize_rate':s('sanitize')/max(n,1),'mean_components':s('components')/max(n,1),'duplicate_edge_errors':s('duplicate_edge_errors')},'by_error':dict(Counter(x['sanitize_error'] for x in results if not x['sanitize'])), 'gate':{'checkpoint_unchanged':file_digest(INITIAL)==INITIAL_SHA,'sealed_test_read':False,'proposal_target_independent':True,'production_go':False},'samples':results};OUT.mkdir(parents=True,exist_ok=True);(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'output':str(OUT/'report.json'),**report['aggregate'],'gate':report['gate']}))

if __name__=='__main__':main()
