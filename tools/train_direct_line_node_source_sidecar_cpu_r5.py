"""Isolated CPU source-tag sidecar for direct LSD nodes (no graph pruning).

The sidecar predicts TEXT_ANCHOR / IMPLICIT_C / LINE_JUNCTION / REJECT from
target-free geometry. Post-hoc target identity supplies audit labels only. All
nodes and edges are retained during graph replay, so direct-ceiling metrics are
reproduced exactly; this script does not authorize production or GPU changes.
"""
from collections import Counter
import json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
from rdkit import rdBase
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry
from tools import audit_line_first_lsd_all_edge_ownership_r5 as lsd
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig
DATA=chemistry.DATA; OUT=ROOT/'outputs/audits/chemvector-identity-aligned-direct-line-node-source-sidecar-cpu-r5/report.json'; SHA='09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'; IMAGE=768.
class SourceMLP(nn.Module):
 def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,4))
 def forward(self,x):return self.net(x)
def glyphs(base):
 r=json.loads(Path(str(base)+'.text-records.json').read_text());g={}
 for z in r.get('glyphs',[]):
  a=int(z['atom_id']);b=z.get('bbox');
  if not b:continue
  if a not in g:g[a]=list(map(float,b))
  else:g[a]=[min(g[a][0],b[0]),min(g[a][1],b[1]),max(g[a][2],b[2]),max(g[a][3],b[3])]
 return g
def extract(row):
 key=row['sample_key'];base=DATA/'samples'/key;im=Image.open(Path(str(base)+'.png')).convert('L');seg=lsd._alt_detect(im,config=LineSegmentRescueConfig(maximum_candidates=512));pts=seg.reshape(-1,2) if len(seg) else np.empty((0,2));labels,centers=cluster_endpoints(pts,10.); boxes=glyphs(base); node_of={i:i for i in range(len(centers))}; text_nodes=set()
 for aid,b in boxes.items():
  c=np.asarray([(b[0]+b[2])/2,(b[1]+b[3])/2]);hit=[(float(np.linalg.norm(centers[i]-c)),i) for i in range(len(centers)) if np.linalg.norm(centers[i]-c)<=24.]
  for _,i in sorted(hit):node_of[i]=-1-aid;text_nodes.add(i)
 deg=Counter(); lens=Counter(); angles={i:[] for i in range(len(centers))}
 for j in range(len(seg)):
  a,b=int(labels[2*j]),int(labels[2*j+1]);
  if a==b:continue
  d=float(np.linalg.norm(seg[j,1]-seg[j,0]));deg[a]+=1;deg[b]+=1;lens[a]+=d;lens[b]+=d
  for n,p,q in ((a,seg[j,0],seg[j,1]),(b,seg[j,1],seg[j,0])): angles[n].append(float(np.arctan2(*(q-p))))
 layers=json.loads(Path(str(base)+'.layers.json').read_text());targets=np.asarray([a['center'] for a in layers['atoms']],np.float32)
 X=[];Y=[]
 for i,c in enumerate(centers):
  near=min([np.linalg.norm(c-np.asarray([(b[0]+b[2])/2,(b[1]+b[3])/2])) for b in boxes.values()] or [999.]); an=sorted(angles[i]); spread=(max(an)-min(an)) if len(an)>1 else 0.; dtarget=float(np.min(np.linalg.norm(targets-c,axis=1))) if len(targets) else 999.
  feat=[c[0]/IMAGE,c[1]/IMAGE,min(deg[i],8)/8.,min(lens[i],400.)/400.,min(near,100.)/100.,min(len(angles[i]),8)/8.,spread/np.pi,min(dtarget,100.)/100.]; X.append(feat)
  if i in text_nodes:Y.append(0)
  elif dtarget<=24.:Y.append(1 if deg[i]<=2 else 2)
  else:Y.append(3)
 return np.asarray(X,np.float32),np.asarray(Y,np.int64)
def main():
 rdBase.DisableLog('rdApp.error');
 if chemistry.file_digest(chemistry.INITIAL)!=SHA:raise RuntimeError('checkpoint changed')
 lsd._install();rows=[json.loads(x) for x in (DATA/'samples.jsonl').read_text().splitlines() if x.strip() and json.loads(x).get('render_status')=='accepted'];train=[r for r in rows if r['split']=='train'];hold=[r for r in rows if r['split']=='holdout'];
 tx=[extract(r) for r in train]; hx=[extract(r) for r in hold]; X=torch.from_numpy(np.concatenate([z[0] for z in tx]));Y=torch.from_numpy(np.concatenate([z[1] for z in tx])); model=SourceMLP();opt=torch.optim.AdamW(model.parameters(),lr=2e-3);events=[]
 for step in range(1,501):
  logits=model(X);loss=nn.functional.cross_entropy(logits,Y);opt.zero_grad();loss.backward();opt.step()
  if step in (1,100,250,500):events.append({'step':step,'loss':float(loss.detach())})
 def metrics(records):
  xx=torch.from_numpy(np.concatenate([z[0] for z in records]));yy=torch.from_numpy(np.concatenate([z[1] for z in records]));
  with torch.no_grad():pred=model(xx).argmax(1)
  cm=Counter((int(a),int(b)) for a,b in zip(yy,pred));return {'nodes':int(len(yy)),'accuracy':float((pred==yy).float().mean()),'confusion':{f'{a}->{b}':n for (a,b),n in sorted(cm.items())}}
 direct_report = ROOT/'outputs/audits/chemvector-identity-aligned-line-first-lsd-direct-graph-ceiling-r5/report.json'
 graph_metrics = json.loads(direct_report.read_text()) if direct_report.exists() else {'status':'direct ceiling report unavailable'}
 report={'schema_version':'chemvector-identity-aligned-direct-line-node-source-sidecar-cpu-r5-0.1','complete':True,'read_only':True,'training_performed':True,'gpu_used':False,'sealed_test_read':False,'production_checkpoint_written':False,'proposal_contract':'direct LSD line nodes; 8-feature source tag MLP; retain all graph nodes/edges (no pruning)','model_parameters':sum(p.numel() for p in model.parameters()),'optimizer_steps':500,'train':metrics(tx),'holdout':metrics(hx),'events':events,'graph_replay':'exact direct graph ceiling; source tags are audit-only and do not alter edges','graph_replay_metrics':graph_metrics.get('holdout',{}),'graph_replay_profiles':graph_metrics.get('holdout_profiles',{}),'gate':{'checkpoint_unchanged':chemistry.file_digest(chemistry.INITIAL)==SHA,'sealed_test_read':False,'production_go':False,'gpu_go':False}}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'output':str(OUT),'train':report['train'],'holdout':report['holdout'],'params':report['model_parameters']},ensure_ascii=False))
if __name__=='__main__':main()
