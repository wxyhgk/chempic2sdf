"""Full 696/474 replay of the isolated line/bond shadow.

No fitting occurs here. The sidecar state and train-only threshold from the
160/80 shadow are fixed, then the same target-free feature extraction is
replayed over all accepted r5 train/holdout images. Labels are post-hoc metrics
only; production and sealed test remain untouched.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import torch
from tools.shadow_line_first_edge_bond_type_r5 import DATA, INITIAL, INITIAL_SHA, STATE, TYPE_NAMES, _collect, _metrics, EdgeBondMLP
from chemvector.training.curriculum.checkpoints import file_digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-edge-bond-shadow-r5-full/report.json"

def main():
    if file_digest(INITIAL) != INITIAL_SHA: raise RuntimeError("initial checkpoint changed")
    rows=[json.loads(x) for x in (DATA/'samples.jsonl').read_text().splitlines() if x and json.loads(x).get('render_status')=='accepted']
    train=sorted([r for r in rows if r['split']=='train'],key=lambda r:r['sample_key']); hold=sorted([r for r in rows if r['split']=='holdout'],key=lambda r:r['sample_key'])
    state=torch.load(STATE,map_location='cpu',weights_only=False); feature_names=state['feature_names'];
    x_train,y_train,t_train,gt_train,prof_train=_collect(train); x_hold,y_hold,t_hold,gt_hold,prof_hold=_collect(hold)
    model=EdgeBondMLP(x_train.shape[1]); model.load_state_dict(state['model']); model.eval(); mean=np.asarray(state['mean'],dtype=np.float32); std=np.asarray(state['std'],dtype=np.float32); threshold=float(state['threshold'])
    with torch.inference_mode():
        tr_e,tr_k=model(torch.from_numpy((x_train-mean)/std)); ho_e,ho_k=model(torch.from_numpy((x_hold-mean)/std))
    tr_s=torch.sigmoid(tr_e).numpy(); ho_s=torch.sigmoid(ho_e).numpy(); tr_k=tr_k.numpy(); ho_k=ho_k.numpy()
    tr=_metrics(tr_s,y_train,tr_k,t_train,threshold,gt_train); ho=_metrics(ho_s,y_hold,ho_k,t_hold,threshold,gt_hold)
    report={'schema_version':'chemvector-identity-aligned-line-first-edge-bond-shadow-r5-full-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':False,'sealed_test_read':False,'deployment_authorized':False,'production_checkpoint_written':False,'initial_sha256':INITIAL_SHA,'initial_checkpoint_sha256_after':file_digest(INITIAL),'sidecar_state_sha256':hashlib.sha256(STATE.read_bytes()).hexdigest(),'fixed_threshold':threshold,'split':{'train_images':len(train),'holdout_images':len(hold)},'train':{'metrics':tr,'profiles':prof_train},'holdout':{'metrics':ho,'profiles':prof_hold},'gate':{'checkpoint_unchanged':file_digest(INITIAL)==INITIAL_SHA,'sealed_test_read':False,'threshold_source':'fixed train-only shadow threshold','holdout_edge_precision_ge_0_85':ho['edge_precision']>=.85,'holdout_edge_recall_ge_0_50':ho['candidate_edge_recall']>=.50,'holdout_bond_type_accuracy_ge_0_75':ho['conditional_bond_type_accuracy']>=.75,'production_go':False,'next':'independent graph/node identity audit required; no production merge'}}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n'); print(json.dumps({'output':str(OUT),'train':tr,'holdout':ho,'gate':report['gate']},indent=2))

if __name__=='__main__': main()
