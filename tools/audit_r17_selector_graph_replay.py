#!/usr/bin/env python3
"""Frozen graph replay after selector-retention sidecar proposals."""
from __future__ import annotations
from collections import Counter
import hashlib,json,os,sys
from pathlib import Path
import torch
from PIL import Image
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.components.graph.joint.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA
from tools.audit_direct_graph_pair_scorer_chemistry_r9 import sanitize
from tools.train_element_context_head_cpu_r17 import ElementContextHead, ELEMENTS
from tools.train_type_residual_cpu_r17 import TypeContextHead
from tools.train_graph_context_aromatic_cpu_r17 import AromaticContextHead
from tools.audit_graph_context_aromatic_compile_r17 import cycle_edges

CACHE=ROOT/"outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m128-t0p72-c1-panel128/feature-cache.pt"
SOURCE=ROOT/os.environ.get("R17_SELECTOR_SOURCE", "outputs/datasets/structure-expansion-batch-12272-20260923/samples")
MANIFEST_PATH = os.environ.get("R17_SELECTOR_MANIFEST")
TRAIN_MOLECULES = int(os.environ.get("R17_SELECTOR_TRAIN_MOLECULES", "0"))
HOLDOUT_MOLECULES = int(os.environ.get("R17_SELECTOR_HOLDOUT_MOLECULES", "0"))
STATE=ROOT/os.environ.get("R17_SELECTOR_STATE","outputs/audits/chemvector-r17-selector-retention-m128-v4/sidecar-state.pt")
ELEMENT_STATE=ROOT/"outputs/audits/chemvector-element-context-head-cpu-r17-all-m128-panel128/head-state.pt"
TYPE_STATE=ROOT/"outputs/audits/chemvector-type-residual-cpu-r17-all-m128-panel128/sidecar-state.pt"
AROMATIC_STATE=ROOT/"outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m128-t0p72-c1-panel128/sidecar-state.pt"
OUT=ROOT/os.environ.get("R17_SELECTOR_GRAPH_OUT","outputs/audits/chemvector-r17-selector-graph-replay-m128-v1")
PAIR_THRESHOLD=.72; MAX_NEW=16
EXCLUDE_NEAR_SELECTED = os.environ.get("R17_SELECTOR_EXCLUDE_NEAR_SELECTED", "1") == "1"
THRESHOLD_OVERRIDE = os.environ.get("R17_SELECTOR_THRESHOLD")
SAFE_COMPILE_ONLY = os.environ.get("R17_SELECTOR_SAFE_COMPILE", "0") == "1"
PERTURB_TARGET = os.environ.get("R17_SELECTOR_PERTURB_TARGET", "0") == "1"
NMS_RADIUS = float(os.environ.get("R17_SELECTOR_NMS_RADIUS", "0"))

class SelectorSidecar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.LayerNorm(133),torch.nn.Linear(133,32),torch.nn.GELU(),torch.nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def target(key):
    layers=json.loads((SOURCE/f"{key}.layers.json").read_text());graph=json.loads((SOURCE/f"{key}.graph.json").read_text())
    atoms=sorted(layers['atoms'],key=lambda x:int(x['atom_id'])); centers=torch.tensor([a['center'] for a in atoms],dtype=torch.float32)/768.0
    edges={tuple(sorted((int(b['begin_atom']),int(b['end_atom'])))) for b in graph['bonds']}; types={tuple(sorted((int(b['begin_atom']),int(b['end_atom'])))):str(b.get('chemical_type','SINGLE')).upper() for b in graph['bonds']};
    if PERTURB_TARGET: return [], torch.zeros_like(centers), set(), {}
    return atoms,centers,edges,types

def load_records():
    """Load the fixed cache, or a deterministic public release slice.

    The release mode only reads the already-audited manifest and sample
    payloads needed for the frozen forward replay. It never changes labels,
    routing, or the production checkpoint.
    """
    if not MANIFEST_PATH:
        return torch.load(CACHE,map_location='cpu',weights_only=False)
    manifest=json.loads(Path(MANIFEST_PATH).read_text())
    rows=manifest['rows']
    grouped={split:{} for split in ('train','holdout')}
    for row in rows:
        split='fresh_holdout' if row['split']=='holdout' else row['split']
        grouped.setdefault(split,{}).setdefault(row['molecule_id'],[]).append(row)
    limits={'train':TRAIN_MOLECULES,'fresh_holdout':HOLDOUT_MOLECULES}
    records={'train':[],'dev':[],'fresh_holdout':[]}
    for split,by_molecule in grouped.items():
        if split not in limits: continue
        ids=sorted(by_molecule)
        limit=limits[split]
        if limit<=0: raise ValueError(f'{split} molecule limit must be positive in release mode')
        chosen=ids[:limit]
        records[split]=[row for mid in chosen for row in sorted(by_molecule[mid],key=lambda x:x['sample_key'])]
    return {'initial_checkpoint_sha256':SHA,'records':records,
            'release_manifest':str(Path(MANIFEST_PATH).resolve()),
            'release_molecule_limits':limits}
def match(points,centers):
    if not len(points) or not len(centers): return {}
    d=torch.cdist(points.float()*768.,centers.float()*768.); cand=[(float(d[i,j]),i,j) for i in range(d.size(0)) for j in range(d.size(1)) if float(d[i,j])<=16.]
    out={};used=set()
    for _,i,j in sorted(cand):
        if i not in out and j not in used:out[i]=j;used.add(j)
    return out

def score_nms(indices, points, scores, radius):
    """Target-free NMS over sidecar candidates, preserving score order."""
    if radius <= 0 or len(indices) <= 1:
        return indices
    order = indices[torch.argsort(scores[indices],descending=True)]
    kept=[]
    for index in order.tolist():
        if not kept:
            kept.append(index); continue
        distance=torch.linalg.vector_norm((points[kept]-points[index])*768.0,dim=1)
        if bool((distance > radius).all()): kept.append(index)
    return torch.tensor(kept,device=indices.device,dtype=indices.dtype)
def mapped_edges(points,active,pairs,probs,centers,truth):
    compact=match(points,centers); mapping={int(active[i]):j for i,j in compact.items()}; out=set()
    for i in torch.where(probs>=PAIR_THRESHOLD)[0].tolist():
        a,b=(int(x) for x in pairs[i].tolist());
        if a in mapping and b in mapping and mapping[a]!=mapping[b]: out.add(tuple(sorted((mapping[a],mapping[b]))))
    return out,mapping

def compile_prediction(raw, direct, mask, type_head, element_head, aromatic_head, aromatic_threshold, nodes, centers=None, target_atoms=None, target_types=None, oracle_elements=False, oracle_types=False, presence_override=None):
    # ``presence_override`` is an optional target-free routing tensor aligned
    # with ``direct`` pairs.  Chemical/type heads continue consuming the
    # frozen direct logits; only edge admission/context ownership changes.
    route_presence = direct['presence_logits'] if presence_override is None else presence_override
    active=torch.where(mask)[0]
    pairs, sample_indices=enumerate_unordered_atom_pairs(mask[None])
    # This helper is called with direct output generated for the same pairs.
    point=raw['atom_geometry'][0,:, :2][pairs]
    delta=point[:,1]-point[:,0]
    distance=torch.linalg.vector_norm(delta*768.,dim=-1)
    scores=raw['atom_scores'][0,pairs]
    type_features=torch.cat((direct['hidden'].float(),direct['presence_logits'][:,None].float(),point.mean(1),distance[:,None],scores.float()),dim=-1)
    typed=type_head(type_features).argmax(-1)
    raw_pair_rows=[(int(a),int(b),i) for i,(a,b) in enumerate(pairs.tolist())]
    raw_presence={(a,b) for a,b,i in raw_pair_rows if float(route_presence[i].sigmoid())>=PAIR_THRESHOLD}
    cycle_selected=cycle_edges(raw_pair_rows,raw_presence)
    adjacency={int(node):set() for node in nodes}
    for a,b in raw_presence: adjacency[a].add(b);adjacency[b].add(a)
    aromatic_rows=[]
    for i,(a,b) in enumerate(pairs.tolist()):
        midpoint=point[i].mean(0); dxy=delta[i]; dist=distance[i:i+1]; deg=torch.tensor([float(len(adjacency.get(int(a),()))),float(len(adjacency.get(int(b),())))],device=type_features.device)
        arom_feat=torch.cat((direct['hidden'][i].float(),direct['chemical_logits'][i].float(),raw['atom_features'][0,a].float(),raw['atom_features'][0,b].float(),midpoint,dxy,dist,scores[i].float(),torch.tensor([float((int(a),int(b)) in cycle_selected),deg[0],deg[1]],device=type_features.device)))
        aromatic_rows.append(arom_feat)
    aromatic_logits=aromatic_head(torch.stack(aromatic_rows)) if aromatic_rows else torch.empty((0,),device=type_features.device)
    by_node={int(node):[] for node in nodes}
    for i,(a,b) in enumerate(pairs.tolist()):
        # Match the element-context head's training contract: context is
        # aggregated over presence-admitted edges only, not every unordered
        # pair.  All-pair aggregation is a severe OOD mismatch and can turn
        # otherwise correct C/N/O predictions into Si/P noise.
        if float(route_presence[i].sigmoid()) < PAIR_THRESHOLD:
            continue
        by_node[int(a)].append(type_features[i]);by_node[int(b)].append(type_features[i])
    atom_rows=[]
    for node in nodes:
        vals=by_node[int(node)]
        if vals:
            mean_context=torch.stack(vals).mean(0); max_context=torch.stack(vals).max(0).values
        else:
            # Fail closed for a degree-zero node while preserving the fixed
            # 653-D head contract; this is never a target-dependent fallback.
            mean_context=type_features.new_zeros(type_features.shape[-1]); max_context=mean_context
        atom_rows.append(torch.cat((raw['atom_features'][0,node].float(),mean_context,max_context,torch.tensor([float(len(vals))/8.0],device=type_features.device))))
    elements=[ELEMENTS[i] for i in element_head(torch.stack(atom_rows)).argmax(-1).tolist()]
    node_mapping = match(raw['atom_geometry'][0,nodes,:2].detach().cpu(), centers) if centers is not None else {}
    if oracle_elements and target_atoms is not None:
        elements=[str(target_atoms[node_mapping.get(i, -1)].get('element') or 'C') if i in node_mapping else elements[i] for i,node in enumerate(nodes)]
    # ``sanitize`` addresses nodes by their compact position in ``nodes``;
    # AtomPair indices, however, are global raw-slot ids.  Keep the slot
    # space for model lookup, then explicitly compact it at the compiler
    # boundary so non-contiguous active slots cannot silently drop edges.
    local_index={int(node):i for i,node in enumerate(nodes)}
    edge_map={}
    for i,(a,b) in enumerate(pairs.tolist()):
        if float(route_presence[i].sigmoid()) >= PAIR_THRESHOLD:
            kind=int(typed[i])+1
            # The type head's aromatic class is provisional. The frozen
            # aromatic sidecar owns aromatic membership, just as in the
            # training-cache compiler audit.
            if kind==4: kind=1
            if (int(a),int(b)) in cycle_selected and float(aromatic_logits[i].sigmoid()) >= aromatic_threshold:
                kind=4
            if oracle_types and target_types is not None and i < len(pairs):
                a,b=(int(x) for x in pairs[i].tolist()); ta=node_mapping.get(list(nodes).index(a)); tb=node_mapping.get(list(nodes).index(b)); truth=target_types.get(tuple(sorted((int(ta),int(tb))))) if ta is not None and tb is not None else None
                kind={'SINGLE':1,'DOUBLE':2,'TRIPLE':3,'AROMATIC':4}.get(truth,kind)
            if int(a) in local_index and int(b) in local_index:
                edge_map[(local_index[int(a)],local_index[int(b)])]=kind
    aromatic_nodes={u for e,k in edge_map.items() if k==4 for u in e}
    ok,error=sanitize([(int(node),elements[j],j in aromatic_nodes) for j,node in enumerate(nodes)],edge_map)
    prediction={'slots':[int(n) for n in nodes], 'elements':elements,
                'edges':[[int(nodes[a]),int(nodes[b]),kind] for (a,b),kind in sorted(edge_map.items())]}
    return ok,error,prediction

def strict_identity(prediction, raw, centers, atoms, types):
    """Evaluation-only; every predicted edge contributes to precision.

    Unmapped edges cannot silently disappear from the denominator. Exact
    typed topology below is NOT full molecular identity (charge/isotope/
    stereo are absent from this compiler).
    """
    slots=prediction['slots']
    compact=match(raw['atom_geometry'][0,slots,:2].detach().cpu(),centers)
    mapping={slot:compact[i] for i,slot in enumerate(slots) if i in compact}
    mapped={};unmapped=0
    for a,b,kind in prediction['edges']:
        if a not in mapping or b not in mapping:
            unmapped+=1;continue
        mapped[tuple(sorted((mapping[a],mapping[b])))]=kind
    names={'SINGLE':1,'DOUBLE':2,'TRIPLE':3,'AROMATIC':4}
    typed_truth={edge:names.get(kind,0) for edge,kind in types.items()}
    true=sum(edge in typed_truth for edge in mapped)
    typed_true=sum(typed_truth.get(edge)==kind for edge,kind in mapped.items())
    element_correct=sum(prediction['elements'][i]==str(atoms[t].get('element') or 'C') for i,t in compact.items())
    return {'predicted_atoms':len(slots),'mapped_atoms':len(compact),'target_atoms':len(atoms),
            'element_correct':element_correct,'predicted_edges':len(prediction['edges']),
            'unmapped_edges':unmapped,'true_edges':true,'typed_true_edges':typed_true,
            'target_edges':len(types),'exact_typed_topology':bool(len(compact)==len(slots)==len(atoms)
            and unmapped==0 and mapped==typed_truth),
            'exact_elements_and_typed_topology':bool(len(compact)==len(slots)==len(atoms)
            and unmapped==0 and mapped==typed_truth and element_correct==len(atoms))}

def run():
    if OUT.exists():raise FileExistsError(OUT)
    if digest(INITIAL)!=SHA:raise RuntimeError('checkpoint changed')
    cache=load_records();state=torch.load(STATE,map_location='cpu',weights_only=False)
    if cache.get('initial_checkpoint_sha256')!=SHA or state.get('initial_checkpoint_sha256')!=SHA:raise RuntimeError('binding mismatch')
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');model,config,_=load_bundle_for_probe(INITIAL,SHA);model.to(device).eval().requires_grad_(False)
    mean=state['mean'].to(device);std=state['std'].to(device);threshold=float(THRESHOLD_OVERRIDE) if THRESHOLD_OVERRIDE is not None else float(state['threshold']); scorer=SelectorSidecar().to(device);scorer.load_state_dict(state['state_dict']);scorer.eval(); element_state=torch.load(ELEMENT_STATE,map_location='cpu',weights_only=False);type_state=torch.load(TYPE_STATE,map_location='cpu',weights_only=False);aromatic_state=torch.load(AROMATIC_STATE,map_location='cpu',weights_only=False);element_head=ElementContextHead().to(device);element_head.load_state_dict(element_state['state_dict']);element_head.eval();type_head=TypeContextHead(262).to(device);type_head.load_state_dict(type_state['state_dict']);type_head.eval();aromatic_head=AromaticContextHead(526).to(device);aromatic_head.load_state_dict(aromatic_state['state_dict']);aromatic_head.eval();aromatic_threshold=float(aromatic_state.get('threshold',.5))
    rows=[]; decisions=[]
    with torch.inference_mode():
        for split,records in cache['records'].items():
            for rec in records:
                key=rec['sample_key'];target_atoms,centers,truth,target_types=target(key);pix=image_tensor(Image.open(SOURCE/f'{key}.png').convert('RGB'))[None].to(device);outputs=structure_forward(model,pix,config,-.25);vision,raw,base,setout=outputs[0],outputs[3],outputs[6],outputs[5]
                raw_mask=raw['atom_mask'][0].bool();base_mask=base['atom_mask'][0].bool();raw_slots=torch.where(raw_mask)[0]
                raw_features=raw['atom_features'][0,raw_slots].float();xy=raw['atom_geometry'][0,raw_slots,:2].float();scores=raw['atom_scores'][0,raw_slots].float();pres=setout['atom_presence_logits'][0,raw_slots].float();pre=setout.get('atom_pre_presence_logits',setout['atom_presence_logits'])[0,raw_slots].float();x=torch.cat((raw_features,scores[:,None],xy,pres[:,None],pre[:,None]),-1);p=scorer((x-mean)/std).sigmoid();eligible=(p>=threshold)&(~base_mask[raw_slots]);candidate_pre_owner=torch.where(eligible)[0]
                if EXCLUDE_NEAR_SELECTED and int(base_mask.sum())>0:
                    selected_points=raw['atom_geometry'][0,torch.where(base_mask)[0],:2]
                    min_distance=torch.cdist(raw['atom_geometry'][0,raw_slots,:2],selected_points).min(-1).values*768.0
                    eligible &= min_distance > 16.0
                owner_excluded=0
                if EXCLUDE_NEAR_SELECTED and int(base_mask.sum())>0:
                    # Keep this count separate from NMS so the 16 px owner
                    # contract is auditable rather than hidden in one mask.
                    owner_excluded=int(len(candidate_pre_owner)-int(eligible.sum()))
                candidate_post_owner=torch.where(eligible)[0]
                candidate=score_nms(candidate_post_owner,xy,p,NMS_RADIUS)
                if len(candidate)>MAX_NEW:candidate=candidate[torch.argsort(p[candidate],descending=True)[:MAX_NEW]]
                def graph(mask):
                    pairs,si=enumerate_unordered_atom_pairs(mask[None]);direct=model.pair(vision,raw['atom_features'],raw['atom_geometry'][...,:2],raw['atom_scores'],pairs,si)
                    active=torch.where(mask)[0].cpu(); points=raw['atom_geometry'][0,active,:2].detach().cpu(); probs=direct['presence_logits'].sigmoid().cpu(); pred=compile_prediction(raw,direct,mask,type_head,element_head,aromatic_head,aromatic_threshold,active.tolist(),centers,target_atoms,target_types); oe=compile_prediction(raw,direct,mask,type_head,element_head,aromatic_head,aromatic_threshold,active.tolist(),centers,target_atoms,target_types,oracle_elements=True); ot=compile_prediction(raw,direct,mask,type_head,element_head,aromatic_head,aromatic_threshold,active.tolist(),centers,target_atoms,target_types,oracle_types=True); oa=compile_prediction(raw,direct,mask,type_head,element_head,aromatic_head,aromatic_threshold,active.tolist(),centers,target_atoms,target_types,oracle_elements=True,oracle_types=True); return mapped_edges(points,active,pairs.cpu(),probs,centers,truth)[0],int(len(pairs)),pred,oe,ot,oa
                base_edges,base_pairs,base_compile,base_oe,base_ot,base_oa=graph(base_mask)
                aug_mask=base_mask.clone(); accepted_count=0
                if SAFE_COMPILE_ONLY and base_compile[0]:
                    for index in candidate[torch.argsort(p[candidate],descending=True)].tolist():
                        trial=aug_mask.clone(); trial[raw_slots[index]]=True
                        trial_result=graph(trial)
                        if trial_result[2][0]:
                            aug_mask=trial; accepted_count+=1
                        if accepted_count>=MAX_NEW: break
                elif not SAFE_COMPILE_ONLY:
                    aug_mask[raw_slots[candidate]]=True
                aug_edges,aug_pairs,aug_compile,aug_oe,aug_ot,aug_oa=graph(aug_mask)
                if not torch.equal(aug_mask & base_mask,base_mask):
                    raise RuntimeError('selected prefix changed')
                base_identity=strict_identity(base_compile[2],raw,centers,target_atoms,target_types)
                aug_identity=strict_identity(aug_compile[2],raw,centers,target_atoms,target_types)
                rows.append({'split':split,'sample_key':key,'baseline_pairs':base_pairs,'augmented_pairs':aug_pairs,'candidate_pre_owner':int(len(candidate_pre_owner)),'owner_excluded':owner_excluded,'candidate_post_nms':int(len(candidate)),'new_atoms':int(accepted_count if SAFE_COMPILE_ONLY else len(candidate)),'baseline_sanitize':bool(base_compile[0]),'augmented_sanitize':bool(aug_compile[0]),'baseline_oracle_elements':bool(base_oe[0]),'augmented_oracle_elements':bool(aug_oe[0]),'baseline_oracle_types':bool(base_ot[0]),'augmented_oracle_types':bool(aug_ot[0]),'baseline_oracle_all':bool(base_oa[0]),'augmented_oracle_all':bool(aug_oa[0]),'baseline_compile_error':base_compile[1],'augmented_compile_error':aug_compile[1],'baseline_edges':[list(x) for x in base_edges],'augmented_edges':[list(x) for x in aug_edges],'target_edges':[list(x) for x in truth]})
                rows[-1].update({'baseline_strict_identity':base_identity,'augmented_strict_identity':aug_identity,
                                'baseline_prediction':base_compile[2],'augmented_prediction':aug_compile[2]})
                decisions.append({'split':split,'sample_key':key,'baseline_pairs':base_pairs,'augmented_pairs':aug_pairs,'candidate_pre_owner':int(len(candidate_pre_owner)),'owner_excluded':owner_excluded,'candidate_post_nms':int(len(candidate)),'new_atoms':int(accepted_count if SAFE_COMPILE_ONLY else len(candidate)),'baseline_sanitize':bool(base_compile[0]),'augmented_sanitize':bool(aug_compile[0])})
                decisions[-1].update({'baseline_prediction':base_compile[2],'augmented_prediction':aug_compile[2]})
    summary={}
    for split in ('train','dev','fresh_holdout'):
      rs=[r for r in rows if r['split']==split]; truth=[set(map(tuple,r['target_edges'])) for r in rs];be=[set(map(tuple,r['baseline_edges'])) for r in rs];ae=[set(map(tuple,r['augmented_edges'])) for r in rs];
      summary[split]={'images':len(rs),'baseline_pairs':sum(r['baseline_pairs'] for r in rs),'augmented_pairs':sum(r['augmented_pairs'] for r in rs),'candidate_pre_owner':sum(r['candidate_pre_owner'] for r in rs),'owner_excluded':sum(r['owner_excluded'] for r in rs),'candidate_post_nms':sum(r['candidate_post_nms'] for r in rs),'new_atoms':sum(r['new_atoms'] for r in rs),'target_edges':sum(map(len,truth)),'baseline_predicted_edges':sum(map(len,be)),'augmented_predicted_edges':sum(map(len,ae)),'baseline_true_edges':sum(len(a&b) for a,b in zip(be,truth)),'augmented_true_edges':sum(len(a&b) for a,b in zip(ae,truth)),'baseline_sanitize_rate':sum(r['baseline_sanitize'] for r in rs)/max(len(rs),1),'augmented_sanitize_rate':sum(r['augmented_sanitize'] for r in rs)/max(len(rs),1)}
      for name in ('oracle_elements','oracle_types','oracle_all'):
        summary[split][f'baseline_{name}_sanitize_rate']=sum(r[f'baseline_{name}'] for r in rs)/max(len(rs),1);summary[split][f'augmented_{name}_sanitize_rate']=sum(r[f'augmented_{name}'] for r in rs)/max(len(rs),1)
      summary[split].update({'baseline_precision':summary[split]['baseline_true_edges']/max(summary[split]['baseline_predicted_edges'],1),'augmented_precision':summary[split]['augmented_true_edges']/max(summary[split]['augmented_predicted_edges'],1),'baseline_recall':summary[split]['baseline_true_edges']/max(summary[split]['target_edges'],1),'augmented_recall':summary[split]['augmented_true_edges']/max(summary[split]['target_edges'],1)})
      for arm in ('baseline','augmented'):
        counts=Counter()
        for row in rs: counts.update(row[f'{arm}_strict_identity'])
        counts.update({'edge_precision':counts['true_edges']/max(counts['predicted_edges'],1),
                       'typed_edge_precision':counts['typed_true_edges']/max(counts['predicted_edges'],1),
                       'typed_edge_recall':counts['typed_true_edges']/max(counts['target_edges'],1),
                       'element_accuracy':counts['element_correct']/max(counts['mapped_atoms'],1)})
        summary[split][f'{arm}_strict_identity']=dict(counts)
    decision_digest=hashlib.sha256(json.dumps(decisions,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    fresh=summary['fresh_holdout']; bs=fresh['baseline_strict_identity']; as_=fresh['augmented_strict_identity']
    result={'schema_version':'chemvector-r17-selector-graph-replay-0.1','complete':True,'read_only':True,'training_performed':False,'gpu_used':device.type=='cuda','device':str(device),'sealed_test_read':False,'production_checkpoint_written':False,'initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':digest(INITIAL),'threshold':threshold,'exclude_near_selected':EXCLUDE_NEAR_SELECTED,'owner_radius_px':16.0,'nms_radius_px':NMS_RADIUS,'safe_compile_only':SAFE_COMPILE_ONLY,'target_perturbation':PERTURB_TARGET,'context_scope':'presence_admitted_edges','strict_identity_attribution':True,'decision_digest':decision_digest,'summary':summary,'rows':rows,'gate':{'checkpoint_unchanged':digest(INITIAL)==SHA,'selected_prefix_retention':True,'fresh_recall_no_drop':fresh['augmented_recall']>=fresh['baseline_recall'],'fresh_precision_ge_0.90':fresh['augmented_precision']>=.90,'fresh_precision_no_drop_gt_0.01':fresh['augmented_precision']>=fresh['baseline_precision']-.01,'fresh_strict_edge_precision_no_drop_gt_0.01':as_['edge_precision']>=bs['edge_precision']-.01,'fresh_strict_typed_precision_no_drop_gt_0.01':as_['typed_edge_precision']>=bs['typed_edge_precision']-.01,'fresh_strict_typed_recall_no_drop':as_['typed_edge_recall']>=bs['typed_edge_recall'],'fresh_sanitize_no_drop':fresh['augmented_sanitize_rate']>=fresh['baseline_sanitize_rate'],'production_go':False}}
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'report.json').write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({'output':str(OUT/'report.json'),'summary':summary,'gate':result['gate']},indent=2))
if __name__=='__main__':run()
