#!/usr/bin/env python3
"""Read-only per-profile type/repair audit for the frozen CPU r13 sidecar."""
from __future__ import annotations
from collections import Counter
import hashlib, json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA, rows, _extract, DirectPairResidual
from tools.train_direct_graph_type_profile_residual_cpu_r13 import TypeResidual
from tools.diagnose_aromatic_cycle_veto_r13 import veto_item, type_stats
from tools.audit_direct_graph_pair_scorer_chemistry_r9 import one, finish

OUT = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-type-profile-stats-r13"
PRESENCE = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r10-coverage/scorer-state.pt"
TYPE = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-type-profile-residual-cpu-r13/type-state.pt"
THRESHOLD = .72

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def main():
    if digest(INITIAL) != SHA: raise RuntimeError('production checkpoint changed')
    model, cfg, _ = load_bundle_for_probe(INITIAL, SHA)
    model.eval().requires_grad_(False)
    hold = sorted((r for r in rows() if r.get('split') == 'fresh_holdout'), key=lambda r: r['sample_key'])
    items = [_extract(r, model, cfg, need_labels=True) for r in hold]
    ps = torch.load(PRESENCE, map_location='cpu', weights_only=False)
    presence = DirectPairResidual(262); presence.load_state_dict(ps['state_dict']); presence.eval()
    ts = torch.load(TYPE, map_location='cpu', weights_only=False)
    heads = {p: TypeResidual(262) for p in ts['heads']}
    for p, h in heads.items(): h.load_state_dict(ts['heads'][p]); h.eval()
    by = {}
    for profile in sorted({i['row']['profile_id'] for i in items}):
        subset = [i for i in items if i['row']['profile_id'] == profile]
        rec = {}
        for name, repair in [('cycle_only', False), ('cycle_plus_valence_repair', True)]:
            metrics, types = Counter(), Counter()
            for item in subset:
                q, changed = veto_item(item, presence, heads, repair_valence=repair)
                metrics.update(one(q, presence, THRESHOLD)); types.update(type_stats(item, q, presence))
            rec[name] = {'metrics': finish(metrics), 'type_stats': dict(types)}
        by[profile] = rec
    result = {'schema_version':'chemvector-identity-aligned-direct-graph-type-profile-stats-r13-0.1',
              'complete':True,'read_only':True,'training_performed':False,'gpu_used':False,
              'sealed_test_read':False,'production_checkpoint_written':False,'holdout_images':len(items),
              'profiles':by,'initial_checkpoint_sha256':SHA,'initial_checkpoint_sha256_after':digest(INITIAL),
              'gate':{'checkpoint_unchanged':digest(INITIAL)==SHA,'sealed_test_read':False,'production_go':False}}
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'output':str(OUT/'report.json'),'profiles':by,'gate':result['gate']},indent=2,ensure_ascii=False))
if __name__ == '__main__': main()
