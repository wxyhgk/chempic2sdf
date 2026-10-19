"""CPU visual bond-type audit for the direct line-node graph ceiling."""
from collections import Counter, defaultdict
import json, sys, os
from pathlib import Path
import numpy as np
from PIL import Image
from rdkit import rdBase
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry
from tools import audit_line_first_lsd_all_edge_ownership_r5 as lsd
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig
DATA = chemistry.DATA
OUT = ROOT / 'outputs/audits/chemvector-identity-aligned-line-first-lsd-direct-graph-type-audit-r5/report.json'
SHA = '09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'
OFFSET_TOL = float(os.environ.get('DIRECT_OFFSET_TOL', '5.0'))
CLUSTER_RADIUS = float(os.environ.get('DIRECT_CLUSTER_RADIUS', '10.0'))

def glyph_boxes(base):
    raw = json.loads(Path(str(base) + '.text-records.json').read_text()); out = {}
    for g in raw.get('glyphs', []):
        aid, b = int(g['atom_id']), g.get('bbox')
        if not b: continue
        if aid not in out: out[aid] = list(map(float, b))
        else: out[aid] = [min(out[aid][0], b[0]), min(out[aid][1], b[1]), max(out[aid][2], b[2]), max(out[aid][3], b[3])]
    return out

def finish(c):
    d = dict(c); d['edge_precision'] = c['true_edges'] / max(c['mapped_edges'], 1); d['edge_recall'] = c['true_edges'] / max(c['gt_bonds'], 1)
    d['type_accuracy'] = c['type_correct'] / max(c['type_total'], 1); d['type_accuracy_non_aromatic'] = c['type_correct_non_arom'] / max(c['type_total_non_arom'], 1); d['oracle_sanitize_rate'] = c['sanitize_ok'] / max(c['images'], 1); return d

def evaluate(row):
    key = row['sample_key']; base = DATA / 'samples' / key; image = Image.open(Path(str(base) + '.png')).convert('L')
    segments = lsd._alt_detect(image, config=LineSegmentRescueConfig(maximum_candidates=512)); points = segments.reshape(-1, 2) if len(segments) else np.empty((0, 2)); labels, centers = cluster_endpoints(points, CLUSTER_RADIUS); node_of = {i: i for i in range(len(centers))}
    for aid, b in glyph_boxes(base).items():
        c = np.asarray([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]); hits = [(float(np.linalg.norm(centers[i] - c)), i) for i in range(len(centers)) if np.linalg.norm(centers[i] - c) <= 24.0]
        if hits:
            for _, i in sorted(hits): node_of[i] = -1 - aid
    pair_offsets = defaultdict(list)
    for i in range(len(segments)):
        a, b = int(labels[2 * i]), int(labels[2 * i + 1]); na, nb = node_of.get(a, a), node_of.get(b, b)
        if na == nb: continue
        pair = tuple(sorted((na, nb))); u = centers[b] - centers[a]; norm = float(np.linalg.norm(u))
        if norm < 1e-6: continue
        perp = np.asarray([-u[1], u[0]]) / norm; midpoint = (segments[i, 0] + segments[i, 1]) / 2.0
        pair_offsets[pair].append(float(np.dot(midpoint - (centers[a] + centers[b]) / 2.0, perp)))
    # A bond stroke can be fragmented into many LSD segments. Cluster signed
    # perpendicular offsets first; the number of separated rails is the visual
    # multiplicity estimate, rather than raw segment count.
    predicted = {}
    for pair, offsets in pair_offsets.items():
        rails = []
        for value in sorted(offsets):
            if not rails or value - rails[-1][-1] > OFFSET_TOL: rails.append([value])
            else: rails[-1].append(value)
        n = min(len(rails), 3); predicted[pair] = n
    layers = json.loads(Path(str(base) + '.layers.json').read_text()); graph = json.loads(Path(str(base) + '.graph.json').read_text()); targets = np.asarray([a['center'] for a in layers['atoms']], np.float32)
    gt = {tuple(sorted((int(z['begin_atom']), int(z['end_atom'])))): str(z.get('chemical_type', 'OTHER')).upper() for z in graph['bonds']}
    def target(n):
        n = int(n)
        if n < 0: return -n - 1
        ds = np.linalg.norm(targets - centers[n], axis=1) if len(targets) else np.asarray([])
        return int(np.argmin(ds)) if len(ds) and float(ds.min()) <= 24.0 else None
    mapped = {}
    for pair, kind in predicted.items():
        a, b = target(pair[0]), target(pair[1])
        if a is not None and b is not None and a != b: mapped[tuple(sorted((a, b)))] = kind
    true = set(mapped) & set(gt); correct = correct_na = total_na = 0
    for edge in true:
        name = gt[edge]; wanted = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3}.get(name, 0); hit = int(bool(wanted) and mapped[edge] == wanted); correct += hit
        if name not in ('AROMATIC', 'AROM'): total_na += 1; correct_na += hit
    nodes = sorted({x for edge in mapped for x in edge}); idx = {x: i for i, x in enumerate(nodes)}; edges = {(idx[a], idx[b]): kind for (a, b), kind in mapped.items()}; elems = [str(layers['atoms'][x].get('element') or 'C') for x in nodes]; aromatic = [bool(layers['atoms'][x].get('aromatic', False)) for x in nodes]; ok, _, _ = chemistry._sanitize_graph([(x, elems[i], aromatic[i]) for i, x in enumerate(nodes)], edges)
    return Counter(images=1, mapped_edges=len(mapped), true_edges=len(true), gt_bonds=len(gt), type_correct=correct, type_total=len(true), type_correct_non_arom=correct_na, type_total_non_arom=total_na, sanitize_ok=int(ok), segments=len(segments))

def main():
    rdBase.DisableLog('rdApp.error')
    if chemistry.file_digest(chemistry.INITIAL) != SHA: raise RuntimeError('checkpoint changed')
    lsd._install(); rows = sorted([json.loads(x) for x in (DATA / 'samples.jsonl').read_text().splitlines() if x.strip() and json.loads(x).get('render_status') == 'accepted'], key=lambda x: x['sample_key']); outputs = {}
    for split in sorted({r['split'] for r in rows}):
        total = Counter(); profiles = defaultdict(Counter)
        for row in [r for r in rows if r['split'] == split]:
            value = evaluate(row); total.update(value); profiles[row['profile_id']].update(value)
        outputs[split] = {'all': finish(total), 'by_profile': {k: finish(v) for k, v in sorted(profiles.items())}}
    hold, hp = outputs['holdout']['all'], outputs['holdout']['by_profile']; gate = {'checkpoint_unchanged': chemistry.file_digest(chemistry.INITIAL) == SHA, 'sealed_test_read': False, 'holdout_precision_ge_0.95': hold['edge_precision'] >= .95, 'holdout_recall_ge_0.80': hold['edge_recall'] >= .80, 'holdout_type_non_aromatic_ge_0.80': hold['type_accuracy_non_aromatic'] >= .80, 'all_profiles_recall_ge_0.75': all(v['edge_recall'] >= .75 for v in hp.values()), 'all_profiles_sanitize_ge_0.90': all(v['oracle_sanitize_rate'] >= .90 for v in hp.values()), 'production_go': False, 'gpu_go': False}
    report = {'schema_version': 'chemvector-identity-aligned-line-first-lsd-direct-graph-type-audit-r5-0.1', 'complete': True, 'read_only': True, 'training_performed': False, 'gpu_used': False, 'sealed_test_read': False, 'production_checkpoint_written': False, 'proposal_contract': 'direct LSD nodes + glyph alias; segment multiplicity heuristic (synthetic visual ceiling)', 'splits': outputs, 'holdout': hold, 'holdout_profiles': hp, 'gate': gate, 'interpretation': {'aromatic': 'not inferred by multiplicity heuristic; excluded from non-aromatic type gate', 'labels': 'target labels used post-hoc only'}}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n'); print(json.dumps({'output': str(OUT), 'holdout': hold, 'gate': gate}, ensure_ascii=False))

if __name__ == '__main__': main()
