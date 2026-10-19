"""Fixed58 CPU shadow review of saved predicted edges, with separate label scoring."""

import argparse
from collections import Counter
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, __version__ as pillow_version

from chemvector.bond_object import BOND_TYPES
from chemvector.training.curriculum import runtime
from tools import pair_local_evidence as evidence
from tools.preview_pair_false_edges import read, ROOT, TRIAL

OUTPUT = ROOT / 'outputs/audits/chemvector-pair-local-evidence58-20260921-r1'
LABELS = ROOT / 'outputs/audits/chemvector-pair-regression-diagnostics58-20260921'
FOCUS = {'curr_4ef506af64e8f0b8__bond_short': (15,18),
         'curr_08d02946e879406f__font_large': (4,6)}
FLAGS = ('interior_candidate', 'alternate_corridor_path', 'raster_gap', 'carbon_conflict', 'combined_review')
CUTOFF = math.log(.85/.15)


def write(path, obj):
    payload = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(payload + '\n')


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only diagnostic required')
    if not OUTPUT.is_dir(): raise ValueError('Fresh budgeted output required')
    hashes = {}
    def data(path, digest=None): return json.loads(read(path, hashes, digest))
    manifest = data(ROOT/'outputs/datasets/chemvector-selector-retention-fit-20260919/manifest.json',
                    'f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111')
    post = data(TRIAL/'postcheck.json', 'b09b82c98fe5280de0d79a6cca858a2854398fa66dca1d29205997f4cece1f54')
    prior = data(LABELS/'report.json', '7bede56f8f8cbf5e7bc03a1e70cf964efb638bebff80840888bab146464789a2')
    for path in (Path(__file__), Path(evidence.__file__), Path(runtime.__file__),
                 ROOT/'tools/preview_pair_false_edges.py', ROOT/'tests/chemvector/test_pair_local_evidence.py',
                 ROOT/'chemvector/components/bonds/objects/bond_object.py'):
        read(path, hashes)
    rows = manifest['training_samples']
    if (len(rows) != 58 or len({r['sample_key'] for r in rows}) != 58
            or any(r['split'] != 'train' for r in rows)
            or Counter(r['fit_domain'] for r in rows) != {'witness':26, 'old_full_molecule':32}):
        raise ValueError('Fixed58 scope mismatch')
    write(OUTPUT/'plan.json', dict(schema_version='pair-local-evidence-plan-0.1',
        config=evidence.DEFAULT.to_dict(), pair_threshold=.85,
        sample_keys=[r['sample_key'] for r in rows], phases=[0,50],
        query_scope='All saved raw-present edges, plus two fixed diagnostic probes when absent',
        labels_scope='Only exact sampled-label receipts on 50 updated images; unsampled is unclassified, not negative',
        hypothesis='Combined warning = neutral-carbon conflict AND (wide raster gap OR predicted local alternate path)',
        policy='No graph edits, threshold tuning, parameter search, new supervision or deployment',
        after_the_fact_exploration=True, independent_accuracy_claim=False,
        wall_seconds=120, gpu_used=False, model_forwards=0, optimizer_steps=0, input_sha256=dict(hashes)))
    counts = {str(step):Counter() for step in (0,50)}
    flagged_true, flagged_unclassified, focus, artifacts = [], [], [], {}
    for row in rows:
        key = row['sample_key']; folder = Path(manifest['sources'][row['source']])/'samples'
        meta = data(folder/(key+'.meta.json'), row['meta_sha256'])
        png = read(folder/(key+'.png'), hashes, meta['sha256']['.png'])
        image = Image.open(io.BytesIO(png))
        if image.size != (768,768): raise ValueError('Expected same-source 768px square raster')
        gray = np.asarray(image.convert('L'))
        # Ground truth / sampled labels are NOT passed to the evidence helper.
        label_case = data(LABELS/(key+'.json'), prior['case_sha256'][key+'.json'])
        replay = label_case['presence_loss_replay']
        sampled = ({} if replay is None else {r['row_id']:r['label']
                    for r in replay['arms']['initial_logits']['per_sampled_row']})
        results = {}
        for step in (0,50):
            name = f'check-{step:06d}--{key}.json'
            record = data(TRIAL/name, post['artifact_sha256'][name])
            if record['metrics']['key'] != row['source']+':'+key:
                raise ValueError('Wrong saved sample')
            graph = evidence.prediction_graph(record['trace'], image.size)
            pairs, logits, chemical = record['pairs'], record['presence_logits'], record['chemical_logits']
            if (len(pairs) != len(logits) or len(pairs) != len(chemical)
                    or any(not math.isfinite(x) for x in logits)
                    or any(len(c) != len(BOND_TYPES) or not all(math.isfinite(x) for x in c) for c in chemical)):
                raise ValueError('Invalid saved pair logits')
            raw_present = {tuple(pair) for pair,z in zip(pairs,logits) if z >= CUTOFF}
            if raw_present != set(graph['edges']):
                raise ValueError('Raw presence and saved scene topology differ')
            stats = counts[str(step)]; stats['images'] += 1
            stats['images_with_sampled_labels'] += replay is not None
            output = []
            for index,pair_list in enumerate(pairs):
                pair = tuple(pair_list); present = pair in raw_present
                if not present and pair != FOCUS.get(key): continue
                kind = BOND_TYPES[max(range(len(BOND_TYPES)), key=chemical[index].__getitem__)]
                result = evidence.inspect_pair(gray, graph, pair, kind)
                # Only after prediction-only inspection is complete do we add labels for scoring.
                label = ('sampled_positive' if sampled[index] else 'sampled_negative') if index in sampled else 'unclassified'
                result.update(row_id=index, raw_present=present, probability=1/(1+math.exp(-logits[index])),
                              scoring_label=label)
                output.append(result)
                if present:
                    stats['present_edges'] += 1; stats[label] += 1
                    stats['pixel_unavailable'] += not result['pixels']['available']
                    for flag,value in result['flags'].items():
                        stats[flag] += value
                        stats[label+'__'+flag] += value
                    if result['flags']['combined_review']:
                        summary = dict(sample_key=key, step=step, pair=pair, row_id=index,
                                       probability=result['probability'], flags=result['flags'])
                        if label == 'sampled_positive': flagged_true.append(summary)
                        elif label == 'unclassified': flagged_unclassified.append(summary)
                if pair == FOCUS.get(key): focus.append(dict(sample_key=key, step=step, **result))
            results[str(step)] = output
        path = OUTPUT/(key+'.json')
        write(path, dict(sample_key=key, group=row['fit_domain'], phases=results))
        artifacts[path.name] = sha256(path.read_bytes()).hexdigest()
    if ([counts[str(s)]['images'] for s in (0,50)] != [58,58]
            or [counts[str(s)]['present_edges'] for s in (0,50)] != [1093,1126]
            or len(focus) != 4): raise ValueError('Incomplete saved scope')
    for path,digest in list(hashes.items()): read(path, hashes, digest)
    write(OUTPUT/'report.json', dict(schema_version='pair-local-evidence58-0.1',
        config=evidence.DEFAULT.to_dict(), totals=counts, fixed_probes=focus,
        combined_warning_known_true_edges=flagged_true,
        combined_warning_unclassified_edges=flagged_unclassified,
        input_sha256=hashes, case_sha256=artifacts,
        versions=dict(numpy=np.__version__, pillow=pillow_version),
        training_performed=False, model_forwards=0, checkpoint_loads=0, optimizer_steps=0, gpu_used=False,
        graph_edits=0, production_wired=False, original_acceptance_changed=False,
        sealed_test_read=False, validation_sample_artifacts_read=False, new_identity_queries=0,
        limitations=['Post-hoc exploratory rules; not an independently validated edge classifier',
            'Only sampled rows have recovered exact labels; unclassified must not be counted as false',
            'Neutral-carbon conflict implicates all incident edges, not necessarily the incorrect one',
            'A predicted alternate path can itself contain wrong nodes or edges',
            'RGB evidence is not feature/attention attribution; text boxes and anchors may be inaccurate',
            'No deletion, repair or accuracy improvement has been performed']))
    print(json.dumps(dict(totals=counts, fixed_probes=[dict(sample_key=f['sample_key'], step=f['step'],
        flags=f['flags'], max_gap=(f['pixels'].get('profiles') or {}).get('wide')) for f in focus]), indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--worker',action='store_true')
    args=parser.parse_args(argv)
    if args.worker: worker(); return 0
    receipt=runtime.run_budgeted([sys.executable,'-m','tools.audit_pair_local_evidence','--worker'],
                                 output=OUTPUT,budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt,indent=2))
    return 0 if receipt['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
