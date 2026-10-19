"""Replay canonical tie selection on exactly seven already saved CPU heatmaps."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

from chemvector.training.curriculum import runtime
from chemvector.components.graph.joint.predicted_incidence import _local_peak_topk
from tools.audit_selector_stable_subset import Inputs, write
from tools import selector_peak_tie_contract as tie

ROOT = Path(__file__).resolve().parents[1]
DENSE = ROOT / 'outputs/audits/chemvector-selector-dense-observation7-20260920'
DENSE_SHA = '7ba69b5f75241fced837611a9dbdef236a1855fdbbd290f01ff50154f78ec1ff'
OUTPUT = ROOT / 'outputs/audits/chemvector-selector-peak-tie-replay7-20260921'


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU saved-heatmap replay only')
    if not OUTPUT.is_dir() or (OUTPUT / 'plan.json').exists():
        raise ValueError('Fresh one-shot output required')
    torch.set_num_threads(2)
    inputs = Inputs()
    prior = inputs.json(DENSE / 'report.json', DENSE_SHA)
    if len(prior['snapshot_sha256']) != 7 or prior['focused_targets'] != 10:
        raise ValueError('Original seven-image scope changed')
    for path in (Path(__file__), Path(tie.__file__), Path(runtime.__file__),
                 ROOT / 'chemvector/components/graph/joint/predicted_incidence.py',
                 ROOT / 'tests/chemvector/test_selector_peak_tie_contract.py',
                 ROOT / 'tests/chemvector/test_selector_canonical_adapters.py'):
        inputs.read(path, prior['input_sha256'].get(str(path)))
    write(OUTPUT / 'plan.json', dict(policy=tie.POLICY, snapshots=list(prior['snapshot_sha256']),
        maximum_atoms=128, threshold=.01, wall_seconds=120, automatic_retries=0,
        model_forwards=0, optimizer_steps=0, gpu_devices=[], training_authorized=False,
        same_complete_saved_dense_tensor=True, production_decoder_changed=False))
    cases, counts = [], Counter()
    for name, digest in prior['snapshot_sha256'].items():
        key = name[:-4]
        with np.load(BytesIO(inputs.read(DENSE / name, digest)), allow_pickle=False) as saved:
            snapshot = {k: torch.from_numpy(saved[k].copy()) for k in saved.files}
        z = snapshot['heatmap_logits']
        old = tuple(snapshot['raw_'+k] for k in ('scores', 'indices', 'mask'))
        reproduced = _local_peak_topk(z, maximum=128, threshold=.01)
        if any(not torch.equal(a, b) for a, b in zip(old, reproduced)):
            raise ValueError('Saved old CPU decoder did not reproduce')
        canonical = tie.stable_local_peak_topk(z, maximum=128, threshold=.01)
        difference = tie.selection_difference(z, old, canonical, maximum=128, threshold=.01)
        second = tie.stable_local_peak_topk(z, maximum=128, threshold=.01)
        if any(not torch.equal(a, b) for a, b in zip(canonical, second)):
            raise ValueError('Canonical decoding of the same tensor changed')
        path = ROOT / 'outputs/experiments/chemvector-selector-full-loss-control-20260920' / ('geometry--'+key+'.json')
        geometry = inputs.json(path, prior['input_sha256'][str(path)])
        dense_xy = snapshot['anchors'][0].flatten(1).T
        def relations(result):
            ids = result[1][0, result[2][0]].tolist()
            return {(i, t) for i in ids for t, xy in enumerate(geometry['target_anchors'])
                    if math.dist(dense_xy[i].tolist(), xy)*768 <= 6}
        before, after = relations(old), relations(canonical)
        old_covered, new_covered = ({t for i, t in edges} for edges in (before, after))
        detail = dict(sample_key=key, **difference, legacy_snapshot_reproduced=True,
            repeated_canonical_result_bitwise_equal=True,
            lost_eligible_grid_target_relations=sorted(before-after),
            gained_eligible_grid_target_relations=sorted(after-before),
            raw_missing_before=sorted(set(range(len(geometry['target_anchors'])))-old_covered),
            raw_missing_after=sorted(set(range(len(geometry['target_anchors'])))-new_covered))
        cases.append(detail)
        batch = difference['batches'][0]
        counts.update(images=1, active_set_changed=not batch['active_set_unchanged'],
            active_order_changed=not batch['active_order_unchanged'],
            removed_candidates=len(batch['removed_flat_ids']), added_candidates=len(batch['added_flat_ids']),
            lost_eligible_relations=len(before-after), gained_eligible_relations=len(after-before),
            raw_missing_before=len(detail['raw_missing_before']), raw_missing_after=len(detail['raw_missing_after']))
    for path, digest in list(inputs.hashes.items()): inputs.read(path, digest)
    write(OUTPUT / 'report.json', dict(schema_version='selector-peak-tie-replay7-0.1',
        input_sha256=inputs.hashes, counts=counts, cases=cases, model_forwards=0,
        optimizer_steps=0, gpu_used=False, training_authorized=False, default_changed=False,
        production_decoder_changed=False, validation_sample_artifacts_read=False, sealed_test_read=False,
        limitations=['Only seven existing full heatmaps, not all58 images or a new accuracy evaluation',
            'No selector/bond/SDF forward and no new candidate features reconstructed',
            'The previous first-image parity failure has no full heatmap in its saved snapshot',
            'Exact tie ordering does not remove flat background candidates or ensure CPU/GPU dense numeric equality']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_selector_peak_tie_replay', '--worker'],
        output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
