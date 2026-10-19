"""Fixed7 target/provenance audit and saved-logit heatmap derivatives on CPU."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from chemvector.training.curriculum import runtime
from tools.audit_selector_stable_subset import Inputs, write
from tools import audit_selector_dense_observation as prior
from tools import selector_anchor_contract as contract

ROOT = prior.ROOT
DENSE_SHA = '7ba69b5f75241fced837611a9dbdef236a1855fdbbd290f01ff50154f78ec1ff'
OUTPUT = ROOT / 'outputs/audits/chemvector-selector-anchor-contract7-20260920'
REVISION = '1141b40d621f4707b4aaad550d6d71a91eafeb0f'
CHECKPOINTS = (
    ('joint30k', 'outputs/checkpoints/chemvector-v0971-joint-proposal-objects-full50k/samples-030000.pt',
     'f8935e92d64e90a2b03b5f1924813cf19c08e7f6bfe382357b03e0eccd047f75'),
    ('calibration', 'outputs/checkpoints/chemvector-v0972-bond-presence-calibration-10k/best.pt',
     'e689237cc3aa9e880f416789f7d5907af986f7402b34920fabaf239ef82f88a4'),
    ('style_set', 'outputs/checkpoints/chemvector-v0101-set-first-style10k/selected.pt',
     'e982057d57dd4c03864eae890df12e1d8710fce60dfc452b94511d86a5f967ac'),
    ('initial', str(prior.INITIAL.relative_to(ROOT)), prior.INITIAL_SHA),
)
HISTORICAL_FILES = {
    'Data/sdf/chemscene_v0_1.py': 'Data/sdf/chemscene_v0_1.py',
    'Data/sdf/chemscene_v0_2.py': 'Data/sdf/chemscene_v0_2.py',
    'Data/sdf/chemsvg_sequence_v4.py': 'Data/sdf/chemsvg_sequence_v4.py',
    'Data/sdf/build_chemvision_layers.py': 'Data/sdf/build_chemvision_layers.py',
    'chemvector/train_atom_bond_scene.py': 'chemvector/training/graph/scenes/train_atom_bond_scene.py',
    'chemvector/spatial_atom_object.py': 'chemvector/components/atoms/objects/spatial_atom_object.py',
    'chemvector/vertex.py': 'chemvector/components/vision/vertex.py',
    'chemvector/train_joint_proposal_objects_full.py': 'chemvector/training/graph/proposals/train_joint_proposal_objects_full.py',
}


def tensor_hash(state):
    result = sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def provenance(inputs):
    records, previous = [], None
    for label, relative, expected in CHECKPOINTS:
        path = ROOT / relative
        inputs.read(path, expected)
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        state = ({k.removeprefix('atom.'): v for k, v in payload['model'].items() if k.startswith('atom.')}
                 if label == 'initial' else payload['atom_head'])
        equal = previous is None or (set(state) == set(previous)
                                     and all(torch.equal(state[k], v) for k, v in previous.items()))
        if len(state) != 40 or not equal:
            raise ValueError('Recorded atom checkpoint chain differs')
        if label == 'initial':
            if (payload['progress']['optimizer_steps'] != 0
                    or payload['provenance']['checkpoints']['source']['sha256'] != CHECKPOINTS[2][2]
                    or tensor_hash(payload['model']) != prior.INITIAL_STATE):
                raise ValueError('Initial bundle source/state differs')
        if label == 'style_set' and payload.get('code_git_revision') != REVISION:
            raise ValueError('Recorded historical revision differs')
        args = payload.get('training_args', {})
        records.append(dict(stage=label, path=str(path), file_sha256=expected, atom_state_sha256=tensor_hash(state),
            atom_tensor_count=len(state), equal_to_previous=equal if previous is not None else None,
            samples_seen=payload.get('samples_seen'), recorded_code_revision=payload.get('code_git_revision'),
            joint_training_atom_loss_weight=args.get('atom_loss_weight'),
            joint_training_atom_learning_rate=args.get('atom_learning_rate')))
        previous = {k: v.clone() for k, v in state.items()}
        del payload
    code = []
    for old, current in HISTORICAL_FILES.items():
        historical = subprocess.check_output(['git', 'show', REVISION + ':' + old], cwd=ROOT)
        active = inputs.read(ROOT / current)
        if historical != active:
            raise ValueError('Historical/current target-path implementation differs: ' + current)
        code.append(dict(revision=REVISION, historical_path=old, current_path=current,
                         byte_equal=True, sha256=sha256(active).hexdigest()))
    return dict(checkpoints=records, historical_code=code,
        checkpoint_payloads_deserialized=4, model_modules_created=0, model_forwards=0,
        old_training_shards_opened=0,
        limitation='Recorded later revision is available; original joint checkpoint records no code revision')


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only audit required')
    if not OUTPUT.is_dir() or (OUTPUT / 'plan.json').exists():
        raise ValueError('Fresh budgeted output required')
    torch.set_num_threads(2)
    inputs = Inputs()
    _, manifest, scope = prior.preflight(inputs)
    dense = inputs.json(prior.OUTPUT / 'report.json', DENSE_SHA)
    for module in (Path(__file__), Path(contract.__file__), Path(prior.__file__),
                   Path(sys.modules[Inputs.__module__].__file__), Path(runtime.__file__),
                   ROOT / 'chemvector/training/curriculum/data.py',
                   ROOT / 'Data/sdf/text_source_renderer/v2/layers.py',
                   ROOT / 'Data/sdf/chemvector_curriculum_v0_1/rendering.py',
                   ROOT / 'frontend/inference_profiles.json'):
        inputs.read(module, dense['input_sha256'].get(str(module)))
    write(OUTPUT / 'plan.json', dict(schema_version='selector-anchor-contract-plan-0.1',
        scope=[dict(sample_key=r['sample_key'], target_ids=t) for r, t in scope],
        wall_budget_seconds=120, gpu_devices=[], model_forwards=0, optimizer_steps=0,
        checkpoint_reads='four explicit CPU weights_only payloads for parameter equality; no nn.Module or optimizer',
        derivative_scope='existing heatmap loss versus detached saved logits, all source targets included',
        old_training_shards_opened=0, target_comparison='same current annotations through both actual collators',
        training_authorized=False, threshold_or_labels_changed=False))
    lineage = provenance(inputs)
    counts, cases = Counter(), {}
    for row, focus in scope:
        key = row['sample_key']
        root = Path(manifest['sources'][row['source']]) / 'samples'
        meta = inputs.json(root / (key + '.meta.json'), row['meta_sha256'])
        records = {suffix: inputs.json(root / (key + suffix), meta['sha256'][suffix])
                   for suffix in ('.graph.json', '.layers.json', '.text-records.json')}
        geometry_path = prior.CONTROL / ('geometry--' + key + '.json')
        geometry = inputs.json(geometry_path, dense['input_sha256'][str(geometry_path)])
        case_path = prior.OUTPUT / (key + '.json')
        prior_case = inputs.json(case_path, dense['case_sha256'][case_path.name])
        if prior_case['target_ids'] != focus or not prior_case['parity']['historical_discrete_contract_reproduced']:
            raise ValueError('Fixed focus/reproduced scope differs')
        snapshot_path = prior.OUTPUT / (key + '.npz')
        inputs.read(snapshot_path, dense['snapshot_sha256'][snapshot_path.name])
        with np.load(snapshot_path, allow_pickle=False) as saved:
            snapshot = {k: torch.from_numpy(saved[k].copy()) for k in saved.files}
        paths = contract.compare_target_paths(records['.graph.json'], records['.layers.json'],
                                             records['.text-records.json'], meta)
        if (not paths['anchors_equal'] or not paths['heatmaps_equal']
                or not torch.equal(paths['current_anchors'][0], torch.tensor(geometry['target_anchors']))):
            raise ValueError('Target path does not reproduce saved target anchors/heatmap')
        derivative = contract.saved_heatmap_probe(snapshot, geometry, focus, records['.layers.json'],
                                                  paths['current_heatmap'])
        for label in ('current_anchors', 'current_heatmap', 'legacy_text_centers', 'current_text_centers'):
            paths.pop(label)
        counts.update(images=1, atoms=paths['atom_count'], visible_text_atoms=paths['visible_text_count'],
                      changed_text_center_targets=len(paths['source_ids_with_changed_text_center']))
        for node in derivative['targets']:
            counts['focused_targets'] += 1
            counts['gt_heatmap_positive'] += node['gt_heatmap_target'] == 1
            counts['gt_logit_direct_descent_up'] += node['gt_dloss_dlogit'] < 0
            counts['local_max_neighbors'] += len(node['local_max_neighbors'])
            counts['neighbor_is_other_positive'] += sum(bool(n['positive_source_atom_ids']) for n in node['local_max_neighbors'])
            counts['neighbor_direct_descent_down'] += sum(n['dloss_dlogit'] > 0 for n in node['local_max_neighbors'])
            counts['local_margin_direct_descent_improves'] += sum(
                n['independent_descent_gt_minus_neighbor_margin_direction'] > 0 for n in node['local_max_neighbors'])
        result = dict(sample_key=key, target_paths=paths, saved_logit_probe=derivative,
                      training_authorized=False, model_forwards=0, uses_ground_truth=True)
        path = OUTPUT / (key + '.json')
        write(path, result)
        cases[path.name] = sha256(path.read_bytes()).hexdigest()
    for path, digest in list(inputs.hashes.items()):
        inputs.read(path, digest)
    if torch.cuda.is_initialized():
        raise ValueError('Unexpected CUDA initialization')
    write(OUTPUT / 'report.json', dict(schema_version='selector-anchor-contract-0.1',
        input_sha256=inputs.hashes, case_sha256=cases, provenance=lineage, counts=counts,
        model_forwards=0, model_modules_created=0, saved_logits_autograd=True,
        parameter_gradients_computed=False, optimizer_steps=0, training_authorized=False,
        training_performed=False, gpu_used=False, labels_changed=False, default_changed=False,
        old_training_shards_opened=0, validation_sample_artifacts_read=False, new_identity_queries=0,
        limitations=['Bridge tests the two target adapters on current fixed7; it does not replay historical training examples',
            'Heatmap and text-center supervision are separate; their differing text-center contract is not heatmap anchor mixing',
            'Saved-logit gradient signs do not establish shared-parameter update direction or recovery',
            'No evidence of a target-definition swap is not proof of no renderer/style domain shift']))
    print(json.dumps(dict(counts=counts, atom_state_equal_across_four_checkpoints=True), ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    prior.preflight(Inputs())
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_selector_anchor_contract', '--worker'],
        output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
