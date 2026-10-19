"""Prepare a finite >=10k image exposure plan from an existing release, CPU only.

Never loads a model, opens validation media, changes admission or starts training.
Only released train metadata, chemical graphs and PNGs are opened. This is not
a repeat of target compatibility / private identity review / full payload audit.
"""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
import time

from PIL import Image

from chemvector.training.curriculum.admission import identities, require_training_admission
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.interface_diagnostic import write_json
from chemvector.training.curriculum.pair_scale_plan import (
    BOND_TYPES, coverage, deduplicate, digest, draft_protocol, exposure_plan, sample_id)
from chemvector.training.curriculum.runtime import ResourceBudget, run_budgeted


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'outputs/datasets/chemvector-expanded-release-20260915'
MANIFEST_SHA = '67ccb1307f7af3ca19f16ee6fd900d1a2e8c3c62dd0b9c0ec28704ddd93aa05e'
ADMISSION_SHA = 'da6d4f2639c5294768a9d186edc060d104998a261eecd45471c12ed7b9a2e076'
INITIAL = ROOT / 'outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt'
INITIAL_SHA = '09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'
OUTPUT = ROOT / 'outputs/audits/chemvector-pair-scale-preparation-20260924'
IMPLEMENTATIONS = (
    'chemvector/training/curriculum/pair_scale_plan.py',
    'chemvector/training/curriculum/pair_scale_policy.py',
    'tools/prepare_pair_scale_training.py',
    'tests/chemvector/test_pair_scale_plan.py',
)


def bound_bytes(path, expected):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError('Explicit non-aliased path required')
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError('Changed bound input: ' + str(path))
    return raw


def check_partition(rows):
    """Only already-released manifest identities, not any private identity index."""
    cache, assigned, seen = {}, {}, set()
    for row in rows:
        key = sample_id(row)
        if key in seen or row['split'] not in ('train', 'validation'):
            raise ValueError('Invalid release row/split')
        seen.add(key)
        smiles = row['canonical_smiles']
        if smiles not in cache:
            cache[smiles] = identities(smiles)
        identity = cache[smiles]
        if identity['group_key'] != row['group_key']:
            raise ValueError('Changed manifest chemical group')
        for kind, value in (('group', identity['group_key']), ('connection', identity['connectivity_prefix']),
                            ('molecule', row['molecule_id'])):
            previous = assigned.setdefault((kind, value), row['split'])
            if previous != row['split']:
                raise ValueError('Published train/development identity overlap')
    return dict(checked_manifest_rows=len(rows), distinct_manifest_structures=len(cache),
                grouping='existing_core_scaffold_and_connectivity_and_molecule_id',
                train_development_overlap=0, validation_media_read=False, private_identity_queries=0)


def read_train_record(manifest, row):
    # Deny before inspecting the source path: validation sources may be sealed
    # or tar-backed. Their manifest rows never reach any media loader here.
    if row['split'] != 'train' or row['domain'] not in ('curriculum', 'full_molecule'):
        raise ValueError('Training-only inventory')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', row['sample_key']):
        raise ValueError('Unsafe sample key')
    source = Path(manifest['sources'][row['source']])
    prefix = source / 'samples' / row['sample_key']
    meta_path = Path(str(prefix) + '.meta.json')
    meta = json.loads(bound_bytes(meta_path, row['meta_sha256']))
    if (meta['split'] != 'train' or meta['training_authorized'] is not True
            or meta['external_overlap_status'] != 'checked'
            or any(meta[k] != row[k] for k in ('sample_key', 'molecule_id', 'canonical_smiles'))):
        raise ValueError('Metadata is not this released train record')
    graph_path, png_path = Path(str(prefix) + '.graph.json'), Path(str(prefix) + '.png')
    graph = json.loads(bound_bytes(graph_path, meta['sha256']['.graph.json']))
    png = bound_bytes(png_path, meta['sha256']['.png'])
    if (graph['split'] != 'train' or graph['training_authorized'] is not True
            or graph['canonical_smiles'] != row['canonical_smiles'] or graph['molecule_id'] != row['molecule_id']):
        raise ValueError('Chemical graph does not match release')
    with Image.open(BytesIO(png)) as image:
        if image.size != (768, 768):
            raise ValueError('Original image size changed')
        rgb = image.convert('RGB')
        pixels = sha256(rgb.tobytes()).hexdigest()
    counts = dict(Counter(b['chemical_type'] for b in graph['bonds']))
    if set(counts) - set(BOND_TYPES):
        raise ValueError('New unsupported chemical bond supervision')
    heavy_atoms = sum(a['element'] != 'H' for a in graph['atoms'])
    if heavy_atoms != meta['heavy_atoms']:
        raise ValueError('Graph/metadata size mismatch')
    return dict(row, pixel_sha256=pixels, png_sha256=meta['sha256']['.png'],
                graph_sha256=meta['sha256']['.graph.json'], profile_id=meta['profile_id'],
                heavy_atoms=heavy_atoms, bond_counts=counts,
                elements=dict(Counter(a['element'] for a in graph['atoms'])),
                display_bonds=dict(Counter(b['direction'] for b in graph['bonds'])),
                stages=meta['stages'])


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only preparation required')
    started = time.monotonic()
    implementations = {p: file_digest(ROOT / p) for p in IMPLEMENTATIONS}
    admission = json.loads(bound_bytes(RELEASE / 'admission.json', ADMISSION_SHA))
    require_training_admission(admission)
    if admission['approved_manifest_sha256'] != MANIFEST_SHA:
        raise ValueError('Release admission binding changed')
    manifest = json.loads(bound_bytes(RELEASE / 'manifest.json', MANIFEST_SHA))
    if file_digest(INITIAL) != INITIAL_SHA:
        raise ValueError('Initial checkpoint changed')
    partition = check_partition(manifest['samples'])
    training = [r for r in manifest['samples'] if r['split'] == 'train']
    development = [r for r in manifest['samples'] if r['split'] == 'validation']
    if len(training) != 13151 or Counter(r['domain'] for r in development) != {'full_molecule': 800, 'curriculum': 144}:
        raise ValueError('Fixed released population changed')
    records = []
    for ordinal, row in enumerate(training, 1):
        if time.monotonic() - started > 550:
            raise TimeoutError('CPU preparation ceiling; no automatic retry')
        records.append(read_train_record(manifest, row))
        if ordinal % 1000 == 0:
            print(json.dumps(dict(event='train_inventory', checked=ordinal, total=len(training),
                                  elapsed_seconds=time.monotonic()-started)), flush=True)
    retained, aliases = deduplicate(records)
    if len(retained) != 12985 or len(aliases) != 166:
        raise ValueError('RGB unique count differs from the historical release; review before planning')
    schedule = exposure_plan(retained)
    protocol = draft_protocol(len(schedule))
    prefix_counts = {str(n): coverage(retained, schedule[:n]) for n in
                     sorted({29, 58, 800, *protocol['checkpoint_attempts']})}
    totals = coverage(retained, schedule)
    source_bindings = dict(manifest_sha256=MANIFEST_SHA, admission_sha256=ADMISSION_SHA,
                           initial_sha256=INITIAL_SHA)
    artifacts = {
        'training-inventory.json': dict(source_bindings=source_bindings, records=records),
        'unique-training-rows.json': dict(source_bindings=source_bindings, records=retained),
        'duplicate-aliases.json': aliases,
        'schedule.json': dict(rows_sha256=digest(retained), schedule=schedule, coverage=totals, prefixes=prefix_counts),
        'development-reference.json': dict(source_manifest_sha256=MANIFEST_SHA, records=development,
                                           unchanged=True, media_read=False),
        'protocol-draft.json': protocol,
    }
    for name, value in artifacts.items():
        write_json(OUTPUT / name, value)
    # Verify metadata still binds the inventory. PNG/graph bytes were checked
    # above; a future execution must recheck all loader payloads per access.
    for row in records:
        path = Path(manifest['sources'][row['source']]) / 'samples' / (row['sample_key'] + '.meta.json')
        bound_bytes(path, row['meta_sha256'])
    bound_bytes(RELEASE / 'manifest.json', MANIFEST_SHA)
    bound_bytes(RELEASE / 'admission.json', ADMISSION_SHA)
    for path, value in implementations.items():
        if file_digest(ROOT / path) != value:
            raise ValueError('Preparation implementation changed')
    write_json(OUTPUT / 'report.json', dict(schema_version='pair-scale-preparation-0.1',
        complete=True, status='prepared_not_runnable', source_bindings=source_bindings,
        implementation_sha256=implementations, artifact_sha256={name: file_digest(OUTPUT/name) for name in artifacts},
        inventory_rows=len(records), unique_images=len(retained), duplicate_rows=len(aliases),
        partition_check=partition, coverage=totals, prefixes=prefix_counts,
        training_authorized=False, deployment_authorized=False, training_performed=False,
        model_loads=0, model_forwards=0, optimizer_steps=0, gpu_used=False,
        validation_media_read=False, sealed_test_read=False, new_identity_queries=0,
        old_release_changed=False, elapsed_seconds=time.monotonic()-started,
        pending=['Implement bounded streaming worker and same-invocation throughput gate',
                 'Synthetic and integration tests for worker, cache and complete-pass accounting',
                 'Confirm new run protocol/budget before creating execution authorization'],
        limitations=['Only metadata/graph/PNG rechecked; not a fresh full payload/target audit',
                     'No development media or model throughput measured',
                     'Ground-truth bond counts are not predicted-candidate reachable supervision',
                     'One fixed pass tests training feasibility, not convergence or independent test accuracy',
                     'Reuses the existing development split and its documented external-evaluation limitations']))
    print(json.dumps(dict(status='prepared_not_runnable', coverage=totals, proposed_budget=protocol['proposed_budget'])), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    receipt = run_budgeted([sys.executable, '-m', 'tools.prepare_pair_scale_training', '--worker'],
                          output=OUTPUT, budget=ResourceBudget(600))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
