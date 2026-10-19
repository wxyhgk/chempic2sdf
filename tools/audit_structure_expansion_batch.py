"""Independent artifact audit and one-image-per-structure CPU target preflight.

No checkpoints, training, model forwards, sealed identities or default changes.
Public old candidate batches are compared by their hash-bound index only.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

from PIL import Image
from rdkit import Chem
import torch

from Data.sdf.case_diversity_v1.build import accounting
from Data.sdf.case_diversity_v1.catalog import inventory
from Data.sdf.case_diversity_v1.coverage import partition, visible_features
from Data.sdf.case_diversity_v1.profiles import pair_coverage
from Data.sdf.chemvector_curriculum_v0_2.coverage import visible_coverage
from Data.sdf.structure_expansion_v1.batch import (
    digest, load_fixed_candidates, selected_profiles, measured_coverage, verify_hashes,
)
from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.data import load_sample, CurriculumCollator, _point
from chemvector.training.curriculum.interface_diagnostic import write_json
from chemvector.training.curriculum.runtime import ResourceBudget, run_budgeted
from tools.check_display_label_candidate import compare_labels
from tools.preflight_curriculum_v02_targets import validate_targets, source_paths


SUFFIXES = ('.png','.svg','.text.png','.bond.png','.auxiliary.png','.semantic.png',
            '.graph.json','.layers.json','.text-records.json','.sdf','.meta.json')


def prior_indices():
    """Two explicit public candidate batches, not all historical training data."""
    case = Path('outputs/datasets/case-diversity-v1-12288-20260922')
    ca = Path('outputs/audits/case-diversity-v1-12288-20260922/report.json')
    foundation = Path('outputs/datasets/curriculum-v0.2-expanded-20260919')
    fa = Path('outputs/audits/curriculum-v02-expanded-independent-20260919.json')
    sources = {str(ca):'598ab01eac2ecadee70d2a0dc64daaeea2ddc49027b7c4db4e1a1d89b25ff823',
               str(fa):'b327fdba991436901c3c456ee12ccd2fcb1b78319c38747673ffdcd039cf65c2'}
    verify_hashes(sources)
    case_audit = json.loads(ca.read_bytes()); old_audit = json.loads(fa.read_bytes())
    sources[str(case/'samples.json')] = case_audit['sample_index_sha256']
    for name in ('samples.jsonl','sample-sha256.json'):
        sources[str(foundation/name)] = old_audit['manifest_sha256'][name]
    verify_hashes(sources)
    cases = json.loads((case/'samples.json').read_bytes())
    rows = [json.loads(line) for line in (foundation/'samples.jsonl').read_bytes().splitlines()]
    hashes = json.loads((foundation/'sample-sha256.json').read_bytes())
    rgb, png = defaultdict(set), defaultdict(set)
    for row in cases:
        rgb[row['rgb_sha256']].add(row['molecule_id'])
    for row in rows:
        png[hashes['samples/'+row['sample_key']+'.png']].add(row['molecule_id'])
    return {'case_diversity_v1_RGB':('rgb_sha256',rgb), 'foundation_v02_PNG_bytes':('png_sha256',png)}, sources


def compare_prior(rows, indices):
    result = {}
    for name, (field,index) in indices.items():
        matches, conflict = [], []
        for row in rows:
            old_ids = index.get(row[field],set())
            if old_ids:
                matches.append(row['sample_key'])
                if old_ids != {row['molecule_id']}:
                    conflict.append(dict(sample_key=row['sample_key'],molecule_id=row['molecule_id'],
                                         prior_molecule_ids=sorted(old_ids)))
        result[name] = dict(matching_images=len(matches), matching_sample_keys=matches,
                            different_identity_pixel_matches=conflict, comparison_field=field)
    return result


def audit(dataset, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only audit required')
    torch.set_num_threads(1)
    root, output = Path(dataset), Path(output)
    summary = json.loads((root/'summary.json').read_bytes())
    plan = json.loads((root/'plan.json').read_bytes())
    if (not summary['complete'] or summary['training_authorized']
            or summary['external_overlap_status'] != 'unchecked'):
        raise ValueError('Completed quarantine batch required')
    entries, _ = load_fixed_candidates(smoke=plan['smoke'])
    if json.loads((root/'molecules.json').read_bytes()) != entries:
        raise ValueError('Batch membership differs from fixed candidate manifest')
    ps = selected_profiles(); by_profile = {p.profile_id:p for p in ps}
    profile_records = [asdict(p) for p in ps]
    if plan['profiles'] != profile_records or json.loads((root/'profiles.json').read_bytes()) != profile_records:
        raise ValueError('Fixed style design changed')
    paths = set(source_paths()) | {Path(__file__), Path('tools/check_display_label_candidate.py'),
        Path('chemvector/components/ocr/display_label_candidate.py'), Path('chemvector/core/ocr.py'),
        Path('tests/data_sdf/test_structure_batch_audit.py')}
    sources = {**plan['input_sha256'], **{str(p):digest(p) for p in paths}}
    manifests = {name:digest(root/name) for name in ('plan.json','summary.json','molecules.json','profiles.json',
        'samples.json','rejections.json','sample-sha256.json','quality-partition.json','progress.jsonl')}
    verify_hashes(sources)
    hashes = json.loads((root/'sample-sha256.json').read_bytes())
    rows = json.loads((root/'samples.json').read_bytes()); rejected = json.loads((root/'rejections.json').read_bytes())
    expected_names = {'samples/'+r['sample_key']+suffix for r in rows for suffix in SUFFIXES}
    if set(hashes) != expected_names:
        raise ValueError('Payload hash manifest omits or adds sample files')
    for i,(name,value) in enumerate(hashes.items(),1):
        if digest(root/name) != value:
            raise ValueError('Generated file changed: ' + name)
        if i % 30000 == 0:
            print(json.dumps(dict(verified_files=i)),flush=True)
    accounting(entries,ps,rows,rejected)
    journal = [json.loads(line) for line in (root/'progress.jsonl').read_bytes().splitlines()]
    accepted_journal, rejected_journal = [], []
    for r in journal:
        status = r.pop('status')
        if status == 'accepted': accepted_journal.append(r)
        elif status == 'quality_rejected': rejected_journal.append(r)
        else: raise ValueError('Unknown journal status')
    if accepted_journal != rows or rejected_journal != rejected:
        raise ValueError('Journal and final accounting disagree')
    quality = json.loads((root/'quality-partition.json').read_bytes())
    retained = set(quality['retained']); by_id = {e['molecule_id']:e for e in entries}
    panel = {}
    for row in sorted(rows,key=lambda r:r['sample_key']):
        if row['sample_key'] in retained:
            panel.setdefault(row['molecule_id'],row['sample_key'])
    panel_keys = set(panel.values())
    config = AtomBondSceneConfig(); collate = CurriculumCollator(config,purpose='diagnostic')
    targets, target_errors, reconstructed, collisions = [], [], [], []
    parser = {k:Counter() for k in ('counts','newly_supported','abstained','field_mismatches','changed_previously_parsed')}
    maximum = Counter()
    for i,row in enumerate(rows,1):
        key = row['sample_key']; sample = load_sample(root,key,purpose='diagnostic')
        meta, graph, layers, records = (sample[k] for k in ('meta','graph','layers','text_records'))
        e = by_id[row['molecule_id']]; profile = by_profile[row['profile_id']]
        if (any(meta.get(k) != e[k] for k in e) or meta['profile'] != asdict(profile)
                or any(meta[k] != row[k] for k in ('sample_key','molecule_id','profile_id'))):
            raise ValueError('Source entry/profile differs from sample metadata')
        for obj in (meta,graph,layers):
            if obj['split'] != 'quarantine' or obj['training_authorized'] or obj['external_overlap_status'] != 'unchecked':
                raise ValueError('Sample lifecycle changed')
        if set(meta['sha256']) != set(SUFFIXES)-{'.meta.json'}:
            raise ValueError('Inner payload set differs')
        if any(hashes['samples/'+key+suffix] != value for suffix,value in meta['sha256'].items()):
            raise ValueError('Inner and outer hashes disagree')
        if (layers['renderer']['config'] != asdict(profile) or layers['renderer']['font']['family'] != profile.font_family
                or any(profile.font_family not in g['svg_attributes']['style'] for g in records['glyphs'])):
            raise ValueError('Rendered style differs from fixed design')
        if visible_features(layers,records) != row['visible'] or visible_coverage(layers) != row['geometry']:
            raise ValueError('Measured visible coverage changed')
        mol = Chem.MolFromMolBlock((root/'samples'/(key+'.sdf')).read_text().split('$$$$')[0],removeHs=False,strictParsing=True)
        if (mol is None or Chem.MolToSmiles(mol) != e['canonical_smiles']
                or mol.GetNumAtoms() != len(graph['atoms']) or mol.GetNumBonds() != len(graph['bonds'])):
            raise ValueError('SDF identity/stereo/count roundtrip differs')
        for a,g in zip(mol.GetAtoms(),graph['atoms'],strict=True):
            if (a.GetIdx(),a.GetSymbol(),a.GetFormalCharge(),a.GetIsotope()) != (g['atom_id'],g['element'],g['formal_charge'],g['isotope']):
                raise ValueError('SDF atom index identity differs')
        for b,g in zip(mol.GetBonds(),graph['bonds'],strict=True):
            if (b.GetIdx(),{b.GetBeginAtomIdx(),b.GetEndAtomIdx()},str(b.GetBondType())) != (g['bond_id'],{g['begin_atom'],g['end_atom']},g['chemical_type']):
                raise ValueError('SDF bond index identity differs')
        rgb = sha256(sample['image'].tobytes()).hexdigest()
        if rgb != row['rgb_sha256'] or row['png_sha256'] != meta['sha256']['.png']:
            raise ValueError('Original PNG/RGB differs')
        reconstructed.append(dict(sample_key=key,molecule_id=row['molecule_id'],rgb_sha256=rgb))
        cells = Counter(tuple(min(191,int(v/1023*192)) for v in _point(a['center'])) for a in layers['atoms'])
        if any(n > 1 for n in cells.values()): collisions.append(key)
        maximum['atoms'] = max(maximum['atoms'],len(graph['atoms']))
        maximum['bonds'] = max(maximum['bonds'],len(graph['bonds']))
        if key in retained:
            for field, values in compare_labels(layers['atoms']).items(): parser[field].update(values)
        if key in panel_keys:
            try:
                batch = collate([sample])
                target = validate_targets(sample,batch,config)
            except ValueError as error:
                target_errors.append(dict(sample_key=key,reason=str(error)))
            else:
                targets.append(dict(sample_key=key,**target))
            try:
                load_sample(root,key,purpose='training')
            except ValueError as error:
                if 'not admitted for training' not in str(error): raise
            else: raise ValueError('Training loader accepted quarantine')
            try:
                CurriculumCollator(config,purpose='training')([{**sample,'purpose':'training'}])
            except ValueError as error:
                if 'Training admission is required' not in str(error): raise
            else: raise ValueError('Training collator accepted quarantine')
        if i % 1000 == 0:
            print(json.dumps(dict(verified_samples=i,target_panel_passed=len(targets))),flush=True)
    if partition(reconstructed) != quality: raise ValueError('Internal quality partition differs')
    kept = [r for r in rows if r['sample_key'] in retained]; kept_ids = {r['molecule_id'] for r in kept}
    counts = Counter(r['molecule_id'] for r in rows)
    measured = dict(planned_images=len(entries)*len(ps),unique_molecules=len(entries),accepted_images=len(rows),
        rejected_images=len(rejected),rejection_reasons=dict(Counter(r['reason'] for r in rejected)),
        retained_different_images=len(kept),retained_different_molecules=len(kept_ids),
        conflict_images=len(quality['conflict_identity_quarantine']),duplicate_images=len(quality['same_identity_duplicates']),
        distinct_png=len({r['png_sha256'] for r in rows}),distinct_decoded_rgb=len({r['rgb_sha256'] for r in rows}),
        molecules_with_all8_accepted=sum(counts[e['molecule_id']]==8 for e in entries),
        molecules_with_zero_accepted=sum(counts[e['molecule_id']]==0 for e in entries),
        selected_inventory=inventory(entries),retained_inventory=inventory([e for e in entries if e['molecule_id'] in kept_ids]),
        actual_coverage=dict(accepted=measured_coverage(rows),retained_proposal=measured_coverage(kept)),
        verified_sample_file_hashes=len(hashes))
    if any(summary[k] != v for k,v in measured.items()): raise ValueError('Summary does not match recomputed measurements')
    prior, prior_sources = prior_indices(); comparison = compare_prior(rows,prior)
    verify_hashes({**sources,**prior_sources,**{str(root/name):value for name,value in manifests.items()}})
    write_json(output/'target-panel.json',dict(selection='lexicographically_first_retained_image_per_structure',
        selected=panel,passed=targets,failed=target_errors,all_images_collated=False,training_authorized=False))
    write_json(output/'report.json',dict(schema_version='structure-expansion-batch-audit-1',integrity_passed=True,
        dataset=str(root),manifest_sha256=manifests,input_sha256=sources,prior_index_sha256=prior_sources,
        verified_files=len(hashes),verified_SDF_and_label_contracts=len(rows),
        accepted_images=len(rows),rejected_images=len(rejected),retained_images=len(kept),retained_structures=len(kept_ids),
        target_grid_collision_samples=collisions,target_query_maximum=dict(maximum),
        target_panel_images=len(panel),target_panel_passed=len(targets),target_panel_failures=target_errors,
        target_config=asdict(config),all_images_collated=False,
        training_loader_and_collator_rejected_quarantine=len(panel),
        retained_factor_pair_coverage=pair_coverage([p for p in ps if any(r['profile_id']==p.profile_id for r in kept)]),
        correct_string_parser_check={k:dict(v) for k,v in parser.items()},cross_batch_comparison=comparison,
        cross_batch_scope='two_explicit_public_candidate_batches_only; foundation_PNG_not_redecoded_RGB',
        blocking_cross_batch_conflicts=any(v['different_identity_pixel_matches'] for v in comparison.values()),
        external_identity_review=False,model_loads=0,model_forwards=0,gpu_used=False,training_authorized=False,
        predicted_candidate_reachability='not_measured',default_parser_changed=False,
        limits=['artifact_and_target_compatibility_not_accuracy', 'one_collated_image_per_retained_structure_not_all_styles',
                'candidate_parser_truth_strings_not_noisy_OCR_safety','no_full_historical_or_evaluation_dedup']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true'); args=parser.parse_args(argv)
    if args.worker:
        audit(args.dataset,args.output); return 0
    command=[sys.executable,'-m','tools.audit_structure_expansion_batch','--worker',
             '--dataset',str(args.dataset),'--output',str(args.output)]
    receipt=run_budgeted(command,output=args.output,budget=ResourceBudget(600))
    print(json.dumps(receipt,indent=2)); return 0 if receipt['status']=='completed' else 1


if __name__ == '__main__': raise SystemExit(main())
