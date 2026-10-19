"""Independent read-only audit of the new isolated case bundle, no inference."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
import sys

from PIL import Image
from rdkit import Chem

from Data.sdf.case_diversity_v1.build import accounting
from Data.sdf.case_diversity_v1.coverage import partition, visible_features
from Data.sdf.case_diversity_v1.profiles import profiles, FACTORS
from Data.sdf.chemvector_curriculum_v0_2.coverage import visible_coverage
from chemvector.core.ocr import parse_atom_display_text
from chemvector.training.curriculum.data import validate_records, _point
from chemvector.training.curriculum.runtime import run_budgeted, ResourceBudget
from chemvector.training.curriculum.interface_diagnostic import write_json


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def worker(dataset, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only audit required')
    root, output = Path(dataset), Path(output)
    summary = json.loads((root / 'summary.json').read_bytes())
    if not summary['complete'] or summary['training_authorized']: raise ValueError('Completed quarantine batch required')
    plan = json.loads((root / 'plan.json').read_bytes()); hashes = json.loads((root / 'sample-sha256.json').read_bytes())
    for name, expected in plan['input_sha256'].items():
        if digest(name) != expected: raise ValueError('Source/font binding changed: ' + name)
    for i, (name, expected) in enumerate(hashes.items(), 1):
        if digest(root / name) != expected: raise ValueError('Generated file changed: ' + name)
        if i % 30000 == 0: print(json.dumps(dict(verified_sample_files=i)), flush=True)
    entries = json.loads((root / 'molecules.json').read_bytes()); by_id = {e['molecule_id']: e for e in entries}
    rows = json.loads((root / 'samples.json').read_bytes()); rejected = json.loads((root / 'rejections.json').read_bytes())
    ps = profiles(); ps_by_id = {p.profile_id: p for p in ps}; accounting(entries, ps, rows, rejected)
    quality = json.loads((root / 'quality-partition.json').read_bytes())
    retained = set(quality['retained']); reconstructed = []
    label_counts = {g: Counter() for g in ('all_accepted','retained')}
    layout_counts = {g: Counter() for g in label_counts}
    unsupported = {g: Counter() for g in label_counts}
    mismatch_states = {g: Counter() for g in label_counts}
    target_collisions, identity_checks = [], 0
    for i, row in enumerate(rows, 1):
        key = row['sample_key']; base = root / 'samples' / key
        meta = json.loads(Path(str(base) + '.meta.json').read_bytes())
        if (meta['split'] != 'quarantine' or meta['training_authorized'] or meta['external_overlap_status'] != 'unchecked'
                or meta['canonical_smiles'] != by_id[row['molecule_id']]['canonical_smiles']):
            raise ValueError('Chemical identity or lifecycle differs')
        for suffix, expected in meta['sha256'].items():
            if hashes['samples/' + key + suffix] != expected: raise ValueError('Inner/outer payload hashes disagree')
        graph, layers, records = (json.loads(Path(str(base) + s).read_bytes()) for s in ('.graph.json','.layers.json','.text-records.json'))
        validate_records(graph, layers, records)
        if layers['text_records_sha256'] != meta['sha256']['.text-records.json'] or records['source_png_sha256'] != meta['sha256']['.png']:
            raise ValueError('Same-source record binding differs')
        profile = ps_by_id[row['profile_id']]
        if layers['renderer']['font']['family'] != profile.font_family:
            raise ValueError('Actual font family differs from planned profile')
        if any(profile.font_family not in g['svg_attributes']['style'] for g in records['glyphs']):
            raise ValueError('Visible glyph font differs')
        if visible_features(layers, records) != row['visible'] or visible_coverage(layers) != row['geometry']:
            raise ValueError('Visible coverage counters differ')
        mol = Chem.MolFromMolBlock(Path(str(base) + '.sdf').read_text().split('$$$$')[0], removeHs=False, strictParsing=True)
        if mol is None or Chem.MolToSmiles(mol) != meta['canonical_smiles'] or mol.GetNumAtoms() != len(graph['atoms']):
            raise ValueError('SDF identity/index roundtrip differs')
        if mol.GetNumBonds() != len(graph['bonds']): raise ValueError('Bond count differs')
        for atom, truth in zip(mol.GetAtoms(), graph['atoms'], strict=True):
            if (atom.GetSymbol(), atom.GetFormalCharge(), atom.GetIsotope()) != (truth['element'], truth['formal_charge'], truth['isotope']):
                raise ValueError('Atom index identity differs')
        for bond, truth in zip(mol.GetBonds(), graph['bonds'], strict=True):
            if ({bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}, str(bond.GetBondType())) != ({truth['begin_atom'], truth['end_atom']}, truth['chemical_type']):
                raise ValueError('Bond identity differs')
        identity_checks += 1
        with Image.open(str(base) + '.png') as image:
            if image.mode != 'RGB' or image.size != (768,768): raise ValueError('Input raster changed')
            rgb = sha256(image.tobytes()).hexdigest()
        if rgb != row['rgb_sha256'] or row['png_sha256'] != meta['sha256']['.png']: raise ValueError('Pixel identity differs')
        reconstructed.append(dict(sample_key=key, molecule_id=row['molecule_id'], rgb_sha256=rgb))
        cells = Counter(tuple(min(191, int(v / 1023 * 192)) for v in _point(a['center'])) for a in layers['atoms'])
        if any(n > 1 for n in cells.values()): target_collisions.append(key)
        for group in ('all_accepted', 'retained') if key in retained else ('all_accepted',):
            label_counts[group].update(row['visible']['labels']); layout_counts[group].update(row['visible']['layouts'])
            for a in layers['atoms']:
                if not a['visible_text']: continue
                parsed = parse_atom_display_text(a['text'])
                if parsed is None: unsupported[group][a['text']] += 1
                elif (parsed['element'], parsed['formal_charge'], parsed['isotope']) != (a['element'], a['formal_charge'], a['isotope']):
                    mismatch_states[group][a['text']] += 1
        if i % 1000 == 0: print(json.dumps(dict(verified_full_samples=i)), flush=True)
    if partition(reconstructed) != quality: raise ValueError('Internal quality partition differs')
    if len(rows) + len(rejected) != summary['planned_images'] or len(hashes) != summary['verified_sample_file_hashes']:
        raise ValueError('Complete denominator differs')
    counts = Counter(s['molecule_id'] for s in rows if s['sample_key'] in retained)
    pair_support = {}
    retained_profiles = {s['profile_id'] for s in rows if s['sample_key'] in retained}
    for a, b in combinations(FACTORS, 2):
        pair_support[a + ' x ' + b] = len({(getattr(ps_by_id[p], a), getattr(ps_by_id[p], b)) for p in retained_profiles})
    write_json(output / 'report.json', dict(schema_version='case-diversity-audit-1', integrity_passed=True,
        source_plan_sha256=digest(root / 'plan.json'), summary_sha256=digest(root / 'summary.json'),
        sample_index_sha256=digest(root / 'samples.json'), sample_hash_manifest_sha256=digest(root / 'sample-sha256.json'),
        checker_sha256=digest(Path(__file__)), verified_files=len(hashes), verified_SDF_and_label_contracts=identity_checks,
        accepted_images=len(rows), rejected_images=len(rejected), retained_different_images=len(retained),
        retained_structures=len(counts), retained_structures_with_all16=sum(n == 16 for n in counts.values()),
        retained_global_style_pair_coverage=pair_support, labels={k:dict(v) for k,v in label_counts.items()},
        visible_layouts={k:dict(v) for k,v in layout_counts.items()},
        current_display_parser_unsupported={k:dict(v) for k,v in unsupported.items()},
        current_display_parser_core_mismatches={k:dict(v) for k,v in mismatch_states.items()},
        target_grid_collision_samples=target_collisions, predicted_candidate_reachability='not_measured',
        parser_check_is_not_OCR_accuracy=True, training_authorized=False, external_identity_review=False,
        model_loads=0, model_forwards=0, gpu_used=False, independent_accuracy_measured=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true');args=parser.parse_args(argv)
    if args.worker: worker(args.dataset,args.output);return 0
    command=[sys.executable,'-m','tools.audit_case_diversity_v1','--worker','--dataset',str(args.dataset),'--output',str(args.output)]
    receipt=run_budgeted(command,output=args.output,budget=ResourceBudget(300))
    print(json.dumps(receipt,indent=2));return 0 if receipt['status']=='completed' else 1


if __name__ == '__main__': raise SystemExit(main())
