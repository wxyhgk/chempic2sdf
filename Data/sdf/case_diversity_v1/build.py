"""Bounded, CPU-only case generation; never grants data or training admission."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageOps

from Data.sdf.chemvector_curriculum_v0_1.bounded import bounded_map
from Data.sdf.chemvector_curriculum_v0_1.build import QUALITY_REJECTIONS
from Data.sdf.chemvector_curriculum_v0_1.gallery import write_gallery
from Data.sdf.chemvector_curriculum_v0_2.coverage import summarize, visible_coverage
from chemvector.training.curriculum.data import validate_records
from chemvector.training.curriculum.runtime import ResourceBudget, run_budgeted
from chemvector.training.curriculum.interface_diagnostic import write_json
from .catalog import pool, select, inventory, MANIFEST, MANIFEST_SHA
from .profiles import profiles, font_record, FAMILIES, FACTORS, pair_coverage
from .rendering import render
from .coverage import visible_features, partition


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def bindings():
    roots = [Path(__file__).parent, Path('Data/sdf/chemvector_curriculum_v0_1'),
             Path('Data/sdf/chemvector_curriculum_v0_2'), Path('Data/sdf/text_source_renderer')]
    paths = {p for root in roots for p in root.rglob('*.py')}
    paths.update(Path('chemvector/training/curriculum') / name for name in
                 ('data.py', 'runtime.py', 'admission.py', 'interface_diagnostic.py'))
    paths.add(Path('tests/data_sdf/test_case_diversity_v1.py'))
    result = {str(p): digest(p) for p in sorted(paths)}
    result[str(MANIFEST)] = MANIFEST_SHA
    for family in FAMILIES:
        font = font_record(family); result[font['path']] = font['sha256']
    return result


def molecule_job(job):
    entry, selected = job; results = []
    for profile in selected:
        try:
            files, meta = render(entry, profile)
            graph, layers, records = (json.loads(files[s]) for s in ('.graph.json', '.layers.json', '.text-records.json'))
            validate_records(graph, layers, records)
            with Image.open(io.BytesIO(files['.png'])) as image:
                rgb_hash = sha256(image.convert('RGB').tobytes()).hexdigest()
            results.append(dict(files=files, meta=meta, failure=None, rgb_sha256=rgb_hash,
                                visible=visible_features(layers, records), geometry=visible_coverage(layers)))
        except ValueError as error:
            if str(error) not in QUALITY_REJECTIONS: raise
            results.append(dict(files=None, failure=dict(molecule_id=entry['molecule_id'],
                canonical_smiles=entry['canonical_smiles'], profile_id=profile.profile_id, reason=str(error))))
    return results


def accounting(entries, ps, samples, rejected):
    expected = {(e['molecule_id'], p.profile_id) for e in entries for p in ps}
    actual = [(r['molecule_id'], r['profile_id']) for r in [*samples, *rejected]]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Every planned case, including rejections, must remain accounted for')


def contact_sheet(output, entries, samples, ps):
    indexed = {s['sample_key']: s for s in samples}
    candidates = [e for e in entries if any(r.startswith('diversity_seed_00') for r in e['source_record_ids'])]
    if not candidates: return None
    e = candidates[0]
    canvas = Image.new('RGB', (1040, 1080), 'white'); draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), 'QUARANTINE / style matrix / foreground thumbnails only / ' + e['canonical_smiles'], fill='black')
    for i, p in enumerate(ps):
        x, y = i % 4 * 260, 40 + i // 4 * 260
        draw.text((x + 8, y), f'{p.profile_id} {p.font_family}', fill='black')
        draw.text((x + 8, y + 18), f'font={p.font_size} angle={p.rotation_degrees:g} bond={p.bond_length_cap:g} stroke={p.bond_line_width:g}', fill='black')
        key = e['molecule_id'] + '__' + p.profile_id
        if key not in indexed:
            draw.text((x + 30, y + 100), 'QUALITY REJECTED', fill='red'); continue
        sample = indexed[key]
        with Image.open(output / 'samples' / (key + '.png')) as image:
            b = sample['measured']['foreground_bbox']
            crop = image.crop((max(0, b[0]-8), max(0, b[1]-8), min(768, b[2]+8), min(768, b[3]+8)))
            thumb = ImageOps.contain(crop, (240, 205)); canvas.paste(thumb, (x+(260-thumb.width)//2, y+43))
    name = 'style-matrix.png'; canvas.save(output / name); return name


def run(output, *, smoke=False, workers=12):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError('Explicit bounded CPU-only renderer required')
    output = Path(output)
    if output.exists(): raise FileExistsError('Fresh dataset directory required')
    entries, provenance = pool()
    if smoke:
        seeds = [e for e in entries if any(r.startswith('diversity_seed_') for r in e['source_record_ids'])]
        ids = {e['molecule_id'] for e in seeds}
        molecules = seeds + [e for e in entries if e['source_kind'] == 'existing_train_identity' and e['molecule_id'] not in ids][:4]
    else:
        molecules = select(entries, 768)
    ps = profiles(); sources = bindings()
    output.mkdir(parents=True, exist_ok=False); (output / 'samples').mkdir()
    write_json(output / 'plan.json', dict(schema_version='case-diversity-plan-1', smoke=smoke,
        planned_molecules=len(molecules), planned_images=len(molecules)*16, pool_inventory=inventory(entries),
        selected_inventory=inventory(molecules), source_provenance=provenance,
        profiles=[asdict(p) for p in ps], planned_factor_pair_coverage=pair_coverage(ps),
        selection='all_12_label_seeds_and_fixed_train32_then_diminishing_feature_support; deterministic_ID_ties',
        workers=workers, per_molecule_seconds=180, journal_commit_unit='all16_styles_of_one_structure',
        automatic_retries=0, training_authorized=False, external_overlap_status='unchecked',
        split='quarantine', gpu_used=False, model_forwards=0, input_sha256=sources))
    write_json(output / 'molecules.json', molecules); write_json(output / 'profiles.json', [asdict(p) for p in ps])
    write_json(output / 'planned-chemistry.json', summarize(molecules))
    samples, rejected, hashes = [], [], {}
    stream = bounded_map(molecule_job, [(e, ps) for e in molecules], workers=workers, timeout_seconds=180)
    try:
        with (output / 'progress.jsonl').open('x') as journal:
            for i, batch in enumerate(stream, 1):
                for result in batch:
                    if result['failure'] is not None:
                        rejected.append(result['failure']); record = dict(status='quality_rejected', **result['failure'])
                    else:
                        meta, files = result['meta'], result['files']
                        for suffix, data in files.items():
                            name = 'samples/' + meta['sample_key'] + suffix
                            with (output / name).open('xb') as handle: handle.write(data)
                            hashes[name] = sha256(data).hexdigest()
                        sample = {k: meta[k] for k in ('sample_key', 'molecule_id', 'profile_id', 'stages', 'measured')}
                        sample.update(rgb_sha256=result['rgb_sha256'], png_sha256=meta['sha256']['.png'],
                                      visible=result['visible'], geometry=result['geometry'])
                        samples.append(sample); record = dict(status='accepted', **sample)
                    journal.write(json.dumps(record, sort_keys=True) + '\n'); journal.flush()
                if i % 16 == 0 or i == len(molecules):
                    print(json.dumps(dict(completed_molecules=i, planned_molecules=len(molecules),
                        accepted=len(samples), rejected=len(rejected))), flush=True)
    finally:
        stream.close()
    accounting(molecules, ps, samples, rejected)
    for name, expected in hashes.items():
        if digest(output / name) != expected: raise ValueError('Generated artifact hash mismatch')
    for name, expected in sources.items():
        if digest(name) != expected: raise ValueError('Generator/font/source changed during generation')
    quality = partition(samples); retained_keys = set(quality['retained'])
    retained = [s for s in samples if s['sample_key'] in retained_keys]
    retained_ids = {s['molecule_id'] for s in retained}
    actual = {}
    for group, rows in [('accepted', samples), ('retained_proposal', retained)]:
        visible = {field: dict(sum((Counter(s['visible'][field]) for s in rows), Counter()))
                   for field in ('labels', 'layouts', 'features')}
        by_profile = {p.profile_id: dict(Counter(s['profile_id'] for s in rows)).get(p.profile_id, 0) for p in ps}
        actual[group] = dict(visible=visible, images_by_profile=by_profile,
            bond_types=dict(sum((Counter(s['geometry']['chemical_bond_types']) for s in rows), Counter())),
            drawing_directions=dict(sum((Counter(s['geometry']['drawing_directions']) for s in rows), Counter())))
    counts = Counter(s['molecule_id'] for s in samples)
    summary = dict(schema_version='case-diversity-summary-1', complete=True, status='quarantine_cases_only',
        planned_images=len(molecules)*16, unique_molecules=len(molecules), accepted_images=len(samples),
        rejected_images=len(rejected), rejection_reasons=dict(Counter(r['reason'] for r in rejected)),
        molecules_with_all16_accepted=sum(counts[e['molecule_id']] == 16 for e in molecules),
        molecules_with_zero_accepted=sum(counts[e['molecule_id']] == 0 for e in molecules),
        distinct_png=len({s['png_sha256'] for s in samples}), distinct_decoded_rgb=len({s['rgb_sha256'] for s in samples}),
        retained_different_images=len(retained), retained_different_molecules=len(retained_ids),
        conflict_images=len(quality['conflict_identity_quarantine']), duplicate_images=len(quality['same_identity_duplicates']),
        selected_inventory=inventory(molecules), retained_inventory=inventory([e for e in molecules if e['molecule_id'] in retained_ids]),
        actual_coverage=actual, verified_sample_file_hashes=len(hashes),
        training_authorized=False, external_overlap_status='unchecked', gpu_used=False, model_forwards=0,
        evaluation_sample_files_read=False, new_identity_queries=0, independent_accuracy_measured=False,
        limitations=['new_style_cases_mostly_existing_template_identities_not_new_independent_molecules',
            'one_RDKit_layout_engine_768px_clean_rasters_no_real_scans',
            'factor_pair_coverage_is_design_not_guaranteed_after_rejections',
            'chemistry_and_visible_coverage_not_model_candidate_reachability',
            'internal_quality_partition_does_not_authorize_training_or_external_identity_clearance'])
    write_json(output / 'samples.json', samples); write_json(output / 'rejections.json', rejected)
    write_json(output / 'sample-sha256.json', hashes); write_json(output / 'quality-partition.json', quality)
    write_json(output / 'accepted-chemistry.json', summarize(molecules, samples))
    contact_sheet(output, molecules, samples, ps)
    write_gallery(output, molecules, samples, summary)
    write_json(output / 'summary.json', summary)  # Completion marker LAST.
    print(json.dumps({k: summary[k] for k in ('planned_images','accepted_images','rejected_images','retained_different_images')}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=12); parser.add_argument('--wall-seconds', type=int, default=1200)
    parser.add_argument('--worker', action='store_true'); args = parser.parse_args(argv)
    if args.worker:
        run(args.output, smoke=args.smoke, workers=args.workers); return 0
    if args.output.exists(): raise FileExistsError('Fresh dataset directory required')
    command = [sys.executable, '-m', 'Data.sdf.case_diversity_v1.build', '--worker', '--output', str(args.output), '--workers', str(args.workers)]
    if args.smoke: command.append('--smoke')
    receipt = run_budgeted(command, output=Path(str(args.output) + '-runtime'), budget=ResourceBudget(args.wall_seconds))
    print(json.dumps(receipt, indent=2)); return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__': raise SystemExit(main())
