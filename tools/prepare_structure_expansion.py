"""Bounded CPU preparation of new structure candidates and optional preview."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from Data.sdf.case_diversity_v1.build import bindings
from Data.sdf.case_diversity_v1.catalog import pool
from Data.sdf.structure_expansion_v1.catalog import new_candidates
from Data.sdf.structure_expansion_v1.preview import select_preview, render_preview
from chemvector.training.curriculum.interface_diagnostic import write_json
from chemvector.training.curriculum.runtime import run_budgeted, ResourceBudget


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def worker(output, preview):
    output = Path(output)
    sources = bindings()
    paths = [*Path('Data/sdf/structure_expansion_v1').glob('*.py'), Path(__file__),
             Path('tests/data_sdf/test_structure_expansion_v1.py'), Path('tests/data_sdf/test_structure_preview.py')]
    sources.update({str(p):digest(p) for p in paths})
    reference, provenance = pool()
    entries, report = new_candidates(reference)
    write_json(output/'plan.json', dict(source_sha256=sources, reference_provenance=provenance,
        reference_inventory_sha256=sha256(json.dumps(reference,sort_keys=True).encode()).hexdigest(),
        planned_preview_images=128 if preview else 0, full_catalog_generation=False,
        gpu_used=False, model_forwards=0, training_authorized=False))
    write_json(output/'candidates.json', entries)
    selected = select_preview(entries)
    write_json(output/'preview-selection.json', selected)
    summary = render_preview(selected, output/'preview') if preview else None
    if any(digest(p) != value for p, value in sources.items()):
        raise ValueError('Source changed during candidate preparation')
    write_json(output/'report.json', dict(schema_version='structure-expansion-1',
        status='quarantine_candidates_prepared', **report, preview=summary,
        candidate_manifest_sha256=digest(output/'candidates.json'),
        plan_sha256=digest(output/'plan.json'), preview_selection_sha256=digest(output/'preview-selection.json'),
        sources_verified=len(sources), new_identity_queries=0, sealed_sample_files_read=False,
        full_catalog_rendered=False, default_inference_changed=False, model_forwards=0, gpu_used=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.output,args.preview)
        return 0
    command = [sys.executable,'-m','tools.prepare_structure_expansion','--worker','--output',str(args.output)]
    if args.preview:
        command.append('--preview')
    receipt = run_budgeted(command, output=args.output, budget=ResourceBudget(240))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
