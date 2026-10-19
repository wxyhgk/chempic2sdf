"""Plan-only missing-scaffold inventory; no additional renders or identity queries."""

from hashlib import sha256
import json
from pathlib import Path
import sys

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from Data.sdf.case_diversity_v1.catalog import pool, inventory
from chemvector.training.curriculum.interface_diagnostic import write_json
from chemvector.training.curriculum.runtime import run_budgeted, ResourceBudget


SOURCE = Path('outputs/datasets/case-diversity-v1-12288-20260922')
OUTPUT = Path('outputs/audits/case-diversity-scaffold-gaps-20260922')


def scaffold(entry):
    mol = Chem.MolFromSmiles(entry['canonical_smiles']); Chem.RemoveStereochemistry(mol)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def missing_representatives(entries, selected):
    covered = {s for e in selected if (s := scaffold(e))}
    groups = {}
    for e in entries:
        key = scaffold(e)
        if key and key not in covered: groups.setdefault(key, []).append(e)
    return [dict(scaffold=key, available_structures=len(rows),
                 entry=min(rows, key=lambda e: (e['heavy_atoms'], e['molecule_id'])))
            for key, rows in sorted(groups.items())]


def worker():
    data = (SOURCE / 'molecules.json').read_bytes(); selected = json.loads(data)
    entries, provenance = pool(); missing = missing_representatives(entries, selected)
    write_json(OUTPUT / 'candidates.json', missing)
    write_json(OUTPUT / 'report.json', dict(status='plan_only_not_generated_or_admitted',
        reference='planned768_not_final_quality_retained_cohort',
        selected_molecules_sha256=sha256(data).hexdigest(), source_provenance=provenance,
        current_inventory=inventory(selected), proposed_structures=len(missing),
        projected_inventory=inventory(selected + [r['entry'] for r in missing]),
        selection='one_per_uncovered_nonempty_Murcko_scaffold; minimum_heavy_atoms_then_ID; no_model_scores',
        planned_images_if_future16_profiles=len(missing)*16, images_generated=0,
        training_authorized=False, external_overlap_status='unchecked',
        model_forwards=0, gpu_used=False, external_identity_queries=0,
        caveats=['not_a_generic_topology_count; heteroatom_Murcko_definition',
                 'acyclic_diversity_needs_separate_backbone_stratification',
                 'recompute_after_quality_partition_before_any_future_generation',
                 'same_existing_template_pool_not_new_external_molecule_sources'],
        candidates_sha256=sha256((OUTPUT / 'candidates.json').read_bytes()).hexdigest(),
        implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest()))


def main():
    if sys.argv[1:] == ['--worker']: worker(); return 0
    receipt = run_budgeted([sys.executable, '-m', 'tools.plan_case_scaffold_gaps', '--worker'],
                           output=OUTPUT, budget=ResourceBudget(60))
    print(json.dumps(receipt, indent=2)); return 0 if receipt['status']=='completed' else 1


if __name__ == '__main__': raise SystemExit(main())
