"""CPU-only text-interface comparison on the already audited quarantine cases.

Truth atom fields score the parser AFTER parsing; they never enter the parser.
This does not run OCR or establish deployment safety on mistaken OCR strings.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

from chemvector.components.ocr.display_label_candidate import parse_display_label_candidate
from chemvector.core.ocr import parse_atom_display_text
from chemvector.training.curriculum.interface_diagnostic import write_json
from chemvector.training.curriculum.runtime import run_budgeted, ResourceBudget


DATASET = Path('outputs/datasets/case-diversity-v1-12288-20260922')
AUDIT = Path('outputs/audits/case-diversity-v1-12288-20260922/report.json')
AUDIT_SHA = '598ab01eac2ecadee70d2a0dc64daaeea2ddc49027b7c4db4e1a1d89b25ff823'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def compare_labels(atoms):
    counts = Counter()
    unsupported, added, mismatches, changed = Counter(), Counter(), Counter(), Counter()
    for atom in atoms:
        if not atom['visible_text']:
            continue
        text = atom['text']
        old = parse_atom_display_text(text)
        new = parse_display_label_candidate(text)  # TEXT ONLY, no graph evidence.
        counts['visible_labels'] += 1
        if old is not None:
            counts['legacy_parsed'] += 1
            if old != new:
                changed[text] += 1
        if new is None:
            unsupported[text] += 1
            continue
        counts['candidate_parsed'] += 1
        if old is None:
            added[text] += 1
        if any(new[k] != atom[k] for k in ('element','formal_charge','isotope','radical_electrons')):
            mismatches[text + ':core'] += 1
        # Absence of H in a visible label does NOT establish zero total H.
        # Check only positive visible-H declarations against total graph H.
        if new['explicit_h_count']:
            counts['positive_visible_H_checked'] += 1
            if new['explicit_h_count'] != atom['explicit_h_count'] + atom['implicit_h_count']:
                mismatches[text + ':hydrogens'] += 1
    return dict(counts=dict(counts), newly_supported=dict(added), abstained=dict(unsupported),
                field_mismatches=dict(mismatches), changed_previously_parsed=dict(changed))


def worker(output):
    if digest(AUDIT) != AUDIT_SHA:
        raise ValueError('Frozen source audit changed')
    audit = json.loads(AUDIT.read_bytes())
    for name, key in (('samples.json','sample_index_sha256'), ('sample-sha256.json','sample_hash_manifest_sha256'),
                      ('summary.json','summary_sha256'), ('plan.json','source_plan_sha256')):
        if digest(DATASET / name) != audit[key]:
            raise ValueError('Audited input changed: ' + name)
    sources = {p:digest(p) for p in (str(Path(__file__)), 'chemvector/core/ocr.py',
        'chemvector/components/ocr/display_label_candidate.py', 'tests/chemvector/test_display_label_candidate.py')}
    hashes = json.loads((DATASET/'sample-sha256.json').read_bytes())
    rows = json.loads((DATASET/'samples.json').read_bytes())
    totals = {k:Counter() for k in ('counts','newly_supported','abstained','field_mismatches','changed_previously_parsed')}
    for row in rows:
        name = 'samples/' + row['sample_key'] + '.layers.json'
        data = (DATASET/name).read_bytes()
        if sha256(data).hexdigest() != hashes[name]:
            raise ValueError('Audited labels changed')
        result = compare_labels(json.loads(data)['atoms'])
        for key, values in result.items():
            totals[key].update(values)
    if any(digest(p) != value for p, value in sources.items()):
        raise ValueError('Parser/checker changed during comparison')
    write_json(Path(output)/'report.json', dict(schema_version='display-label-candidate-check-1',
        source_audit_sha256=AUDIT_SHA, implementation_sha256=sources,
        verified_layers=len(rows), **{k:dict(v) for k,v in totals.items()},
        observed_label_compatibility_passed=not totals['field_mismatches'] and not totals['changed_previously_parsed'],
        truth_used_only_for_scoring=True, plain_text_ambiguous_isotopes_abstain=True,
        implicit_H_without_visible_H_not_checked=True, end_to_end_graph_recompile=False,
        default_parser_changed=False, inference_wiring_changed=False, deployment_authorized=False,
        gpu_used=False, model_forwards=0, training_authorized=False,
        caveat='Correct-string compatibility, not OCR accuracy or noisy-string false-positive safety'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.output)
        return 0
    receipt = run_budgeted([sys.executable,'-m','tools.check_display_label_candidate',
        '--worker','--output',str(args.output)], output=args.output, budget=ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
