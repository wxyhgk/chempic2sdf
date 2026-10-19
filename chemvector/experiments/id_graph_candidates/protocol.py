"""Freeze a new bounded shadow experiment; prior artifacts remain immutable."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from rdkit import rdBase

from chemvector.external_ocr_r0_contracts import FROZEN_ARM_BUNDLE_SHA256, arm_bundle_sha256
from chemvector.experiments.id_graph.roundtrip import require_hash, sha256, write_json
from .decode import Budget

PREDICTION = Path('outputs/experiments/id-graph-prediction-20260913')
TRUTH = Path('outputs/experiments/id-graph-roundtrip-20260913')
BASELINE_TEXT = Path('outputs/evaluations/text-entry-comparison-20260909/predictions.jsonl')
BEAMS = Path('outputs/evaluations/external-ocr-r0-predicted-bbox/arms/ctc_v0109_top8.jsonl')
PREDICTION_SHA = '8d54de90db5ead287777e1d6a49bcb1798f360acbe69b8af9c2202851dadc303'


def verify(config):
    for group in ('sources', 'inputs'):
        for name, digest in config[group].items():
            require_hash(name, digest)


def freeze(output):
    if output.exists():
        raise FileExistsError('Use a fresh candidate experiment directory')
    require_hash(PREDICTION / 'protocol.json', PREDICTION_SHA)
    if arm_bundle_sha256(BEAMS) != FROZEN_ARM_BUNDLE_SHA256['ctc_v0109_top8']:
        raise ValueError('Historical CTC top8 bundle changed')
    previous = json.loads((PREDICTION / 'protocol.json').read_text())
    sources = dict(previous['sources'])
    for path in sorted(Path(__file__).parent.glob('*.py')) + [Path('tests/chemvector/test_id_graph_candidates.py')]:
        sources[str(path)] = sha256(path)
    inputs = dict(previous['inputs'])
    for root in (PREDICTION, TRUTH):
        inputs[str(root / 'report.json')] = sha256(root / 'report.json')
        for name, digest in json.loads((root / 'report.json').read_text())['artifacts'].items():
            inputs[str(root / name)] = digest
    for path in (BEAMS, BEAMS.with_suffix('.metadata.json')):
        inputs[str(path)] = sha256(path)
    if str(BASELINE_TEXT) not in inputs:
        raise ValueError('Frozen greedy OCR input is missing')
    receipt = json.loads((PREDICTION / 'prediction_receipt.json').read_text())
    config = {'schema': 'id-graph-shadow-candidates-0.1', 'frozen_utc': datetime.now(timezone.utc).isoformat(),
              'sources': sources, 'inputs': inputs, 'sample_order': previous['sample_order'],
              'rdkit_version': rdBase.rdkitVersion, 'budget': asdict(Budget()),
              'bond_threshold_logit': receipt['bond_threshold_logit'],
              'node_contract': 'preserve all IDs, positions, observations; no node adds/deletes',
              'attribute_contract': 'same element/isotope/radicals; only top8-backed charge/H/no_implicit alternatives',
              'attribute_budget_order': 'beam-rank breadth first, then candidate ID; no cross-crop confidence claim',
              'bond_contract': 'bottom4 selected presence DELETE; bottom4 chemical top2-margin RETYPE to other three classes',
              'edit_budget': 'one atom attribute alternative OR one DELETE OR one RETYPE, never combined edits',
              'decision': 'KEEP every valid baseline; shadow-only candidate sets for invalid baseline; never auto adopt',
              'deduplication': 'full sanitized SDF isomeric identity retaining charge/isotope; preserve all action provenance',
              'scores': 'text_role_score, within-crop CTC logp, and bond logits separate; no cross-source weighted sum',
              'truth_boundary': 'save all candidates/receipt before scorer reads truth',
              'primary': 'candidate core-graph coverage and wrong-valid risk; operational exact unchanged by design',
              'normalization': 'all common uniquely geometrically matched edges, including raw-equal pairs',
              'controls': '200 baseline replays and truth core exact; no-op and input hash checks',
              'scope': 'post-hoc known-development shadow pilot; not unseen evaluation or deployable correction',
              'denominator': 200, 'training_performed': False, 'promotion_authorized': False}
    if len(config['sample_order']) != 200 or len(set(config['sample_order'])) != 200:
        raise ValueError('Require original 200 distinct samples')
    verify(config)
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', config)
    return config
