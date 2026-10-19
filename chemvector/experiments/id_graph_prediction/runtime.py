"""Freeze and replay the unchanged detector/AtomPair head; reuse frozen OCR strings."""

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch
from rdkit import rdBase

from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from chemvector.direct_scene_inference import load_direct_model
from chemvector.external_ocr_r0_contracts import SOURCE_CHECKPOINT, GROUNDED_GRAPH_CHECKPOINT, read_jsonl
from chemvector.image_io import image_tensor
from chemvector.predicted_incidence import PredictedProposalConfig, decode_predicted_atom_proposals, decode_predicted_bond_proposals
from chemvector.text_entry_predictions import model_digest
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_spatial_incidence import forward_frozen_bond_head
from chemvector.evaluation.audits.ocr.set_rejected_text.checks import check_deployed
from chemvector.evaluation.audits.ocr.set_rejected_text.inputs import selected_images
from chemvector.evaluation.audits.ocr.set_rejected_text.protocol import BASELINE, DATASET, source_hashes
from chemvector.experiments.e2e_bottleneck.protocol import DIRECT
from chemvector.experiments.id_graph.roundtrip import SELECTION, SELECTION_SHA, require_hash, sha256, write_json
from .adapter import BOND_TYPES, from_prediction, assert_connectivity_preserved

TRUTH = Path('outputs/experiments/id-graph-roundtrip-20260913')
TRUTH_PROTOCOL_SHA = 'c6e3b4a09bede764f0c63fedfd80f40a41be68314fdf994623b353dbbdeda9b1'


def verify(config):
    for group in ('sources', 'checkpoints', 'shards', 'inputs'):
        for path, digest in config[group].items():
            require_hash(path, digest)


def freeze(output):
    if output.exists():
        raise FileExistsError('Use a new prediction directory; no implicit retries')
    require_hash(SELECTION, SELECTION_SHA)
    require_hash(TRUTH / 'protocol.json', TRUTH_PROTOCOL_SHA)
    old = json.loads(SELECTION.read_text())
    control = json.loads((TRUTH / 'report.json').read_text())
    if not control['compiler_control_pass'] or control['counts']['all_checks_pass'] != 200:
        raise ValueError('Require the completed 200/200 ID-graph control')
    if len(old['sample_order']) != 200 or len(set(old['sample_order'])) != 200:
        raise ValueError('Require the original 200 distinct sample identities')
    for path, digest in old['sources'].items():
        require_hash(path, digest)
    for path, digest in json.loads((TRUTH / 'protocol.json').read_text())['sources'].items():
        require_hash(path, digest)
    inputs = {**old['inputs'], str(SELECTION): SELECTION_SHA,
              str(TRUTH / 'report.json'): sha256(TRUTH / 'report.json')}
    for name, digest in control['artifacts'].items():
        inputs[str(TRUTH / name)] = digest
    sources = source_hashes()
    sources['tests/chemvector/test_id_graph_prediction.py'] = sha256('tests/chemvector/test_id_graph_prediction.py')
    config = {'schema': 'id-graph-prediction-diagnostic-0.1',
              'frozen_utc': datetime.now(timezone.utc).isoformat(),
              'sources': sources, 'inputs': inputs, 'checkpoints': old['checkpoints'], 'shards': old['shards'],
              'sample_order': old['sample_order'], 'sample_metadata': old['sample_metadata'], 'image_index': old['image_index'],
              'denominator': 200, 'batch_size': 6, 'precision': 'CUDA BF16',
              'gpu': torch.cuda.get_device_name(0), 'torch_version': torch.__version__, 'rdkit_version': rdBase.rdkitVersion,
              'atom_threshold': 'unchanged source.training_args.atom_decision_threshold',
              'bond_threshold': 'unchanged direct.validation.threshold converted to logit',
              'bond_classes': list(BOND_TYPES), 'ocr': 'reuse frozen atom_conditioned raw strings; no new OCR inference',
              'adapter': 'preserve candidate IDs and every thresholded bond; no coordinate deduplication',
              'chemistry': 'same carbon defaults and OCR parser; unspecified stereo; no aromatic repair or carbon fallback',
              'geometry': 'unquantized predicted pixel coordinates; identity is candidate ID',
              'matching_radius_px': 6., 'assignment': 'maximum cardinality then minimum distance; geometry only',
              'ambiguity': 'attribute/bond comparison restricted to mutually unique radius-eligible pairs',
              'attributes': ['element', 'formal_charge', 'isotope', 'radical_electrons'],
              'representation_only': ['explicit_h_count', 'no_implicit', 'aromatic'],
              'compiler_comparison': 'strict ID baseline; NOT equivalent to legacy repaired/fallback SDF baseline',
              'truth_boundary': 'all raw predictions and receipt saved before graph truth is decoded for scoring',
              'scope': 'descriptive errors on reused development; not causal bottleneck ranking or unseen evaluation',
              'training_performed': False, 'promotion_authorized': False, 'factorial_executed': False}
    verify(config)
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', config)
    return config


def predict(output, config):
    from chemvector.bond_object import BOND_TYPES as native_classes

    if tuple(native_classes) != BOND_TYPES:
        raise ValueError('Bond class ordering changed')
    torch.set_num_threads(4)
    device = torch.device('cuda')
    direct, payload = load_direct_model(str(DIRECT), device)
    if (payload['training_args']['source_checkpoint'] != str(SOURCE_CHECKPOINT)
            or payload['training_args']['grounded_graph_checkpoint'] != str(GROUNDED_GRAPH_CHECKPOINT)):
        raise ValueError('Direct model dependency differs')
    vision, atom, bond, selector, source = load_components(str(SOURCE_CHECKPOINT), str(GROUNDED_GRAPH_CHECKPOINT), device)
    models = (vision, atom, bond, selector, direct)
    for model in models:
        model.eval().requires_grad_(False)
    before = model_digest(models)
    proposals = PredictedProposalConfig(**source['proposal_config'])
    threshold = float(source['training_args']['atom_decision_threshold'])
    probability = float(payload['validation']['threshold'])
    bond_threshold = math.log(probability / (1. - probability))
    ocr, deployed = defaultdict(list), defaultdict(list)
    for row in read_jsonl(BASELINE / 'predictions.jsonl'):
        if row['arm'] == 'atom_conditioned':
            ocr[row['sample_key']].append({'id': int(row['object_key'].rsplit(':p', 1)[1]),
                                          'text': row['text'], 'score': row['score'], 'bbox': row['bbox']})
    for row in read_jsonl(DATASET / 'atom-objects.jsonl'):
        if row['sample_key'] in config['sample_metadata']:
            deployed[row['sample_key']].append(row)
    (output / 'raw').mkdir()
    (output / 'graphs').mkdir()
    artifacts, completed = {}, []

    def consume(samples):
        pixels = torch.stack([image_tensor(s['image']) for s in samples]).to(device)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
            atom_output = atom(features)
            atoms = decode_predicted_atom_proposals(atom_output, proposals)
            bonds = decode_predicted_bond_proposals(forward_frozen_bond_head(vision, bond, features), proposals)
            selected = selector(atoms['atom_features'], atoms['atom_geometry'], atoms['atom_scores'], atoms['atom_mask'],
                                bonds['bond_features'], bonds['bond_endpoints'], bonds['bond_types'], bonds['bond_scores'], bonds['bond_mask'])
            atoms['atom_mask'] = atoms['atom_mask'] & (selected['atom_presence_logits'] >= threshold)
        for i, sample in enumerate(samples):
            key = sample['key']
            check_deployed(key, atoms['atom_geometry'][i].float().cpu().numpy(), atoms['atom_mask'][i].cpu().numpy(), deployed[key])
            local = {name: value[i:i + 1] for name, value in atoms.items()}
            pairs, indices = enumerate_unordered_atom_pairs(local['atom_mask'])
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                predicted = direct(features[i:i + 1], local['atom_features'], local['atom_geometry'][..., :2],
                                   local['atom_scores'], pairs, indices)
            candidates = torch.where(local['atom_mask'][0])[0].tolist()
            pair_ids = pairs.cpu().tolist()
            presence = predicted['presence_logits'].float().cpu().tolist()
            chemical = predicted['chemical_logits'].float().cpu().tolist()
            record = {'sample_key': key, 'atoms': [
                {'id': j, 'xy': (local['atom_geometry'][0, j, :2].float().cpu() * 768).tolist(),
                 'score': float(local['atom_scores'][0, j])} for j in candidates],
                'pairs': [{'id': j, 'atom_ids': pair, 'presence_logit': presence[j], 'chemical_logits': chemical[j]}
                          for j, pair in enumerate(pair_ids)], 'ocr': ocr[key]}
            raw_path = output / 'raw' / f'{key}.json'
            write_json(raw_path, record)
            graph, diagnostics = from_prediction(record, bond_threshold)
            assert_connectivity_preserved(record, graph, bond_threshold)
            graph_path = output / 'graphs' / f'{key}.json'
            write_json(graph_path, graph)
            for path in (raw_path, graph_path):
                artifacts[str(path.relative_to(output))] = sha256(path)
            completed.append({'sample_key': key, 'adapter_preserved': True, 'deployed_replay_exact': True, **diagnostics})
        print(f'Prediction adapter: {len(completed)}/200', flush=True)

    try:
        pending = []
        for sample in selected_images(config):
            pending.append(sample)
            if len(pending) == config['batch_size']:
                consume(pending)
                pending = []
        if pending:
            consume(pending)
        if model_digest(models) != before:
            raise ValueError('Frozen model state changed')
        verify(config)
        receipt = {'sample_count': len(completed), 'models_unchanged': True, 'model_digest': before,
                   'atom_threshold': threshold, 'bond_threshold_logit': bond_threshold, 'bond_threshold_probability': probability,
                   'rows': completed, 'artifacts': artifacts, 'truth_used_in_prediction': False,
                   'protocol_sha256': sha256(output / 'protocol.json')}
        write_json(output / 'prediction_receipt.json', receipt)
        return receipt
    except Exception as error:
        write_json(output / 'failure.json', {'stage': 'prediction', 'completed': len(completed),
                                           'denominator': 200, 'error': str(error), 'experiment_valid': False})
        raise
