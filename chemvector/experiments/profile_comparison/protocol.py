"""Pin existing recipes and the already exposed 50-molecule development set."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from chemvector.experiments.id_graph.roundtrip import require_hash, sha256, write_json
from chemvector.evaluation.audits.ocr.set_rejected_text.protocol import source_hashes

REFERENCE = Path('outputs/experiments/id-graph-prediction-20260913')
TRUTH = Path('outputs/experiments/id-graph-roundtrip-20260913')
REFERENCE_SHA = '8d54de90db5ead287777e1d6a49bcb1798f360acbe69b8af9c2202851dadc303'
TRUTH_SHA = 'c6e3b4a09bede764f0c63fedfd80f40a41be68314fdf994623b353dbbdeda9b1'
REGISTRY = Path('frontend/inference_profiles.json')
PROFILES = {'v01010': 'structure-v01010-direct-aromatic-ocr',
            'v01016': 'structure-v01016-structural-twin-ocr'}
ARMS = ('strict_id_reference', *PROFILES)


def validate_selection(previous):
    keys, metadata = previous['sample_order'], previous['sample_metadata']
    allowed = {f'Data/sdf/chemvector_joint_v1/fresh_eval_12k/shards/fresh_dev-{i:06d}.tar' for i in (0, 1)}
    if len(keys) != 200 or len(set(keys)) != 200 or set(metadata) != set(keys):
        raise ValueError('Require the original 200 distinct samples')
    if set(previous['shards']) != allowed or set(previous['image_index']) != set(keys):
        raise ValueError('Unexpected input shards/index')
    molecules = defaultdict(list)
    for key in keys:
        if Path(key).name != key or not key.startswith('fresh_dev_'):
            raise ValueError('Invalid sample identity')
        entry = previous['image_index'][key]
        if entry['shard'] not in allowed or entry['member'] != f'{key}.png':
            raise ValueError('Unexpected PNG member')
        molecules[metadata[key]['molecule_id']].append(metadata[key]['variant'])
    variants = Counter(('original', 'rdkit', 'coordgen', 'stress'))
    if len(molecules) != 50 or any(Counter(v) != variants for v in molecules.values()):
        raise ValueError('Require 50 molecules with four complete variants')


def validate_profiles(profiles):
    """The paired recipes may differ only by the existing suppression action."""
    if set(profiles) != set(PROFILES):
        raise ValueError('Unexpected profile set')
    stripped = {}
    for arm, profile in profiles.items():
        if profile['module'] != 'chemvector.infer_direct_atom_pair_bond':
            raise ValueError('Unexpected inference entrypoint')
        args = list(profile['arguments'])
        for flag, width in (('--model-version', 2), ('--action-ranking-checkpoint', 2),
                            ('--structural-twin-suppression', 1)):
            if flag in args:
                index = args.index(flag)
                del args[index:index + width]
        stripped[arm] = args
    if stripped['v01010'] != stripped['v01016']:
        raise ValueError('Profiles differ beyond suppression/model-version')
    if '--structural-twin-suppression' in profiles['v01010']['arguments']:
        raise ValueError('Baseline unexpectedly enables suppression')
    if '--structural-twin-suppression' not in profiles['v01016']['arguments']:
        raise ValueError('Candidate does not enable suppression')


def verify(protocol):
    for group in ('sources', 'checkpoints', 'shards', 'inputs'):
        for name, digest in protocol[group].items():
            require_hash(name, digest)


def freeze(output):
    import torch
    from rdkit import rdBase

    if output.exists():
        raise FileExistsError('Use a fresh output directory; no implicit retries')
    require_hash(REFERENCE / 'protocol.json', REFERENCE_SHA)
    require_hash(TRUTH / 'protocol.json', TRUTH_SHA)
    previous = json.loads((REFERENCE / 'protocol.json').read_text())
    validate_selection(previous)
    for name, digest in previous['sources'].items():
        require_hash(name, digest)
    registry = json.loads(REGISTRY.read_text())['profiles']
    profiles = {arm: registry[key] for arm, key in PROFILES.items()}
    validate_profiles(profiles)
    checkpoints = dict(previous['checkpoints'])
    args = profiles['v01016']['arguments']
    ranking = args[args.index('--action-ranking-checkpoint') + 1]
    checkpoints[ranking] = profiles['v01016']['action_ranking_checkpoint_sha256']
    for profile in profiles.values():
        args = profile['arguments']
        for index, arg in enumerate(args):
            if arg.endswith('-checkpoint') or arg == '--checkpoint':
                if args[index + 1] not in checkpoints:
                    raise ValueError('Unbound profile checkpoint')
    inputs = {str(REGISTRY): sha256(REGISTRY)}
    for root in (REFERENCE, TRUTH):
        for filename in ('protocol.json', 'report.json'):
            inputs[str(root / filename)] = sha256(root / filename)
        report = json.loads((root / 'report.json').read_text())
        for name, digest in report['artifacts'].items():
            inputs[str(root / name)] = digest
    sources = source_hashes()
    test = Path('tests/chemvector/test_profile_comparison.py')
    sources[str(test)] = sha256(test)
    protocol = {
        'schema': 'frontend-profile-common-sdf-comparison-0.1',
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'sample_order': previous['sample_order'], 'sample_metadata': previous['sample_metadata'],
        'image_index': previous['image_index'], 'shards': previous['shards'],
        'sources': sources, 'inputs': inputs, 'checkpoints': checkpoints,
        'profiles': profiles, 'arms': list(ARMS), 'denominator': 200,
        'device': 'cuda', 'gpu': torch.cuda.get_device_name(0),
        'torch_version': torch.__version__, 'rdkit_version': rdBase.rdkitVersion,
        'runtime': 'unmodified CLI main, batch 1, CUDA BF16; memoize checkpoint loading only',
        'parity': 'cold CLI: first sample of each variant in both arms, plus first executed suppression if any',
        'parity_metric_exclusion': ['inference_seconds'],
        'model_state_check': 'every cached module state_dict digest before/after all inference',
        'scoring': 'one canonical sanitized SDF scorer; core removes stereo but retains isotopes/charge',
        'failures': 'no exclusion/retry; report CLI errors separately and keep full denominator',
        'primary_pair': ['v01010', 'v01016'],
        'strict_reference_caveat': 'historical batch6 detector plus CPU FP32 frozen OCR; not a compiler-only A/B',
        'fallback': 'retain actual legacy behavior; named carbon fallback does not reset all elements to C',
        'truth_boundary': 'predict both profiles and seal receipt before reading truth graph/signatures',
        'scope': 'reused 50-molecule development comparison, not unseen validation or architecture ranking',
        'training_performed': False, 'promotion_authorized': False,
    }
    verify(protocol)
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', protocol)
    return protocol
