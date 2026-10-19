"""Content-bound receipt for the COMPLETE frozen inputs of one selector call.

This receipt is not a tensor cache: a future producer must also persist all the
raw/bond tensors. No sample labels, GT, optimizer, model loading or file IO here.
"""

from copy import deepcopy
from hashlib import sha256
import json
import math

import torch

SCHEMA = 'selector-input-cache-receipt-0.1'
RAW_FIELDS = ('atom_scores', 'atom_indices', 'atom_features', 'atom_geometry', 'atom_mask')
BOND_FIELDS = ('bond_scores', 'bond_indices', 'bond_features', 'bond_endpoints', 'bond_types', 'bond_mask')
CONTEXT_FIELDS = {'sample_key', 'image_sha256', 'checkpoint_sha256', 'implementation_sha256',
                  'atom_grid_shape', 'bond_grid_shape', 'decoder', 'numerics'}
DECODER_FIELDS = {'atom_policy', 'bond_policy', 'maximum_atoms', 'maximum_bonds',
                  'atom_threshold', 'bond_threshold', 'image_size'}


def json_digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _hash_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _validate_context(context):
    if not isinstance(context, dict) or set(context) != CONTEXT_FIELDS:
        raise ValueError('Exact non-label provenance fields required')
    if (not isinstance(context['sample_key'], str) or not context['sample_key']
            or any(not _hash_string(context[k]) for k in ('image_sha256', 'checkpoint_sha256', 'implementation_sha256'))
            or not isinstance(context['decoder'], dict) or set(context['decoder']) != DECODER_FIELDS):
        raise ValueError('Explicit sample, content hashes and decoder context required')
    decoder = context['decoder']
    if (any(not isinstance(decoder[k], str) or not decoder[k] for k in ('atom_policy', 'bond_policy'))
            or any(type(decoder[k]) is not int or decoder[k] < 1 for k in ('maximum_atoms', 'maximum_bonds', 'image_size'))
            or any(isinstance(decoder[k], bool) or not isinstance(decoder[k], (int, float))
                   or not math.isfinite(decoder[k]) or not 0 <= decoder[k] <= 1
                   for k in ('atom_threshold', 'bond_threshold'))):
        raise ValueError('Decoder capacity, thresholds and policies must be explicit')
    for key in ('atom_grid_shape', 'bond_grid_shape'):
        shape = context[key]
        if not isinstance(shape, (list, tuple)) or len(shape) != 3 or any(type(v) is not int or v < 1 for v in shape):
            raise ValueError('Grid identity must include channel, height and width')
    if context['atom_grid_shape'][0] != 1:
        raise ValueError('Atom indices use a single-channel spatial grid')
    settings = context['numerics']
    if (not isinstance(settings, dict) or set(settings) != {'device', 'dtype', 'torch_version', 'autocast', 'tf32'}
            or settings['device'] != 'cpu' or settings['dtype'] != 'float32'
            or not isinstance(settings['torch_version'], str) or not settings['torch_version']
            or settings['autocast'] is not False or settings['tf32'] is not False):
        raise ValueError('This diagnostic receipt is for CPU FP32 without autocast/TF32')


def _tensor_receipts(tree, prefix, context):
    fields = RAW_FIELDS if prefix == 'atom' else BOND_FIELDS
    if not isinstance(tree, dict) or set(tree) != set(fields):
        raise ValueError('Complete image-only proposal tensors required; scores alone/labels are forbidden')
    maximum = context['decoder']['maximum_atoms' if prefix == 'atom' else 'maximum_bonds']
    ids, mask, scores = (tree[prefix+'_'+name] for name in ('indices', 'mask', 'scores'))
    for name, value in tree.items():
        expected_type = torch.int64 if name.endswith('_indices') else torch.bool if name.endswith('_mask') else torch.float32
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                or value.requires_grad or value.dtype != expected_type
                or not torch.isfinite(value).all() or value.shape[:2] != (1, maximum)):
            raise ValueError('Complete detached finite one-image CPU proposal tensors required')
    if (ids.shape != (1, maximum) or mask.shape != ids.shape or scores.shape != ids.shape
            or not torch.equal(mask, scores >= context['decoder'][prefix+'_threshold'])
            or (scores < -1).any() or (scores > 1).any()
            or (ids < 0).any() or (ids >= math.prod(context[prefix+'_grid_shape'])).any()
            or len(set(ids[mask].tolist())) != int(mask.sum())):
        raise ValueError('Scores, masks or complete flat identities disagree')
    features = tree[prefix+'_features']
    if features.ndim != 3 or features.shape[-1] < 1:
        raise ValueError('Candidate features are mandatory, including unselected candidates')
    if prefix == 'atom':
        if tree['atom_geometry'].shape != (1, maximum, 8):
            raise ValueError('Full atom geometry required')
    elif (tree['bond_endpoints'].shape != (1, maximum, 2, 2)
          or tree['bond_types'].ndim != 3 or tree['bond_types'].shape[-1] < 1):
        raise ValueError('Full bond endpoints and type features required')
    result = {}
    for name, value in sorted(tree.items()):
        metadata = dict(dtype=str(value.dtype), shape=list(value.shape))
        digest = sha256(json.dumps(metadata, sort_keys=True).encode())
        digest.update(value.contiguous().numpy().tobytes())
        result[name] = dict(**metadata, sha256=digest.hexdigest())
    return result


def make_receipt(raw, bonds, context):
    _validate_context(context)
    body = dict(schema_version=SCHEMA, context=deepcopy(context),
        raw=_tensor_receipts(raw, 'atom', context), bonds=_tensor_receipts(bonds, 'bond', context))
    return dict(**body, input_identity_sha256=json_digest(body), training_authorized=False,
                contains_labels=False, tensors_persisted_by_this_function=False)


def require_exact_replay(receipt, raw, bonds, context):
    fresh = make_receipt(raw, bonds, context)
    if receipt != fresh:
        raise ValueError('Selector input/provenance changed; do not reuse slot labels or saved selector features')
    return fresh['input_identity_sha256']
