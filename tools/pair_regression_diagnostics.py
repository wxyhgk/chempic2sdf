"""Read-only saved-pair diagnostics, not new labels or replacement acceptance."""

from rdkit import Chem
import torch
from torch.nn import functional as F

from chemvector.bond_object import BOND_TYPES
from chemvector.training.curriculum.validation_truth import canonical_molblock

PAIR_THRESHOLD = .85


def validate_pair_records(before, after):
    if before['metrics']['key'] != after['metrics']['key']:
        raise ValueError('Different samples cannot form a paired diagnostic')
    for field in ('candidate_label_sha256', 'text_sha256', 'target_bonds', 'reachable_positive_pairs'):
        if before['metrics'][field] != after['metrics'][field]:
            raise ValueError('Frozen signatures or target denominators changed')
    for field in ('roles', 'ocr', 'compiled_candidate_indices'):
        if before['trace'][field] != after['trace'][field]:
            raise ValueError('Text or candidate lineage changed')
    if before['pairs'] != after['pairs']:
        raise ValueError('Cannot reuse pair row IDs across differing candidates')
    pairs = before['pairs']
    if len(set(tuple(p) for p in pairs)) != len(pairs) or any(
            len(p) != 2 or any(type(v) is not int or v < 0 for v in p) or p[0] >= p[1] for p in pairs):
        raise ValueError('Distinct ordered nonnegative pairs required')
    for record in (before, after):
        logits = torch.tensor(record['presence_logits'], dtype=torch.float32)
        chemistry = torch.tensor(record['chemical_logits'], dtype=torch.float32).reshape(-1, len(BOND_TYPES))
        if (logits.shape != (len(pairs),) or chemistry.shape != (len(pairs), len(BOND_TYPES))
                or not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(chemistry).all())):
            raise ValueError('Finite aligned saved pair outputs required')


def scene_edges(record):
    trace = record['trace']
    lineage = trace['compiled_candidate_indices']
    scene = trace['scene_before_fallback']
    atoms = scene['atom_objects']
    if len(lineage) != len(atoms) or len(set(lineage)) != len(lineage):
        raise ValueError('Unique compiled candidate lineage required')
    if [a['id'] for a in atoms] != list(range(len(atoms))):
        raise ValueError('Packed scene atom IDs required')
    edges = {}
    for edge in scene['bond_objects']:
        refs = edge['atom_refs']
        if len(refs) != 2 or any(type(i) is not int or not 0 <= i < len(lineage) for i in refs):
            raise ValueError('Invalid scene endpoints')
        pair = tuple(sorted(lineage[i] for i in refs))
        if pair[0] == pair[1] or pair in edges:
            raise ValueError('Duplicate or self scene edge')
        edges[pair] = edge['chemical_type']
    return edges


def compiled_edges(record):
    """Use exact q1023 lineage; never infer missing/ambiguous correspondences."""
    if record['prediction']['molblock'] is None:
        return None
    molecule = Chem.MolFromMolBlock(record['prediction']['molblock'], sanitize=True, removeHs=False)
    if molecule is None:
        raise ValueError('Saved nonempty MolBlock is not valid')
    trace = record['trace']
    anchors = {tuple(a['anchor']): slot for a, slot in zip(
        trace['scene_before_fallback']['atom_objects'], trace['compiled_candidate_indices'])}
    if len(anchors) != len(trace['compiled_candidate_indices']) or molecule.GetNumAtoms() != len(anchors):
        raise ValueError('Quantized collision or changed compiled atom count')
    slots = {}
    conformer = molecule.GetConformer()
    for atom in molecule.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        scaled = (position.x * 100, -position.y * 100)
        point = tuple(round(v) for v in scaled)
        if point not in anchors or any(abs(a - b) > 1e-5 for a, b in zip(scaled, point)):
            raise ValueError('Compiled coordinate no longer has exact candidate lineage')
        slots[atom.GetIdx()] = anchors[point]
    if len(set(slots.values())) != len(slots):
        raise ValueError('Compiled lineage is not one-to-one')
    return {tuple(sorted((slots[b.GetBeginAtomIdx()], slots[b.GetEndAtomIdx()]))): str(b.GetBondType())
            for b in molecule.GetBonds()}


def compare_decoding(before, after):
    validate_pair_records(before, after)
    old_scene, new_scene = scene_edges(before), scene_edges(after)
    old_final, new_final = compiled_edges(before), compiled_edges(after)
    output = []
    p0, p1 = [torch.tensor(r['presence_logits'], dtype=torch.float32).sigmoid() for r in (before, after)]
    c0, c1 = [torch.tensor(r['chemical_logits'], dtype=torch.float32).reshape(-1, len(BOND_TYPES)).argmax(-1)
              for r in (before, after)]
    for i, value in enumerate(before['pairs']):
        pair = tuple(value)
        present0, present1 = bool(p0[i] >= PAIR_THRESHOLD), bool(p1[i] >= PAIR_THRESHOLD)
        changed_type = c0[i] != c1[i] and (present0 or present1)
        compiled_change = (old_final is not None and new_final is not None
                           and old_final.get(pair) != new_final.get(pair))
        if (present0 != present1 or changed_type or old_scene.get(pair) != new_scene.get(pair)
                or compiled_change):
            output.append(dict(row_id=i, pair=list(pair), probability_before=float(p0[i]),
                probability_after=float(p1[i]), present_before=present0, present_after=present1,
                raw_type_before=BOND_TYPES[int(c0[i])], raw_type_after=BOND_TYPES[int(c1[i])],
                scene_type_before=old_scene.get(pair), scene_type_after=new_scene.get(pair),
                compiled_type_before=None if old_final is None else old_final.get(pair),
                compiled_type_after=None if new_final is None else new_final.get(pair),
                compiled_before_available=old_final is not None, compiled_after_available=new_final is not None))
    identity0, identity1 = [canonical_molblock(r['prediction']['molblock']) for r in (before, after)]
    return dict(changed_pairs=output, pairs=len(p0),
        raw_presence_added=int(((p1 >= PAIR_THRESHOLD) & (p0 < PAIR_THRESHOLD)).sum()),
        raw_presence_removed=int(((p0 >= PAIR_THRESHOLD) & (p1 < PAIR_THRESHOLD)).sum()),
        raw_chemical_changes_on_present_pairs=int(((c0 != c1) & ((p0 >= PAIR_THRESHOLD) | (p1 >= PAIR_THRESHOLD))).sum()),
        final_structure_equal=identity0 is not None and identity0 == identity1,
        compiled_edge_maps_equal=old_final is not None and new_final is not None and old_final == new_final,
        raw_typed_bond_ids_lost=sorted(set(before['metrics']['correct_typed_bond_ids'])
                                     - set(after['metrics']['correct_typed_bond_ids'])),
        baseline_exact_broken=before['metrics']['sdf_exact'] and not after['metrics']['sdf_exact'],
        new_invalid_sdf=before['metrics']['sdf_valid'] and not after['metrics']['sdf_valid'],
        new_fallback=not before['prediction']['fallback'] and after['prediction']['fallback'],
        final_graphs_available=[old_final is not None, new_final is not None],
        original_acceptance_changed=False)


def recovered_sample_labels(event, pair_count):
    """Only for SHA-bound single-image legacy sampler: positives precede negatives.

    Unselected is NOT negative or unknown; no full known-mask reconstruction is
    claimed. Chemical labels and historical parameter gradients are absent.
    """
    selected = event['sampled_row_ids']
    positives = event['positive_rows']
    if (event.get('updated') is not True or event.get('sampled_unknown_pairs') != 0
            or type(positives) is not int or not 0 <= positives <= len(selected)
            or len(selected) != event['sampled_rows'] or len(selected) != len(set(selected))
            or any(type(i) is not int or not 0 <= i < pair_count for i in selected)):
        raise ValueError('Invalid single-image completed sampling receipt')
    return dict(zip(selected, [True] * positives + [False] * (len(selected) - positives)))


def presence_replay(before, after, event):
    """Differentiate saved logits only; no actual model, optimizer or new labels."""
    validate_pair_records(before, after)
    if event['key'] != before['metrics']['key']:
        raise ValueError('Sampler receipt belongs to a different image')
    labels_by_row = recovered_sample_labels(event, len(before['pairs']))
    selected = list(labels_by_row)
    if not selected:
        raise ValueError('Empty branches cannot have an optimizer receipt')
    labels = torch.tensor(list(labels_by_row.values()), dtype=torch.float32)
    positive_count = labels.sum().clamp_min(1)
    negative_count = (1 - labels).sum().clamp_min(1)
    balance = (negative_count / positive_count).clamp(1, 16)
    arms = {}
    for name, record in (('initial_logits', before), ('final_logits', after)):
        logits = torch.tensor(record['presence_logits'], dtype=torch.float32, requires_grad=True)
        loss = F.binary_cross_entropy_with_logits(logits[selected], labels, pos_weight=balance)
        gradient, = torch.autograd.grad(loss, logits)
        sampled_gradient = gradient[selected]
        unsampled = torch.ones(len(logits), dtype=torch.bool); unsampled[selected] = False
        if (not bool(torch.isfinite(gradient).all()) or not bool(torch.isfinite(loss))
                or bool((gradient[unsampled] != 0).any())
                or bool((sampled_gradient[labels == 0] < 0).any())
                or bool((sampled_gradient[labels == 1] > 0).any())):
            raise ValueError('Unexpected direct-logit loss direction')
        arms[name] = dict(presence_loss=float(loss.detach()), positive_weight=float(balance),
            per_sampled_row=[dict(row_id=i, label=labels_by_row[i],
                probability=float(logits[i].detach().sigmoid()), direct_logit_gradient=float(gradient[i]))
                for i in selected], unsampled_gradient_zero=True)
    return dict(step=event['step'], attempt=event['attempt'], arms=arms,
        observed_pre_own_step_presence_loss=event['losses']['presence_loss'],
        historical_parameter_gradients_recovered=False,
        scope='Same saved sampled rows evaluated at initial/final logits; not a replay of optimizer steps')
