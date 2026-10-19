"""Opt-in training-side perturbation contract. No launcher or data admission.

Targets guard augmentation identity only. They never replace predicted anchors,
features, candidates or original labels. Existing training/inference is unchanged.
"""

import torch

from chemvector.direct_atom_pair_bond import direct_atom_pair_loss

OFFSETS = ((0., 0.),) + tuple((x, y) for y in (-4., 0., 4.) for x in (-4., 0., 4.) if x or y)


def plan_views(anchors, selected, mapping, targets, pairs, known, chosen):
    """Fixed58-compatible, single-image CPU planning from existing correspondence.

    Every original chosen row survives in view zero. Extra views require both
    endpoints within 6 px of their ORIGINAL owner, strictly nearer to that owner
    than other targets, and inside the image. Never re-match or relabel a shift.
    """
    tensors = (anchors, selected, mapping, targets, pairs, known, chosen)
    if (any(t.device.type != "cpu" for t in tensors) or anchors.ndim != 2 or anchors.shape[-1] != 2
            or targets.ndim != 2 or targets.shape[-1] != 2 or not len(targets)
            or selected.shape != anchors.shape[:1] or selected.dtype != torch.bool
            or mapping.shape != selected.shape or mapping.dtype != torch.long
            or pairs.ndim != 2 or pairs.shape[-1] != 2 or pairs.dtype != torch.long
            or known.shape != pairs.shape[:1] or known.dtype != torch.bool
            or chosen.ndim != 1 or chosen.dtype != torch.long
            or any(not t.is_floating_point() or not bool(torch.isfinite(t).all()) for t in (anchors, targets))):
        raise ValueError("Finite single-image CPU planning tensors required")
    if (bool(((mapping < -1) | (mapping >= len(targets))).any())
            or bool((mapping[~selected] != -1).any())
            or len(mapping[mapping >= 0].unique()) != int((mapping >= 0).sum())
            or bool(((pairs < 0) | (pairs >= len(anchors))).any())
            or bool((pairs[:, 0] >= pairs[:, 1]).any())
            or len(pairs.unique(dim=0)) != len(pairs)
            or not bool(selected[pairs].all())
            or not torch.equal(known, (mapping[pairs] >= 0).all(-1))
            or bool(((chosen < 0) | (chosen >= len(pairs))).any())
            or len(chosen.unique()) != len(chosen) or not bool(known[chosen].all())):
        raise ValueError("Unique original selected correspondence and known chosen rows required")
    views = []
    owners = mapping.clamp_min(0)
    slots = torch.arange(len(anchors))
    for dx, dy in OFFSETS:
        shifted = anchors.detach().clone()
        shifted[selected] += shifted.new_tensor((dx/768., dy/768.))
        distance2 = ((shifted.double()[:, None]-targets.double()[None])*768).square().sum(-1)
        own = distance2[slots, owners]
        other = distance2.clone(); other[slots, owners] = float("inf")
        endpoint_reasons = dict(
            outside_image=((shifted < 0) | (shifted > 1)).any(-1),
            outside_owner_radius=own > 36.,
            not_unique_nearest=own >= other.min(-1).values,
            unmatched_owner=mapping < 0,
        )
        pair_reasons = {name: values[pairs].any(-1) for name, values in endpoint_reasons.items()}
        safe = ~torch.stack(list(pair_reasons.values())).any(0)
        # Keep the historical base loss even if ownership is ambiguous. Such
        # cases get no unsafe augmented label, not a silent base-label rewrite.
        kept = chosen if dx == dy == 0 else chosen[safe[chosen]]
        views.append(dict(offset_pixels=[dx, dy], anchors=shifted, rows=kept.clone(),
            eligible_mask=safe, exclusion_masks=pair_reasons))
    return views


def typed_augmentation_loss(base_output, augmented_output, presence, chemical, augmented_positions, *, weight=1.):
    """Original loss + one equally weighted shifted-view loss, NOT mean logits.

    Positions index the original sampled rows; chemical targets are unchanged.
    A caller still needs admitted data, fresh frozen caches and run authorization.
    Weight zero is a parity test; this module never creates or steps an optimizer.
    """
    count = len(presence)
    if (type(weight) not in (int, float) or weight not in (0., 1.) or not count
            or presence.dtype != torch.bool or presence.shape != (count,)
            or chemical.dtype != torch.long or chemical.shape != (count,)
            or chemical.device != presence.device or augmented_positions.device != presence.device
            or bool(((chemical[presence] < 0) | (chemical[presence] >= 4)).any())
            or augmented_positions.dtype != torch.long or augmented_positions.ndim != 1
            or bool(((augmented_positions < 0) | (augmented_positions >= count)).any())
            or len(augmented_positions.unique()) != len(augmented_positions)):
        raise ValueError("Bound original and augmented label rows required")
    for output, size in ((base_output, count), (augmented_output, len(augmented_positions))):
        if size == 0:
            if output is not None: raise ValueError("Empty augmentation must skip its forward")
            continue
        if (output is None or output["presence_logits"].shape != (size,)
                or output["chemical_logits"].shape != (size, 4)
                or any(output[k].device != presence.device or not output[k].is_floating_point()
                       for k in ("presence_logits", "chemical_logits"))
                or not all(bool(torch.isfinite(output[k]).all()) for k in ("presence_logits", "chemical_logits"))):
            raise ValueError("Finite outputs aligned to planned rows required")
    original = direct_atom_pair_loss(base_output, presence, chemical)
    extra = None if not len(augmented_positions) else direct_atom_pair_loss(
        augmented_output, presence[augmented_positions], chemical[augmented_positions])
    total = original["loss"] if extra is None or weight == 0 else original["loss"]+extra["loss"]
    return dict(loss=total, original=original, augmented=extra, augmentation_applied=bool(extra is not None and weight))


def other_nodes_in_ribbon(anchors, pairs, owners, targets):
    """Geometry-only ambiguity flag; keep hard negatives, do not mask by it.

    An interior nonendpoint target within the existing +/-12px ribbon is NOT
    proof of another visible stroke or of a wrong graph label.
    """
    a, b = anchors[pairs[:, 0]].double()*768, anchors[pairs[:, 1]].double()*768
    delta = b-a; squared = delta.square().sum(-1).clamp_min(1e-12)
    rel = targets.double()[None]*768-a[:, None]
    fraction = (rel*delta[:, None]).sum(-1)/squared[:, None]
    transverse = (rel[..., 0]*delta[:, None, 1]-rel[..., 1]*delta[:, None, 0]).abs()/squared.sqrt()[:, None]
    ids = torch.arange(len(targets))[None]
    external = (ids != owners[pairs[:, 0], None]) & (ids != owners[pairs[:, 1], None])
    return (external & (fraction >= .04) & (fraction <= .96) & (transverse <= 12.)).any(-1)
