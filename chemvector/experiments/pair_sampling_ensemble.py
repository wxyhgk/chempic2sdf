"""Prediction-only fixed translation mixture; experimental, not deployed."""

from collections import Counter
from itertools import product
import math

import torch

OFFSETS = ((0., 0.),) + tuple((x, y) for y in (-4., 0., 4.) for x in (-4., 0., 4.) if x or y)


def translated_anchors(anchors, selected_mask, offset, image_size):
    """Translate sampling origins only; never clip, alter masks or mutate inputs."""
    if (anchors.ndim != 3 or anchors.shape[-1] != 2 or selected_mask.shape != anchors.shape[:2]
            or selected_mask.dtype != torch.bool or len(offset) != 2
            or not math.isfinite(image_size) or image_size <= 0
            or not all(math.isfinite(v) for v in offset) or not bool(torch.isfinite(anchors).all())):
        raise ValueError("Finite predicted anchors, mask and pixel translation required")
    result = anchors.clone()
    result[selected_mask] += anchors.new_tensor(offset)/image_size
    if bool(((result[selected_mask] < 0) | (result[selected_mask] > 1)).any()):
        raise ValueError("Translation leaves image; no clipping or per-case fallback")
    return result


def mean_joint_predictions(presence_logits, chemical_logits):
    """Equal-weight nine-view marginalization, not a maximum or learned gate.

    p(edge)=mean(p_v); p(type|edge)=sum(p_v*q_v)/sum(p_v).
    Return logits compatible with the unchanged aromatic/threshold rules.
    """
    if (presence_logits.ndim != 2 or presence_logits.shape[0] != len(OFFSETS)
            or chemical_logits.shape != (*presence_logits.shape, 4)
            or not bool(torch.isfinite(presence_logits).all() and torch.isfinite(chemical_logits).all())):
        raise ValueError("Exactly nine finite aligned presence/type outputs required")
    p = presence_logits.double().sigmoid()
    joint = (p[..., None]*chemical_logits.double().softmax(-1)).mean(0)
    epsilon = torch.finfo(torch.float64).eps
    mean = p.mean(0).clamp(epsilon, 1-epsilon)
    conditional = joint.clamp_min(epsilon)
    conditional = conditional/conditional.sum(-1, keepdim=True)
    return dict(presence_logits=(mean.log()-torch.log1p(-mean)).float(),
                chemical_logits=conditional.log().float())


def _counts(bonds):
    return dict(targets=len(bonds), stages=dict(Counter(b["stage"] for b in bonds)))


def summarize(cases):
    cells = {(c["molecule_id"], tuple(c["bits"])) for c in cases}
    mids = {c["molecule_id"] for c in cases}
    if (len(cases) != len(cells) or len(mids) != 4
            or cells != {(m, b) for m in mids for b in product((0, 1), repeat=4)}):
        raise ValueError("Complete fixed 64-image panel required")
    transitions, focus = [], []
    for case in cases:
        old = {b["bond_id"]: b for b in case["baseline_metrics"]["bonds"]}
        new = {b["bond_id"]: b for b in case["metrics"]["bonds"]}
        if (len(old) != len(case["baseline_metrics"]["bonds"]) or len(new) != len(case["metrics"]["bonds"])
                or old.keys() != new.keys() or any(old[i]["candidates"] != new[i]["candidates"]
                or old[i]["expected_type"] != new[i]["expected_type"] for i in old)):
            raise ValueError("Candidate and target lineage must remain fixed")
        po = case["po_bond_ids"]
        if (len(po) != len(set(po)) or not set(po) <= set(old)
                or sorted(old[i]["expected_type"] for i in po) != ["DOUBLE", "SINGLE", "SINGLE"]):
            raise ValueError("All three P-O controls required")
        for bid in sorted(old):
            a, b = old[bid], new[bid]
            record = dict(sample_key=case["sample_key"], bond_id=bid, kind=a["expected_type"],
                before=a["stage"], after=b["stage"], before_probability=a["presence_probability"],
                after_probability=b["presence_probability"])
            transitions.append(record)
            if bid in po: focus.append(dict(**record, font_size=12 if case["bits"][1] == 0 else 24))
    gained = [r for r in transitions if r["before"] != "reachable_correct" and r["after"] == "reachable_correct"]
    lost = [r for r in transitions if r["before"] == "reachable_correct" and r["after"] != "reachable_correct"]
    focus_summary = []
    for font in (12, 24):
        for kind in ("SINGLE", "DOUBLE"):
            rows = [r for r in focus if r["font_size"] == font and r["kind"] == kind]
            focus_summary.append(dict(font_size=font, kind=kind, targets=len(rows),
                baseline_correct=sum(r["before"] == "reachable_correct" for r in rows),
                mixture_correct=sum(r["after"] == "reachable_correct" for r in rows),
                gained_correct=sum(r["before"] != "reachable_correct" and r["after"] == "reachable_correct" for r in rows),
                lost_correct=sum(r["before"] == "reachable_correct" and r["after"] != "reachable_correct" for r in rows)))
    negatives = {stage: sum(c[key]["confirmed_false_positive_pairs"] for c in cases)
                 for stage, key in (("baseline", "baseline_metrics"), ("mixture", "metrics"))}
    unknown = {stage: sum(c[key]["unknown_present_pairs"] for c in cases)
               for stage, key in (("baseline", "baseline_metrics"), ("mixture", "metrics"))}
    large = next(r for r in focus_summary if r["font_size"] == 24 and r["kind"] == "SINGLE")
    gate = dict(large_po_improved=large["mixture_correct"] > large["baseline_correct"],
        no_original_correct_bond_lost=not lost, no_confirmed_fp_increase=negatives["mixture"] <= negatives["baseline"],
        no_unknown_present_increase=unknown["mixture"] <= unknown["baseline"])
    return dict(images=64, structures=4,
        baseline=_counts([b for c in cases for b in c["baseline_metrics"]["bonds"]]),
        mixture=_counts([b for c in cases for b in c["metrics"]["bonds"]]),
        focus_by_font=focus_summary, gained_correct=len(gained), lost_correct=len(lost),
        gained_bonds=gained, lost_bonds=lost, confirmed_false_positives=negatives, unknown_present=unknown,
        panel_gate=gate, panel_gate_passed=all(gate.values()),
        scope="Four previously observed structures; no whole-molecule accuracy or deployment authorization")
