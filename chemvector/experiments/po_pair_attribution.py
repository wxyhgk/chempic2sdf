"""Frozen pair-input interventions; privileged diagnostics, never inference rules."""

from itertools import product
import statistics

import torch

from chemvector.bond_object import BOND_TYPES

BRANCHES = ("visual", "atom", "score")
MASKS = tuple(product((0, 1), repeat=3))
ATOL = 2e-5


def capture_shared(head, call):
    """Observe the actual production concatenation without replacing any input."""
    captured = []
    handle = head.shared.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone()))
    try:
        output = call()
    finally:
        handle.remove()
    if len(captured) != 1 or not bool(torch.isfinite(captured[0]).all()):
        raise ValueError("Exactly one finite shared input required")
    return output, captured[0]


def branch_widths(head):
    # These are actual layer dimensions, not checkpoint-dependent constants.
    return (head.config.hidden_dim, head.atom_encoder[-2].out_features,
            head.score_encoder[0].out_features)


def mix_inputs(low, high, widths, bits):
    if (low.ndim != 2 or low.shape != high.shape or len(widths) != 3
            or any(w < 1 for w in widths) or low.shape[-1] != sum(widths)
            or tuple(bits) not in MASKS
            or not bool(torch.isfinite(low).all() and torch.isfinite(high).all())):
        raise ValueError("Aligned finite three-branch inputs required")
    left, right = low.split(widths, dim=-1), high.split(widths, dim=-1)
    return torch.cat([right[i] if bit else left[i] for i, bit in enumerate(bits)], dim=-1)


def readout(head, encoded):
    hidden = head.shared(encoded)
    return dict(presence_logits=head.presence(hidden).squeeze(-1),
                chemical_logits=head.chemical(hidden))


def result_rows(output):
    logits, chemistry = output["presence_logits"], output["chemical_logits"]
    if (logits.ndim != 1 or chemistry.shape != (len(logits), len(BOND_TYPES))
            or not bool(torch.isfinite(logits).all() and torch.isfinite(chemistry).all())):
        raise ValueError("Finite complete head outputs required")
    return [dict(presence_logit=float(logits[i]), presence_probability=float(logits[i].sigmoid()),
                 chemical_logits=chemistry[i].tolist(), raw_type=BOND_TYPES[int(chemistry[i].argmax())])
            for i in range(len(logits))]


def oracle_anchors(anchors, focus, target_anchors):
    """Change only matched focus coordinates using the existing quantized targets.

    This explicitly uses truth and is not a deployable correction. It does not
    re-extract atom features/scores, introduce candidates or change their mask.
    """
    result = anchors.clone()
    assigned = {}
    for bond in focus:
        for candidate, slot in zip(bond["candidates"], bond["target_slots"]):
            if (type(candidate) is not int or type(slot) is not int
                    or not 0 <= candidate < anchors.shape[1]
                    or not 0 <= slot < target_anchors.shape[1]
                    or candidate in assigned and assigned[candidate] != slot):
                raise ValueError("Unambiguous reachable endpoint mapping required")
            assigned[candidate] = slot
            result[0, candidate] = target_anchors[0, slot]
    return result


def count_rows(rows, threshold):
    if not rows:
        raise ValueError("No diagnostic rows")
    missing = sum(r["presence_probability"] < threshold for r in rows)
    wrong = sum(r["presence_probability"] >= threshold and r["raw_type"] != r["expected_type"] for r in rows)
    return dict(targets=len(rows), correct=len(rows)-missing-wrong, missing=missing, wrong_type=wrong,
                mean_probability=statistics.mean(r["presence_probability"] for r in rows))


def summarize(cases, comparisons, threshold):
    cells = {(c["molecule_id"], tuple(c["bits"])): c for c in cases}
    mids = {c["molecule_id"] for c in cases}
    if (len(cells) != len(cases) or len(mids) != 4
            or set(cells) != {(m, b) for m in mids for b in product((0, 1), repeat=4)}):
        raise ValueError("Full fixed 64-image cohort required")
    expected = {(c["sample_key"], b["bond_id"]) for c in cases if c["bits"][1] == 0 for b in c["bonds"]}
    actual = {(r["low_key"], r["bond_id"]) for r in comparisons}
    if len(comparisons) != 96 or len(actual) != len(comparisons) or actual != expected:
        raise ValueError("All 96 aligned font comparisons required")
    for case in cases:
        if sorted(r["expected_type"] for r in case["bonds"]) != ["DOUBLE", "SINGLE", "SINGLE"]:
            raise ValueError("Two singles and one double control required")
    lookup = {c["sample_key"]: c for c in cases}
    for row in comparisons:
        low, high = lookup[row["low_key"]], lookup[row["high_key"]]
        bits = list(low["bits"]); bits[1] = 1
        if (low["bits"][1] != 0 or bits != high["bits"] or low["molecule_id"] != high["molecule_id"]
                or set(row["hybrids"]) != {"".join(map(str, b)) for b in MASKS}):
            raise ValueError("Matched font-only comparison and all eight branches required")
    by_font = []
    for bit in (0, 1):
        for kind in ("SINGLE", "DOUBLE"):
            rows = [b for c in cases if c["bits"][1] == bit for b in c["bonds"] if b["expected_type"] == kind]
            baseline = [{**r["baseline"], "expected_type": kind} for r in rows]
            oracle = [{**r["oracle"], "expected_type": kind} for r in rows]
            correct = lambda r: r["presence_probability"] >= threshold and r["raw_type"] == kind
            by_font.append(dict(font_size=12 if bit == 0 else 24, kind=kind,
                baseline=count_rows(baseline, threshold), oracle=count_rows(oracle, threshold),
                gained_correct=sum(not correct(a) and correct(b) for a, b in zip(baseline, oracle)),
                lost_correct=sum(correct(a) and not correct(b) for a, b in zip(baseline, oracle)),
                mean_probability_delta=statistics.mean(b["presence_probability"]-a["presence_probability"]
                    for a, b in zip(baseline, oracle))))
    hybrids = []
    for bits in MASKS:
        name = "".join(map(str, bits))
        for kind in ("SINGLE", "DOUBLE"):
            rows = [r for r in comparisons if r["expected_type"] == kind]
            scores = [{**r["hybrids"][name], "expected_type": kind} for r in rows]
            deltas = [r["hybrids"][name]["presence_probability"]-r["hybrids"]["111"]["presence_probability"] for r in rows]
            hybrids.append(dict(bits=name, kind=kind, **count_rows(scores, threshold),
                mean_delta_from_large=statistics.mean(deltas),
                probability_increased_vs_large=sum(d > ATOL for d in deltas)))
    return dict(images=64, structures=4, font_comparisons=96, oracle_by_font=by_font, hybrids=hybrids,
                scope="Matched-target oracle/activation interventions; not deployable predictions or whole-graph accuracy")
