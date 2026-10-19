"""Synthetic-only intervention contracts; no real images/checkpoints are loaded."""

from copy import deepcopy
from itertools import product

import pytest
import torch

from chemvector.direct_atom_pair_bond import DirectAtomPairBond, DirectAtomPairBondConfig
from chemvector.experiments import po_pair_attribution as probe
from tools import diagnose_po_pair_attribution as runner


def test_capture_reconstruction_and_no_input_mutation():
    torch.manual_seed(1)
    head = DirectAtomPairBond(DirectAtomPairBondConfig(feature_channels=8, atom_feature_dim=6,
        longitudinal_samples=8, transverse_samples=3, hidden_dim=16)).eval().requires_grad_(False)
    features = torch.rand(1, 8, 16, 16)
    atom = torch.rand(1, 3, 6); anchors = torch.rand(1, 3, 2); scores = torch.rand(1, 3)
    pairs = torch.tensor([[0, 1], [0, 2], [1, 2]]); indices = torch.zeros(3, dtype=torch.long)
    original = anchors.clone()
    with torch.inference_mode():
        output, encoded = probe.capture_shared(head, lambda: head(features, atom, anchors, scores, pairs, indices))
        assert probe.branch_widths(head) == (16, 128, 32)
        rebuilt = probe.readout(head, encoded)
        for name in ("presence_logits", "chemical_logits"):
            torch.testing.assert_close(output[name], rebuilt[name], atol=probe.ATOL, rtol=0)
        assert len(probe.result_rows(rebuilt)) == 3
    assert torch.equal(anchors, original) and not head.shared._forward_pre_hooks
    assert all(p.grad is None for p in head.parameters())


def test_capture_cleans_hook_on_exception():
    head = DirectAtomPairBond()
    def fail(): raise RuntimeError("synthetic")
    with pytest.raises(RuntimeError): probe.capture_shared(head, fail)
    assert not head.shared._forward_pre_hooks
    with pytest.raises(ValueError, match="Exactly one"): probe.capture_shared(head, lambda: None)


def test_all_eight_hybrids_change_only_selected_branches():
    low, high = torch.arange(12.).reshape(2, 6), -torch.arange(12.).reshape(2, 6)
    left, right = low.clone(), high.clone()
    widths = (1, 2, 3)
    for bits in probe.MASKS:
        mixed = probe.mix_inputs(low, high, widths, bits)
        for i, section in enumerate(mixed.split(widths, -1)):
            assert torch.equal(section, (high if bits[i] else low).split(widths, -1)[i])
    assert torch.equal(left, low) and torch.equal(right, high)


@pytest.mark.parametrize("problem", ["nan", "width", "mask", "shape"])
def test_invalid_hybrid_is_rejected(problem):
    low, high, widths, bits = torch.zeros(2, 6), torch.ones(2, 6), (1, 2, 3), (0, 1, 0)
    if problem == "nan": low[0, 0] = float("nan")
    elif problem == "width": widths = (1, 1, 1)
    elif problem == "mask": bits = (0, 2, 1)
    else: high = high[:1]
    with pytest.raises(ValueError): probe.mix_inputs(low, high, widths, bits)


def test_oracle_changes_only_focus_anchors_and_rejects_ambiguous_mapping():
    anchors, target = torch.zeros(1, 5, 2), torch.ones(1, 3, 2)
    focus = [dict(candidates=[1, 3], target_slots=[0, 2])]
    changed = probe.oracle_anchors(anchors, focus, target)
    assert not bool(anchors.any())
    assert torch.equal(changed[0, [1, 3]], target[0, [0, 2]])
    assert not bool(changed[0, [0, 2, 4]].any())
    with pytest.raises(ValueError):
        probe.oracle_anchors(anchors, focus+[dict(candidates=[1, 2], target_slots=[1, 2])], target)


def fixture():
    cases, comparisons = [], []
    for mid in "abcd":
        for bits in product((0, 1), repeat=4):
            key = mid+"".join(map(str, bits))
            bonds = []
            for bid, kind in enumerate(("DOUBLE", "SINGLE", "SINGLE")):
                baseline = dict(presence_probability=.2 if bits[1] else .95, raw_type=kind)
                bonds.append(dict(bond_id=bid, expected_type=kind, baseline=baseline, oracle=dict(baseline)))
            cases.append(dict(sample_key=key, molecule_id=mid, bits=list(bits), bonds=bonds))
            if not bits[1]:
                high = list(bits); high[1] = 1
                for bid, kind in enumerate(("DOUBLE", "SINGLE", "SINGLE")):
                    comparisons.append(dict(low_key=key, high_key=mid+"".join(map(str, high)),
                        bond_id=bid, expected_type=kind,
                        hybrids={"".join(map(str, mask)): dict(presence_probability=.2 if mask[0] else .95,
                            raw_type=kind) for mask in probe.MASKS}))
    return cases, comparisons


def test_fixed_summary_with_controls_no_false_repair_claim():
    cases, comparisons = fixture(); before = deepcopy((cases, comparisons))
    result = probe.summarize(cases, comparisons, .85)
    assert (cases, comparisons) == before
    for row in result["oracle_by_font"]:
        assert row["baseline"] == row["oracle"] and row["gained_correct"] == row["lost_correct"] == 0
    for row in result["hybrids"]:
        assert row["targets"] == (64 if row["kind"] == "SINGLE" else 32)
        assert row["correct"] == (0 if row["bits"][0] == "1" else row["targets"])


@pytest.mark.parametrize("problem", ["missing_case", "duplicate_comparison", "missing_hybrid", "wrong_font"])
def test_incomplete_or_misaligned_summary_fails(problem):
    cases, comparisons = fixture()
    if problem == "missing_case": cases.pop()
    elif problem == "duplicate_comparison": comparisons[-1] = comparisons[0]
    elif problem == "missing_hybrid": comparisons[0]["hybrids"].pop("010")
    else: comparisons[0]["high_key"] = comparisons[0]["low_key"]
    with pytest.raises(ValueError): probe.summarize(cases, comparisons, .85)


def test_cpu_guard_before_model_or_source_read(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"): runner.run()
