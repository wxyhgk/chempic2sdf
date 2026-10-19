"""Synthetic prediction-only sampling, aggregation and regression guards."""

from copy import deepcopy
from itertools import product
from types import SimpleNamespace

import pytest
import torch

from chemvector.experiments import pair_sampling_ensemble as method
from tools import diagnose_po_pair_translation as runner


def test_fixed_offsets_are_symmetric_and_include_single_zero_control():
    assert method.OFFSETS[0] == (0., 0.) and len(method.OFFSETS) == len(set(method.OFFSETS)) == 9
    assert set(method.OFFSETS) == set(product((-4., 0., 4.), repeat=2))
    assert torch.tensor(method.OFFSETS).sum(0).tolist() == [0., 0.]


def test_translation_preserves_direction_length_unselected_and_original():
    anchors = torch.tensor([[[.3, .4], [.5, .6], [.01, .99]]])
    mask = torch.tensor([[True, True, False]]); before = anchors.clone()
    moved = method.translated_anchors(anchors, mask, (-4., 4.), 768.)
    torch.testing.assert_close(moved[0, 1]-moved[0, 0], anchors[0, 1]-anchors[0, 0])
    torch.testing.assert_close(moved[mask]-anchors[mask], torch.tensor([[-4/768, 4/768]]).expand(2, -1))
    assert torch.equal(moved[~mask], anchors[~mask]) and torch.equal(anchors, before)
    assert torch.equal(method.translated_anchors(anchors, mask, (0., 0.), 768.), anchors)


@pytest.mark.parametrize("issue", ["outside", "nan", "size", "mask"])
def test_bad_translation_fails_without_clipping(issue):
    anchors, mask, size = torch.zeros(1, 2, 2), torch.ones(1, 2, dtype=torch.bool), 768.
    if issue == "nan": anchors[0, 0, 0] = float("nan")
    elif issue == "size": size = 0.
    elif issue == "mask": mask = mask.float()
    with pytest.raises(ValueError): method.translated_anchors(anchors, mask, (-4., 0.), size)


def test_mean_is_not_max_and_chemical_mixture_is_presence_weighted():
    p = torch.tensor([[.99]]+[[.01]]*8)
    presence = torch.logit(p)
    chemistry = torch.tensor([[[20., -20., -20., -20.]]]+[[[-20., 20., -20., -20.]]]*8)
    output = method.mean_joint_predictions(presence, chemistry)
    assert output["presence_logits"].sigmoid().item() == pytest.approx((.99+8*.01)/9, abs=1e-7)
    assert output["presence_logits"].sigmoid().item() < .85
    q = output["chemical_logits"].softmax(-1)[0]
    assert q[0] == pytest.approx(.99/(.99+.08), abs=1e-6)
    assert int(q.argmax()) == 0  # Not the unweighted eight-view type majority.


def test_identical_views_preserve_probabilities_and_permutation_is_equivalent():
    torch.manual_seed(2)
    p, c = torch.randn(1, 5), torch.randn(1, 5, 4)
    output = method.mean_joint_predictions(p.expand(9, -1), c.expand(9, -1, -1))
    torch.testing.assert_close(output["presence_logits"].sigmoid(), p[0].sigmoid())
    torch.testing.assert_close(output["chemical_logits"].softmax(-1), c[0].softmax(-1))
    p, c = torch.randn(9, 5), torch.randn(9, 5, 4)
    a, b = method.mean_joint_predictions(p, c), method.mean_joint_predictions(p.flip(0), c.flip(0))
    for key in a: torch.testing.assert_close(a[key], b[key])


@pytest.mark.parametrize("issue", ["missing_view", "misaligned", "nan", "extreme"])
def test_aggregate_shapes_and_numeric_guards(issue):
    p, c = torch.zeros(9, 2), torch.zeros(9, 2, 4)
    if issue == "missing_view": p, c = p[:8], c[:8]
    elif issue == "misaligned": c = c[:, :1]
    elif issue == "nan": p[0, 0] = float("nan")
    else:
        p[:, 0] = 1000; p[:, 1] = -1000; c[..., 0] = 1000
        assert all(bool(torch.isfinite(t).all()) for t in method.mean_joint_predictions(p, c).values())
        return
    with pytest.raises(ValueError): method.mean_joint_predictions(p, c)


def test_predictor_reads_all_pairs_and_only_predicted_inputs():
    class Head:
        config = SimpleNamespace(image_size=768.)
        def __init__(self): self.calls = []
        def __call__(self, features, atom_features, anchors, scores, pairs, rows):
            self.calls.append(anchors.clone())
            logits = anchors[rows, pairs[:, 0], 0]+anchors[rows, pairs[:, 1], 1]
            return dict(presence_logits=logits, chemical_logits=torch.zeros(len(pairs), 4))
    head = Head()
    selected = dict(atom_mask=torch.tensor([[True, False, True, True]]),
        atom_geometry=torch.full((1, 4, 8), .5), atom_features=torch.ones(1, 4, 2), atom_scores=torch.ones(1, 4))
    before = {k: v.clone() for k, v in selected.items()}
    pairs, rows = runner.METRICS.enumerate_unordered_atom_pairs(selected["atom_mask"])
    baseline = dict(pairs=pairs, output=head(None, selected["atom_features"], selected["atom_geometry"][..., :2],
        selected["atom_scores"], pairs, rows))
    output, saved = runner.predict_views(head, None, selected, baseline)
    assert len(head.calls) == 9 and output["presence_logits"].shape == (3,)
    assert torch.tensor(saved["presence_logits"]).shape == (9, 3)
    assert all(torch.equal(selected[k], before[k]) for k in before)
    for offset, anchors in zip(method.OFFSETS, head.calls):
        torch.testing.assert_close(anchors, method.translated_anchors(before["atom_geometry"][..., :2],
            selected["atom_mask"], offset, 768.))
    with pytest.raises(ValueError): runner.predict_views(head, None, selected, {**baseline, "pairs": pairs[:1]})


def fixture():
    cases = []
    for mid in "abcd":
        for bits in product((0, 1), repeat=4):
            bonds = [dict(bond_id=i, candidates=[0, i+1], expected_type=kind,
                stage="reachable_missing" if bits[1] and i == 1 else "reachable_correct",
                presence_probability=.2 if bits[1] and i == 1 else .95)
                for i, kind in enumerate(("DOUBLE", "SINGLE", "SINGLE"))]
            before = dict(bonds=bonds, confirmed_false_positive_pairs=0, unknown_present_pairs=1)
            after = deepcopy(before)
            for bond in after["bonds"]: bond.update(stage="reachable_correct", presence_probability=.95)
            cases.append(dict(sample_key=mid+"".join(map(str, bits)), molecule_id=mid, bits=list(bits),
                po_bond_ids=[0, 1, 2], baseline_metrics=before, metrics=after))
    return cases


def test_gate_requires_improvement_and_no_regression_not_merely_net_gain():
    cases = fixture()
    result = method.summarize(cases)
    assert result["panel_gate_passed"] and result["gained_correct"] == 32 and result["lost_correct"] == 0
    cases[0]["metrics"]["bonds"][0]["stage"] = "reachable_wrong_type"
    result = method.summarize(cases)
    assert result["gained_correct"] == 32 and result["lost_correct"] == 1 and not result["panel_gate_passed"]


@pytest.mark.parametrize("issue", ["fp", "unknown", "missing", "duplicate", "lineage", "no_gain"])
def test_safety_and_complete_panel_guards(issue):
    cases = fixture()
    if issue == "fp": cases[0]["metrics"]["confirmed_false_positive_pairs"] = 1
    elif issue == "unknown": cases[0]["metrics"]["unknown_present_pairs"] += 1
    elif issue == "missing": cases.pop()
    elif issue == "duplicate": cases[-1] = cases[0]
    elif issue == "lineage": cases[0]["metrics"]["bonds"][0]["candidates"] = [1, 2]
    else:
        for c in cases: c["metrics"] = deepcopy(c["baseline_metrics"])
    if issue in ("missing", "duplicate", "lineage"):
        with pytest.raises(ValueError): method.summarize(cases)
    else: assert not method.summarize(cases)["panel_gate_passed"]


def test_cpu_guard_before_any_work(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"): runner.run()
