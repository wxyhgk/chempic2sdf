"""Synthetic saved-cache checks; no checkpoint, data release, or model forward."""

from copy import deepcopy
from hashlib import sha256
from itertools import product
import json

import numpy as np
import pytest
import torch

from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.training.curriculum import supervision_contract, upstream_diagnostic
from chemvector.train_direct_atom_pair_bond import match_atom_targets
from tools import audit_curriculum_v02_selector_matching as audit


@pytest.mark.parametrize("shape", [(0, 0), (0, 2), (3, 0)])
def test_empty_assignment(shape):
    assert audit.radius_assignment(np.zeros(shape), 6) == []


def test_post_assignment_gate_loses_available_correspondence():
    cost = [[68.8663, 3.8118], [94.1752, 23.5994]]
    rows, cols = audit.linear_sum_assignment(cost)
    assert not any(cost[r][c] <= 6 for r, c in zip(rows, cols))
    assert audit.radius_assignment(cost, 6) == [(0, 1)]


def test_cardinality_first_not_greedy_and_inclusive_boundary():
    assert set(audit.radius_assignment([[5, 6], [6, 100]], 6)) == {(0, 1), (1, 0)}
    assert audit.radius_assignment([[6, 6.000001]], 6) == [(0, 0)]
    assert audit.radius_assignment([[6.000001]], 6) == []
    result = audit.radius_assignment([[1], [1], [2]], 6)
    assert len(result) == 1 and result[0][0] in {0, 1}


def test_counterfactual_matches_exhaustive_small_assignments():
    rng = np.random.default_rng(5)
    for n, m in product(range(1, 5), repeat=2):
        cost = rng.uniform(0, 12, (n, m))
        options = []
        for columns in product(range(-1, m), repeat=n):
            used = [c for c in columns if c >= 0]
            if len(set(used)) != len(used):
                continue
            pairs = [(r, c) for r, c in enumerate(columns) if c >= 0]
            if all(cost[r, c] <= 6 for r, c in pairs):
                options.append((-len(pairs), sum(cost[r, c] for r, c in pairs)))
        result = audit.radius_assignment(cost, 6)
        assert (-len(result), sum(cost[r, c] for r, c in result)) == pytest.approx(min(options))


@pytest.mark.parametrize("cost,radius", [([1], 6), ([[float("nan")]], 6),
    ([[-1]], 6), ([[float("inf")]], 6), ([[1]], 0), ([[1]], float("nan"))])
def test_invalid_assignment(cost, radius):
    with pytest.raises(ValueError):
        audit.radius_assignment(cost, radius)


def fixture(*, rejected=False, duplicate=False, unknown=False):
    positions = [[.2, .3], [.7, .3], [.201, .3] if duplicate else [.9, .9]]
    logits = [-1 if rejected else 1, 1, 1 if duplicate or unknown else -1]
    trace = dict(raw_count=3, selected_count=sum(v >= -.25 for v in logits), candidates=[
        dict(candidate=i, geometry=xy + [0.] * 6, proposal_score=.5,
             selection_logit=logits[i], selected=logits[i] >= -.25) for i, xy in enumerate(positions)])
    config = PredictedProposalConfig(maximum_atoms=3, maximum_bonds=1)
    raw, selected, _ = audit.restore_candidates(trace, 3, -.25)
    batch = dict(purpose="diagnostic", training_authorized=False, atom_mask=torch.tensor([[True, True]]),
        atom_anchors=torch.tensor([positions[:2]]), bond_mask=torch.tensor([[True]]),
        bond_atom_refs=torch.tensor([[[0, 1]]]), bond_chemical=torch.tensor([[0]]))
    coverage = upstream_diagnostic.labeled_coverage(raw, selected, batch, config)
    mapping, _ = match_atom_targets(selected, batch, config)
    contract = supervision_contract.pair_targets_with_unknown(selected, mapping, batch)
    slots = {int(mapping[0, i]): i for i in selected['atom_mask'][0].nonzero().flatten().tolist()
             if int(mapping[0, i]) >= 0}
    case = dict(sample_key="synthetic", canonical_smiles="CC", candidates=trace,
        metrics=dict(coverage=coverage, pair_contract=contract['coverage'],
            bonds=[dict(target_slots=[0, 1], candidates=[slots.get(0), slots.get(1)])]))
    return case, batch, config


def test_saved_replay_preserves_unknown_and_inputs():
    case, batch, config = fixture(unknown=True)
    before = deepcopy(case)
    result = audit.analyze_case(case, batch, config, -.25)
    assert result["original_pair_contract"]["unknown_pairs"] == 2
    assert result["original_pair_contract"]["known_negative_pairs"] == 0
    assert result["counts"]["raw_negative_selected"] == 1
    assert case == before


def test_rejected_near_candidate_has_positive_label_and_upward_local_gradient():
    case, batch, config = fixture(rejected=True)
    result = audit.analyze_case(case, batch, config, -.25)
    assert result["counts"]["raw_positive_rejected"] == 1
    assert result["counts"]["lost_targets_no_nearby"] == 1
    row = result["relevant_candidates"][0]
    assert row["raw_positive"] and row["atom_only_logit_derivatives"]["atom_only_total"] < 0


def test_duplicate_negative_raw_label_does_not_imply_selected_match_failure():
    case, batch, config = fixture(rejected=True, duplicate=True)
    result = audit.analyze_case(case, batch, config, -.25)
    assert result["counts"]["raw_positive_rejected"] == 1
    assert result["counts"]["raw_negative_selected"] == 1
    assert result["original_coverage"]["selected_matched"] == 2
    duplicate = next(r for r in result["relevant_candidates"] if r["candidate"] == 2)
    assert not duplicate["raw_positive"] and duplicate["selected_target"] == 0


@pytest.mark.parametrize("change", ["duplicate", "nan", "threshold", "count", "geometry", "out_of_range"])
def test_reconstruction_fails_closed(change):
    case, _, _ = fixture()
    trace = case["candidates"]
    if change == "duplicate": trace["candidates"][1]["candidate"] = 0
    elif change == "nan": trace["candidates"][0]["selection_logit"] = float("nan")
    elif change == "threshold": trace["candidates"][0]["selected"] = False
    elif change == "count": trace["raw_count"] += 1
    elif change == "geometry": trace["candidates"][0]["geometry"].pop()
    else: trace["candidates"][0]["candidate"] = 3
    with pytest.raises(ValueError):
        audit.restore_candidates(trace, 3, -.25)


@pytest.mark.parametrize("field", ["coverage", "pair_contract", "bonds"])
def test_replay_rejects_changed_saved_metrics(field):
    case, batch, config = fixture()
    if field == "coverage": case["metrics"][field]["selected_matched"] += 1
    elif field == "pair_contract": case["metrics"][field]["unknown_pairs"] += 1
    else: case["metrics"][field][0]["candidates"] = [1, 0]
    with pytest.raises(ValueError, match="reproduce"):
        audit.analyze_case(case, batch, config, -.25)


def test_logit_derivatives_are_detached_from_input_and_mask_inactive_slots():
    logits = torch.tensor([[-.6, -.8, -.09, 99.]], requires_grad=True)
    before = logits.detach().clone()
    info, grads = audit.atom_loss_probe(logits, torch.tensor([[True, True, False, False]]),
                                      torch.tensor([[True, True, True, False]]))
    assert info["target_count"] == 2 and logits.grad is None and torch.equal(logits, before)
    assert all(g[0, 3] == 0 for g in grads.values())
    assert grads["weighted_ranking"][0, 0] == 0  # Only the worst positive is active.
    assert grads["weighted_ranking"][0, 1] < 0 < grads["weighted_ranking"][0, 2]


def test_artifact_binding_and_symlink_guards(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"saved": True}))
    expected = sha256(path.read_bytes()).hexdigest()
    assert audit.read_bound(path, expected) == {"saved": True}
    with pytest.raises(ValueError, match="SHA"):
        audit.read_bound(path, "0" * 64)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="nonsealed"):
        audit.digest(link)
    with pytest.raises(ValueError, match="nonsealed"):
        audit.digest(tmp_path / "sealed_test" / "anything")


def test_cpu_duration_and_no_overwrite_guards(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        audit.run(tmp_path / "unused")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="60"):
        audit.run(tmp_path / "unused", max_seconds=61)
    with pytest.raises(FileExistsError):
        audit.run(tmp_path)
