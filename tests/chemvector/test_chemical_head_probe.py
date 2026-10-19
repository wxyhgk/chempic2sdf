import torch

from chemvector.training.curriculum.chemical_head_probe import (
    _decision,
    _presence_signature,
)


def record(key, domain="full_molecule", exact=False, valid=True, fallback=False):
    return {"key": key, "domain": domain, "exact": exact, "valid": valid, "fallback": fallback}


def test_decision_requires_strict_full_exact_gain_and_zero_new_fallback():
    before = [record("a", exact=False), record("b", exact=True), record("c", domain="curriculum", exact=False)]
    after = [record("a", exact=True), record("b", exact=True), record("c", domain="curriculum", exact=False)]
    result = _decision(before, after, complete=True)
    assert result["candidate_eligible"]
    assert result["rescued_keys"] == ["a"]
    assert result["broken_keys"] == []


def test_decision_rejects_new_fallback_even_with_exact_gain():
    before = [record("a", exact=False), record("b", exact=True)]
    after = [record("a", exact=True), record("b", exact=True, fallback=True)]
    result = _decision(before, after, complete=True)
    assert not result["candidate_eligible"]
    assert "new_fallbacks" in result["reasons"]


def test_presence_signature_includes_logits_and_selected_pairs():
    output = {"presence_logits": torch.tensor([0.0, 1.0])}
    pairs = torch.tensor([[0, 1], [0, 2]])
    first = _presence_signature(output, pairs, .85)
    changed = _presence_signature({"presence_logits": torch.tensor([0.0, 1.0001])}, pairs, .85)
    assert first != changed
