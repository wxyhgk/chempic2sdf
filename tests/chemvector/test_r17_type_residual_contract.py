import torch
import pytest


def test_type_residual_zero_state_preserves_frozen_chemical_logits():
    from tools.train_r17_t3f_type_correction_cpu import TypeResidual, residual_logits

    head = TypeResidual(266)
    features = torch.randn(7, 266)
    frozen = features[:, -4:].clone()
    with torch.no_grad():
        out = residual_logits(head, features, frozen)
    assert torch.equal(out, frozen)


def test_feature_cache_rejects_foreign_binding_and_wrong_split():
    from tools.train_r17_t3f_type_correction_cpu import validate_cache

    binding = {"manifest_sha256": "abc", "train_keys": ["a"], "holdout_keys": ["b"]}
    cache = {"binding": binding, "train": [{"key": "a"}], "holdout": [{"key": "b"}]}
    validate_cache(cache, binding)
    with pytest.raises(RuntimeError, match="binding"):
        validate_cache(cache, {**binding, "manifest_sha256": "changed"})
    cache["holdout"] = [{"key": "a"}]
    with pytest.raises(RuntimeError, match="split/key"):
        validate_cache(cache, binding)
