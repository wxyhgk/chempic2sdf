from copy import deepcopy

import pytest

from chemvector.training.curriculum.protocol import require_run_protocol, selection_records, validate_protocol


@pytest.fixture
def fixed():
    return {"training_protocol": {"schema_version": "chemvector-fixed-continuation-protocol-0.1",
        "max_steps": 6000, "batch_size": 1, "validate_every": 600, "patience": 3,
        "wall_seconds": 43200, "max_gpus": 1, "selection_domains": ["full_molecule"], "automatic_resume": False}}


def test_curriculum_improvement_cannot_mask_full_molecule_regression(fixed):
    baseline = [{"domain": "curriculum", "exact": False}, {"domain": "full_molecule", "exact": True}]
    current = [{"domain": "curriculum", "exact": True}, {"domain": "full_molecule", "exact": False}]
    assert sum(r["exact"] for r in selection_records(fixed, baseline)) == 1
    assert sum(r["exact"] for r in selection_records(fixed, current)) == 0
    assert selection_records({}, baseline) == baseline


@pytest.mark.parametrize("field,value", [("max_steps", 7000), ("batch_size", 2), ("wall_seconds", 1000),
                                        ("validate_every", 1), ("patience", 10)])
def test_cli_cannot_override_reviewed_recipe(fixed, field, value):
    with pytest.raises(ValueError, match="hash-bound"):
        require_run_protocol(fixed, **{field: value})
    require_run_protocol(fixed, **{field: fixed["training_protocol"][field]})


@pytest.mark.parametrize("field,value", [("wall_seconds", 43201), ("max_gpus", 2), ("automatic_resume", True),
    ("selection_domains", ["curriculum", "full_molecule"]), ("max_steps", True), ("batch_size", 0)])
def test_invalid_protocol_is_rejected(fixed, field, value):
    value_dict = deepcopy(fixed["training_protocol"])
    value_dict[field] = value
    with pytest.raises(ValueError):
        validate_protocol(value_dict)


def test_full_validation_cannot_disappear(fixed):
    with pytest.raises(ValueError, match="No full-molecule"):
        selection_records(fixed, [{"domain": "curriculum"}])
