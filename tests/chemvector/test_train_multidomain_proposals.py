from chemvector.train_multidomain_proposals import (
    add_proposal_summary,
    freeze_proposal_object_features,
    proposal_checkpoint_score,
)
from chemvector.dense_bond_object import (
    DenseBondObjectConfig,
    MultiScaleDenseBondObjectDetector,
)
from chemvector.spatial_atom_object import (
    SpatialAtomObjectConfig,
    SpatialAtomObjectDetector,
)


def test_proposal_summary_uses_weakest_domain() -> None:
    metrics = {
        "domains": {
            "a": {"atom_proposal_recall": 0.99, "bond_proposal_recall": 0.98},
            "b": {"atom_proposal_recall": 0.95, "bond_proposal_recall": 0.96},
            "c": {"atom_proposal_recall": 0.90, "bond_proposal_recall": 0.80},
        }
    }

    add_proposal_summary(metrics)

    assert metrics["proposal_summary"] == {
        "minimum_domain_atom_proposal_recall": 0.90,
        "minimum_domain_bond_proposal_recall": 0.80,
        "minimum_domain_joint_proposal_recall": 0.80,
        "mean_domain_atom_proposal_recall": (0.99 + 0.95 + 0.90) / 3,
        "mean_domain_bond_proposal_recall": (0.98 + 0.96 + 0.80) / 3,
    }


def test_proposal_checkpoint_score_prefers_weak_domain_coverage() -> None:
    first = {
        "proposal_summary": {
            "minimum_domain_joint_proposal_recall": 0.81,
            "minimum_domain_bond_proposal_recall": 0.81,
            "minimum_domain_atom_proposal_recall": 0.90,
            "mean_domain_bond_proposal_recall": 0.90,
            "mean_domain_atom_proposal_recall": 0.95,
        }
    }
    second = {
        "proposal_summary": {
            "minimum_domain_joint_proposal_recall": 0.82,
            "minimum_domain_bond_proposal_recall": 0.82,
            "minimum_domain_atom_proposal_recall": 0.89,
            "mean_domain_bond_proposal_recall": 0.89,
            "mean_domain_atom_proposal_recall": 0.94,
        }
    }

    assert proposal_checkpoint_score(second) > proposal_checkpoint_score(first)


def test_freeze_proposal_object_features_keeps_output_heads_trainable() -> None:
    atom = SpatialAtomObjectDetector(SpatialAtomObjectConfig())
    bond = MultiScaleDenseBondObjectDetector(DenseBondObjectConfig())

    freeze_proposal_object_features(atom, bond)

    assert not any(parameter.requires_grad for parameter in atom.context.parameters())
    assert not any(parameter.requires_grad for parameter in bond.base.context.parameters())
    assert not any(parameter.requires_grad for parameter in bond.stride8_adapter.parameters())
    assert any(parameter.requires_grad for parameter in atom.heatmap.parameters())
    assert any(parameter.requires_grad for parameter in bond.base.heatmap.parameters())
    assert any(parameter.requires_grad for parameter in bond.base.display.parameters())
