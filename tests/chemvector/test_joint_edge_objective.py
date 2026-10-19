import torch

from chemvector.training.curriculum.joint_edge_objective import joint_edge_loss


def _inputs():
    output = {
        "presence_logits": torch.tensor([0.2, -0.4, 0.7, 1.1], requires_grad=True),
        "chemical_logits": torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], requires_grad=True),
    }
    known = torch.tensor([True, True, False, True])
    presence = torch.tensor([True, False, False, True])
    chemical = torch.tensor([0, 1, 2, 3])
    return output, known, presence, chemical


def test_joint_objective_masks_unknown_and_separates_terms():
    output, known, presence, chemical = _inputs()
    loss, details = joint_edge_loss(output, known, presence, chemical)
    assert loss is not None
    assert details["known_rows"] == 3
    assert details["positive_rows"] == 2
    assert details["negative_rows"] == 1
    assert details["presence_only"] is False
    loss.backward()
    # Unknown row 2 must not receive either presence or chemical supervision.
    assert output["presence_logits"].grad[2].item() == 0.0
    assert torch.equal(output["chemical_logits"].grad[2], torch.zeros(4))
    # Confirmed non-edge receives presence gradient; positive receives chemical.
    assert output["presence_logits"].grad[1].item() != 0.0
    assert output["chemical_logits"].grad[0].abs().sum().item() > 0.0


def test_joint_objective_replay_and_aromatic_require_matching_context():
    output, known, presence, chemical = _inputs()
    ref_presence = (output["presence_logits"].detach() + 0.3).requires_grad_()
    ref_chemical = output["chemical_logits"].detach().requires_grad_()
    pairs = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=torch.long)
    samples = torch.zeros(4, dtype=torch.long)
    loss, details = joint_edge_loss(
        output, known, presence, chemical,
        pair_indices=pairs, sample_indices=samples,
        replay_weight=0.2, aromatic_safety_weight=0.5,
        reference_presence_logits=ref_presence,
        reference_chemical_logits=ref_chemical,
    )
    assert loss is not None and torch.isfinite(loss)
    assert details["replay_loss"] >= 0.0
    assert details["aromatic_safety_loss"] >= 0.0
    loss.backward()
    assert ref_presence.grad is None
    assert ref_chemical.grad is None


def test_replay_is_zero_at_identity_and_positive_after_perturbation():
    output, known, presence, chemical = _inputs()
    ref_presence = output["presence_logits"].detach()
    ref_chemical = output["chemical_logits"].detach()
    same, same_details = joint_edge_loss(
        output, known, presence, chemical, replay_weight=1.0,
        reference_presence_logits=ref_presence,
        reference_chemical_logits=ref_chemical,
    )
    assert same is not None and same_details["replay_loss"] < 1e-7
    perturbed = {
        "presence_logits": output["presence_logits"] + 0.5,
        "chemical_logits": output["chemical_logits"] + torch.tensor(0.0),
    }
    changed, changed_details = joint_edge_loss(
        perturbed, known, presence, chemical, replay_weight=1.0,
        reference_presence_logits=ref_presence,
        reference_chemical_logits=ref_chemical,
    )
    assert changed is not None and changed_details["replay_loss"] > 0.0


def test_chemical_replay_ignores_confirmed_non_edges():
    output, known, presence, chemical = _inputs()
    ref_presence = output["presence_logits"].detach()
    ref_chemical = output["chemical_logits"].detach()
    perturbed = {
        "presence_logits": output["presence_logits"].detach(),
        "chemical_logits": output["chemical_logits"].detach().clone(),
    }
    # Row 1 is a confirmed non-edge.  Its chemical field is undefined and
    # must not contribute to replay; perturbing it leaves the replay value
    # unchanged when positive rows are held fixed.
    perturbed["chemical_logits"].data[1] += torch.tensor([4.0, -3.0, 2.0, -1.0])
    base, base_details = joint_edge_loss(
        output, known, presence, chemical, replay_weight=1.0,
        reference_presence_logits=ref_presence,
        reference_chemical_logits=ref_chemical,
    )
    changed, changed_details = joint_edge_loss(
        perturbed, known, presence, chemical, replay_weight=1.0,
        reference_presence_logits=ref_presence,
        reference_chemical_logits=ref_chemical,
    )
    assert base is not None and changed is not None
    assert abs(base_details["replay_loss"] - changed_details["replay_loss"]) < 1e-7


def test_aromatic_anchor_is_directional_and_positive_only():
    output, known, presence, chemical = _inputs()
    pairs = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=torch.long)
    samples = torch.zeros(4, dtype=torch.long)
    ref_presence = output["presence_logits"].detach()
    reference = output["chemical_logits"].detach().clone()
    # Row 3 is the only confirmed aromatic positive.  Lowering its current
    # aromatic probability should activate the anchor; improving it should not.
    chemical_targets = chemical.clone(); chemical_targets[3] = 3
    current = output["chemical_logits"].detach().clone().requires_grad_()
    low = current.clone(); low[3] = torch.tensor([4.0, 4.0, 4.0, -4.0]); low.requires_grad_()
    loss_low, details_low = joint_edge_loss(
        {"presence_logits": output["presence_logits"].detach(), "chemical_logits": low},
        known, presence, chemical_targets, pair_indices=pairs, sample_indices=samples,
        aromatic_anchor_weight=1.0, reference_presence_logits=ref_presence,
        reference_chemical_logits=reference,
    )
    assert loss_low is not None and details_low["aromatic_anchor_loss"] > 0.0
    high = current.clone(); high[3] = torch.tensor([-4.0, -4.0, -4.0, 4.0]); high.requires_grad_()
    loss_high, details_high = joint_edge_loss(
        {"presence_logits": output["presence_logits"].detach(), "chemical_logits": high},
        known, presence, chemical_targets, pair_indices=pairs, sample_indices=samples,
        aromatic_anchor_weight=1.0, reference_presence_logits=ref_presence,
        reference_chemical_logits=reference,
    )
    assert loss_high is not None and details_high["aromatic_anchor_loss"] == 0.0


def test_aromatic_ring_safety_has_directional_gradient():
    pairs = torch.tensor([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 5],
    ], dtype=torch.long)
    output = {
        "presence_logits": torch.full((6,), 4.0, requires_grad=True),
        "chemical_logits": torch.tensor([[0.0, 0.0, 0.0, -4.0]] * 6, requires_grad=True),
    }
    known = torch.ones(6, dtype=torch.bool)
    presence = torch.ones(6, dtype=torch.bool)
    chemical = torch.full((6,), 3, dtype=torch.long)
    samples = torch.zeros(6, dtype=torch.long)
    loss, details = joint_edge_loss(
        output, known, presence, chemical, pair_indices=pairs,
        sample_indices=samples, aromatic_safety_weight=1.0,
    )
    assert loss is not None and details["rings"] == 1
    assert details["aromatic_safety_loss"] > 0.0
    loss.backward()
    assert output["chemical_logits"].grad[:, 3].mean().item() < 0.0


def test_aromatic_anchor_only_protects_confirmed_aromatic_targets():
    """The asymmetric anchor must not invent labels for non-aromatic rows.

    A reference aromatic probability of .9 should push a current .1
    aromatic probability upward when the row is a confirmed aromatic edge.
    Confirmed single-bond rows and unknown rows must receive no anchor
    gradient, even when their reference logits are aromatic-heavy.
    """
    output = {
        "presence_logits": torch.zeros(3, requires_grad=True),
        "chemical_logits": torch.tensor([
            [-2.0, -2.0, -2.0, 2.0],  # current aromatic-heavy target row
            [2.0, -2.0, -2.0, -2.0],  # current single-heavy non-aromatic row
            [-2.0, -2.0, -2.0, 2.0],  # unknown row (must be ignored)
        ], requires_grad=True),
    }
    # Make the aromatic target row current aromatic probability low, while
    # keeping the reference high enough to activate the asymmetric hinge.
    output["chemical_logits"].data[0] = torch.tensor([2.0, 2.0, 2.0, -2.0])
    reference = torch.tensor([
        [-2.0, -2.0, -2.0, 2.0],
        [-2.0, -2.0, -2.0, 2.0],
        [-2.0, -2.0, -2.0, 2.0],
    ])
    known = torch.tensor([True, True, False])
    presence = torch.tensor([True, True, False])
    chemical = torch.tensor([3, 0, 3])
    loss, details = joint_edge_loss(
        output, known, presence, chemical,
        aromatic_anchor_weight=1.0,
        reference_chemical_logits=reference,
    )
    assert loss is not None
    assert details["aromatic_anchor_loss"] > 0.0
    loss.backward()
    grad = output["chemical_logits"].grad
    assert grad[0, 3].item() < 0.0
    assert torch.equal(grad[2], torch.zeros_like(grad[2]))


def test_joint_objective_rejects_unknown_only_batch():
    output, _, presence, chemical = _inputs()
    loss, details = joint_edge_loss(
        output, torch.zeros(4, dtype=torch.bool), presence, chemical,
    )
    assert loss is None
    assert details["known_rows"] == 0


def test_joint_objective_keeps_presence_only_known_negative_batch():
    output, _, _, chemical = _inputs()
    known = torch.tensor([True, True, False, False])
    presence = torch.tensor([False, False, False, False])
    loss, details = joint_edge_loss(output, known, presence, chemical)
    assert loss is not None
    assert details["presence_only"] is True
    loss.backward()
    assert output["presence_logits"].grad[:2].abs().sum().item() > 0.0
    assert output["chemical_logits"].grad is None
