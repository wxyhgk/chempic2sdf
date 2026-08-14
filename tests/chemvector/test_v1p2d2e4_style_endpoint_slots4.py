import torch

from chemvector.training.bonds.objects.train_v1p2d2e3_style_endpoint_field import (
    INITIAL_CHECKPOINT,
)
from chemvector.training.bonds.objects.train_v1p2d2e4_style_endpoint_slots4 import (
    expand_detector_slots,
)


def test_slots4_expansion_preserves_all_existing_output_slots() -> None:
    initial = torch.load(INITIAL_CHECKPOINT, map_location="cpu")
    detector = expand_detector_slots(initial)
    state = detector.state_dict()
    assert detector.config.slots == 4
    assert torch.equal(state["heatmap.weight"][:3], initial["detector"]["heatmap.weight"])
    assert torch.equal(state["heatmap.bias"][:3], initial["detector"]["heatmap.bias"])
    assert torch.equal(state["offset.weight"][:6], initial["detector"]["offset.weight"])
    assert torch.equal(state["offset.bias"][:6], initial["detector"]["offset.bias"])
    assert state["heatmap.weight"].shape[0] == 4
    assert state["offset.weight"].shape[0] == 8
