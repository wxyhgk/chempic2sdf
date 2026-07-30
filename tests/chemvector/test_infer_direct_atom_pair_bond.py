from torch import nn
import torch

from chemvector.infer_direct_atom_pair_bond import (
    apply_atom_ocr,
    apply_structural_twin_suppression,
    direct_scene_tensors,
)


class _Ranking(nn.Module):
    def __init__(self, *, first_rank: float = 8.0) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()))
        self.first_rank = first_rank

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        rows = len(features)
        rank = torch.arange(rows, 0, -1, device=features.device).float()
        rank[0] = self.first_rank
        utility = torch.zeros((rows, 4), device=features.device)
        utility[:, 3] = 10.0
        return {
            "safety_logits": torch.full((rows,), 10.0, device=features.device),
            "utility_logits": utility,
            "rank_logits": rank,
        }


def _duplicate_live_outputs(*, chemical_mismatch: bool = False):
    atom_mask = torch.tensor([[False, True, False, True, False, True]])
    atom_features = torch.zeros((1, 6, 128))
    atom_geometry = torch.zeros((1, 6, 8))
    atom_geometry[0, 1, :2] = torch.tensor([0.25, 0.25])
    atom_geometry[0, 3, :2] = torch.tensor([0.251, 0.25])
    atom_geometry[0, 5, :2] = torch.tensor([0.55, 0.25])
    atoms = {
        "atom_mask": atom_mask,
        "atom_features": atom_features,
        "atom_geometry": atom_geometry,
        "atom_scores": torch.full((1, 6), 0.5),
    }
    pairs = torch.tensor([[1, 3], [1, 5], [3, 5]])
    chemical = torch.tensor(
        [[8.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]]
    )
    if chemical_mismatch:
        chemical[2] = torch.tensor([0.0, 8.0, 0.0, 0.0])
    output = {
        "presence_logits": torch.tensor([-8.0, 8.0, 8.0]),
        "chemical_logits": chemical,
        "hidden": torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4),
    }
    return atoms, pairs, output


def test_direct_scene_tensors_encode_pair_incidence_without_distance_filter() -> None:
    atoms = {
        "atom_mask": torch.tensor([[True, True, False, True]]),
        "atom_scores": torch.tensor([[0.9, 0.8, -1.0, 0.7]]),
    }
    pairs = torch.tensor([[0, 1], [0, 3], [1, 3]])
    output = {
        "presence_logits": torch.tensor([2.0, -1.0, 3.0]),
        "chemical_logits": torch.tensor(
            [[4.0, 1.0, 0.0, 0.0], [1.0, 4.0, 0.0, 0.0], [0.0, 1.0, 4.0, 0.0]]
        ),
    }

    result = direct_scene_tensors(atoms, pairs, output)

    assert result["bond_mask"].tolist() == [True, True, True]
    assert result["atom_presence_logits"].tolist() == [1e4, 1e4, -1e4, 1e4]
    assert result["incidence_logits"].argmax(-1).tolist() == [[0, 1], [0, 3], [1, 3]]
    assert result["bond_types"].shape == (3, 8)
    assert torch.allclose(result["bond_types"][:, :4], result["bond_types"][:, 4:])


def test_direct_scene_tensors_only_overrides_changed_chemical_predictions() -> None:
    atoms = {
        "atom_mask": torch.tensor([[True, True, True]]),
        "atom_scores": torch.tensor([[0.9, 0.8, 0.7]]),
    }
    pairs = torch.tensor([[0, 1], [1, 2]])
    output = {
        "presence_logits": torch.tensor([2.0, 2.0]),
        "chemical_logits": torch.tensor([[4.0, 1.0, 0.0, 0.0], [4.0, 1.0, 0.0, 0.0]]),
    }

    result = direct_scene_tensors(atoms, pairs, output, torch.tensor([0, 3]))

    assert result["bond_types"][0, :4].argmax().item() == 0
    assert result["bond_types"][1, :4].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_apply_atom_ocr_updates_chemistry_and_reclips_ports() -> None:
    scene = {
        "schema_version": "chemscene-0.2",
        "molecule_id": "test",
        "units": "q1023",
        "canvas": {"width": 768, "height": 768},
        "atom_objects": [
            {"id": 0, "anchor": [200, 500], "role": "IMPLICIT",
             "chemical": {"element": "C", "formal_charge": 0, "isotope": 0,
                          "aromatic": False, "explicit_h_count": 0, "no_implicit": False,
                          "radical_electrons": 0, "canonical_chiral_tag": "CHI_UNSPECIFIED"},
             "text": None, "incident_ports": []},
            {"id": 1, "anchor": [800, 500], "role": "IMPLICIT",
             "chemical": {"element": "C", "formal_charge": 0, "isotope": 0,
                          "aromatic": False, "explicit_h_count": 0, "no_implicit": False,
                          "radical_electrons": 0, "canonical_chiral_tag": "CHI_UNSPECIFIED"},
             "text": None, "incident_ports": []},
        ],
        "bond_objects": [
            {"id": 0, "atom_refs": [0, 1], "ports": [[200, 500], [800, 500]],
             "display_type": "SINGLE", "chemical_type": "SINGLE", "direction": "NONE",
             "stereo": "STEREONONE", "bbox": None}
        ],
    }
    result = apply_atom_ocr(
        scene, [4, 7],
        {4: {"text": "NH2+", "score": -0.1, "center": [0.2, 0.5],
             "half_extent": [0.04, 0.02]}},
    )
    assert result["parsed_text"] == 1
    assert scene["atom_objects"][0]["chemical"]["element"] == "N"
    assert scene["atom_objects"][0]["chemical"]["formal_charge"] == 1
    assert scene["atom_objects"][0]["chemical"]["explicit_h_count"] == 2
    assert scene["bond_objects"][0]["ports"][0] != [200, 500]


def test_structural_twin_suppression_remaps_queries_and_drops_incident_pairs() -> None:
    atoms, pairs, output = _duplicate_live_outputs()
    filtered_pairs, filtered_output, diagnostics = apply_structural_twin_suppression(
        atoms,
        pairs,
        output,
        _Ranking(),
        presence_threshold=0.85,
    )
    assert diagnostics["executed"]
    assert diagnostics["removed_candidate_index"] == 1
    assert diagnostics["removed_compact_index"] == 0
    assert diagnostics["witness_compact_indices"] == [1]
    assert atoms["atom_mask"].tolist() == [[False, False, False, True, False, True]]
    assert filtered_pairs.tolist() == [[3, 5]]
    assert filtered_output["presence_logits"].tolist() == [8.0]
    assert filtered_output["hidden"].tolist() == output["hidden"][2:].tolist()


def test_structural_twin_suppression_protects_chemical_mismatch() -> None:
    atoms, pairs, output = _duplicate_live_outputs(chemical_mismatch=True)
    filtered_pairs, filtered_output, diagnostics = apply_structural_twin_suppression(
        atoms,
        pairs,
        output,
        _Ranking(),
        presence_threshold=0.85,
    )
    assert not diagnostics["executed"]
    assert diagnostics["certified_actions"] == 0
    assert torch.equal(filtered_pairs, pairs)
    assert filtered_output is output
