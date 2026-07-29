import torch

from chemvector.infer_direct_atom_pair_bond import apply_atom_ocr, direct_scene_tensors


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
