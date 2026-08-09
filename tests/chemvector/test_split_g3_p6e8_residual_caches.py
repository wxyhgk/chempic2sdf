import torch

from chemvector.evaluation.audits.atoms.split_g3_p6e8_residual_caches import split_cache


def test_p6e8_split_cache_keeps_only_requested_rows() -> None:
    cache = {
        "split": ["train", "holdout"],
        "image_crops": torch.arange(2)[:, None],
        "score_crops": torch.arange(2)[:, None],
        "candidate_offsets": torch.zeros(2, 8, 2),
        "candidate_peak_logits": torch.zeros(2, 8),
        "candidate_mask": torch.ones(2, 8, dtype=torch.bool),
        "offset_targets": torch.zeros(2, 2),
        "sample_key": ["a", "b"], "molecule_id": ["m1", "m2"], "scaffold": ["s1", "s2"],
        "variant": ["rdkit", "stress"], "target_role": ["TEXT", "IMPLICIT"],
        "degree_bucket": ["degree1", "degree2"], "target_index": [1, 2], "required_residual_cap": [0.0, 1.0],
        "p6e1_checkpoint_sha256": "a", "eligible_manifest_sha256": "b", "vision_checkpoint_sha256": "c",
    }
    result = split_cache(cache, "holdout")
    assert result["sample_key"] == ["b"]
    assert result["image_crops"].flatten().tolist() == [1]
