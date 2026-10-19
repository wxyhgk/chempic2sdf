"""Frozen-initial, forward-only audit for a text endpoint-pair field.

This is a quarantine diagnostic, not a training launcher.  It replays a small
deterministic slice of the public training cohort with ``initial.pt`` and
records whether raw-missing visible O/N atoms have usable text-center cells
and center-to-anchor offsets.  No optimizer, checkpoint, selector update, or
sealed identity is touched.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.components.graph.joint.predicted_incidence import PredictedProposalConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets


ROOT = Path(__file__).resolve().parents[1]
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
COHORT = ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json"
DEFAULT_OUTPUT = ROOT / "outputs/audits/chemvector-text-endpoint-pair-admission-20261006"
PROFILES = ("font_large", "serif")
IMAGE_SIZE = 768.0
FIELD_SIZE = 192
OFFSET_MAX_PX = 32.0


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values.flatten(), descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(order.numel(), device=values.device)
    return ranks.reshape(values.shape) + 1


def _select_rows(rows: list[dict], count: int) -> list[dict]:
    """Stable balanced selection; does not inspect image bytes or sealed data."""
    grouped = {profile: [] for profile in PROFILES}
    for row in rows:
        # The release manifest intentionally carries identity-only fields;
        # profile is bound in the sample key/meta and is not trusted from a
        # caller-provided extra column.
        profile = str(row["sample_key"]).rsplit("__", 1)[-1]
        if profile in grouped:
            grouped[profile].append(row)
    for profile in grouped:
        grouped[profile].sort(key=lambda row: hashlib.sha256(
            row["sample_key"].encode("utf-8")).hexdigest())
    quota, remainder = divmod(count, len(PROFILES))
    selected: list[dict] = []
    for index, profile in enumerate(PROFILES):
        take = quota + int(index < remainder)
        selected.extend(grouped[profile][:take])
    selected.sort(key=lambda row: row["sample_key"])
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} rows available for requested {count}")
    return selected


def _cell(point: torch.Tensor, size: int) -> tuple[int, int]:
    x = min(size - 1, max(0, int(float(point[0]) * size)))
    y = min(size - 1, max(0, int(float(point[1]) * size)))
    return x, y


def _replay_row(model, config, payload, loader, collator, manifest, row) -> dict:
    sample = loader(manifest, row)
    batch = collator([sample])
    tensors = {key: value.cuda() for key, value in batch.items() if isinstance(value, torch.Tensor)}
    with torch.no_grad():
        _, atom_output, _, raw, _, _, _ = structure_forward(
            model, tensors["pixel_values"], config,
            float(payload["provenance"]["atom_threshold"]),
        )
    mapping, _ = match_atom_targets(raw, tensors, config)
    matched = {int(value) for value in mapping[0].tolist() if int(value) >= 0}
    target_count = int(tensors["atom_mask"].sum())
    logits = atom_output["heatmap_logits"][0, 0].float()
    ranks = _rank(logits.sigmoid())
    peaks = logits.sigmoid() == F.max_pool2d(logits[None, None], 3, 1, 1)[0, 0]
    h, w = logits.shape
    anchors = tensors["atom_anchors"][0, :target_count]
    centers = tensors["atom_text_center"][0, :target_count]
    text_mask = tensors["atom_text_mask"][0, :target_count]
    elements = [str(atom["element"]) for atom in sample["layers"]["atoms"][:target_count]]
    records: list[dict] = []
    center_cells: list[tuple[int, int]] = []
    for target_id in range(target_count):
        if target_id in matched or not bool(text_mask[target_id]) or elements[target_id] not in {"O", "N"}:
            continue
        center_cell = _cell(centers[target_id], h)
        anchor_cell = _cell(anchors[target_id], h)
        cy, cx = center_cell[1], center_cell[0]
        center_index = cy * w + cx
        # The frozen atom field's anchor map is only an observation.  It is
        # deliberately not used as a training target or injected into model input.
        predicted_anchor = atom_output["anchors"][0, :, cy, cx]
        offset_distance = float(torch.linalg.vector_norm(predicted_anchor - anchors[target_id]) * IMAGE_SIZE)
        center_anchor_distance = float(torch.linalg.vector_norm(centers[target_id] - anchors[target_id]) * IMAGE_SIZE)
        center_cells.append(center_cell)
        records.append({
            "target_id": target_id,
            "element": elements[target_id],
            "center_cell": list(center_cell),
            "anchor_cell": list(anchor_cell),
            "center_rank": int(ranks[cy, cx]),
            "center_is_local_peak": bool(peaks[cy, cx]),
            "center_anchor_distance_px": center_anchor_distance,
            "predicted_anchor_offset_distance_px": offset_distance,
            "offset_label_valid": center_anchor_distance <= OFFSET_MAX_PX,
            "strict_frozen_proxy": bool(peaks[cy, cx] and offset_distance <= 6.0),
        })
    collisions = len(center_cells) - len(set(center_cells))
    return {
        "sample_key": row["sample_key"],
        "profile_id": str(sample["meta"]["profile_id"]),
        "target_count": target_count,
        "raw_matched": len(matched),
        "raw_missing_text_on": records,
        "text_on_raw_gap_count": len(records),
        "text_center_cell_collisions": collisions,
    }


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or initial checkpoint contract unavailable")
    fixed = json.loads(COHORT.read_text(encoding="utf-8"))
    if fixed.get("sealed_test_read"):
        raise ValueError("Cohort is not training-only")
    fixed_keys = {row["sample_key"] for row in fixed["per_sample"]}
    sources = source_checks()
    manifest = sources["manifest"]
    rows = [row for row in manifest["samples"] if row["sample_key"] in fixed_keys]
    selected = _select_rows(rows, args.max_images)
    model, model_config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    proposal_config = PredictedProposalConfig(
        maximum_atoms=128,
        maximum_bonds=model_config.maximum_bonds,
        atom_threshold=0.01,
        bond_threshold=0.01,
        atom_match_radius_px=6.0,
        bond_match_radius_px=12.0,
        image_size=768.0,
    )
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=model_config.maximum_atoms,
                            bond_queries=model_config.maximum_bonds),
        purpose="training",
    )
    records = []
    with torch.no_grad():
        for index, row in enumerate(selected, 1):
            records.append(_replay_row(model, proposal_config, payload, loader, collator, manifest, row))
            if index % 16 == 0:
                print(json.dumps({"processed": index, "total": len(selected)}), flush=True)
    all_targets = [target for record in records for target in record["raw_missing_text_on"]]
    by_profile = Counter(record["profile_id"] for record in records)
    positives = len(all_targets)
    strict = sum(int(target["strict_frozen_proxy"]) for target in all_targets)
    valid_offsets = sum(int(target["offset_label_valid"]) for target in all_targets)
    collisions = sum(int(record["text_center_cell_collisions"]) for record in records)
    result = {
        "schema_version": "chemvector-text-endpoint-pair-admission-0.1",
        "complete": True,
        "forward_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA,
        "images": len(records),
        "profiles": dict(by_profile),
        "summary": {
            "raw_missing_o_n_targets": positives,
            "strict_frozen_proxy": strict,
            "strict_frozen_proxy_precision": strict / max(positives, 1),
            "valid_center_anchor_offsets": valid_offsets,
            "offset_label_fraction": valid_offsets / max(positives, 1),
            "center_cell_collisions": collisions,
        },
        "contract": {
            "baseline": "frozen initial raw proposals; no selector or sidecar",
            "target": "raw-missing visible O/N only",
            "field_size": [FIELD_SIZE, FIELD_SIZE],
            "offset_limit_px": OFFSET_MAX_PX,
            "sealed_test_queries": 0,
        },
        "records": records,
        "gate": {
            "minimum_raw_missing_o_n_50": positives >= 50,
            "offset_labels_valid": valid_offsets == positives,
            "center_cell_collision_free": collisions == 0,
            "forward_only": True,
            "bounded_training_authorized": False,
            "pass": positives >= 50 and valid_offsets == positives and collisions == 0,
        },
    }
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": result["summary"], "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    if not 1 <= args.max_images <= 64:
        raise SystemExit("--max-images must be in [1, 64]")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
