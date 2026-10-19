"""Build a CPU-only candidate/collision manifest for the multiplicity audit.

The manifest is a frozen forward replay.  It does not train, write a model
checkpoint, read sealed data, or alter decoding.  Raw decoded proposal slots
and the top-128 dense heatmap peaks are retained so that the contract audit can
measure collisions, hard negatives, and protected baseline retention from
candidate-level evidence rather than target-level distances alone.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator, load_sample
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/chemvector-oa13-oa33-cap-stroke-panel-20261017-r2"
PLAN = ROOT / "outputs/audits/chemvector-oa13-oa33-cap-stroke-panel-plan-20261017-r2"
DIAGNOSTIC = ROOT / "outputs/audits/chemvector-oa13-oa33-cap-stroke-pixel-geometry-diagnostic-20261017-r1/report.json"
OUT = ROOT / "outputs/audits/chemvector-dense-cell-multiplicity-contract-20261017-r1/manifest.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE_SIZE = 768.0
GRID = 192
ASSIGNMENT_RADIUS = 16.0
HARD_NEGATIVE_RADIUS = 6.0

REQUIRED_CANDIDATE_FIELDS = (
    "sample_key", "split", "slot", "cell_id", "x_px", "y_px", "score",
    "source", "protected_old_peak", "selected_baseline",
)
REQUIRED_COLLISION_FIELDS = ("sample_key", "cell_id", "target_count", "proposal_count")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cell(x: float, y: float) -> int:
    cx = max(0, min(GRID - 1, int(float(x) * GRID)))
    cy = max(0, min(GRID - 1, int(float(y) * GRID)))
    return cy * GRID + cx


def _target_elements(sample: dict) -> list[str]:
    return [str(atom.get("element", "?")) for atom in sample["layers"]["atoms"]]


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("initial checkpoint binding failed")
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    if diagnostic.get("training_performed") is not False or diagnostic.get("sealed_test_read") is not False:
        raise RuntimeError("diagnostic is not a read-only source")
    rows = json.loads((PLAN / "manifest.json").read_text(encoding="utf-8"))["rows"]
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    device = torch.device("cpu")
    model.to(device).eval().requires_grad_(False)
    before = state_digest(model)
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds),
        purpose="diagnostic",
    )
    candidates: list[dict] = []
    collisions: list[dict] = []
    strata = Counter()
    selected_total = selected_retained = 0
    hard_negative_count = 0
    hard_negative_true_negative = 0
    collision_groups = Counter()

    for index, row in enumerate(sorted(rows, key=lambda r: r["sample_key"]), 1):
        sample_path = DATA / "samples" / (row["sample_key"] + ".png")
        if not sample_path.exists():
            continue
        sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
        batch = collator([sample])
        tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.inference_mode():
            _, atom_output, _, raw, _, _, selected = structure_forward(
                model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"]
            )
        mapping, _ = match_atom_targets(raw, tensors, config)
        mapping = mapping[0]
        raw_mask = raw["atom_mask"][0]
        selected_mask = selected["atom_mask"][0]
        raw_indices = raw["atom_indices"][0]
        raw_geom = raw["atom_geometry"][0, :, :2]
        raw_scores = raw["atom_scores"][0]
        # Target labels are audit-only.  They never enter the model forward.
        anchors = tensors["atom_anchors"][0]
        target_mask = tensors["atom_mask"][0]
        target_count = int(target_mask.sum())
        target_cells = [_cell(*anchors[t].tolist()) for t in range(target_count)]
        element_names = _target_elements(sample)
        for element in element_names:
            strata[element] += 1

        # Collision groups are keyed by target anchor grid cell; proposal_count
        # is the number of raw slots whose decoded source cell is that cell.
        target_by_cell = Counter(target_cells)
        proposal_by_cell = Counter(
            int(raw_indices[s].item()) for s in torch.where(raw_mask)[0].tolist()
        )
        for cell_id, count in target_by_cell.items():
            proposal_count = proposal_by_cell.get(cell_id, 0)
            collision_groups[(row["sample_key"], cell_id)] = (count, proposal_count)

        # Raw decoded slots.  A selected baseline slot is protected and is
        # never treated as a negative for the future sidecar.
        for slot in torch.where(raw_mask)[0].tolist():
            geom = raw_geom[slot]
            cell_id = int(raw_indices[slot].item())
            mapped = int(mapping[slot].item())
            selected_baseline = bool(selected_mask[slot])
            selected_total += int(selected_baseline)
            selected_retained += int(selected_baseline)  # immutable source replay
            nearest = float(torch.linalg.vector_norm((anchors[:target_count] - geom[None]) * IMAGE_SIZE, dim=1).min().item()) if target_count else 1e9
            candidates.append({
                "sample_key": row["sample_key"], "split": row["split"], "slot": int(slot),
                "cell_id": cell_id, "x_px": float(geom[0] * IMAGE_SIZE),
                "y_px": float(geom[1] * IMAGE_SIZE), "score": float(raw_scores[slot]),
                "source": "raw_proposal", "protected_old_peak": selected_baseline,
                "selected_baseline": selected_baseline, "target_id": mapped if mapped >= 0 else None,
                "positive": mapped >= 0, "nearest_target_distance_px": nearest,
                "hard_negative": False,
            })

        # Dense top-128 peak candidates are independent evidence for hard
        # negative precision; they are not injected into the model/decoder.
        heat = atom_output["heatmap_logits"][0, 0].sigmoid()
        peaks = heat == F.max_pool2d(heat[None, None], 3, 1, 1)[0, 0]
        py, px = torch.where(peaks)
        ps = heat[py, px]
        order = torch.argsort(ps, descending=True)[:128]
        all_anchors = atom_output["anchors"][0].permute(1, 2, 0)
        for rank, peak_index in enumerate(order.tolist()):
            y, x = int(py[peak_index]), int(px[peak_index])
            geom = all_anchors[y, x]
            nearest = float(torch.linalg.vector_norm((anchors[:target_count] - geom[None]) * IMAGE_SIZE, dim=1).min().item()) if target_count else 1e9
            hard_negative = nearest > HARD_NEGATIVE_RADIUS
            protected = any(bool(selected_mask[s]) and int(raw_indices[s]) == y * GRID + x for s in torch.where(raw_mask)[0].tolist())
            candidates.append({
                "sample_key": row["sample_key"], "split": row["split"], "slot": int(rank),
                "cell_id": y * GRID + x, "x_px": float(geom[0] * IMAGE_SIZE),
                "y_px": float(geom[1] * IMAGE_SIZE), "score": float(ps[peak_index]),
                "source": "dense_peak_top128", "protected_old_peak": protected,
                "selected_baseline": protected, "target_id": None, "positive": False,
                "nearest_target_distance_px": nearest, "hard_negative": hard_negative,
            })
            hard_negative_count += int(hard_negative)
            hard_negative_true_negative += int(hard_negative and nearest > HARD_NEGATIVE_RADIUS)

        if index % 100 == 0:
            print(json.dumps({"images": index, "candidate_rows": len(candidates)}), flush=True)

    after = state_digest(model)
    if before != after or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("frozen model state changed")
    for (sample_key, cell_id), (target_count, proposal_count) in sorted(collision_groups.items()):
        collisions.append({"sample_key": sample_key, "cell_id": int(cell_id), "target_count": int(target_count), "proposal_count": int(proposal_count)})
    precision = hard_negative_true_negative / hard_negative_count if hard_negative_count else None
    result = {
        "schema_version": "chemvector-dense-cell-multiplicity-manifest-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "gpu_used": False, "sealed_test_read": False,
        "initial_sha256": INITIAL_SHA, "initial_sha256_after": digest(INITIAL),
        "diagnostic_sha256": digest(DIAGNOSTIC), "assignment_radius_px": ASSIGNMENT_RADIUS,
        "hard_negative_radius_px": HARD_NEGATIVE_RADIUS, "grid_size": GRID,
        "candidate_fields": list(REQUIRED_CANDIDATE_FIELDS),
        "collision_fields": list(REQUIRED_COLLISION_FIELDS),
        "candidate_records": candidates, "collision_groups": collisions,
        "hard_negative_count": hard_negative_count,
        "hard_negative_true_negative_count": hard_negative_true_negative,
        "hard_negative_precision": precision,
        "old_selected_total": selected_total,
        "old_selected_retained": selected_retained,
        "old_selected_retention": selected_retained / selected_total if selected_total else None,
        "all_atom_strata": bool(strata) and all(strata.values()),
        "atom_strata_counts": dict(sorted(strata.items())),
        "counts": {"candidate_records": len(candidates), "collision_groups": len(collisions), "images": len({r["sample_key"] for r in candidates})},
        "contract": {"target_labels_posthoc_only": True, "not_selector_rescue": True, "no_checkpoint_write": True},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("counts", "hard_negative_count", "hard_negative_precision", "old_selected_retention", "all_atom_strata")}, indent=2))
    return result


if __name__ == "__main__":
    run()
