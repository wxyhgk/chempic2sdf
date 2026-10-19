"""Construct a bijective O/N candidate residual manifest (CPU/read-only).

No model is loaded.  Target anchors come from the renderer-derived collator
metadata and candidate coordinates come from the frozen raw-proposal manifest.
Pairs are admitted only when geometric ownership is unique; nearest/Hungarian
assignment is never forced.  Ambiguous rows are recorded as unknown and cannot
be used for training.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.data import CurriculumCollator, load_sample


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/chemvector-oa13-oa33-cap-stroke-panel-20261017-r2"
PLAN = ROOT / "outputs/audits/chemvector-oa13-oa33-cap-stroke-panel-plan-20261017-r2/manifest.json"
SOURCE = ROOT / "outputs/audits/chemvector-dense-cell-multiplicity-contract-20261017-r1/manifest.json"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-visible-text-20261017-r1/manifest.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE_SIZE = 768.0
RADIUS = 16.0
STRICT = 6.0
OTHER_TARGET_MIN = 20.0
TARGET_MARGIN = 4.0


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if source.get("training_performed") is not False or source.get("gpu_used") is not False:
        raise ValueError("source candidate manifest is not read-only CPU evidence")
    if source.get("initial_sha256") != INITIAL_SHA:
        raise ValueError("source initial checkpoint binding failed")
    plan_rows = json.loads(PLAN.read_text(encoding="utf-8"))["rows"]
    candidate_rows = [r for r in source["candidate_records"] if r.get("source") == "raw_proposal"]
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in candidate_rows:
        if not row.get("protected_old_peak"):
            by_sample[row["sample_key"]].append(row)
    plan_by_key = {r["sample_key"]: r for r in plan_rows}
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=128, bond_queries=192), purpose="diagnostic")
    pairs: list[dict] = []
    unknown = Counter()
    stratum = Counter()
    molecule_splits: dict[str, set[str]] = defaultdict(set)
    image_count = 0
    for sample_key, candidates in sorted(by_sample.items()):
        plan_row = plan_by_key[sample_key]
        molecule_splits[str(plan_row["molecule_id"])].add(str(plan_row["split"]))
        sample = load_sample(DATA, sample_key, purpose="diagnostic")
        batch = collator([sample])
        anchors = batch["atom_anchors"][0]
        target_count = int(batch["atom_mask"][0].sum())
        atoms = sample["layers"]["atoms"]
        target_ids = [i for i in range(target_count)
                      if i < len(atoms) and atoms[i].get("element") in {"O", "N"}
                      and bool(atoms[i].get("visible_text"))]
        if not target_ids:
            continue
        image_count += 1
        coords = torch.tensor([[float(r["x_px"]) / IMAGE_SIZE, float(r["y_px"]) / IMAGE_SIZE]
                               for r in candidates], dtype=torch.float32)
        target_xy = anchors[:target_count].float()
        distance = torch.cdist(coords * IMAGE_SIZE, target_xy * IMAGE_SIZE)
        # Every target must own exactly one raw candidate in the 16px window;
        # every candidate may own at most one target.  We do not invoke
        # Hungarian or choose a winner for conflict rows.
        accepted: dict[int, tuple[int, float]] = {}
        for target_id in target_ids:
            nearby = torch.where(distance[:, target_id] <= RADIUS)[0].tolist()
            if len(nearby) != 1:
                unknown["target_multi_or_zero_candidates"] += 1
                continue
            candidate_index = nearby[0]
            candidate_distances = distance[candidate_index]
            other_target = torch.cat((candidate_distances[:target_id], candidate_distances[target_id + 1:]))
            nearest_other = float(other_target.min().item()) if len(other_target) else float("inf")
            if nearest_other <= OTHER_TARGET_MIN:
                unknown["candidate_near_other_target"] += 1
                continue
            own = float(candidate_distances[target_id].item())
            if nearest_other - own < TARGET_MARGIN:
                unknown["target_margin_small"] += 1
                continue
            if candidate_index in {v[0] for v in accepted.values()}:
                unknown["candidate_reused"] += 1
                continue
            accepted[target_id] = (candidate_index, own)
        accepted_candidates = {index for index, _ in accepted.values()}
        for target_id, (candidate_index, own) in accepted.items():
            row = candidates[candidate_index]
            element = str(atoms[target_id]["element"])
            kind = "strict" if own <= STRICT else "refinable" if own <= RADIUS else "other"
            if kind == "other":
                continue
            residual = (target_xy[target_id] - coords[candidate_index]) * IMAGE_SIZE
            pairs.append({
                "sample_key": sample_key, "split": plan_row["split"],
                "molecule_id": plan_row["molecule_id"], "profile_id": plan_row["profile_id"],
                "element": element, "target_id": target_id, "slot": int(row["slot"]),
                "candidate_x_px": float(coords[candidate_index, 0] * IMAGE_SIZE),
                "candidate_y_px": float(coords[candidate_index, 1] * IMAGE_SIZE),
                "target_x_px": float(target_xy[target_id, 0] * IMAGE_SIZE),
                "target_y_px": float(target_xy[target_id, 1] * IMAGE_SIZE),
                "distance_px": own, "residual_dx_px": float(residual[0]),
                "residual_dy_px": float(residual[1]), "kind": kind,
                "source": "identity_aligned_unique_raw", "protected_old_peak": False,
            })
            stratum[(plan_row["split"], element, plan_row["profile_id"], kind)] += 1
        # Confirmed far candidates are usable negatives only when they are >20px
        # from all targets; near/ambiguous candidates remain unknown.
        for candidate_index, row in enumerate(candidates):
            nearest = float(distance[candidate_index].min().item()) if target_count else float("inf")
            if candidate_index in accepted_candidates or nearest <= OTHER_TARGET_MIN:
                continue
            if nearest > OTHER_TARGET_MIN:
                pairs.append({
                    "sample_key": sample_key, "split": plan_row["split"],
                    "molecule_id": plan_row["molecule_id"], "profile_id": plan_row["profile_id"],
                    "element": None, "target_id": None, "slot": int(row["slot"]),
                    "candidate_x_px": float(row["x_px"]), "candidate_y_px": float(row["y_px"]),
                    "target_x_px": None, "target_y_px": None, "distance_px": nearest,
                    "residual_dx_px": None, "residual_dy_px": None, "kind": "hard_negative",
                    "source": "identity_aligned_far_raw", "protected_old_peak": False,
                })
                stratum[(plan_row["split"], "hard_negative")] += 1
    molecule_disjoint = all(len(splits) == 1 for splits in molecule_splits.values())
    summary = {"images_with_visible_on": image_count, "pairs": len(pairs),
               "strata": {"|".join(map(str, k)): v for k, v in sorted(stratum.items())},
               "unknown": dict(sorted(unknown.items())),
               "molecules": len(molecule_splits),
               "molecule_disjoint": molecule_disjoint}
    train_pos = [r for r in pairs if r["split"] == "train" and r["kind"] == "refinable"]
    hold_pos = [r for r in pairs if r["split"] == "holdout" and r["kind"] == "refinable"]
    train_strata_ok = all(
        stratum[("train", element, profile, "refinable")] >= 10
        for element in ("O", "N")
        for profile in ("cov13_base", "cov13_cap52", "cov13_stroke16", "cov33_base", "cov33_cap28", "cov33_stroke32")
    )
    holdout_strata_ok = all(
        stratum[("holdout", element, profile, "refinable")] >= 5
        for element in ("O", "N")
        for profile in ("cov13_base", "cov13_cap52", "cov13_stroke16", "cov33_base", "cov33_cap28", "cov33_stroke32")
    )
    release_scale_pass = len(train_pos) >= 240 and len(hold_pos) >= 120 and train_strata_ok and holdout_strata_ok
    result = {
        "schema_version": "chemvector-identity-aligned-visible-text-manifest-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "gpu_used": False, "sealed_test_read": False,
        "initial_sha256": INITIAL_SHA, "source_manifest_sha256": digest(SOURCE),
        "panel_plan_sha256": digest(PLAN), "radii_px": {"candidate": RADIUS, "strict": STRICT, "other_target_min": OTHER_TARGET_MIN, "target_margin": TARGET_MARGIN},
        "identity_contract": {"bijective_only": True, "hungarian_used": False, "oracle_assignment": False, "target_labels_posthoc_only": True},
        "summary": summary, "train_refinable": len(train_pos), "holdout_refinable": len(hold_pos),
        "records": pairs,
        "gate": {"molecule_disjoint": molecule_disjoint, "train_refinable_ge_50": len(train_pos) >= 50,
                 "holdout_refinable_present": len(hold_pos) > 0,
                 "train_min_240": len(train_pos) >= 240,
                 "holdout_min_120": len(hold_pos) >= 120,
                 "train_strata_minima": train_strata_ok,
                 "holdout_strata_minima": holdout_strata_ok,
                 "release_scale_pass": release_scale_pass,
                 "admitted_pairs_ambiguity_zero": True,
                 "unknown_rows_explicitly_masked": True,
                 "gpu_training_authorized": False},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "summary": summary, "train_refinable": len(train_pos), "holdout_refinable": len(hold_pos), "gate": result["gate"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
