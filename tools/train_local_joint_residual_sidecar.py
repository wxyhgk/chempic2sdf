"""One bounded residual sidecar trial for below-threshold local atom pairs.

The production initial and its pair head are frozen.  Only known strict pair
rows whose frozen presence probability is below 0.85 enter this isolated
sidecar.  Existing baseline edges are never rescored or replaced, unknown
rows are excluded, and no production checkpoint is written.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.bond_object import BOND_TYPES
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.pair_scale_plan import sample_id
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.interface_diagnostic import tensor_tree_digest
from chemvector.training.curriculum.supervision_contract import pair_targets_with_unknown
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets
from tools.current_initial_cohort import split_group_disjoint


ROOT = Path(__file__).resolve().parents[1]
AUDIT_REPORT = ROOT / "outputs/audits/chemvector-local-joint-candidate-20261009-r2/report.json"
AUDIT_GATE = ROOT / "outputs/audits/chemvector-local-joint-candidate-20261009-r2/gate.json"
AUDIT_REPORT_SHA256 = "ef11357878d10594b94edc71dfbcec6d4d285b01ce1847c8bfc4b1ea75b08712"
AUDIT_GATE_SHA256 = "e5c65d21f35cef44b95b4b4ac4a974aa35a7454007142341574ecf95f8a0713e"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_LOCAL_RESIDUAL_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-local-joint-residual-sidecar-20261009"),
))
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
BASELINE_THRESHOLD = 0.85
RESIDUAL_PRECISION_FLOOR = 0.95
MAX_ACCEPTED_PER_IMAGE = 16
MAX_STEPS = 800
SEED = 20261009
THRESHOLD_GRID = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98)


class ResidualPairSidecar(nn.Module):
    """Small target-independent residual scorer (under 10k parameters)."""

    def __init__(self, input_dim: int = 11, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        if sum(p.numel() for p in self.parameters()) >= 10_000:
            raise ValueError("Residual sidecar exceeds parameter budget")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Feature 0 is the frozen pair logit.  The sidecar only learns a
        # residual correction, leaving baseline edges outside this cache.
        return features[:, 0] + self.net(features).squeeze(-1)


def _cache_digest(cache: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in ("features", "labels", "chemical", "image_indices"):
        digest.update(cache[key].numpy().tobytes())
    return digest.hexdigest()


def _average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    if not len(labels) or not bool(labels.any()):
        return None
    order = torch.argsort(scores, descending=True, stable=True)
    y = labels[order].float()
    precision = y.cumsum(0) / torch.arange(1, len(y) + 1, dtype=torch.float32)
    return float((precision * y).sum() / y.sum())


def _capacity_mask(scores: torch.Tensor, image_indices: torch.Tensor, threshold: float) -> torch.Tensor:
    selected = scores >= threshold
    selected = selected.clone()
    for image in image_indices.unique(sorted=True).tolist():
        rows = torch.where(selected & (image_indices == image))[0]
        if len(rows) > MAX_ACCEPTED_PER_IMAGE:
            keep = rows[torch.argsort(scores[rows], descending=True)[:MAX_ACCEPTED_PER_IMAGE]]
            selected[rows] = False
            selected[keep] = True
    return selected


def _evaluate(scores: torch.Tensor, labels: torch.Tensor, image_indices: torch.Tensor, threshold: float | None) -> dict:
    result = {
        "candidates": int(len(labels)), "positives": int(labels.sum()),
        "auprc": _average_precision(scores, labels), "threshold": threshold,
    }
    if threshold is None:
        result.update({"selected": 0, "matched": 0, "false_positives": 0,
                       "precision": 0.0, "recall": 0.0})
        return result
    selected = _capacity_mask(scores, image_indices, threshold)
    matched = int((selected & labels).sum())
    count = int(selected.sum())
    result.update({"selected": count, "matched": matched,
                   "false_positives": count - matched,
                   "precision": matched / max(count, 1),
                   "recall": matched / max(int(labels.sum()), 1)})
    return result


def _choose_threshold(scores: torch.Tensor, labels: torch.Tensor, image_indices: torch.Tensor) -> dict:
    rows = []
    for threshold in THRESHOLD_GRID:
        result = _evaluate(scores, labels, image_indices, threshold)
        rows.append(result)
    admissible = [row for row in rows if row["precision"] >= RESIDUAL_PRECISION_FLOOR]
    selected = max(admissible, key=lambda row: (row["recall"], -row["threshold"])) if admissible else None
    return {"selected": selected, "grid": rows, "admissible": selected is not None}


def _features(pair_output: dict[str, torch.Tensor], selected: dict[str, torch.Tensor], pairs: torch.Tensor,
              rows: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    first = pairs[:, 0]; second = pairs[:, 1]
    first_points = selected["atom_geometry"][rows, first, :2]
    second_points = selected["atom_geometry"][rows, second, :2]
    delta = second_points - first_points
    first_scores = selected["atom_scores"][rows, first]
    second_scores = selected["atom_scores"][rows, second]
    score_stats = torch.stack((first_scores, second_scores, first_scores + second_scores), dim=1)
    return torch.cat((
        pair_output["presence_logits"][:, None].float(),
        distances[:, None].float() / 192.0,
        delta.float(), score_stats.float(), pair_output["chemical_logits"].float(),
    ), dim=1)


def _build_cache(model, config, rows, loader, collator, device, atom_threshold) -> dict[str, torch.Tensor | tuple[str, ...]]:
    features, labels, chemical, image_indices, keys = [], [], [], [], []
    for image_index, row in enumerate(rows):
        sample = loader(loader.manifest, row)
        batch = collator([sample])
        tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            vision_features, _, _, _, _, _, selected = structure_forward(
                model, tensors["pixel_values"], config, atom_threshold
            )
            mapping, _ = match_atom_targets(selected, tensors, config)
            contract = pair_targets_with_unknown(selected, mapping, tensors)
            pairs, sample_indices = contract["pair_indices"], contract["sample_indices"]
            endpoints = selected["atom_geometry"][sample_indices[:, None], pairs, :2]
            distances = torch.linalg.vector_norm(endpoints[:, 0] - endpoints[:, 1], dim=-1) * 768.0
            local = (distances >= 8.0) & (distances <= 192.0)
            if local.any():
                local_rows = torch.where(local)[0]
                out = model.pair(
                    vision_features, selected["atom_features"], selected["atom_geometry"][..., :2],
                    selected["atom_scores"], pairs[local_rows], sample_indices[local_rows],
                )
                known = contract["known_mask"][local_rows]
                base_prob = out["presence_logits"].sigmoid()
                keep = known & (base_prob < BASELINE_THRESHOLD)
                if keep.any():
                    kept = torch.where(keep)[0]
                    kept_output = {key: value[kept] for key, value in out.items()}
                    feats = _features(kept_output, selected, pairs[local_rows][kept],
                                      sample_indices[local_rows][kept], distances[local_rows][kept])
                    features.append(feats.cpu())
                    labels.append(contract["legacy_targets"]["presence"][local_rows][kept].cpu().bool())
                    chemical.append(contract["legacy_targets"]["chemical"][local_rows][kept].cpu().long())
                    image_indices.append(torch.full((len(kept),), image_index, dtype=torch.long))
        keys.append(row["sample_key"])
    if not features:
        raise ValueError("No known below-threshold residual candidates")
    return {
        "features": torch.cat(features), "labels": torch.cat(labels),
        "chemical": torch.cat(chemical), "image_indices": torch.cat(image_indices),
        "sample_keys": tuple(keys),
    }


def _split_cache(cache: dict, rows: list[dict], train_rows: list[dict], holdout_rows: list[dict]) -> tuple[dict, dict]:
    key_to_split = {r["sample_key"]: "train" for r in train_rows}
    key_to_split.update({r["sample_key"]: "holdout" for r in holdout_rows})
    train_ids = {i for i, key in enumerate(cache["sample_keys"]) if key_to_split[key] == "train"}
    train_mask = torch.tensor([int(i) in train_ids for i in cache["image_indices"].tolist()])
    holdout_mask = ~train_mask
    result = []
    for partition, mask in (("train", train_mask), ("holdout", holdout_mask)):
        result.append({
            **{key: value[mask] for key, value in cache.items() if isinstance(value, torch.Tensor)},
            "sample_keys": tuple(key for key in cache["sample_keys"] if key_to_split[key] == partition),
        })
    return result[0], result[1]


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("Production initial checkpoint contract changed")
    if file_digest(AUDIT_REPORT) != AUDIT_REPORT_SHA256 or file_digest(AUDIT_GATE) != AUDIT_GATE_SHA256:
        raise RuntimeError("Bound local-pair audit artifacts changed")
    audit = json.loads(AUDIT_REPORT.read_text(encoding="utf-8"))
    gate = json.loads(AUDIT_GATE.read_text(encoding="utf-8"))
    if (audit["source_sha256"]["initial.pt"] != INITIAL_SHA
            or audit.get("sealed_test_read")
            or audit.get("training_performed")
            or not gate["gate"]["training_admissible"]):
        raise RuntimeError("Local joint audit gate is not admissible")
    if audit["candidate_policy"]["max_distance_px"] != 192.0:
        raise RuntimeError("Residual trial requires the bound 192px local audit")
    fixed = json.loads((ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json").read_text())
    fixed_keys = {row["sample_key"] for row in fixed["per_sample"]}
    sources = source_checks(); manifest = sources["manifest"]
    rows = [row for row in manifest["samples"] if row["sample_key"] in fixed_keys]
    train_rows, holdout_rows, split_meta = split_group_disjoint(rows, 480)
    if any(split_meta[k] != 0 for k in ("group_overlap", "molecule_overlap", "canonical_smiles_overlap")):
        raise RuntimeError(f"Split overlap: {split_meta}")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    atom_threshold = payload["provenance"]["atom_threshold"]
    model.to("cuda:0").eval().requires_grad_(False)
    frozen_before = tensor_tree_digest(model.state_dict())
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds), purpose="training")
    cache = _build_cache(model, config, rows, loader, collator, torch.device("cuda:0"), atom_threshold)
    train_cache, holdout_cache = _split_cache(cache, rows, train_rows, holdout_rows)
    if int(train_cache["labels"].sum()) < 50 or int(holdout_cache["labels"].sum()) < 14:
        raise RuntimeError("Residual positive-count gate failed before optimizer creation")
    torch.manual_seed(SEED)
    sidecar = ResidualPairSidecar().to("cuda:0").train()
    optimizer = torch.optim.AdamW(sidecar.parameters(), lr=2e-3, weight_decay=1e-4)
    x = train_cache["features"].to("cuda:0"); y = train_cache["labels"].float().to("cuda:0")
    pos_weight = torch.tensor(min(float((~train_cache["labels"]).sum()) / max(float(train_cache["labels"].sum()), 1.0), 20.0), device="cuda:0")
    generator = torch.Generator(device="cuda:0").manual_seed(SEED)
    batch_size = min(4096, len(x)); events = []
    for step in range(1, MAX_STEPS + 1):
        indices = torch.randint(len(x), (batch_size,), generator=generator, device="cuda:0")
        logits = sidecar(x[indices])
        loss = F.binary_cross_entropy_with_logits(logits, y[indices], pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(sidecar.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
        if step in (1, 200, 400, 600, 800): events.append({"step": step, "loss": float(loss.detach())})
    with torch.no_grad():
        train_scores = sidecar(train_cache["features"].to("cuda:0")).sigmoid().cpu()
        holdout_scores = sidecar(holdout_cache["features"].to("cuda:0")).sigmoid().cpu()
    threshold = _choose_threshold(train_scores, train_cache["labels"], train_cache["image_indices"])
    selected = threshold["selected"]
    threshold_value = None if selected is None else float(selected["threshold"])
    metrics = {
        "parameters": sum(p.numel() for p in sidecar.parameters()), "steps": MAX_STEPS, "events": events,
        "train": _evaluate(train_scores, train_cache["labels"], train_cache["image_indices"], threshold_value),
        "holdout": _evaluate(holdout_scores, holdout_cache["labels"], holdout_cache["image_indices"], threshold_value),
        "train_threshold": threshold,
    }
    frozen_after = tensor_tree_digest(model.state_dict())
    if frozen_before != frozen_after:
        raise RuntimeError("Frozen production model changed during residual trial")
    result = {
        "schema_version": "chemvector-local-joint-residual-sidecar-0.1", "complete": True,
        "training_performed": True, "in_memory_sidecar_only": True,
        "production_checkpoint_written": False, "sealed_test_read": False,
        "deployment_authorized": False, "initial_sha256": INITIAL_SHA,
        "audit_report": str(AUDIT_REPORT.relative_to(ROOT)), "audit_gate": str(AUDIT_GATE.relative_to(ROOT)),
        "audit_report_sha256": AUDIT_REPORT_SHA256, "audit_gate_sha256": AUDIT_GATE_SHA256,
        "split": split_meta,
        "contract": {"baseline_threshold": BASELINE_THRESHOLD, "candidate_source": "known_strict_local_pairs_below_frozen_presence_0.85",
                     "unknown_excluded": True, "old_pair_head_frozen": True, "baseline_edges_immutable": True,
                     "max_accepted_per_image": MAX_ACCEPTED_PER_IMAGE, "max_steps": MAX_STEPS,
                     "feature_dim": int(train_cache["features"].shape[1])},
        "cache": {"train_candidates": int(len(train_cache["labels"])), "train_positives": int(train_cache["labels"].sum()),
                  "holdout_candidates": int(len(holdout_cache["labels"])), "holdout_positives": int(holdout_cache["labels"].sum()),
                  "train_digest": _cache_digest(train_cache), "holdout_digest": _cache_digest(holdout_cache)},
        "metrics": metrics,
        "frozen_state": {"before_sha256": frozen_before, "after_sha256": frozen_after,
                         "unchanged": frozen_before == frozen_after},
        "gate": {"prefix_immutable": True, "baseline_selected_retention": 1.0,
                 "train_positive_count_sufficient": int(train_cache["labels"].sum()) >= 50,
                 "holdout_positive_count_sufficient": int(holdout_cache["labels"].sum()) >= 14,
                 "train_precision_sufficient": bool(selected is not None and selected["precision"] >= RESIDUAL_PRECISION_FLOOR),
                 "holdout_precision_sufficient": metrics["holdout"]["precision"] >= RESIDUAL_PRECISION_FLOOR,
                 "holdout_matched_sufficient": metrics["holdout"]["matched"] >= 14,
                 "admissible": bool(frozen_before == frozen_after
                                      and int(train_cache["labels"].sum()) >= 50
                                      and int(holdout_cache["labels"].sum()) >= 14
                                      and selected is not None and selected["precision"] >= RESIDUAL_PRECISION_FLOOR
                                      and metrics["holdout"]["precision"] >= RESIDUAL_PRECISION_FLOOR
                                      and metrics["holdout"]["matched"] >= 14)},
    }
    OUTPUT.mkdir(parents=True)
    torch.save({"schema_version": "chemvector-local-joint-residual-cache-0.1", "initial_sha256": INITIAL_SHA,
                "audit_report_sha256": AUDIT_REPORT_SHA256,
                "train": {**{k: v for k, v in train_cache.items() if isinstance(v, torch.Tensor)},
                          "sample_keys": list(train_cache["sample_keys"])},
                "holdout": {**{k: v for k, v in holdout_cache.items() if isinstance(v, torch.Tensor)},
                            "sample_keys": list(holdout_cache["sample_keys"])}},
               OUTPUT / "candidate_cache.pt")
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False)); return result


if __name__ == "__main__":
    run()
