"""Small fixed CPU preflight for a bond/port-context atom candidate scorer.

The frozen initial model is replayed only on a deterministic small slice of
the style-factorial stream (120 train-core samples and 40 samples per eval
variant).  Candidate features combine frozen atom scores, candidate-group
competition, and distances to *predicted* bond ports.  No rendered target
distance is read as an input.  Cache labels are used only to form the strict
positive/hard-negative training rows and to audit the result; protected and
near-unknown rows are masked from the loss.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.data import ChemVectorIterableDataset
from chemvector.train_atom_bond_scene import AtomBondSceneCollator
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/candidates.jsonl"
CACHE_AUDIT = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/cache-audit.json"
CORE = ROOT / "outputs/audits/chemvector-style-factorial-core-stream-20261007-r1"
DATA = ROOT / "Data/sdf/chemvector_joint_v1/style_train_100k/shards"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-bond-context-cpu-preflight-20261007-r2/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
SEED = 20261014
TRAIN_LIMIT = 120
DEV_LIMIT = 40
STEPS = 300
BATCH = 512
PORT_RADIUS_PX = 12.0
IMAGE_SIZE = 768.0


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Head(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dimensions), nn.Linear(dimensions, 48), nn.GELU(), nn.Linear(48, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_manifests() -> dict[str, dict]:
    result = {}
    for split, name in (("train_core", "train-core-manifest.json"), ("dev_core", "dev-core-manifest.json"), ("dev_stress", "dev-stress-manifest.json")):
        for row in json.loads((CORE / name).read_text(encoding="utf-8"))["rows"]:
            result[row["sample_key"]] = {**row, "replay_split": split}
    return result


def load_cache(wanted: set[str]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["sample_key"] in wanted:
                groups[row["sample_key"]].append(row)
    return groups


def candidate_features(rows: list[dict], bonds: dict[str, torch.Tensor], set_output: dict[str, torch.Tensor]) -> np.ndarray:
    """Use predicted ports and candidate competition only; no GT fields."""
    points = np.asarray([[float(row["x_norm"]), float(row["y_norm"])] for row in rows], dtype=np.float32) * IMAGE_SIZE
    raw = np.asarray([float(row["raw_score"]) for row in rows], dtype=np.float32)
    presence = np.asarray([float(row["presence_logit"]) for row in rows], dtype=np.float32)
    count = len(rows)
    if count > 1:
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        distances += np.eye(count, dtype=np.float32) * 1e6
        nearest_candidate = distances.min(axis=1)
        local_count = (distances < 32.0).sum(axis=1).astype(np.float32)
    else:
        nearest_candidate, local_count = np.asarray([768.0], dtype=np.float32), np.asarray([0.0], dtype=np.float32)
    bond_mask = bonds["bond_mask"][0].bool()
    # structure_forward's frozen bond head exposes endpoint pairs under
    # ``bond_endpoints`` (the scene decoder calls the same objects ports).
    ports = bonds["bond_endpoints"][0, bond_mask].detach().cpu().numpy().reshape(-1, 2) * IMAGE_SIZE
    bond_logits = set_output["bond_presence_logits"][0, bond_mask].detach().cpu().numpy()
    if len(ports):
        port_distances = np.linalg.norm(points[:, None, :] - ports[None, :, :], axis=-1)
        nearest_port = port_distances.min(axis=1)
        port_count = (port_distances <= PORT_RADIUS_PX).sum(axis=1).astype(np.float32)
        endpoint_logit = np.repeat(bond_logits, 2)
        order = np.argsort(port_distances, axis=1)
        nearest_logit = endpoint_logit[order[:, 0]]
        second_logit = endpoint_logit[order[:, min(1, endpoint_logit.size - 1)]]
        port_score = np.exp(-nearest_port / PORT_RADIUS_PX).astype(np.float32)
    else:
        nearest_port = np.full(count, IMAGE_SIZE, dtype=np.float32)
        port_count = np.zeros(count, dtype=np.float32)
        nearest_logit = np.full(count, -20.0, dtype=np.float32)
        second_logit = np.full(count, -20.0, dtype=np.float32)
        port_score = np.zeros(count, dtype=np.float32)
    rank = np.argsort(np.argsort(-presence, kind="stable"), kind="stable").astype(np.float32)
    return np.column_stack((raw, presence, rank / max(count, 1),
                            nearest_candidate / IMAGE_SIZE, local_count / 128.0,
                            nearest_port / IMAGE_SIZE, port_score,
                            port_count / 32.0, nearest_logit, second_logit,
                            nearest_logit - second_logit)).astype(np.float32)


def evaluate(rows: list[dict], logits: np.ndarray) -> dict:
    strict = np.asarray([row["label"] == "strict_positive" for row in rows], dtype=bool)
    near = np.asarray([row["label"] == "near_unknown" for row in rows], dtype=bool)
    hard = np.asarray([row["label"] == "hard_negative" for row in rows], dtype=bool)
    protected = np.asarray([row["label"] == "protected" for row in rows], dtype=bool)
    # Protected rows are retained by the frozen baseline and are not scorer
    # candidates. Keep them as context diagnostics only.
    scorer_rows = ~protected
    selected = (logits >= 0.0) & scorer_rows
    true = int((selected & strict).sum())
    return {"candidates": int(scorer_rows.sum()), "protected_context": int(protected.sum()),
            "strict_positive": int(strict.sum()), "near_unknown": int(near.sum()),
            "hard_negative": int(hard.sum()),
            "selected": int(selected.sum()), "strict_selected": true,
            "near_unknown_selected": int((selected & near).sum()),
            "hard_negative_selected": int((selected & hard).sum()),
            "protected_context_not_scored": int(protected.sum()),
            "strict_precision_conservative": true / max(int(selected.sum()), 1),
            "strict_recall": true / max(int(strict.sum()), 1)}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit = json.loads(CACHE_AUDIT.read_text(encoding="utf-8"))
    if not audit["gate"]["jsonl_complete_and_valid"] or not audit["gate"]["matches_forward_aggregate"]:
        raise ValueError("candidate cache audit is not ready")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    manifests = load_manifests()
    by_variant: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key, row in manifests.items():
        by_variant[row["replay_split"], row["variant"]].append(key)
    chosen: set[str] = set()
    # Lexicographic fixed slices are registered before inference; no label
    # filtering is used to choose samples.
    for variant in ("train_core",):
        chosen.update(sorted(by_variant["train_core", "coordgen"])[0:TRAIN_LIMIT])
        chosen.update(sorted(by_variant["train_core", "original"])[0:TRAIN_LIMIT])
        chosen.update(sorted(by_variant["train_core", "rdkit"])[0:TRAIN_LIMIT])
    for variant in ("coordgen", "original", "rdkit"):
        chosen.update(sorted(by_variant["dev_core", variant])[:DEV_LIMIT])
    chosen.update(sorted(by_variant["dev_stress", "stress"])[:DEV_LIMIT])
    groups = load_cache(chosen)
    skipped = sorted(chosen - set(groups))
    # Coordinate-collision samples absent from replay are excluded fail-closed;
    # they are not reconstructed or silently treated as negatives.
    chosen.intersection_update(groups)
    manifests = {key: value for key, value in manifests.items() if key in chosen}
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to("cpu").eval().requires_grad_(False)
    collator = AtomBondSceneCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds))
    shards = sorted(str(path) for path in DATA.glob("train-*.tar"))
    if len(shards) != 20:
        raise ValueError("expected 20 style shards")
    rows_by_key: dict[str, list[dict]] = {}
    seen: set[str] = set()
    pending = []
    def flush(items: list[dict]) -> None:
        if not items:
            return
        batch = collator(items)
        with torch.inference_mode():
            _, _, _, _, bonds, set_output, _ = structure_forward(model, batch["pixel_values"], config, payload["provenance"]["atom_threshold"])
        for index, sample in enumerate(items):
            # Slice frozen batch bond outputs; candidate rows never expose GT.
            b = {name: value[index:index + 1] for name, value in bonds.items()}
            so = {name: value[index:index + 1] for name, value in set_output.items()}
            rows_by_key[sample["key"]] = [{**row, "features": feature.tolist()} for row, feature in zip(groups[sample["key"]], candidate_features(groups[sample["key"]], b, so))]
    for sample in ChemVectorIterableDataset(shards, sequence_version="0.4", serialize_target=False, load_raster_layers=False):
        if sample["key"] not in chosen:
            continue
        pending.append(sample)
        seen.add(sample["key"])
        if len(pending) == 4:
            flush(pending); pending = []
    flush(pending)
    if seen != chosen:
        raise ValueError(f"shards missing {len(chosen - seen)} chosen samples")
    train_rows = [row for key in chosen if manifests[key]["replay_split"] == "train_core" for row in rows_by_key[key] if row["label"] in ("strict_positive", "hard_negative")]
    x = np.asarray([row["features"] for row in train_rows], dtype=np.float32)
    y = np.asarray([row["label"] == "strict_positive" for row in train_rows], dtype=np.float32)
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-5)
    pos, neg = np.flatnonzero(y > 0), np.flatnonzero(y == 0)
    if not len(pos) or not len(neg):
        raise ValueError("small slice lacks both strict and hard rows")
    head = Head(x.shape[1]); optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
    xt, yt = torch.tensor((x - mean) / std), torch.tensor(y); rng = np.random.default_rng(SEED); events = []
    for step in range(1, STEPS + 1):
        index = np.concatenate((rng.choice(pos, BATCH // 2, replace=True), rng.choice(neg, BATCH // 2, replace=True))); rng.shuffle(index)
        logits = head(xt[torch.tensor(index)]); loss = F.binary_cross_entropy_with_logits(logits, yt[torch.tensor(index)])
        optimizer.zero_grad(set_to_none=True); loss.backward(); norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0, error_if_nonfinite=True); optimizer.step()
        events.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm)})
    # Fixed train-prior correction for balanced batches; no threshold scan.
    gate = math.log(float(y.mean()) / (1.0 - float(y.mean())))
    reports = {}
    head.eval()
    with torch.no_grad():
        for split, variants in (("train_core", ("coordgen", "original", "rdkit")), ("dev_core", ("coordgen", "original", "rdkit")), ("dev_stress", ("stress",))):
            for variant in variants:
                keys = [key for key in chosen if manifests[key]["replay_split"] == split and manifests[key]["variant"] == variant]
                rows = [row for key in keys for row in rows_by_key[key]]
                logits = head(torch.tensor((np.asarray([row["features"] for row in rows], dtype=np.float32) - mean) / std)).numpy() + gate
                reports[f"{split}/{variant}"] = evaluate(rows, logits)
    core = [reports[f"dev_core/{v}"] for v in ("coordgen", "original", "rdkit")]
    result = {"schema_version": "chemvector-style-factorial-bond-context-cpu-preflight-0.1", "complete": True,
              "training_performed": True, "training_scope": "frozen_bond_port_context_cpu_only", "gpu_used": False,
              "optimizer_steps": STEPS, "batch_size": BATCH, "checkpoint_written": False, "sealed_test_read": False,
              "initial_sha256": INITIAL_SHA, "cache_sha256": digest(CACHE), "cache_audit_sha256": digest(CACHE_AUDIT),
              "seed": SEED, "sample_limits": {"train_per_variant": TRAIN_LIMIT, "dev_per_variant": DEV_LIMIT},
              "replay_skipped_samples": skipped,
              "feature_names": ["raw_score", "presence_logit", "group_rank", "nearest_candidate", "local_candidate_count", "nearest_predicted_port", "port_proximity", "port_count", "nearest_port_logit", "second_port_logit", "port_logit_margin"],
              "fixed_logit_gate": gate, "train_loss_rows": len(train_rows), "near_unknown_training_rows": 0,
              "reports": reports, "events": events,
              "input_contract": {"gt_distance_used_as_input": False, "target_layers_read": False, "protected_as_negative": False, "near_unknown_as_negative": False},
              "gate": {"each_core_precision_ge_085": all(r["strict_precision_conservative"] >= .85 for r in core), "each_core_recall_ge_050": all(r["strict_recall"] >= .50 for r in core), "stress_separate": True, "training_ready": False, "reason": "small CPU preflight only; no GPU authorization"}}
    result["gate"]["training_ready"] = result["gate"]["each_core_precision_ge_085"] and result["gate"]["each_core_recall_ge_050"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=False); OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "reports": reports, "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
