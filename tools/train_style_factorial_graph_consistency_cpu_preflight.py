"""CPU-only graph-consistency sidecar preflight.

Frozen initial atom/bond/set outputs provide node and predicted-port features;
only a tiny two-layer candidate graph message-passing head is trained.  The
graph is built from candidate-to-candidate proximity, not GT atoms/bonds.
Protected baseline candidates are context and are never scorer negatives;
near-unknown and selected-unmatched rows are masked.  This is a bounded
preflight, not GPU authorization or deployment training.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
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
from tools.train_style_factorial_bond_context_cpu_preflight import (
    CACHE,
    CACHE_AUDIT,
    CORE,
    DATA,
    IMAGE_SIZE,
    INITIAL_SHA,
    PORT_RADIUS_PX,
    candidate_features,
    digest,
    load_cache,
    load_manifests,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-graph-consistency-cpu-preflight-20261007-r1/report.json"
SEED = 20261016
TRAIN_LIMIT = 120
DEV_LIMIT = 40
STEPS = 300
K_NEIGHBORS = 8
HIDDEN = 32


class GraphHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.node = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, HIDDEN), nn.GELU())
        self.layers = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(HIDDEN * 2 + 1), nn.Linear(HIDDEN * 2 + 1, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN))
            for _ in range(2)
        )
        self.out = nn.Linear(HIDDEN, 1)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = self.node(features)
        for layer in self.layers:
            message = torch.matmul(adjacency, hidden)
            degree = adjacency.sum(-1, keepdim=True).clamp_min(1.0)
            mean_message = message / degree
            update = layer(torch.cat((hidden, mean_message, degree / K_NEIGHBORS), dim=-1))
            hidden = hidden + update
        return self.out(hidden).squeeze(-1)


def load_manifests_for_slice() -> dict[str, dict]:
    return load_manifests()


def candidate_adjacency(rows: list[dict]) -> np.ndarray:
    points = np.asarray([[float(row["x_norm"]), float(row["y_norm"])] for row in rows], dtype=np.float32) * IMAGE_SIZE
    n = len(points)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    distances += np.eye(n, dtype=np.float32) * 1e6
    adjacency = np.zeros((n, n), dtype=np.float32)
    for index in range(n):
        neighbors = np.argsort(distances[index], kind="stable")[:K_NEIGHBORS]
        weights = np.exp(-distances[index, neighbors] / 32.0).astype(np.float32)
        adjacency[index, neighbors] = weights
    return adjacency


def evaluate(groups: list[list[dict]], logits: list[np.ndarray]) -> dict:
    selected = strict = hard = near = protected = 0
    strict_total = hard_total = 0
    for rows, values in zip(groups, logits):
        for row, value in zip(rows, values):
            label = row["label"]
            if label == "protected":
                protected += 1
                continue
            if label == "strict_positive":
                strict_total += 1
            elif label == "hard_negative":
                hard_total += 1
            elif label == "near_unknown":
                near += 1
            if value >= 0.0:
                selected += 1
                strict += int(label == "strict_positive")
                hard += int(label == "hard_negative")
    return {
        "scorer_candidates": strict_total + hard_total + near,
        "strict_positive": strict_total,
        "hard_negative": hard_total,
        "near_unknown": near,
        "protected_context_retained": protected,
        "selected": selected,
        "strict_selected": strict,
        "hard_negative_selected": hard,
        "near_unknown_selected": int(selected - strict - hard),
        "strict_precision_conservative": strict / max(selected, 1),
        "strict_recall": strict / max(strict_total, 1),
        "hard_negative_survival": hard / max(hard_total, 1),
        "protected_retention": 1.0,
    }


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit = json.loads(CACHE_AUDIT.read_text(encoding="utf-8"))
    if not audit["gate"]["jsonl_complete_and_valid"] or not audit["gate"]["matches_forward_aggregate"]:
        raise ValueError("candidate cache audit is not ready")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    manifests = load_manifests_for_slice()
    by_variant: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key, row in manifests.items():
        by_variant[row["replay_split"], row["variant"]].append(key)
    chosen: set[str] = set()
    for variant in ("coordgen", "original", "rdkit"):
        chosen.update(sorted(by_variant["train_core", variant])[:TRAIN_LIMIT])
        chosen.update(sorted(by_variant["dev_core", variant])[:DEV_LIMIT])
    chosen.update(sorted(by_variant["dev_stress", "stress"])[:DEV_LIMIT])
    groups = load_cache(chosen)
    skipped = sorted(chosen - set(groups))
    chosen.intersection_update(groups)
    manifests = {key: value for key, value in manifests.items() if key in chosen}

    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to("cpu").eval().requires_grad_(False)
    collator = AtomBondSceneCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds))
    shards = sorted(str(path) for path in DATA.glob("train-*.tar"))
    if len(shards) != 20:
        raise ValueError("expected 20 style shards")
    features_by_key: dict[str, np.ndarray] = {}
    adjacency_by_key: dict[str, np.ndarray] = {}
    seen: set[str] = set()
    pending = []

    def flush(items: list[dict]) -> None:
        if not items:
            return
        batch = collator(items)
        with torch.inference_mode():
            _, _, _, _, bonds, set_output, _ = structure_forward(model, batch["pixel_values"], config, payload["provenance"]["atom_threshold"])
        for index, sample in enumerate(items):
            key = sample["key"]
            bond = {name: value[index:index + 1] for name, value in bonds.items()}
            setter = {name: value[index:index + 1] for name, value in set_output.items()}
            features_by_key[key] = candidate_features(groups[key], bond, setter)
            adjacency_by_key[key] = candidate_adjacency(groups[key])

    for sample in ChemVectorIterableDataset(shards, sequence_version="0.4", serialize_target=False, load_raster_layers=False):
        if sample["key"] not in chosen:
            continue
        pending.append(sample)
        seen.add(sample["key"])
        if len(pending) == 4:
            flush(pending); pending = []
    flush(pending)
    if seen != chosen:
        raise ValueError(f"style shards missing {len(chosen - seen)} chosen samples")

    train_keys = [key for key in sorted(chosen) if manifests[key]["replay_split"] == "train_core"]
    train_rows = [row for key in train_keys for row in groups[key] if row["label"] in ("strict_positive", "hard_negative")]
    # Fit normalization from explicitly labeled train rows without reading any
    # coordinate distance field.
    train_features = []
    for key in train_keys:
        for row, feature in zip(groups[key], features_by_key[key]):
            if row["label"] in ("strict_positive", "hard_negative"):
                train_features.append(feature)
    mean = np.asarray(train_features, dtype=np.float32).mean(0)
    std = np.maximum(np.asarray(train_features, dtype=np.float32).std(0), 1e-5)
    graph = GraphHead(len(mean))
    optimizer = torch.optim.AdamW(graph.parameters(), lr=2e-3, weight_decay=1e-3)
    rng = np.random.default_rng(SEED)
    train_groups = [key for key in train_keys if any(row["label"] == "strict_positive" for row in groups[key]) and any(row["label"] == "hard_negative" for row in groups[key])]
    events = []
    for step in range(1, STEPS + 1):
        key = train_groups[(step - 1) % len(train_groups)]
        values = torch.tensor((features_by_key[key] - mean) / std)
        adjacency = torch.tensor(adjacency_by_key[key])
        logits = graph(values, adjacency)
        mask = torch.tensor([row["label"] in ("strict_positive", "hard_negative") for row in groups[key]])
        labels = torch.tensor([float(row["label"] == "strict_positive") for row in groups[key]])
        pos = mask & (labels > 0.5); neg = mask & ~pos
        # Fixed per-group balanced BCE avoids protected prevalence dominating
        # the rescue objective while retaining protected rows in messages.
        selected = torch.where(pos | neg)[0]
        if not bool(pos.any()) or not bool(neg.any()):
            continue
        pidx = torch.where(pos)[0]; nidx = torch.where(neg)[0]
        count = min(len(pidx), len(nidx), 16)
        index = torch.cat((pidx[:count], nidx[:count]))
        loss = F.binary_cross_entropy_with_logits(logits[index], labels[index])
        optimizer.zero_grad(set_to_none=True); loss.backward(); norm = torch.nn.utils.clip_grad_norm_(graph.parameters(), 5.0, error_if_nonfinite=True); optimizer.step()
        events.append({"step": step, "sample_key": key, "loss": float(loss.detach()), "gradient_norm": float(norm)})

    graph.eval(); logits_by_key = {}
    with torch.no_grad():
        for key in sorted(chosen):
            logits_by_key[key] = graph(torch.tensor((features_by_key[key] - mean) / std), torch.tensor(adjacency_by_key[key])).numpy()
    reports = {}
    for split, variants in (("train_core", ("coordgen", "original", "rdkit")), ("dev_core", ("coordgen", "original", "rdkit")), ("dev_stress", ("stress",))):
        for variant in variants:
            keys = [key for key in sorted(chosen) if manifests[key]["replay_split"] == split and manifests[key]["variant"] == variant]
            reports[f"{split}/{variant}"] = evaluate([groups[key] for key in keys], [logits_by_key[key] for key in keys])
    core = [reports[f"dev_core/{variant}"] for variant in ("coordgen", "original", "rdkit")]
    result = {
        "schema_version": "chemvector-style-factorial-graph-consistency-cpu-preflight-0.1",
        "complete": True, "training_performed": True, "training_scope": "frozen_graph_consistency_sidecar_cpu_only",
        "gpu_used": False, "optimizer_steps": STEPS, "checkpoint_written": False, "sealed_test_read": False,
        "initial_sha256": INITIAL_SHA, "cache_sha256": digest(CACHE), "cache_audit_sha256": digest(CACHE_AUDIT),
        "seed": SEED, "sample_limits": {"train_per_variant": TRAIN_LIMIT, "dev_per_variant": DEV_LIMIT},
        "replay_skipped_samples": skipped, "hidden": HIDDEN, "message_layers": 2, "neighbors": K_NEIGHBORS,
        "fixed_logit_gate": 0.0, "features": ["raw_score", "presence_logit", "rank", "nearest_candidate", "density", "nearest_predicted_port", "port_proximity", "port_count", "port_logits", "port_margin", "relative_xy"],
        "reports": reports, "events": events,
        "input_contract": {"gt_distance_used_as_input": False, "target_layers_used_as_input": False, "protected_as_negative": False, "near_unknown_as_negative": False, "selected_unmatched_as_negative": False},
        "gate": {"each_core_precision_ge_085": all(item["strict_precision_conservative"] >= .85 for item in core), "each_core_recall_ge_050": all(item["strict_recall"] >= .50 for item in core), "each_core_hard_fp_le_015": all(item["hard_negative_survival"] <= .15 for item in core), "protected_retention_exact": all(item["protected_retention"] == 1.0 for item in core), "stress_separate": True, "training_ready": False, "reason": "CPU graph sidecar preflight only; no GPU authorization"},
    }
    result["gate"]["training_ready"] = all(result["gate"][key] for key in ("each_core_precision_ge_085", "each_core_recall_ge_050", "each_core_hard_fp_le_015", "protected_retention_exact"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=False); OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "reports": reports, "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
