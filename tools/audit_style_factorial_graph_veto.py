"""Zero-training graph-constrained rescue upper-bound audit.

This is deliberately a *read-only* audit.  It replays the frozen initial
checkpoint on the fixed style-factorial dev pool and asks whether a dropped
raw atom candidate can be supported by a predicted bond endpoint which is
uniquely assigned and connected to an already retained (protected) atom.
No learned scorer, target geometry, graph/layer supervision, or checkpoint is
used by the rule.  Labels in the candidate cache are used only after the
decision to measure the ceiling and hard-negative survival.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from Data.sdf.chemsvg_sequence_v4 import ChemSVGCoordinateCollisionError
from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.data import ChemVectorIterableDataset
from chemvector.train_atom_bond_scene import AtomBondSceneCollator
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "outputs/audits/chemvector-style-factorial-core-stream-20261007-r1"
CACHE = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/candidates.jsonl"
CACHE_AUDIT = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/cache-audit.json"
DATA = ROOT / "Data/sdf/chemvector_joint_v1/style_train_100k/shards"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-graph-veto-20261016-r1/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE_SIZE = 768.0
BATCH_SIZE = 4
ENDPOINT_RADIUS_PX = 12.0
BOND_LOGIT_GATE = -0.5  # frozen bundle provenance bond_threshold


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifests() -> dict[str, dict]:
    out = {}
    for split, name in (("train_core", "train-core-manifest.json"),
                        ("dev_core", "dev-core-manifest.json"),
                        ("dev_stress", "dev-stress-manifest.json")):
        for row in json.loads((POOL / name).read_text(encoding="utf-8"))["rows"]:
            out[row["sample_key"]] = {**row, "replay_split": split}
    return out


def load_cache() -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    with CACHE.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["replay_split"] != "train_core":
                groups[row["sample_key"]].append(row)
    return groups


def assign_endpoints(rows: list[dict], endpoints: np.ndarray,
                     bond_logits: np.ndarray) -> tuple[dict[int, tuple[int, int, float]], list[dict]]:
    """Greedy deterministic one-to-one endpoint ownership.

    Candidate and endpoint identities are local to one image.  All eligible
    endpoint/candidate edges within the fixed radius are globally sorted by
    distance, then endpoint and candidate index.  No label is consulted.
    """
    points = np.asarray([[float(r["x_norm"]), float(r["y_norm"])] for r in rows], dtype=np.float32) * IMAGE_SIZE
    edges = []
    for bi, (pair, logit) in enumerate(zip(endpoints, bond_logits)):
        if float(logit) <= BOND_LOGIT_GATE:
            continue
        for side, point in enumerate(pair):
            distances = np.linalg.norm(points - point[None, :], axis=1)
            for ci in np.flatnonzero(distances <= ENDPOINT_RADIUS_PX):
                edges.append((float(distances[ci]), bi, side, int(ci), float(logit)))
    edges.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    used_endpoints: set[tuple[int, int]] = set()
    used_candidates: set[int] = set()
    owner: dict[int, tuple[int, int, float]] = {}
    accepted = []
    for distance, bi, side, ci, logit in edges:
        if (bi, side) in used_endpoints or ci in used_candidates:
            continue
        used_endpoints.add((bi, side)); used_candidates.add(ci)
        owner[ci] = (bi, side, distance)
        accepted.append({"bond": bi, "side": side, "candidate": ci,
                         "distance_px": distance, "bond_logit": logit})
    return owner, accepted


def image_decision(rows: list[dict], bonds: dict[str, torch.Tensor], set_output: dict[str, torch.Tensor], index: int) -> dict:
    mask = bonds["bond_mask"][index].bool().detach().cpu().numpy()
    endpoints = bonds["bond_endpoints"][index, mask].detach().cpu().numpy() * IMAGE_SIZE
    logits = set_output["bond_presence_logits"][index, mask].detach().cpu().numpy()
    owner, assignments = assign_endpoints(rows, endpoints, logits)
    retained = {i for i, r in enumerate(rows) if r["label"] == "protected" and r["selected_matched"]}
    # A candidate is connected only when its owned endpoint belongs to an
    # eligible bond whose opposite endpoint is within the same fixed radius
    # of a retained atom.  The opposite endpoint need not steal that atom's
    # ownership; the rescued candidate itself remains one-to-one assigned.
    retained_points = np.asarray([[float(rows[i]["x_norm"]), float(rows[i]["y_norm"])] for i in retained], dtype=np.float32) * IMAGE_SIZE
    rescued = set()
    connected = set()
    for ci, (bi, side, distance) in owner.items():
        if rows[ci]["label"] not in ("strict_positive", "hard_negative"):
            continue
        other = endpoints[bi, 1 - side]
        near_retained = bool(len(retained_points) and np.min(np.linalg.norm(retained_points - other[None, :], axis=1)) <= ENDPOINT_RADIUS_PX)
        if near_retained:
            connected.add(ci)
            rescued.add(ci)
    return {"rescued": rescued, "connected": connected, "owner_count": len(owner),
            "eligible_bonds": int((logits > BOND_LOGIT_GATE).sum()),
            "assignments": assignments, "retained_count": len(retained)}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    cache_audit = json.loads(CACHE_AUDIT.read_text(encoding="utf-8"))
    if not cache_audit["gate"]["jsonl_complete_and_valid"]:
        raise ValueError("candidate cache audit is not complete")
    manifest = manifests()
    groups = load_cache()
    wanted = set(groups)
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    if digest(INITIAL) != INITIAL_SHA:
        raise ValueError("initial checkpoint changed before replay")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval().requires_grad_(False)
    collator = AtomBondSceneCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds))
    shards = sorted(str(p) for p in DATA.glob("train-*.tar"))
    if len(shards) != 20:
        raise ValueError(f"expected 20 shards, got {len(shards)}")
    totals = Counter(); by_variant = defaultdict(Counter); by_split = defaultdict(Counter)
    seen: set[str] = set(); skipped: list[str] = []; pending = []

    def flush(items: list[dict]) -> None:
        if not items:
            return
        try:
            batch = collator(items)
        except ChemSVGCoordinateCollisionError:
            skipped.extend(s["key"] for s in items)
            return
        tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.inference_mode():
            _, _, _, _, bonds, set_output, _ = structure_forward(model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"])
        for i, sample in enumerate(items):
            key = sample["key"]; rows = groups[key]; decision = image_decision(rows,
                {k: v.detach().cpu() for k, v in bonds.items()},
                {k: v.detach().cpu() for k, v in set_output.items()}, i)
            for label in ("strict_positive", "hard_negative"):
                candidates = {j for j, r in enumerate(rows) if r["label"] == label and not r["selected_matched"]}
                survived = candidates & decision["rescued"]
                totals[f"{label}_dropped"] += len(candidates); totals[f"{label}_survived"] += len(survived)
                by_variant[manifest[key]["variant"]][f"{label}_dropped"] += len(candidates)
                by_variant[manifest[key]["variant"]][f"{label}_survived"] += len(survived)
                by_split[manifest[key]["replay_split"]][f"{label}_dropped"] += len(candidates)
                by_split[manifest[key]["replay_split"]][f"{label}_survived"] += len(survived)
            totals["protected"] += sum(r["label"] == "protected" for r in rows)
            totals["protected_retained"] += sum(r["label"] == "protected" and r["selected_matched"] for r in rows)
            totals["images"] += 1; totals["eligible_bonds"] += decision["eligible_bonds"]
            totals["endpoint_assignments"] += decision["owner_count"]

    for sample in ChemVectorIterableDataset(shards, sequence_version="0.4", serialize_target=False, load_raster_layers=False):
        if sample["key"] not in wanted:
            continue
        if sample["key"] in seen:
            raise ValueError(f"duplicate sample {sample['key']}")
        seen.add(sample["key"]); pending.append(sample)
        if len(pending) == BATCH_SIZE:
            flush(pending); pending = []
    flush(pending)
    if seen != wanted:
        raise ValueError(f"manifest/cache samples not replayed: {len(wanted - seen)}")

    def metrics(c: Counter) -> dict:
        ss, sd = int(c["strict_positive_survived"]), int(c["strict_positive_dropped"])
        hs, hd = int(c["hard_negative_survived"]), int(c["hard_negative_dropped"])
        return {"strict_dropped": sd, "strict_rescued": ss, "hard_candidates": hd,
                "hard_survived": hs, "rescued_precision_conservative": ss / max(ss + hs, 1),
                "strict_recall_ceiling": ss / max(sd, 1), "hard_fp_survival": hs / max(hd, 1)}
    variant_report = {k: metrics(v) for k, v in sorted(by_variant.items())}
    split_report = {k: metrics(v) for k, v in sorted(by_split.items())}
    core = [variant_report[v] for v in ("coordgen", "original", "rdkit")]
    gate = {"each_core_precision_ge_085": all(r["rescued_precision_conservative"] >= .85 for r in core),
            "each_core_recall_ge_050": all(r["strict_recall_ceiling"] >= .50 for r in core),
            "each_core_hard_fp_survival_le_015": all(r["hard_fp_survival"] <= .15 for r in core),
            "stress_separate": True, "training_authorized": False}
    report = {"schema_version": "chemvector-style-factorial-graph-veto-0.1", "complete": True,
              "read_only": True, "training_performed": False, "gpu_used": device.type == "cuda",
              "device": str(device), "sealed_test_read": False, "deployment_authorized": False,
              "initial_sha256": INITIAL_SHA, "initial_sha256_after": digest(INITIAL),
              "cache_sha256": digest(CACHE), "cache_audit_sha256": digest(CACHE_AUDIT),
              "samples": int(totals["images"]), "coordinate_collision_samples_skipped": len(skipped),
              "skipped_sample_keys": sorted(skipped), "thresholds": {"endpoint_radius_px": ENDPOINT_RADIUS_PX,
              "bond_presence_logit_gate": BOND_LOGIT_GATE}, "totals": dict(totals),
              "by_variant": variant_report, "by_split": split_report,
              "rule": {"one_to_one_endpoint_assignment": True, "retained_anchor_required": True,
                       "valence_veto": False, "gt_features_used": False,
                       "labels_used_only_for_audit": True}, "gate": gate}
    OUTPUT.parent.mkdir(parents=True, exist_ok=False); OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "samples": report["samples"], "by_variant": variant_report, "gate": gate}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
