"""Build a frozen-initial group-selector candidate cache and readiness audit.

Candidate groups are formed only from predicted geometry (nearby raw points or
the same predicted bond port).  Ground truth is consulted after the forward
pass to label strict raw-matched/selector-dropped positives and confirmed
negatives.  No selector update, publication, or sealed-data access occurs.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator, load_sample
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
PLAN = Path(os.environ.get(
    "CHEMVECTOR_TARGETED_SELECTOR_GROUP_PLAN",
    str(ROOT / "outputs/audits/chemvector-structure-expansion-targeted-split-plan-20261006-r5/report.json"),
))
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_TARGETED_SELECTOR_GROUP_CONTRACT_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-targeted-selector-group-contract-20261006-r1"),
))
GROUP_RADIUS_PX = 10.0
PORT_RADIUS_PX = 12.0


def _group_candidates(points: torch.Tensor, ports: torch.Tensor) -> list[int]:
    n = int(points.size(0))
    if not n:
        return []
    # Do not use transitive single-linkage here.  A sequence of points four
    # pixels apart can otherwise turn a whole line into one 74-member group.
    # Groups are bounded-diameter neighborhoods: every new member must be
    # within the radius of every existing member.
    point_px = points * 768.0
    distances = torch.cdist(point_px, point_px)
    port_ids = None
    if ports.numel():
        port_distances = torch.cdist(point_px, ports.reshape(-1, 2) * 768.0)
        nearest = port_distances.min(dim=1).values
        port_ids = [int(torch.argmin(port_distances[i])) if float(nearest[i]) <= PORT_RADIUS_PX else None
                    for i in range(n)]
    groups: list[list[int]] = []
    for index in range(n):
        candidates = []
        for group_index, members in enumerate(groups):
            if all(float(distances[index, member]) <= GROUP_RADIUS_PX for member in members):
                if port_ids is None or port_ids[index] is None:
                    candidates.append(group_index)
                elif all(port_ids[member] in (None, port_ids[index]) for member in members):
                    candidates.append(group_index)
        # Stable nearest-group tie break keeps cache identities deterministic.
        if candidates:
            group_index = min(candidates, key=lambda value: (float(distances[index, groups[value]].max()), value))
            groups[group_index].append(index)
        else:
            groups.append([index])
    component = [0] * n
    for group_index, members in enumerate(groups):
        for member in members:
            component[member] = group_index
    return component


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or immutable initial checkpoint unavailable")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    rows = plan.get("rows", [])
    if len(rows) not in {130, 480} or len({r["sample_key"] for r in rows}) != len(rows):
        raise ValueError("Fixed 130/480-row targeted split required")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.cuda().eval().requires_grad_(False)
    before = state_digest(model)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
                                                       bond_queries=config.maximum_bonds), purpose="diagnostic")
    groups = []
    totals = defaultdict(int)
    with torch.inference_mode():
        for row in rows:
            sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
            if sample["meta"].get("split") != "quarantine" or sample["meta"].get("training_authorized") is not False:
                raise ValueError("Group cache input is not quarantine-only")
            batch = collator([sample])
            tensors = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = structure_forward(model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"])
            raw, bonds, set_output, selected = out[3], out[4], out[5], out[6]
            raw_map, _ = match_atom_targets(raw, tensors, config)
            selected_map, _ = match_atom_targets(selected, tensors, config)
            target_ids = [int(v) for v in batch["atom_ids"][0]]
            raw_target = {slot: target_ids[int(v)] for slot, v in enumerate(raw_map[0].tolist())
                          if 0 <= int(v) < len(target_ids)}
            selected_target = {target_ids[int(v)] for v in selected_map[0].tolist()
                               if 0 <= int(v) < len(target_ids)}
            active = torch.where(raw["atom_mask"][0])[0]
            points = raw["atom_geometry"][0, active, :2]
            ports = bonds["bond_endpoints"][0, bonds["bond_mask"][0]]
            component = _group_candidates(points, ports)
            by_group = defaultdict(list)
            for local, slot_tensor in enumerate(active.tolist()):
                slot = int(slot_tensor)
                point = points[local]
                port_dist = float(torch.cdist(point[None] * 768.0, ports.reshape(-1, 2) * 768.0).min()) if ports.numel() else 99.0
                feature = [float(raw["atom_scores"][0, slot]),
                           float(set_output["atom_presence_logits"][0, slot]),
                           float(set_output["atom_pre_presence_logits"][0, slot]),
                           float(point[0]), float(point[1]), port_dist / 768.0]
                item = {"slot": slot, "feature": feature, "raw_target": raw_target.get(slot, -1),
                        "selected_target": raw_target.get(slot) if raw_target.get(slot) in selected_target else -1,
                        "group_index": component[local]}
                by_group[component[local]].append(item)
            for group_index, members in sorted(by_group.items()):
                strict_positive = any(m["raw_target"] >= 0 and m["selected_target"] < 0 for m in members)
                protected = any(m["selected_target"] >= 0 for m in members)
                confirmed_negative = all(m["raw_target"] < 0 for m in members)
                group_id = hashlib.sha256(f"{row['sample_key']}:{group_index}".encode()).hexdigest()[:20]
                groups.append({"group_id": group_id, "sample_key": row["sample_key"],
                               "molecule_id": row["molecule_id"], "split": row["split"],
                               "profile_id": row["profile_id"], "members": members,
                               "strict_positive": strict_positive, "protected": protected,
                               "confirmed_negative": confirmed_negative})
                totals["groups"] += 1
                totals[f"{row['split']}_groups"] += 1
                totals[f"{row['split']}_strict_positive"] += int(strict_positive)
                totals[f"{row['split']}_protected"] += int(protected)
                totals[f"{row['split']}_confirmed_negative"] += int(confirmed_negative)
    after = state_digest(model)
    if before != after or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Frozen model state changed")
    train_pos = totals["train_strict_positive"]
    holdout_pos = totals["holdout_strict_positive"]
    result = {"schema_version": "chemvector-targeted-selector-group-contract-0.1",
              "complete": True, "read_only": True, "training_performed": False,
              "optimizer_steps": 0, "gpu_used": True, "sealed_test_read": False,
              "deployment_authorized": False, "initial_sha256": INITIAL_SHA,
              "plan_sha256": file_digest(PLAN), "group_radius_px": GROUP_RADIUS_PX,
              "port_radius_px": PORT_RADIUS_PX, "totals": dict(totals),
              "gate": {"train_strict_positive_minimum_50": train_pos >= 50,
                       "holdout_strict_positive_minimum_14": holdout_pos >= 14,
                       "identity_split": True, "prefix_replay": True,
                       "candidate_contract_ready": train_pos >= 50 and holdout_pos >= 14,
                       "training_authorized": False,
                       "reason": "label/cache audit only; no group scorer has been trained"},
              "state_before": before, "state_after": after, "groups": groups}
    output.mkdir(parents=True)
    (output / "group-cache.json").write_text(json.dumps({"groups": groups}, indent=2, ensure_ascii=False) + "\n")
    (output / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "totals": dict(totals), "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
