"""Artifact-bound identity audit for the oa13/oa33 stress graph artifact.

The artifact itself contains only frozen target-free outputs.  This validator
binds its exact SHA and manifest, then joins public target graphs post hoc to
measure mapped/unmapped edges, duplicate edges, precision and recall.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import torch


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "outputs/audits/chemvector-r17-oa13-oa33-stress-graph-artifact-nms8-v1/graph-artifact.jsonl"
MANIFEST = ROOT / "outputs/audits/chemvector-r17-oa13-oa33-known-clean-stress-manifest-v2/manifest.json"
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923/samples"
OUT = ROOT / "outputs/audits/chemvector-r17-oa13-oa33-stress-artifact-bound-v2"
ARTIFACT_SHA = "8a690c3b92948b51623583c7152fdd44564d02f6662e8827a4352784af387f0c"
RADIUS = 16.0
THRESHOLD = 0.72
TYPE_NAMES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
FORBIDDEN = {"target", "label", "graph", "layers", "molecule_id", "atom_ids"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target(key: str):
    layers = json.loads((SOURCE / f"{key}.layers.json").read_text())
    graph = json.loads((SOURCE / f"{key}.graph.json").read_text())
    atoms = sorted(layers["atoms"], key=lambda value: int(value["atom_id"]))
    centers = torch.tensor([value["center"] for value in atoms], dtype=torch.float32)
    edges = {}
    for bond in graph["bonds"]:
        edge = tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"]))))
        edges[edge] = str(bond.get("chemical_type", "SINGLE")).upper()
    return centers, edges


def nearest(point: torch.Tensor, centers: torch.Tensor):
    distance = torch.linalg.vector_norm(centers - point[None] * 768.0, dim=1)
    value, index = torch.min(distance, dim=0)
    return (int(index), float(value)) if float(value) <= RADIUS else (None, float(value))


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    expected = {row["sample_key"]: (row["split"], row["profile_id"])
                for row in json.loads(MANIFEST.read_text())["rows"]}
    artifact_sha = digest(ARTIFACT)
    rows = 0
    seen = set()
    forbidden_seen = set()
    errors = []
    profile = defaultdict(lambda: defaultdict(int))
    with ARTIFACT.open() as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            key = row.get("sample_key")
            if key in seen:
                errors.append(f"duplicate sample_key {key}")
            seen.add(key)
            forbidden = FORBIDDEN & set(row)
            if forbidden:
                forbidden_seen |= forbidden
                errors.append(f"line {line_no} target fields {sorted(forbidden)}")
            if key not in expected:
                errors.append(f"unknown sample_key {key}")
                continue
            if (row.get("split"), row.get("profile_id")) != expected[key]:
                errors.append(f"identity mismatch {key}")
            centers, truth = target(key)
            block = profile[row["profile_id"]]
            block["images"] += 1
            block["target_atoms"] += len(centers)
            keep = torch.tensor(row["atom_keep_mask"], dtype=torch.bool)
            points = torch.tensor(row["atom_points"], dtype=torch.float32)
            selected = torch.where(keep)[0]
            mapping = {}
            for slot in selected.tolist():
                owner, distance = nearest(points[slot], centers)
                if owner is not None:
                    mapping[slot] = owner
            block["selected_atoms"] += len(selected)
            block["mapped_atoms"] += len(mapping)
            block["target_edges"] += len(truth)
            predicted = set()
            true = typed_true = 0
            unmapped = duplicate = selected_bonds = 0
            for endpoint, probability, type_id, mask in zip(
                    row["bond_endpoints"], row["bond_presence_logits"],
                    row["bond_type_ids"], row["bond_mask"]):
                if not mask or float(probability) < THRESHOLD:
                    continue
                selected_bonds += 1
                left, _ = nearest(torch.tensor(endpoint[0], dtype=torch.float32), centers)
                right, _ = nearest(torch.tensor(endpoint[1], dtype=torch.float32), centers)
                # Endpoint mapping is deliberately checked against selected
                # slots, not only the target center, so unmapped graph edges
                # cannot disappear from the denominator.
                mapped_slots = [slot for slot, owner in mapping.items() if owner == left]
                mapped_slots_right = [slot for slot, owner in mapping.items() if owner == right]
                if left is None or right is None or not mapped_slots or not mapped_slots_right:
                    unmapped += 1
                    continue
                edge = tuple(sorted((left, right)))
                if edge in predicted:
                    duplicate += 1
                    continue
                predicted.add(edge)
                if edge in truth:
                    true += 1
                    tid = int(type_id) - 4 if int(type_id) >= 4 else int(type_id)
                    kind = TYPE_NAMES[tid] if 0 <= tid < len(TYPE_NAMES) else "UNKNOWN"
                    typed_true += int(kind == truth[edge])
            block["selected_bonds"] += selected_bonds
            block["mapped_bonds"] += len(predicted)
            block["unmapped_bonds"] += unmapped
            block["duplicate_bonds"] += duplicate
            block["true_edges"] += true
            block["typed_true_edges"] += typed_true
    for block in profile.values():
        block["edge_precision"] = block["true_edges"] / max(block["mapped_bonds"], 1)
        block["edge_recall"] = block["true_edges"] / max(block["target_edges"], 1)
        block["typed_edge_recall"] = block["typed_true_edges"] / max(block["target_edges"], 1)
        block["endpoint_mapping_rate"] = block["mapped_atoms"] / max(block["selected_atoms"], 1)
    report = {
        "schema_version": "chemvector-r17-oa13-oa33-stress-artifact-bound-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "gpu_used": False, "sealed_test_read": False,
        "production_checkpoint_written": False,
        "artifact": {"sha256": artifact_sha, "expected_sha256": ARTIFACT_SHA,
                     "sha_match": artifact_sha == ARTIFACT_SHA},
        "manifest_sha256": digest(MANIFEST), "rows": rows,
        "manifest_rows": len(expected), "unique_sample_keys": len(seen),
        "target_free": not forbidden_seen, "forbidden_fields_seen": sorted(forbidden_seen),
        "profiles": {key: dict(value) for key, value in sorted(profile.items())},
        "errors": errors[:100],
        "gate": {"artifact_sha_match": artifact_sha == ARTIFACT_SHA,
                 "manifest_exact": rows == len(expected) == len(seen) and not (set(expected) - seen),
                 "schema_and_identity": not errors, "target_free": not forbidden_seen,
                 "production_go": False},
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "profiles": report["profiles"],
                      "gate": report["gate"]}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
