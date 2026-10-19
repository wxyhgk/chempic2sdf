#!/usr/bin/env python3
"""Validate and score the frozen, target-free r17 graph artifact.

This audit consumes the exported JSONL artifact; it never calls the model.
Targets are read only from the public sample graph files for an independent
identity check.  The artifact is therefore byte-bound by both its SHA256 and
the known-clean manifest SHA256.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib, json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "outputs/audits/chemvector-r17-known-clean-graph-artifact-v1/graph-artifact.jsonl"
MANIFEST = ROOT / "outputs/audits/chemvector-r17-selector-known-clean-shadow-cohort-v1/manifest.json"
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923/samples"
OUT = ROOT / "outputs/audits/chemvector-r17-known-clean-artifact-bound-replay-v1/report.json"
ARTIFACT_SHA = "e25c5d3bd7b499e192cea52582452efffdc1a2907b39ab4d8615e2030a1a1441"
RADIUS = 16.0
THRESHOLD = 0.72
REQUIRED = {"sample_key", "split", "profile_id", "atom_points", "atom_keep_mask",
            "atom_raw_mask", "bond_endpoints", "bond_presence_logits",
            "bond_type_ids", "bond_type_logits", "bond_mask"}
FORBIDDEN = {"target", "label", "graph", "layers", "molecule_id", "atom_ids"}
TYPE_NAMES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def target(key: str):
    layers = json.loads((SOURCE / f"{key}.layers.json").read_text())
    graph = json.loads((SOURCE / f"{key}.graph.json").read_text())
    atoms = sorted(layers["atoms"], key=lambda x: int(x["atom_id"]))
    centers = torch.tensor([a["center"] for a in atoms], dtype=torch.float32)
    edges = {}
    for b in graph["bonds"]:
        e = tuple(sorted((int(b["begin_atom"]), int(b["end_atom"]))))
        edges[e] = str(b.get("chemical_type", "SINGLE")).upper()
    return centers, edges

def nearest(point, centers):
    d = torch.linalg.vector_norm(centers - point[None], dim=1)
    value, index = torch.min(d, dim=0)
    return (int(index), float(value)) if float(value) <= RADIUS else (None, float(value))

def run():
    manifest = json.loads(MANIFEST.read_text())["rows"]
    expected = {r["sample_key"]: (r["split"], r["profile_id"]) for r in manifest}
    artifact_sha = sha(ARTIFACT)
    rows = 0; seen = set(); errors = []; profile = defaultdict(lambda: defaultdict(int))
    forbidden_seen = set()
    with ARTIFACT.open() as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip(): continue
            row = json.loads(line); rows += 1; key = row.get("sample_key")
            if key in seen: errors.append(f"duplicate sample_key {key}")
            seen.add(key)
            missing = REQUIRED - set(row); forbidden = FORBIDDEN & set(row)
            if missing: errors.append(f"line {line_no} missing {sorted(missing)}")
            if forbidden: forbidden_seen |= forbidden; errors.append(f"line {line_no} target fields {sorted(forbidden)}")
            if key not in expected: errors.append(f"unknown sample_key {key}"); continue
            if (row.get("split"), row.get("profile_id")) != expected[key]: errors.append(f"identity mismatch {key}")
            lengths = {k: len(row[k]) for k in ("atom_points", "atom_keep_mask", "atom_raw_mask")}
            lengths.update({k: len(row[k]) for k in ("bond_endpoints", "bond_presence_logits", "bond_type_ids", "bond_type_logits", "bond_mask")})
            if lengths["atom_points"] != 128 or any(lengths[k] != 128 for k in ("atom_keep_mask", "atom_raw_mask")): errors.append(f"atom shape {key}: {lengths}")
            if any(lengths[k] != 192 for k in ("bond_endpoints", "bond_presence_logits", "bond_type_ids", "bond_type_logits", "bond_mask")): errors.append(f"bond shape {key}: {lengths}")
            centers, truth = target(key)
            p = profile[row["profile_id"]]; p["images"] += 1; p["target_edges"] += len(truth)
            atom_points = torch.tensor(row["atom_points"], dtype=torch.float32) * 768.0
            atom_keep = torch.tensor(row["atom_keep_mask"], dtype=torch.bool)
            p["selected_atoms"] += int(atom_keep.sum())
            pred = set(); typed_true = 0; true = 0; selected = 0; reachable = 0
            for endpoint, logit, type_id, mask in zip(row["bond_endpoints"], row["bond_presence_logits"], row["bond_type_ids"], row["bond_mask"]):
                # ``bond_scores`` from structure_forward is already a
                # probability; the exporter preserves it under the stable
                # presence-logit field name for schema compatibility.
                if not mask or float(logit) < THRESHOLD: continue
                selected += 1
                a, da = nearest(torch.tensor(endpoint[0], dtype=torch.float32) * 768.0, centers)
                b, db = nearest(torch.tensor(endpoint[1], dtype=torch.float32) * 768.0, centers)
                if a is None or b is None or a == b: continue
                reachable += 1; edge = tuple(sorted((a, b))); pred.add(edge)
                if edge in truth:
                    true += 1
                    # bond_type_ids is argmax over the concatenated
                    # display(0..3)+chemical(4..7) channels.
                    tid = int(type_id) - 4 if int(type_id) >= 4 else int(type_id)
                    kind = TYPE_NAMES[tid] if 0 <= tid < len(TYPE_NAMES) else "UNKNOWN"
                    typed_true += int(kind == truth[edge])
            p["selected_bonds"] += selected; p["reachable_bonds"] += reachable; p["predicted_edges"] += len(pred); p["true_edges"] += true; p["typed_true_edges"] += typed_true
    for p in profile.values():
        p["endpoint_reachability"] = p["reachable_bonds"] / max(p["selected_bonds"], 1)
        p["edge_precision"] = p["true_edges"] / max(p["predicted_edges"], 1)
        p["edge_recall"] = p["true_edges"] / max(p["target_edges"], 1)
        p["typed_edge_recall"] = p["typed_true_edges"] / max(p["target_edges"], 1)
    result = {"schema_version": "chemvector-r17-artifact-bound-graph-replay-0.1", "complete": True,
              "read_only": True, "training_performed": False, "gpu_used": False,
              "sealed_test_read": False, "production_checkpoint_written": False,
              "artifact": {"path": str(ARTIFACT.relative_to(ROOT)), "sha256": artifact_sha, "expected_sha256": ARTIFACT_SHA, "sha_match": artifact_sha == ARTIFACT_SHA},
              "manifest_sha256": sha(MANIFEST), "rows": rows, "manifest_rows": len(manifest), "unique_sample_keys": len(seen),
              "target_free": not forbidden_seen, "forbidden_fields_seen": sorted(forbidden_seen), "profiles": {k: dict(v) for k,v in sorted(profile.items())}, "errors": errors[:100],
              "gate": {"artifact_sha_match": artifact_sha == ARTIFACT_SHA, "manifest_exact": rows == len(manifest) == len(seen) and not (set(expected) - seen), "schema_and_identity": not errors, "target_free": not forbidden_seen, "production_go": False}}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n"); print(json.dumps({"output": str(OUT), "gate": result["gate"], "profiles": result["profiles"]}, indent=2))

if __name__ == "__main__": run()
