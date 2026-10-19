"""Fail-closed CPU gate for the style-factorial core stream.

This is a protocol audit, not a training launcher.  It binds the declared
``original/rdkit/coordgen`` stream to the frozen-initial endpoint replay and
spells out the selector labels that a later candidate replay must publish.
In particular, candidates within 10 px of a rendered atom are
``near_unknown`` and may never enter a negative loss.  Stress is reported as
an independent diagnostic stratum and cannot make the core stream admissible.

The script deliberately does not open raster/shard payloads, checkpoints, or
sealed data, and never grants a GPU receipt.  It is useful before the more
expensive CPU candidate replay because it catches split/style/count drift.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "outputs/audits/chemvector-style-factorial-core-stream-20261007-r1"
REPLAY = ROOT / "outputs/audits/chemvector-style-factorial-endpoint-reachability-20261007-r1/report.json"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-core-cpu-gate-20261010-r1/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
CORE_VARIANTS = ("coordgen", "original", "rdkit")
ALL_VARIANTS = (*CORE_VARIANTS, "stress")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("rows"), list):
        raise ValueError(f"manifest has no rows: {path}")
    return payload["rows"]


def check_manifest(path: Path, expected_split: str, expected_molecules: int,
                   expected_variants: tuple[str, ...]) -> dict:
    data = rows(path)
    counts = Counter(str(row.get("variant")) for row in data)
    molecule_ids = {str(row.get("molecule_id")) for row in data}
    if len(data) != expected_molecules * len(expected_variants):
        raise ValueError(f"{path}: expected {expected_molecules * len(expected_variants)} rows, got {len(data)}")
    if set(counts) != set(expected_variants) or any(counts[v] != expected_molecules for v in expected_variants):
        raise ValueError(f"{path}: style counts changed: {counts}")
    if any(not row.get("sample_key") or not row.get("molecule_id") for row in data):
        raise ValueError(f"{path}: missing identity")
    if len({row["sample_key"] for row in data}) != len(data):
        raise ValueError(f"{path}: duplicate sample key")
    return {"rows": len(data), "molecules": len(molecule_ids),
            "variant_counts": dict(sorted(counts.items())), "split": expected_split}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    stream = json.loads((CORE / "report.json").read_text(encoding="utf-8"))
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    if stream.get("training_performed") or stream.get("training_authorized"):
        raise ValueError("core stream is no longer a read-only proposal")
    if replay.get("sealed_test_read") is not False or replay.get("initial_sha256") != INITIAL_SHA:
        raise ValueError("unsafe or mismatched frozen replay")
    train = check_manifest(CORE / "train-core-manifest.json", "train", 1600, CORE_VARIANTS)
    dev = check_manifest(CORE / "dev-core-manifest.json", "dev", 400, CORE_VARIANTS)
    stress = check_manifest(CORE / "dev-stress-manifest.json", "dev_stress", 400, ("stress",))
    # The endpoint report has one aggregate per variant.  Bind it by exact
    # style name so a future report cannot silently collapse styles.
    by_variant = replay.get("by_variant", {})
    endpoint = {}
    for variant in ALL_VARIANTS:
        row = by_variant.get(variant)
        if row is None:
            raise ValueError(f"missing endpoint row for {variant}")
        target = int(row["target_atoms"])
        endpoint[variant] = {
            "target_atoms": target,
            "raw_recall": float(row["raw_matched"]) / max(target, 1),
            "selected_recall": float(row["selected_matched"]) / max(target, 1),
            "selector_drop_rate": float(row["selector_drop"]) / max(int(row["raw_matched"]), 1),
        }
    # A candidate replay must provide these counts before any scorer is fit.
    # Keeping this explicit prevents near-targets from being relabelled as
    # negatives merely because the strict radius is 6 px.
    label_contract = {
        "coordinate_space": "candidate and rendered atom centers in 768px image coordinates",
        "strict_radius_px": 6.0,
        "near_unknown_radius_px": 10.0,
        "strict_positive": "raw matched <=6px AND selected unmatched; only explicit strict members",
        "confirmed_negative": "raw unmatched AND selected unmatched AND nearest atom >10px",
        "near_unknown": "raw unmatched AND selected unmatched AND nearest atom <=10px",
        "protected": "raw matched AND selected matched (not a loss row)",
        "strict_group_unknown": "excluded from both positive and negative loss",
        "near_unknown_loss_weight": 0.0,
        "hard_negative_loss_weight": 1.0,
        "assignment": "one-to-one Hungarian for audit, no oracle candidate injection",
    }
    # This is intentionally false until a CPU candidate replay emits labels
    # with the above strata.  Endpoint reachability alone cannot establish a
    # precision/recall gate for selector recovery.
    result = {
        "schema_version": "chemvector-style-factorial-core-cpu-gate-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "training_authorized": False, "gpu_used": False, "sealed_test_read": False,
        "initial_sha256": INITIAL_SHA,
        "core_manifest_sha256": {
            "train": digest(CORE / "train-core-manifest.json"),
            "dev": digest(CORE / "dev-core-manifest.json"),
            "stress": digest(CORE / "dev-stress-manifest.json"),
        },
        "replay_sha256": digest(REPLAY), "split": {"train": train, "dev": dev, "stress": stress},
        "endpoint_by_variant": endpoint, "label_contract": label_contract,
        "gate_contract": {
            "per_core_style_strict_precision_ge_085": True,
            "per_core_style_strict_recall_ge_050": True,
            "near_unknown_selected_reported": True,
            "hard_negative_selected_reported": True,
            "prefix_selected_recall_drop_max_pp": 1.0,
            "stress_separate_from_training": True,
        },
        "gate": {
            "manifest_shape": True,
            "identity_disjoint": bool(stream.get("gate", {}).get("identity_disjoint")),
            "endpoint_core_prefix_reachable": all(endpoint[v]["selected_recall"] >= 0.98 for v in CORE_VARIANTS),
            "stress_isolated": True,
            "candidate_label_replay_present": False,
            "training_admissible": False,
            "reason": "CPU candidate replay with strict/near_unknown/hard_negative strata is still required; endpoint recall is not selector precision/recall",
            "next": "run frozen-initial candidate replay on core train/dev; fit CPU visual gate only on strict positives + confirmed far negatives; evaluate every core style and stress separately",
        },
    }
    result["gate"]["prefix_gate_ready"] = bool(result["gate"]["endpoint_core_prefix_reachable"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "gate": result["gate"], "endpoint_by_variant": endpoint}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
