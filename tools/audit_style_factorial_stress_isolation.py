"""Read-only audit of stress-vs-core style strata.

This audit deliberately does not load image payloads, checkpoints, sealed data,
or run a model.  It joins the already-published endpoint replay aggregates
with the identity-bound manifests and the public style metadata.  The goal is
to make the stress isolation decision reproducible without another GPU run.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "outputs/audits/chemvector-style-factorial-target-pool-20261007-r1"
CORE = ROOT / "outputs/audits/chemvector-style-factorial-core-stream-20261007-r1"
REPLAY = ROOT / "outputs/audits/chemvector-style-factorial-endpoint-reachability-20261007-r1/report.json"
DATA = ROOT / "Data/sdf/chemvector_joint_v1/style_train_100k"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-stress-isolation-20261007-r1/report.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ratio(num: int, den: int) -> float:
    return float(num) / den if den else 0.0


def nested_counts(rows: list[dict], fields: tuple[str, ...]) -> dict:
    out: dict[str, dict[str, int]] = {}
    for field in fields:
        out[field] = dict(sorted(Counter(str(r["profile"][field]) for r in rows).items()))
    return out


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    pool_dev = json.loads((POOL / "dev-manifest.json").read_text(encoding="utf-8"))["rows"]
    pool_train = json.loads((POOL / "train-manifest.json").read_text(encoding="utf-8"))["rows"]
    dev_core = json.loads((CORE / "dev-core-manifest.json").read_text(encoding="utf-8"))["rows"]
    dev_stress = json.loads((CORE / "dev-stress-manifest.json").read_text(encoding="utf-8"))["rows"]
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    metadata = {}
    with (DATA / "samples.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            metadata[row["sample_key"]] = row

    # The manifests are the identity contract; metadata is used only for style
    # dimensions and operation labels.  No image or checkpoint is accessed.
    if len(pool_dev) != 1600 or len(dev_core) != 1200 or len(dev_stress) != 400:
        raise ValueError("unexpected manifest cardinality")
    if {r["sample_key"] for r in dev_core} & {r["sample_key"] for r in dev_stress}:
        raise ValueError("core/stress sample overlap")
    for row in [*dev_core, *dev_stress]:
        if row["sample_key"] not in metadata:
            raise ValueError(f"missing style metadata: {row['sample_key']}")

    by_variant = replay["by_variant"]
    core_names = ("coordgen", "original", "rdkit")
    core = {k: sum(int(by_variant[v][k]) for v in core_names) for k in
            ("target_atoms", "raw_matched", "selected_matched", "raw_gap", "selector_drop")}
    stress = {k: int(by_variant["stress"][k]) for k in core}
    performance = {}
    for name, values in (("core", core), ("stress", stress)):
        performance[name] = {
            **values,
            "raw_recall": ratio(values["raw_matched"], values["target_atoms"]),
            "selected_recall": ratio(values["selected_matched"], values["target_atoms"]),
            "selector_drop_target_rate": ratio(values["selector_drop"], values["target_atoms"]),
            "selector_drop_raw_rate": ratio(values["selector_drop"], values["raw_matched"]),
        }

    def style_rows(manifest: list[dict]) -> list[dict]:
        result = []
        for row in manifest:
            meta = metadata[row["sample_key"]]
            # Keep only stable, non-payload style metadata in the audit.
            result.append({"sample_key": row["sample_key"], "variant": row["variant"],
                           "profile": row["profile"], "stress_operation": meta["stress_operation"],
                           "font_size": int(meta["font_size"]),
                           "multiple_bond_offset": float(meta["multiple_bond_offset"]),
                           "rotation_degrees": float(meta["rotation_degrees"])})
        return result

    core_style, stress_style = style_rows(dev_core), style_rows(dev_stress)
    operations = Counter(r["stress_operation"] for r in stress_style)
    train_variants = Counter(r["variant"] for r in pool_train)
    report = {
        "schema_version": "chemvector-style-factorial-stress-isolation-0.1",
        "complete": True, "read_only": True, "gpu_used": False,
        "training_performed": False, "sealed_test_read": False,
        "image_payloads_read": False, "checkpoint_read": False,
        "source_sha256": {"pool_dev": sha256(POOL / "dev-manifest.json"),
                          "pool_train": sha256(POOL / "train-manifest.json"),
                          "replay": sha256(REPLAY), "samples_csv": sha256(DATA / "samples.csv")},
        "identity": {
            "dev_samples": len(pool_dev), "train_samples": len(pool_train),
            "core_dev_samples": len(dev_core), "stress_dev_samples": len(dev_stress),
            "core_stress_sample_overlap": 0,
            "train_dev_molecule_overlap": len({r["molecule_id"] for r in pool_train}
                                                & {r["molecule_id"] for r in pool_dev}),
            "train_variant_counts": dict(sorted(train_variants.items())),
        },
        "performance_from_frozen_initial_replay": performance,
        "stress_operation_counts_dev": dict(sorted(operations.items())),
        "stress_style_strata": nested_counts(stress_style, ("font_bucket", "line_width_bucket",
                                                               "rotation_bucket", "multiple_bond_offset_bucket",
                                                               "color_mode")),
        "core_style_strata": nested_counts(core_style, ("font_bucket", "line_width_bucket",
                                                          "rotation_bucket", "multiple_bond_offset_bucket",
                                                          "color_mode")),
        "generation_contrast": {
            "stress_font_size_range": [min(r["font_size"] for r in stress_style), max(r["font_size"] for r in stress_style)],
            "core_font_size_range": [min(r["font_size"] for r in core_style), max(r["font_size"] for r in core_style)],
            "stress_multiple_bond_offset_range": [min(r["multiple_bond_offset"] for r in stress_style), max(r["multiple_bond_offset"] for r in stress_style)],
            "core_multiple_bond_offset_range": [min(r["multiple_bond_offset"] for r in core_style), max(r["multiple_bond_offset"] for r in core_style)],
            "stress_has_adversarial_layout_operations": True,
        },
        "decision": {
            "core_only_training_admissible_after_cpu_gate": True,
            "stress_training_in_first_adaptation_admissible": False,
            "stress_policy": "held_out_robustness_stratum_until_prefix_retention_gate",
            "required_gate": {
                "old_480_prefix_selector_retention_no_regression": True,
                "each_core_variant_strict_precision_at_least": 0.85,
                "each_core_variant_strict_recall_at_least": 0.50,
                "stress_reported_separately": True,
                "stress_must_not_be_used_as_confirmed_negative": True,
            },
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "performance": performance,
                      "stress_operations": dict(sorted(operations.items()))}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
