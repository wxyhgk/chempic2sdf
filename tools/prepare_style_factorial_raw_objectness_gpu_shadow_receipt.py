"""Prepare (but do not authorize) a bounded raw-objectness GPU shadow receipt."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "outputs/audits/chemvector-style-factorial-core-stream-20261007-r1"
CACHE = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/candidates.jsonl"
CPU = ROOT / "outputs/audits/chemvector-style-factorial-raw-objectness-cpu-preflight-20261010-r1/report.json"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-raw-objectness-gpu-shadow-20261014-r1"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
CORE_STYLES = ("coordgen", "original", "rdkit")
STEPS = 100
WALL = 300


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(output)
    cpu = json.loads(CPU.read_text(encoding="utf-8"))
    stream = json.loads((CORE / "report.json").read_text(encoding="utf-8"))
    if not cpu["gate"].get("strict_training_ready"):
        raise ValueError("CPU raw-objectness gate is not ready")
    if cpu.get("gpu_used") or cpu.get("sealed_test_read") is not False or cpu.get("checkpoint_written"):
        raise ValueError("CPU evidence is unsafe for admission")
    if digest(INITIAL) != INITIAL_SHA:
        raise ValueError("production initial checkpoint changed")
    train = json.loads((CORE / "train-core-manifest.json").read_text(encoding="utf-8"))["rows"]
    dev = json.loads((CORE / "dev-core-manifest.json").read_text(encoding="utf-8"))["rows"]
    stress = json.loads((CORE / "dev-stress-manifest.json").read_text(encoding="utf-8"))["rows"]
    train_ids, dev_ids = {r["molecule_id"] for r in train}, {r["molecule_id"] for r in dev}
    if train_ids & dev_ids:
        raise ValueError("core identity overlap")
    counts = Counter()
    train_values = []
    seen = set()
    with CACHE.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            key = (row["sample_key"], int(row["slot"]))
            if key in seen:
                raise ValueError(f"duplicate candidate key at {line_no}")
            seen.add(key)
            split, label = row["replay_split"], row["label"]
            counts[f"{split}/{row['variant']}/{label}"] += 1
            if split == "train_core" and label in {"protected", "strict_positive", "hard_negative"}:
                train_values.append([float(row["raw_score"]), float(row["presence_logit"]), float(row["x_norm"]), float(row["y_norm"])])
    if not train_values:
        raise ValueError("empty train normalization rows")
    import numpy as np
    values = np.asarray(train_values, dtype=np.float64)
    mean = values.mean(0).tolist(); std = np.maximum(values.std(0), 1e-5).tolist()
    release = {
        "schema_version": "chemvector-style-factorial-raw-objectness-gpu-shadow-release-0.1",
        "purpose": "bounded_shadow_only", "source_initial_sha256": INITIAL_SHA,
        "candidate_cache_sha256": digest(CACHE), "cpu_gate_sha256": digest(CPU),
        "core_stream_sha256": digest(CORE / "report.json"),
        "train_manifest_sha256": digest(CORE / "train-core-manifest.json"),
        "dev_manifest_sha256": digest(CORE / "dev-core-manifest.json"),
        "stress_manifest_sha256": digest(CORE / "dev-stress-manifest.json"),
        "feature_names": ["raw_score", "presence_logit", "x_norm", "y_norm"],
        "forbidden_features": ["nearest_distance_px", "target_atom_id", "layers", "graph"],
        "label_protocol": {"positive": ["protected", "strict_positive"], "negative": ["hard_negative"],
                           "masked": ["near_unknown", "selected_unmatched"]},
        "train_normalization": {"source": "train_core known labels only", "mean": mean, "std": std},
        "model_architecture": {"type": "shared_scalar_mlp_atomness_only", "input_dim": 4,
                               "hidden_dim": 32, "activation": "GELU", "output_heads": ["atomness"]},
        "counts": dict(counts), "identity": {"train_molecules": len(train_ids), "dev_molecules": len(dev_ids), "overlap": 0},
        "core_styles": list(CORE_STYLES), "stress_policy": "holdout_only",
        "production": {"initial_frozen": True, "selector_frozen": True, "ocr_frozen": True, "checkpoint_write": False, "deployment": False},
        "sealed_test_read": False,
    }
    output.mkdir(parents=True)
    (output / "release.json").write_text(json.dumps(release, indent=2, ensure_ascii=False) + "\n")
    release_sha = digest(output / "release.json")
    receipt = {
        "schema_version": "chemvector-style-factorial-raw-objectness-gpu-shadow-receipt-0.1",
        "receipt_type": "proposed_independent_gpu_shadow", "training_run_authorized": False,
        "data_release_authorized": True, "deployment_authorized": False,
        "device_count": 1, "maximum_optimizer_steps": STEPS, "maximum_wall_seconds": WALL,
        "automatic_retry": False, "threshold_search": False, "checkpoint_write": False,
        "model_scope": "scalar atomness sidecar only; frozen initial/selector/OCR",
        "model_architecture": release["model_architecture"],
        "optimizer": {"class": "AdamW", "lr": 0.002, "weight_decay": 0.001, "seed": 20261014, "batch_size": 2048},
        "fixed_gate": 0.0, "train_split": "train_core only", "evaluation": ["dev_core per style", "dev_stress holdout"],
        "release_sha256": release_sha,
        "sealed_test_read": False, "production_checkpoint_sha256": INITIAL_SHA,
        "activation": "requires explicit separate confirmation; this receipt does not start a process",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    report = {"schema_version": "chemvector-style-factorial-raw-objectness-gpu-shadow-preparation-0.1",
              "complete": True, "training_performed": False, "gpu_used": False,
              "sealed_test_read": False, "release_sha256": receipt["release_sha256"],
              "receipt_sha256": digest(output / "receipt.json"), "training_run_authorized": False,
              "deployment_authorized": False, "next": "await explicit activation before bounded GPU shadow"}
    (output / "preparation-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "report": report, "steps": STEPS, "wall_seconds": WALL}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT))
    run(Path(parser.parse_args().output))
