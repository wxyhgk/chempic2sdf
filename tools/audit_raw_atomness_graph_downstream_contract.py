"""Audit the raw-atomness -> frozen-graph downstream interface.

This is a read-only protocol audit.  It deliberately does not run the GPU
shadow, load a model, or manufacture graph metrics from candidate labels.  The
style-factorial candidate cache contains scalar atom proposals only; a graph
downstream experiment is admissible only after a separately replayed artifact
contains the frozen bond/pair outputs aligned by ``sample_key``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "outputs/audits/chemvector-style-factorial-raw-objectness-gpu-shadow-20261014-r3"
DEFAULT_CACHE = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/candidates.jsonl"
DEFAULT_INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
DEFAULT_OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-raw-objectness-graph-downstream-contract-20261016-r1/report.json"

INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
EXPECTED_FEATURES = ("raw_score", "presence_logit", "x_norm", "y_norm")
FORBIDDEN_FEATURES = ("nearest_distance_px", "target_atom_id", "layers", "graph")
ALLOWED_LABELS = {"protected", "strict_positive", "hard_negative", "near_unknown", "selected_unmatched"}
REQUIRED_GRAPH_FIELDS = (
    "sample_key",
    "atom_points",
    "atom_keep_mask",
    "bond_endpoints",
    "bond_presence_logits",
    "bond_type_ids",
    "bond_mask",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stream_cache(path: Path) -> tuple[Counter, set[str], set[str], list[str]]:
    counts: Counter = Counter()
    samples: set[str] = set()
    variants: set[str] = set()
    errors: list[str] = []
    seen: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON: {exc}")
                continue
            key = (str(row.get("sample_key")), int(row.get("slot", -1)))
            if key in seen:
                errors.append(f"line {line_no}: duplicate candidate key {key}")
            seen.add(key)
            missing = [name for name in EXPECTED_FEATURES if name not in row]
            if missing:
                errors.append(f"line {line_no}: missing sidecar features {missing}")
            label = row.get("label")
            if label not in ALLOWED_LABELS:
                errors.append(f"line {line_no}: unknown label {label!r}")
            # These fields may exist for audit only, but are never accepted as
            # sidecar inputs.  Presence here is expected in this historical
            # cache; the report records the separation explicitly.
            counts[f"{row.get('replay_split')}/{row.get('variant')}/{label}"] += 1
            samples.add(str(row.get("sample_key")))
            variants.add(str(row.get("variant")))
    return counts, samples, variants, errors


def _graph_artifact_status(graph_artifact: Path | None) -> dict:
    if graph_artifact is None:
        return {
            "provided": False,
            "available": False,
            "reason": "no separately replayed frozen bond/pair artifact was supplied",
            "required_fields": list(REQUIRED_GRAPH_FIELDS),
        }
    if not graph_artifact.exists():
        return {
            "provided": True,
            "available": False,
            "path": str(graph_artifact),
            "reason": "graph artifact path does not exist",
            "required_fields": list(REQUIRED_GRAPH_FIELDS),
        }
    # The contract accepts JSONL only for this audit.  We inspect one record,
    # never labels or sealed data, and require all graph fields before a run can
    # be admitted.
    first: dict | None = None
    with graph_artifact.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                first = json.loads(line)
                break
    missing = [name for name in REQUIRED_GRAPH_FIELDS if first is None or name not in first]
    return {
        "provided": True,
        "available": not missing,
        "path": str(graph_artifact),
        "sha256": digest(graph_artifact),
        "required_fields": list(REQUIRED_GRAPH_FIELDS),
        "missing_fields": missing,
        "reason": None if not missing else "graph artifact does not satisfy the frozen-output schema",
    }


def audit(*, release_dir: Path = DEFAULT_RELEASE, cache: Path = DEFAULT_CACHE,
          initial: Path = DEFAULT_INITIAL, graph_artifact: Path | None = None) -> dict:
    release_path = release_dir / "release.json"
    receipt_path = release_dir / "receipt.json"
    errors: list[str] = []
    if not release_path.exists() or not receipt_path.exists():
        raise FileNotFoundError(f"release/receipt missing under {release_dir}")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if digest(initial) != INITIAL_SHA:
        errors.append("production initial checkpoint SHA256 changed")
    if release.get("source_initial_sha256") != INITIAL_SHA:
        errors.append("release is not bound to the frozen initial checkpoint")
    if release.get("candidate_cache_sha256") != digest(cache):
        errors.append("release candidate cache SHA256 does not match supplied cache")
    if receipt.get("release_sha256") != digest(release_path):
        errors.append("receipt/release SHA256 binding mismatch")
    if receipt.get("training_run_authorized") is not False:
        errors.append("GPU shadow receipt unexpectedly authorizes training")
    if receipt.get("deployment_authorized") is not False:
        errors.append("GPU shadow receipt unexpectedly authorizes deployment")

    counts, samples, variants, cache_errors = _stream_cache(cache)
    errors.extend(cache_errors)
    release_features = tuple(release.get("feature_names", ()))
    if release_features != EXPECTED_FEATURES:
        errors.append(f"release feature order mismatch: {release_features}")
    forbidden = tuple(release.get("forbidden_features", ()))
    if set(FORBIDDEN_FEATURES) - set(forbidden):
        errors.append("release forbidden-feature contract is incomplete")

    graph = _graph_artifact_status(graph_artifact)
    checks = {
        "initial_frozen": not any("initial" in e for e in errors),
        "release_bound": not any("release" in e or "receipt" in e for e in errors),
        "cache_valid": not cache_errors,
        "sidecar_feature_contract": release_features == EXPECTED_FEATURES and set(FORBIDDEN_FEATURES) <= set(forbidden),
        "graph_artifact_schema": graph["available"],
        "no_training_started": receipt.get("training_run_authorized") is False,
        "no_deployment": receipt.get("deployment_authorized") is False,
    }
    # A graph downstream experiment cannot be admitted until every check is
    # true.  In particular, atomness CPU/GPU metrics alone do not satisfy the
    # graph-level evidence requirement.
    result = {
        "schema_version": "chemvector-style-factorial-raw-objectness-graph-downstream-contract-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "release_dir": str(release_dir),
        "release_sha256": digest(release_path),
        "receipt_sha256": digest(receipt_path),
        "cache_sha256": digest(cache),
        "cache_samples": len(samples),
        "cache_variants": sorted(variants),
        "cache_counts": dict(counts),
        "sidecar_inputs": list(EXPECTED_FEATURES),
        "forbidden_sidecar_inputs": list(FORBIDDEN_FEATURES),
        "graph_artifact": graph,
        "required_graph_metrics": [
            "raw_atom_precision",
            "raw_atom_recall",
            "selected_atom_count",
            "bond_endpoint_reachability",
            "graph_connectedness",
            "sanitize_rate",
            "exact_graph_recovery",
            "component_recovery",
            "stress_holdout",
        ],
        "checks": checks,
        "errors": errors,
        "admission": {
            "graph_downstream_training_ready": all(checks.values()),
            "reason": "frozen graph artifact is required before graph-level evaluation or training",
            "next": "replay frozen bond/pair outputs aligned by sample_key; then run an independent graph report",
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--graph-artifact", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(release_dir=args.release_dir, cache=args.cache, initial=args.initial,
                   graph_artifact=args.graph_artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "checks": result["checks"],
                      "admission": result["admission"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
