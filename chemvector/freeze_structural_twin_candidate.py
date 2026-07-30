#!/usr/bin/env python3
"""Freeze the sole v0.10.15 candidate before route-confirm is opened."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from chemvector.audit_safe_atom_suppression_oracle import DATA_COMMITMENT
from chemvector.audit_structural_twin_certificate import (
    DEFAULT_RANKING,
    RANKING_SHA256,
    _sha256_file,
)
from chemvector.structural_twin_certificate import StructuralTwinCertificateConfig
from chemvector.train_safe_atom_action import ACTION_CACHE_SHA256, DEFAULT_CACHE


DEFAULT_ROUTE_SELECT = (
    "outputs/audits/chemvector-v01015-structural-twin-route-select/frozen_report.json"
)
DEFAULT_OUTPUT = (
    "outputs/candidates/chemvector-v01015-structural-twin/candidate_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-cache", default=DEFAULT_CACHE)
    parser.add_argument("--ranking-checkpoint", default=DEFAULT_RANKING)
    parser.add_argument("--route-select-report", default=DEFAULT_ROUTE_SELECT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"Candidate manifest already exists: {output}")
    if _sha256_file(args.action_cache) != ACTION_CACHE_SHA256:
        raise ValueError("Candidate action cache differs")
    if _sha256_file(args.ranking_checkpoint) != RANKING_SHA256:
        raise ValueError("Candidate ranking checkpoint differs")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if tracked_status:
        raise RuntimeError("Tracked worktree must be clean before candidate freeze")
    route_select_path = Path(args.route_select_report)
    route_select = json.loads(route_select_path.read_text(encoding="utf-8"))
    if route_select.get("status") != "ROUTE-SELECT-GO" or not route_select.get(
        "gate", {}
    ).get("passed"):
        raise ValueError("Route-select candidate did not pass")
    if route_select.get("code_git_revision") != revision:
        raise ValueError("Route-select report was produced by another revision")
    cache = torch.load(args.action_cache, map_location="cpu", weights_only=False)
    manifest = {
        "schema_version": "structural-twin-candidate-0.1",
        "status": "FROZEN-ROUTE-CONFIRM-CANDIDATE",
        "code_git_revision": revision,
        "data_commitment_sha256": DATA_COMMITMENT,
        "action_cache": args.action_cache,
        "action_cache_sha256": ACTION_CACHE_SHA256,
        "route_select_report": str(route_select_path),
        "route_select_report_sha256": _sha256_file(route_select_path),
        "direct_checkpoint": cache["direct_checkpoint"],
        "direct_checkpoint_sha256": cache["direct_checkpoint_sha256"],
        "source_checkpoint": cache["source_checkpoint"],
        "source_checkpoint_sha256": _sha256_file(cache["source_checkpoint"]),
        "grounded_graph_checkpoint": cache["grounded_graph_checkpoint"],
        "grounded_graph_checkpoint_sha256": _sha256_file(
            cache["grounded_graph_checkpoint"]
        ),
        "ranking_checkpoint": args.ranking_checkpoint,
        "ranking_checkpoint_sha256": RANKING_SHA256,
        "atom_threshold": float(cache["atom_threshold"]),
        "presence_threshold": float(cache["presence_threshold"]),
        "certificate": asdict(StructuralTwinCertificateConfig()),
        "maximum_actions_per_image": 1,
        "seed": args.seed,
        "route_confirm_read": False,
        "sealed_test_read": False,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_json_bytes(manifest)
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
