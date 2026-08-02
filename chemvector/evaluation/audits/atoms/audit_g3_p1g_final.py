#!/usr/bin/env python3
"""Independently replay and verify the P1-G final checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import torch

from chemvector.components.atoms.g3_top128_set_ranker import G3Top128SetRanker
from chemvector.components.graph.joint.g3_hierarchical_graph import (
    G3HierarchicalGraphConfig,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1c_top128_set_ranker as p1c
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d
from chemvector.training.atoms import train_g3_p1g_equivalent_set_matching as p1g
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    load_frozen_vision_checkpoint,
    module_state_sha256,
)


DEFAULT_OUTPUT = Path(
    "outputs/audits/chemvector-g3-p1g-equivalent-set-matching-train5k-final"
)
FINAL_CHECKPOINT = p1g.DEFAULT_OUTPUT / "step-006250.pt"
EXPECTED_CHECKPOINT_SHA256 = (
    "1416b06d1cac1cddf6248d5d99d2b8883f26a96f8f20362c7619b98807a9f9a1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config(payload: dict) -> G3HierarchicalGraphConfig:
    allowed = {field.name for field in fields(G3HierarchicalGraphConfig)}
    return G3HierarchicalGraphConfig(
        **{key: value for key, value in payload["g3_config"].items() if key in allowed}
    )


def _assert_equal(expected, actual, path="root") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise AssertionError(f"P1-G audit mapping differs at {path}")
        for key in expected:
            _assert_equal(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError(f"P1-G audit list differs at {path}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _assert_equal(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or abs(expected - actual) > 1e-12:
            raise AssertionError(f"P1-G audit float differs at {path}")
        return
    if expected != actual:
        raise AssertionError(f"P1-G audit value differs at {path}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P1-G final audit requires CUDA BF16 replay")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("P1-G final audit output directory is not empty")
    checkpoint_sha = _sha256(FINAL_CHECKPOINT)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("P1-G final checkpoint SHA differs")
    payload = torch.load(FINAL_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        payload["model_type"] != p1g.MODEL_TYPE
        or payload["global_step"] != 6_250
        or payload["samples_seen"] != 25_000
        or payload["p1g_gate"]["decision"] != "causal_no_go"
    ):
        raise RuntimeError("P1-G final payload identity differs")
    if payload["trainer_sha256"] != _sha256(Path(p1g.__file__)):
        raise RuntimeError("P1-G trainer source differs from checkpoint payload")
    matcher_path = (
        Path(p1g.__file__).parents[2]
        / "components/atoms/g3_equivalent_candidate_matching.py"
    )
    if payload["audit"]["implementation"]["matcher_sha256"] != _sha256(
        matcher_path
    ):
        raise RuntimeError("P1-G matcher source differs from checkpoint payload")

    device = torch.device("cuda")
    config = _config(payload)
    ranker = G3Top128SetRanker(config).to(device)
    ranker.load_state_dict(payload["set_ranker"], strict=True)
    ranker.eval()
    atom_head, _, p1a_payload, _ = p1d.load_p1a_boundary_ranker(device)
    vision_path = Path(p1a_payload["lineage"]["vision"]["path"])
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(vision_path, device)
    vision.requires_grad_(False).eval()
    if (
        module_state_sha256(atom_head)
        != payload["lineage"]["p1a"]["atom_head_state_sha256"]
        or module_state_sha256(vision)
        != payload["lineage"]["vision"]["state_sha256"]
        or vision_sha != payload["lineage"]["vision"]["sha256"]
    ):
        raise RuntimeError("P1-G frozen lineage differs")

    train, _ = p1a.load_train_manifest()
    route, _ = p1a.load_route_manifest()
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    train_geometry = p1g.evaluate_geometry(
        atom_head, ranker, vision, train, collator, device=device
    )
    route_geometry = p1g.evaluate_geometry(
        atom_head, ranker, vision, route, collator, device=device
    )
    route_metrics = p1c.evaluate_route(
        atom_head,
        ranker,
        vision,
        route,
        collator,
        device=device,
        precision="bf16",
        batch_size=4,
        require_zero_parity=False,
    )
    route_metrics["step"] = 6_250
    _assert_equal(payload["train_geometry_metrics"], train_geometry, "train")
    _assert_equal(payload["route_geometry_metrics"], route_geometry, "route_geometry")
    _assert_equal(payload["final_route_metrics"], route_metrics, "route")
    gate = p1g._gate(
        payload["p1g_gate"]["protocol_checks"],
        train_geometry,
        route_geometry,
        route_metrics,
    )
    _assert_equal(payload["p1g_gate"], gate, "gate")
    report = {
        "schema_version": "g3-p1g-final-audit-0.1",
        "checkpoint": str(FINAL_CHECKPOINT.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "trainer_sha256": payload["trainer_sha256"],
        "matcher_sha256": payload["audit"]["implementation"]["matcher_sha256"],
        "checkpoint_schedule": sorted(
            path.name for path in p1g.DEFAULT_OUTPUT.glob("step-*.pt")
        ),
        "train_geometry_replay": train_geometry,
        "route_geometry_replay": route_geometry,
        "route_replay": route_metrics,
        "gate_replay": gate,
        "all_saved_metrics_exact": True,
        "forbidden_data_read": False,
        "decision": "audit_passed_causal_no_go",
    }
    expected_schedule = [
        f"step-{step:06d}.pt" for step in p1g.CHECKPOINT_STEPS
    ]
    if report["checkpoint_schedule"] != expected_schedule:
        raise RuntimeError("P1-G checkpoint schedule differs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"decision": report["decision"], "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
