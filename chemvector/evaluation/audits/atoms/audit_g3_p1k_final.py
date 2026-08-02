#!/usr/bin/env python3
"""Independently replay and verify the P1-K final component ranker."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from chemvector.components.atoms.g3_component_set_ranker import G3ComponentSetRanker
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d
from chemvector.training.atoms import train_g3_p1k_component_set_ranker as p1k
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    load_frozen_vision_checkpoint,
    module_state_sha256,
)


DEFAULT_OUTPUT = Path(
    "outputs/audits/chemvector-g3-p1k-component-set-ranker-train5k-final"
)
FINAL_CHECKPOINT = p1k.DEFAULT_OUTPUT / "step-006250.pt"
EXPECTED_CHECKPOINT_SHA256 = (
    "69243466079b326e94c6c1b356c7dc6c2c5c88aaeeda648b1cac343e0263d4ed"
)
EXPECTED_SCHEDULE = {
    "step-001250.pt": "1d5a915804a674b93f221c2422690a71d79b3443f22d6c475df156d741bf349e",
    "step-002500.pt": "80411c20e7a83f2e86f7b871d743f392c418a2f5919993b874dabd826d785e61",
    "step-003750.pt": "2875f7b45e5d9d0da5a4d0aecdf19586a897c3d2593db8cc40eb68b130795407",
    "step-005000.pt": "09d0546e0e4715375161d848392a3f67067c3cc13006d1dead1442d7ffb74f5c",
    "step-006250.pt": EXPECTED_CHECKPOINT_SHA256,
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_equal(expected, actual, path="root"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise AssertionError(f"P1-K mapping differs at {path}")
        for key in expected:
            _assert_equal(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError(f"P1-K list differs at {path}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _assert_equal(left, right, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or abs(expected - actual) > 1e-12:
            raise AssertionError(f"P1-K float differs at {path}")
    elif expected != actual:
        raise AssertionError(f"P1-K value differs at {path}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P1-K final audit requires CUDA BF16")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("P1-K audit output directory is not empty")
    actual_schedule = {
        name: _sha256(p1k.DEFAULT_OUTPUT / name) for name in EXPECTED_SCHEDULE
    }
    if actual_schedule != EXPECTED_SCHEDULE:
        raise RuntimeError("P1-K checkpoint schedule SHA differs")
    payload = torch.load(FINAL_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        payload["model_type"] != p1k.MODEL_TYPE
        or payload["global_step"] != 6_250
        or payload["samples_seen"] != 25_000
        or payload["p1k_gate"]["decision"] != "causal_go_capability_no_go"
    ):
        raise RuntimeError("P1-K final payload identity differs")
    trainer_sha = _sha256(Path(p1k.__file__))
    component_path = (
        Path(p1k.__file__).parents[2] / "components/atoms/g3_component_set_ranker.py"
    )
    component_sha = _sha256(component_path)
    if (
        payload["trainer_sha256"] != trainer_sha
        or payload["audit"]["implementation"]["component_sha256"] != component_sha
    ):
        raise RuntimeError("P1-K fixed source SHA differs")

    device = torch.device("cuda")
    ranker = G3ComponentSetRanker().to(device)
    ranker.load_state_dict(payload["component_ranker"], strict=True)
    ranker.eval()
    atom_head, _, p1a_payload, _ = p1d.load_p1a_boundary_ranker(device)
    vision_path = Path(p1a_payload["lineage"]["vision"]["path"])
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(vision_path, device)
    vision.requires_grad_(False).eval()
    if (
        module_state_sha256(atom_head) != payload["lineage"]["p1a"]["atom_state_sha256"]
        or module_state_sha256(vision) != payload["lineage"]["vision"]["state_sha256"]
        or vision_sha != payload["lineage"]["vision"]["sha256"]
    ):
        raise RuntimeError("P1-K frozen lineage differs")
    train, _ = p1a.load_train_manifest()
    route, _ = p1a.load_route_manifest()
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    train_metrics = p1k.evaluate_component_ranker(
        atom_head, ranker, vision, train, collator, device=device,
        require_zero_parity=False,
    )
    route_metrics = p1k.evaluate_component_ranker(
        atom_head, ranker, vision, route, collator, device=device,
        require_zero_parity=False,
    )
    _assert_equal(payload["final_train_metrics"], train_metrics, "train")
    _assert_equal(payload["final_route_metrics"], route_metrics, "route")
    gate = p1k._gate(
        payload["p1k_gate"]["protocol_checks"],
        payload["step0_train_metrics"],
        payload["step0_route_metrics"],
        train_metrics,
        route_metrics,
    )
    _assert_equal(payload["p1k_gate"], gate, "gate")
    report = {
        "schema_version": "g3-p1k-final-audit-0.1",
        "checkpoint": str(FINAL_CHECKPOINT.resolve()),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_schedule_sha256": actual_schedule,
        "trainer_sha256": trainer_sha,
        "component_sha256": component_sha,
        "train_replay": train_metrics,
        "route_replay": route_metrics,
        "gate_replay": gate,
        "all_saved_metrics_exact": True,
        "forbidden_data_read": False,
        "decision": "audit_passed_causal_go_capability_no_go",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"decision": report["decision"], "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
