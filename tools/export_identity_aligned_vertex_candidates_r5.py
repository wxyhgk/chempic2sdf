#!/usr/bin/env python3
"""Export scored r5 vertex candidates from an already reviewed shadow state.

This is inference-only.  It does not retrain, read labels, or alter the
production checkpoint.  The older r5 exporter wrote ``score=0`` for every
candidate; this utility repairs that artifact by preserving the vertex
heatmap score emitted by the frozen reviewed head.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.core.model import ChemVectorConfig, ChemVertexDetector
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.data import load_sample
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.identity_vertex_decoder import decode_dense


DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
DEFAULT_STATE = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5/vertex-state.pt"
DEFAULT_OUTPUT = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5-scored/candidates.jsonl"
MAXIMUM = 128


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state(path: Path, head: torch.nn.Module, model: torch.nn.Module) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != "chemvector-r5-vertex-state-0.1":
        raise ValueError("invalid r5 vertex state schema")
    if payload.get("initial_sha256") != INITIAL_SHA:
        raise ValueError("vertex state is not bound to reviewed initial.pt")
    if any(key in payload for key in ("optimizer", "optimizer_state_dict", "sealed_test")):
        raise ValueError("optimizer/sealed state is forbidden in inference artifact")
    head.load_state_dict(payload["head_state_dict"], strict=True)
    block = payload.get("vision_stride4_block_state_dict")
    if block is not None:
        model.vision.stages[0][-1].load_state_dict(block, strict=True)


def run(state: Path = DEFAULT_STATE, output: Path = DEFAULT_OUTPUT, *, cpu: bool = False) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if file_sha(INITIAL) != INITIAL_SHA:
        raise RuntimeError("initial checkpoint digest changed")
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line]
    rows = [row for row in rows if row.get("render_status") == "accepted"]
    if len(rows) != 1170 or len({row["sample_key"] for row in rows}) != 1170:
        raise ValueError("r5 accepted panel changed")
    device = torch.device("cpu" if cpu else "cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --cpu for explicit slow inference")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to(device).eval().requires_grad_(False)
    head = ChemVertexDetector(ChemVectorConfig(vision_dims=(96, 192, 384, 512), vertex_feature_dim=64)).to(device)
    load_state(state, head, model)
    head.eval().requires_grad_(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("x", encoding="utf-8") as handle, torch.inference_mode():
        for row in sorted(rows, key=lambda item: item["sample_key"]):
            sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
            pixels = image_tensor(sample["image"])[None].to(device)
            decoded = decode_dense(
                head(model.vision.forward_stroke_resolution(pixels)),
                threshold=.05, maximum_vertices=MAXIMUM, nms_kernel_size=3,
                with_scores=True,
            )[0]
            record = {
                "sample_key": row["sample_key"], "source": "shadow",
                "candidates": [
                    {"x": float(x), "y": float(y), "score": float(score)}
                    for x, y, score in decoded
                ],
            }
            json.dump(record, handle, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            count += 1
    report = {
        "schema_version": "chemvector-r5-scored-candidate-export-0.1",
        "complete": True, "training_performed": False, "optimizer_steps": 0,
        "sealed_test_read": False, "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA, "vertex_state_sha256": file_sha(state),
        "candidate_artifact_sha256": file_sha(output), "rows": count,
        "score_source": "vertex_heatmap_sigmoid_peak",
        "target_fields_written": False,
        "next": "run isolated integration shadow; do not pass candidates to production graph",
    }
    report_path = output.with_name("export-report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "report": str(report_path), "rows": count}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertex-state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run(args.vertex_state, args.output, cpu=args.cpu)
