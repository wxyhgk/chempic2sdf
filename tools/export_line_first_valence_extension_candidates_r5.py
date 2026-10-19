"""Inference-only export of frozen r5 vertex candidates for the extension."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemvector.core.model import ChemVectorConfig, ChemVertexDetector  # noqa: E402
from chemvector.training.curriculum.chemical_head_probe import INITIAL  # noqa: E402
from chemvector.training.curriculum.checkpoints import file_digest  # noqa: E402
from chemvector.training.curriculum.data import load_sample  # noqa: E402
from chemvector.training.curriculum.pixels import image_tensor  # noqa: E402
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe  # noqa: E402
from tools.identity_vertex_decoder import decode_dense  # noqa: E402

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-valence-extension-r5"
STATE = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5/vertex-state.pt"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-valence-extension-r5/candidates.jsonl"
REPORT = OUT.with_name("candidate-export-report.json")
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> dict:
    if OUT.exists() or REPORT.exists():
        raise FileExistsError(OUT)
    if file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("production checkpoint changed")
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != 216 or len({row["sample_key"] for row in rows}) != 216:
        raise ValueError("extension accounting changed")
    model, _, _ = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    head = ChemVertexDetector(ChemVectorConfig(vision_dims=(96, 192, 384, 512), vertex_feature_dim=64))
    payload = torch.load(STATE, map_location="cpu", weights_only=False)
    if payload.get("initial_sha256") != INITIAL_SHA:
        raise ValueError("vertex state is not bound to reviewed initial")
    head.load_state_dict(payload["head_state_dict"], strict=True)
    block = payload.get("vision_stride4_block_state_dict")
    if block is not None:
        model.vision.stages[0][-1].load_state_dict(block, strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval().requires_grad_(False)
    head.to(device).eval().requires_grad_(False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as handle, torch.inference_mode():
        for index, row in enumerate(sorted(rows, key=lambda x: x["sample_key"]), 1):
            sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
            pixels = image_tensor(sample["image"])[None].to(device)
            decoded = decode_dense(head(model.vision.forward_stroke_resolution(pixels)),
                                   threshold=.05, maximum_vertices=128, nms_kernel_size=3,
                                   with_scores=True)[0]
            json.dump({"sample_key": row["sample_key"], "source": "shadow",
                       "candidates": [{"x": float(x), "y": float(y), "score": float(score)}
                                      for x, y, score in decoded]}, handle, sort_keys=True)
            handle.write("\n")
            if index % 40 == 0:
                print(json.dumps({"progress": index, "device": str(device)}), flush=True)
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-valence-extension-candidate-export-r5-0.1",
        "complete": True, "training_performed": False, "optimizer_steps": 0,
        "gpu_used": device.type == "cuda", "sealed_test_read": False,
        "production_checkpoint_written": False, "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA, "initial_checkpoint_sha256_after": file_digest(INITIAL),
        "vertex_state_sha256": digest(STATE), "candidate_artifact_sha256": digest(OUT),
        "rows": len(rows), "target_fields_written": False,
        "proposal_contract": "frozen r5 vertex head inference only; no labels or target fields in artifact",
        "next": "run extension Hungarian/edge sidecar/valence guard CPU roundtrip; no production merge",
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
