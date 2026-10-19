"""Freeze scope and controls before any candidate fitting or result inspection."""

import json
from pathlib import Path
import platform

import torch

from chemvector.external_ocr_r0_contracts import (
    SOURCE_CHECKPOINT, SOURCE_SHA256, GROUNDED_GRAPH_CHECKPOINT, GROUNDED_GRAPH_SHA256,
    sha256_file, utc_now, validate_frozen_file, write_new_json,
)
from .data import ROOT, SHARD, selection


def freeze(output):
    if output.exists():
        raise FileExistsError("Use a new pilot directory")
    commitment = json.loads((ROOT / "commitment.json").read_text())
    if commitment["commitment_sha256"] != "2082372d253fa309715becfd5cba9e5cdb78eb8d84a5ece4ff1eb73bd67390eb":
        raise ValueError("Unknown training commitment")
    artifacts = [*commitment["tars"], *commitment["samples_csv"]]
    inputs = {str(ROOT / r["path"]): r["sha256"] for r in artifacts
              if r["path"] in ("shards/train-000000.tar", "samples.csv")}
    for path, digest in inputs.items():
        validate_frozen_file(path, digest, "training input")
    inputs[str(ROOT / "commitment.json")] = sha256_file(ROOT / "commitment.json")
    checkpoints = {str(SOURCE_CHECKPOINT): SOURCE_SHA256, str(GROUNDED_GRAPH_CHECKPOINT): GROUNDED_GRAPH_SHA256}
    for path, digest in checkpoints.items():
        validate_frozen_file(path, digest, "frozen checkpoint")
    current = GROUNDED_GRAPH_CHECKPOINT
    for field in ("grounded_selector_checkpoint", "query_checkpoint", "base_checkpoint"):
        current = Path(torch.load(current, map_location="cpu", weights_only=False)[field])
        checkpoints[str(current)] = sha256_file(current)
    paths = sorted(Path("chemvector").rglob("*.py")) + sorted(Path("Data/sdf").glob("*.py"))
    config = {"schema": "recurrent-depth-candidate-pilot-0.1", "frozen_utc": utc_now(),
              "source": str(SOURCE_CHECKPOINT), "graph": str(GROUNDED_GRAPH_CHECKPOINT),
              "sources": {str(p): sha256_file(p) for p in paths}, "inputs": inputs, "checkpoints": checkpoints,
              "selection": selection(), "seeds": [17, 29, 43], "steps": 600, "batch_size": 32,
              "learning_rate": 0.0003, "weight_decay": 0.01, "width": 96, "heads": 4,
              "arms": {"tied1": {"loops": 1, "untied_depth": 1},
                       "tied4": {"loops": 4, "untied_depth": 1},
                       "untied4": {"loops": 4, "untied_depth": 4}},
              "tied4_evaluation_loops": [1, 2, 4, 8], "candidate_radius_px": 6.,
              "selection_logit_threshold": 0., "source_threshold": -.25,
              "training_precision": "CUDA BF16; FP32 weights and loss", "max_training_seconds": 900,
              "gate": {"min_exact_gain_vs_tied1": .01, "max_exact_drop_vs_untied4": .01,
                       "max_fp_increase_per_image": .1, "min_positive_seeds": 2,
                       "max_exact_drop_vs_source": .01},
              "checkpoint_selection": "last fixed step; no validation-based selection or threshold sweep",
              "scope": "internal molecule-disjoint train-pool split; frozen producer may have seen both parts",
              "text_training": False, "graph_or_sdf_evaluated": False, "promotion_authorized": False,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}}
    output.mkdir(parents=True)
    write_new_json(output / "protocol.json", config)
    return config


def load(output):
    config = json.loads((output / "protocol.json").read_text())
    for group in ("sources", "inputs", "checkpoints"):
        for path, digest in config[group].items():
            validate_frozen_file(path, digest, group)
    return config
