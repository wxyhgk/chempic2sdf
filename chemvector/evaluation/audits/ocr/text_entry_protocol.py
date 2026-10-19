"""Frozen inputs for the small, historical text-entry checkpoint comparison."""

from collections import defaultdict
import json
from pathlib import Path

import chemvector.external_ocr_r0_contracts as contract
from chemvector.external_ocr_r0_crop_bundle import load_crop_manifest


DATASET = Path("outputs/evaluations/external-ocr-r0-predicted-bbox/dataset")
TEXT_CHECKPOINT = Path(
    "outputs/checkpoints/chemvector-v089-text-bbox-offset-9k-45k/best.pt"
)
TEXT_SHA256 = "a41902fdd5e7d0246ca9329bea42810f6c61bf6b79a04c64e62fb45b4348bab1"
ARMS = ("atom_conditioned", "independent_text")
MOLECULES = 50


def select_samples(selection, count=MOLECULES):
    grouped = defaultdict(list)
    for key in selection["sample_keys"]:
        grouped[selection["sample_metadata"][key]["molecule_id"]].append(key)
    if len(grouped) < count:
        raise ValueError("Insufficient molecule groups")
    ordered = sorted(grouped, key=lambda identity: (
        contract.sha256_bytes(f"text-entry-v1:{identity}".encode()), identity
    ))[:count]
    result = {}
    for identity in ordered:
        keys = grouped[identity]
        variants = {selection["sample_metadata"][key]["variant"] for key in keys}
        if len(keys) != 4 or variants != {"original", "rdkit", "coordgen", "stress"}:
            raise ValueError("Selection requires four complete variants per molecule")
        result.update({key: selection["sample_metadata"][key] for key in keys})
    return dict(sorted(result.items()))


def source_files():
    paths = sorted(Path("chemvector").rglob("*.py"))
    paths += sorted(Path("Data/sdf").glob("*.py"))
    return {str(path): contract.sha256_file(path) for path in paths}


def freeze(output_dir):
    import torch
    from chemvector.train_joint_bond_set_multidomain import FRESH_DEV_SHARDS
    from chemvector.external_ocr_r0_truth import validate_reveal_inputs

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Freeze requires an empty output directory")
    _, manifest, _ = load_crop_manifest(DATASET)
    validate_reveal_inputs(list(FRESH_DEV_SHARDS))
    selection = json.loads(Path(manifest["selection"]["path"]).read_text())
    contract.validate_frozen_file(TEXT_CHECKPOINT, TEXT_SHA256, "independent text")
    contract.validate_frozen_file(
        contract.CTC_CHECKPOINT, contract.V0109_CTC_SHA256, "shared OCR"
    )
    links = ("grounded_graph_checkpoint", "grounded_selector_checkpoint",
             "query_checkpoint", "base_checkpoint")
    checkpoints = {}
    path = TEXT_CHECKPOINT
    for field in (*links, None):
        checkpoints[str(path)] = contract.sha256_file(path)
        if field is not None:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            path = Path(payload[field])
    checkpoints[str(contract.CTC_CHECKPOINT)] = contract.V0109_CTC_SHA256
    protocol = {
        "schema_version": "text-entry-comparison-v1",
        "frozen_utc": contract.utc_now(),
        "purpose": "paired historical checkpoint diagnostic, not architecture A/B",
        "selection_method": "smallest sha256(text-entry-v1:molecule_id) within R0",
        "sample_metadata": select_samples(selection),
        "molecules": MOLECULES,
        "arms": list(ARMS),
        "dataset_dir": str(DATASET),
        "manifest_sha256": contract.FROZEN_CROP_MANIFEST_SHA256,
        "checkpoints": checkpoints,
        "shards": {path: contract.sha256_file(path) for path in FRESH_DEV_SHARDS},
        "sources": source_files(),
        "detector": {"threshold": 0.3, "maximum_objects": 128,
                     "precision": "cuda_bfloat16", "batch_size": 4},
        "ocr": {"checkpoint": str(contract.CTC_CHECKPOINT), "device": "cpu",
                "precision": "float32", "decoder": "greedy", "batch_size": 32,
                "crop_context_px": 4, "crop_size": [192, 48]},
        "metrics": {"primary_iou": 0.5, "sensitivity_iou": [0.25, 0.5, 0.75],
                    "match": "max feasible cardinality, then max IoU; text blind",
                    "coverage_minimum": 0.9},
        "gates": {"grounded_exact_delta_pp_minimum": 2.0,
                  "precision_regression_pp_maximum": 1.0,
                  "extra_boxes_per_image_increase_maximum": 1.0,
                  "stress_grounded_exact_must_not_decrease": True},
        "training_authorized": False,
        "promotion_authorized": False,
        "graph_integration_performed": False,
        "limitations": ["reused fresh-dev, not unseen confirm",
                        "different detector training histories",
                        "box overlap is not glyph-ink coverage",
                        "no graph/connection accuracy claim",
                        "no new pretrained OCR model evaluated"],
    }
    contract.write_new_json(output / "protocol.json", protocol)
    return {"protocol": str(output / "protocol.json"),
            "sha256": contract.sha256_file(output / "protocol.json"),
            "samples": len(protocol["sample_metadata"])}


def load_protocol(output_dir):
    path = Path(output_dir) / "protocol.json"
    protocol = json.loads(path.read_text())
    if protocol["schema_version"] != "text-entry-comparison-v1":
        raise ValueError("Unexpected protocol")
    for section in ("sources", "checkpoints", "shards"):
        for source, digest in protocol[section].items():
            contract.validate_frozen_file(source, digest, section)
    contract.validate_frozen_file(
        DATASET / "manifest.json", protocol["manifest_sha256"], "baseline manifest"
    )
    if protocol["training_authorized"] or protocol["promotion_authorized"]:
        raise ValueError("This experiment does not authorize training or promotion")
    return protocol
