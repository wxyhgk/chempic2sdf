"""Fixed exposed-development identities, model hashes, and 2x2x2 intervention contract."""

import json
from pathlib import Path
import tarfile

import torch

from chemvector.external_ocr_r0_contracts import sha256_file, validate_frozen_file, utc_now, write_new_json
from chemvector.evaluation.audits.ocr.set_rejected_text.protocol import BASELINE, DATASET, checkpoint_chain, source_hashes
from chemvector.evaluation.audits.ocr.set_rejected_text.inputs import index_selected_images

DIRECT = Path("outputs/checkpoints/chemvector-v0103-direct-atom-pair-style10k/selected.pt")


def freeze(output):
    if output.exists():
        raise FileExistsError("Use a new diagnostic directory")
    old = json.loads((BASELINE / "protocol.json").read_text())
    report = json.loads((BASELINE / "report.json").read_text())
    validate_frozen_file(DATASET / "manifest.json", old["manifest_sha256"], "R0 manifest")
    manifest = json.loads((DATASET / "manifest.json").read_text())
    validate_frozen_file(DATASET / "atom-objects.jsonl", manifest["artifacts"]["atom_objects"]["sha256"], "R0 deployed atoms")
    for name, digest in report["artifacts"].items():
        validate_frozen_file(BASELINE / name, digest, "frozen baseline")
    for path, digest in old["sources"].items():
        validate_frozen_file(path, digest, "historical baseline source")
    for path, digest in old["shards"].items():
        if Path(path).name not in {"fresh_dev-000000.tar", "fresh_dev-000001.tar"}:
            raise ValueError("Unexpected input shard")
        validate_frozen_file(path, digest, "exposed dev shard")
    order, index = index_selected_images(old["shards"], old["sample_metadata"])
    if len(order) != 200:
        raise ValueError("Require fixed 200-image development population")
    checkpoints = checkpoint_chain()
    checkpoints[str(DIRECT)] = sha256_file(DIRECT)
    config = {"schema": "e2e-bottleneck-factorial-0.1", "frozen_utc": utc_now(),
              "sources": source_hashes(), "checkpoints": checkpoints, "shards": old["shards"],
              "sample_order": order, "image_index": index, "sample_metadata": old["sample_metadata"],
              "inputs": {str(BASELINE / n): report["artifacts"][n] for n in ("protocol.json", "predictions.jsonl")},
              "arms": [f"A{a}B{b}C{c}" for a in (0, 1) for b in (0, 1) for c in (0, 1)],
              "A": "exact GT atom set/anchors; sample frozen dense features at anchors; rerun unchanged bond and text heads",
              "B": "GT chemical atom attributes on geometrically matched existing nodes; not raw OCR truth",
              "C": "GT bonds/types on existing matched nodes; cannot add missing atoms or remove extra atoms",
              "baseline": "v0103 Direct AtomPair + frozen R0 A CPU-FP32 greedy OCR records; no twin suppression or ring fusion",
              "detector_precision": "CUDA BF16", "detector_batch_size": 6,
              "ocr": old["ocr"], "matching_radius_px": 6.,
              "primary": "canonical sanitized SDF graph exact excluding stereo, preserving elements/isotopes/charges/bond orders",
              "secondary": "stereochemical SDF graph exact", "full_oracle_required_primary": 200,
              "control_failure": "report control failures; do not rank architecture bottlenecks",
              "inference_failures": "count in full denominator; no cherry-picking or retries",
              "training_performed": False, "promotion_authorized": False,
              "scope": "reused development diagnostic, not causal population effect or deployment candidate",
              "gpu": torch.cuda.get_device_name(0), "torch_version": torch.__version__}
    config["inputs"][str(DATASET / "atom-objects.jsonl")] = sha256_file(DATASET / "atom-objects.jsonl")
    output.mkdir(parents=True)
    write_new_json(output / "protocol.json", config)
    return config


def verify(output):
    config = json.loads((output / "protocol.json").read_text())
    for group in ("sources", "checkpoints", "shards", "inputs"):
        for path, digest in config[group].items():
            validate_frozen_file(path, digest, group)
    return config


def targets(config):
    """Only the fixed selected samples' oracle JSON/SDF are decoded."""
    from rdkit import Chem
    from Data.sdf.chemscene_v0_2 import build_chemscene_v0_2

    archives = {path: tarfile.open(path, "r:") for path in config["shards"]}
    result = {}
    try:
        for key in config["sample_order"]:
            archive = archives[config["image_index"][key]["shard"]]
            payload = {}
            for suffix in ("chemmol.json", "render.json", "layers.json", "sdf"):
                member = archive.getmember(f"{key}.{suffix}")
                if not member.isfile():
                    raise ValueError("Non-regular selected member")
                with archive.extractfile(member) as stream:
                    payload[suffix] = stream.read()
            chemical, render, layers = [json.loads(payload[s]) for s in ("chemmol.json", "render.json", "layers.json")]
            metadata = config["sample_metadata"][key]
            if chemical["molecule_id"] != metadata["molecule_id"] or render["variant"] != metadata["variant"]:
                raise ValueError("Selected target identity mismatch")
            mol = Chem.MolFromMolBlock(payload["sdf"].decode(), sanitize=True, removeHs=True)
            if mol is None:
                raise ValueError(f"Invalid source SDF: {key}")
            result[key] = {"scene": build_chemscene_v0_2(chemical, render, layers), "mol": mol}
    finally:
        for archive in archives.values():
            archive.close()
    return result
