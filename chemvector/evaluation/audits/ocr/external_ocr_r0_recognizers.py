"""Frozen CTC and Paddle recognizer adapters for external OCR R0."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
from PIL import Image

import chemvector.external_ocr_r0_contracts as contract
from chemvector.external_ocr_r0_crop_bundle import load_crop_manifest


def _tensor_from_crop(crop: Image.Image):
    import torch

    array = np.asarray(crop, dtype=np.float32) / 255.0
    return (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"total": 0.0, "p50": None, "p95": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "total": sum(values),
        "p50": statistics.median(ordered),
        "p95": ordered[p95_index],
    }


def recognizer_contract_sha256() -> str:
    ctc_source_path = (
        Path(__file__).resolve().parents[3] / "components" / "ocr" / "ctc_ocr.py"
    )
    payload = {
        "arms": contract.ARM_NAMES,
        "ctc": {
            "checkpoint": contract.V0109_CTC_SHA256,
            "beam_size": contract.CTC_BEAM_SIZE,
            "top_paths": contract.CTC_TOP_PATHS,
            "token_beam": contract.CTC_TOKEN_BEAM,
        },
        "crop_manifest": contract.FROZEN_CROP_MANIFEST_SHA256,
        "paddle_models": contract.PADDLE_MODELS,
        "paddle_model_hashes": contract.PADDLE_MODEL_HASHES,
        "recognize_source": inspect.getsource(recognize_arm),
        "tensor_source": inspect.getsource(_tensor_from_crop),
        "decode_arm_source": inspect.getsource(_decode_ctc_arm),
        "arm_row_source": inspect.getsource(_arm_row),
        "ctc_decoder_source": {
            "path": str(ctc_source_path.relative_to(ctc_source_path.parents[2])),
            "sha256": contract.sha256_file(ctc_source_path),
        },
        "crop_loader_source": inspect.getsource(load_crop_manifest),
    }
    return contract.sha256_bytes(contract.canonical_json_bytes(payload))


def recognize_arm(
    arm: str,
    dataset_dir: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    if arm not in contract.ARM_NAMES:
        raise ValueError(f"Unknown frozen recognizer arm: {arm}")
    root, _, rows = load_crop_manifest(dataset_dir)
    started = contract.utc_now()
    output_rows = []
    model_files: list[dict[str, Any]] = []
    if arm.startswith("ctc_"):
        import torch

        from chemvector.ctc_ocr import decode_ctc_prefix_beam, decode_ctc_with_score
        from chemvector.evaluate_direct_atom_ocr import load_ctc

        contract.validate_frozen_file(
            contract.CTC_CHECKPOINT,
            contract.V0109_CTC_SHA256,
            "CTC checkpoint",
        )
        device = torch.device("cpu")
        model, _ = load_ctc(str(contract.CTC_CHECKPOINT), device)
        batch_size = 64
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            crops = [
                _tensor_from_crop(
                    Image.open(root / row["crop"]["path"]).convert("RGB")
                )
                for row in batch_rows
            ]
            before = time.perf_counter()
            batch = torch.stack(crops).to(device)
            with torch.inference_mode():
                logits = model(batch)
            elapsed_ms = (
                (time.perf_counter() - before) * 1000.0 / max(len(batch_rows), 1)
            )
            for row, model_output in zip(batch_rows, logits.float()):
                candidates = _decode_ctc_arm(arm, model_output)
                output_rows.append(
                    _arm_row(
                        arm,
                        row,
                        "chemvector-v0.10.9-ctc",
                        elapsed_ms,
                        candidates,
                    )
                )
            if start and start % (batch_size * 20) == 0:
                print(f"recognized={start}/{len(rows)} arm={arm}", flush=True)
        model_files = [
            {
                "path": str(contract.CTC_CHECKPOINT),
                "sha256": contract.V0109_CTC_SHA256,
            }
        ]
    else:
        from paddleocr import TextRecognition

        model_name = contract.PADDLE_MODELS[arm]
        recognizer = TextRecognition(
            model_name=model_name,
            device="cpu",
            enable_mkldnn=False,
        )
        batch_size = 64
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            images = [
                np.asarray(
                    Image.open(root / row["crop"]["path"]).convert("RGB")
                )
                for row in batch_rows
            ]
            before = time.perf_counter()
            predictions = list(recognizer.predict(images, batch_size=batch_size))
            elapsed_ms = (
                (time.perf_counter() - before) * 1000.0 / max(len(batch_rows), 1)
            )
            if len(predictions) != len(batch_rows):
                raise RuntimeError(
                    f"Paddle returned {len(predictions)} predictions for "
                    f"{len(batch_rows)} crops"
                )
            for row, prediction in zip(batch_rows, predictions):
                result = prediction.json["res"]
                raw_text = str(result["rec_text"])
                score = float(result["rec_score"])
                if not math.isfinite(score):
                    raise ValueError(
                        f"Paddle returned a non-finite score: {row['object_key']}"
                    )
                candidates = [
                    {
                        "rank": 0,
                        "text": raw_text,
                        "score": score,
                        "score_kind": "paddle_greedy_character_mean",
                    }
                ]
                output_rows.append(
                    _arm_row(arm, row, model_name, elapsed_ms, candidates)
                )
            if start and start % (batch_size * 20) == 0:
                print(f"recognized={start}/{len(rows)} arm={arm}", flush=True)
        close = getattr(recognizer, "close", None)
        if callable(close):
            close()
        cache = Path.home() / ".paddlex" / "official_models" / model_name
        for name in ("inference.yml", "inference.json", "inference.pdiparams"):
            path = cache / name
            if not path.is_file():
                raise FileNotFoundError(f"Cached Paddle model artifact is missing: {path}")
            artifact_sha = contract.sha256_file(path)
            if artifact_sha != contract.PADDLE_MODEL_HASHES[arm][name]:
                raise ValueError(f"Frozen Paddle model artifact SHA256 differs: {path}")
            model_files.append({"path": str(path), "sha256": artifact_sha})

    output_rows.sort(key=lambda row: row["object_key"])
    if [row["object_key"] for row in output_rows] != [
        row["object_key"] for row in rows
    ]:
        raise ValueError("Recognizer output cardinality or keys differ from crop manifest")
    output = Path(output)
    contract.write_new_jsonl(output, output_rows)
    metadata = {
        "schema_version": contract.ARM_SCHEMA_VERSION,
        "arm": arm,
        "phase": "candidate_coverage_only",
        "crop_manifest": {
            "path": str(Path(dataset_dir) / "manifest.json"),
            "sha256": contract.sha256_file(Path(dataset_dir) / "manifest.json"),
        },
        "records": {
            "path": str(output),
            "sha256": contract.sha256_file(output),
            "rows": len(output_rows),
        },
        "model_files": model_files,
        "implementation_contract_sha256": recognizer_contract_sha256(),
        "latency_ms": _latency_summary(
            [float(row["latency_ms"]) for row in output_rows]
        ),
        "started_utc": started,
        "finished_utc": contract.utc_now(),
        "training_performed": False,
        "routing_performed": False,
        "replacement_performed": False,
    }
    metadata_path = output.with_suffix(".metadata.json")
    contract.write_new_json(metadata_path, metadata)
    return {"arm": arm, "records": len(output_rows), "output": str(output)}


def _decode_ctc_arm(arm: str, model_output) -> list[dict[str, Any]]:
    from chemvector.ctc_ocr import decode_ctc_prefix_beam, decode_ctc_with_score

    if arm == "ctc_v0109_greedy":
        text, score = decode_ctc_with_score(model_output)
        return [
            {
                "rank": 0,
                "text": text,
                "score": score,
                "score_kind": "normalized_ctc_log_probability",
            }
        ]
    beam = decode_ctc_prefix_beam(
        model_output,
        beam_size=contract.CTC_BEAM_SIZE,
        top_paths=contract.CTC_TOP_PATHS,
        maximum_bytes=24,
        token_beam=contract.CTC_TOKEN_BEAM,
    )
    return [
        {
            "rank": rank,
            "text": item.text,
            "score": item.log_score,
            "score_kind": "ctc_log_probability",
        }
        for rank, item in enumerate(beam)
    ]


def _arm_row(
    arm: str,
    crop_row: dict[str, Any],
    model_id: str,
    latency_ms: float,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": contract.ARM_SCHEMA_VERSION,
        "arm": arm,
        "object_key": crop_row["object_key"],
        "crop_sha256": crop_row["crop"]["sha256"],
        "model_id": model_id,
        "latency_ms": latency_ms,
        "error": None,
        "candidates": candidates,
    }


def expected_arm_model(arm: str) -> tuple[str, dict[str, str]]:
    if arm.startswith("ctc_"):
        return "chemvector-v0.10.9-ctc", {
            contract.CTC_CHECKPOINT.name: contract.V0109_CTC_SHA256
        }
    return contract.PADDLE_MODELS[arm], contract.PADDLE_MODEL_HASHES[arm]


def validate_arm_artifact(
    arm: str,
    path: str | Path,
    crop_manifest_path: Path,
) -> list[dict[str, Any]]:
    record_path = Path(path)
    if arm not in contract.FROZEN_ARM_BUNDLE_SHA256:
        raise RuntimeError(f"Recognizer arm has not been frozen: {arm}")
    actual_record_sha = contract.sha256_file(record_path)
    metadata_path = record_path.with_suffix(".metadata.json")
    if contract.arm_bundle_sha256(record_path) != contract.FROZEN_ARM_BUNDLE_SHA256[arm]:
        raise ValueError(f"Recognizer records+metadata bundle SHA256 differs: {arm}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != contract.ARM_SCHEMA_VERSION
        or metadata.get("arm") != arm
    ):
        raise ValueError(f"Recognizer metadata identity differs: {arm}")
    if metadata.get("phase") != "candidate_coverage_only":
        raise ValueError(f"Recognizer metadata phase differs: {arm}")
    if metadata.get("implementation_contract_sha256") != recognizer_contract_sha256():
        raise ValueError(f"Recognizer implementation contract differs: {arm}")
    for flag in ("training_performed", "routing_performed", "replacement_performed"):
        if metadata.get(flag) is not False:
            raise ValueError(f"Recognizer protocol flag must be false: {arm}/{flag}")
    rows = contract.read_jsonl(record_path)
    if metadata.get("records") != {
        "path": str(record_path),
        "sha256": actual_record_sha,
        "rows": len(rows),
    }:
        raise ValueError(f"Recognizer metadata record commitment differs: {arm}")
    if (
        metadata.get("crop_manifest", {}).get("sha256")
        != contract.FROZEN_CROP_MANIFEST_SHA256
    ):
        raise ValueError(f"Recognizer metadata crop commitment differs: {arm}")
    if Path(metadata["crop_manifest"]["path"]) != crop_manifest_path:
        raise ValueError(f"Recognizer metadata crop path differs: {arm}")
    _, expected_hashes = expected_arm_model(arm)
    observed_hashes = {
        Path(item["path"]).name: item["sha256"]
        for item in metadata.get("model_files", [])
    }
    if observed_hashes != expected_hashes:
        raise ValueError(f"Recognizer metadata model artifacts differ: {arm}")
    expected_model_id, _ = expected_arm_model(arm)
    for row in rows:
        _validate_arm_row(arm, row, expected_model_id)
    return rows


def _validate_arm_row(
    arm: str,
    row: dict[str, Any],
    expected_model_id: str,
) -> None:
    if row.get("schema_version") != contract.ARM_SCHEMA_VERSION:
        raise ValueError(f"Recognizer row schema differs: {arm}")
    if row.get("arm") != arm or row.get("model_id") != expected_model_id:
        raise ValueError(f"Recognizer row model identity differs: {arm}")
    if row.get("error") is not None:
        raise ValueError(f"Recognizer row contains an error: {arm}/{row.get('object_key')}")
    if not math.isfinite(float(row.get("latency_ms", math.nan))):
        raise ValueError(f"Recognizer row latency is non-finite: {arm}")
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Recognizer row has no candidate: {arm}")
    if arm != "ctc_v0109_top8" and len(candidates) != 1:
        raise ValueError(f"Recognizer row candidate count differs: {arm}")
    if arm == "ctc_v0109_top8" and len(candidates) > contract.CTC_TOP_PATHS:
        raise ValueError("CTC top-8 arm emitted more than eight candidates")
    if [item.get("rank") for item in candidates] != list(range(len(candidates))):
        raise ValueError(f"Recognizer candidate ranks are not contiguous: {arm}")
    texts = [item.get("text") for item in candidates]
    if any(not isinstance(text, str) for text in texts) or len(texts) != len(set(texts)):
        raise ValueError(f"Recognizer candidate texts are invalid or duplicated: {arm}")
    if any(
        not math.isfinite(float(item.get("score", math.nan))) for item in candidates
    ):
        raise ValueError(f"Recognizer candidate score is non-finite: {arm}")
