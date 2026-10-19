"""Execute frozen text detection and common OCR without target-based selection."""

from collections import defaultdict
import hashlib
from pathlib import Path

import chemvector.external_ocr_r0_contracts as contract
from chemvector.external_ocr_r0_crop_bundle import load_crop_manifest
from chemvector.text_entry_protocol import ARMS, TEXT_CHECKPOINT, load_protocol


def selected_images(protocol):
    from chemvector.data import ChemVectorIterableDataset
    from chemvector.external_ocr_r0_truth import validate_reveal_sample

    seen = set()
    for sample in ChemVectorIterableDataset(
        list(protocol["shards"]), sequence_version="0.4",
        serialize_target=False, load_raster_layers=False,
    ):
        key = sample["key"]
        if key not in protocol["sample_metadata"]:
            continue
        if key in seen:
            raise ValueError("Duplicate selected image")
        validate_reveal_sample(sample, protocol["sample_metadata"][key])
        if sample["image"].size != (768, 768):
            raise ValueError("Image is not in the frozen 768px coordinate frame")
        seen.add(key)
        yield sample
        if len(seen) == len(protocol["sample_metadata"]):
            break
    if seen != set(protocol["sample_metadata"]):
        raise ValueError("Missing selected images")


def model_digest(models):
    import torch

    digest = hashlib.sha256()
    for model_index, model in enumerate(models):
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(f"{model_index}:{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1)
                          .view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_detector(device):
    import torch
    from chemvector.pretrain_grounded_graph import load_frozen_grounded_pipeline
    from chemvector.text_object import TextObjectBBoxHead

    payload = torch.load(TEXT_CHECKPOINT, map_location="cpu", weights_only=False)
    if payload["head_type"] != "bbox":
        raise ValueError("Independent checkpoint is not a bbox detector")
    graph = torch.load(payload["grounded_graph_checkpoint"], map_location="cpu",
                       weights_only=False)
    model, _, _, _ = load_frozen_grounded_pipeline(
        Path(graph["grounded_selector_checkpoint"]), device
    )
    head = TextObjectBBoxHead(payload["input_channels"]).to(device)
    head.load_state_dict(payload["text_object_head"], strict=True)
    vision = model.vision_encoder
    for module in (vision, head):
        module.eval().requires_grad_(False)
    return vision, head


def predict(output_dir):
    import torch
    from PIL import Image
    from chemvector.ctc_ocr import bbox_crop, decode_ctc
    from chemvector.evaluate_direct_atom_ocr import load_ctc
    from chemvector.image_io import image_tensor
    from chemvector.text_object import decode_text_objects

    output = Path(output_dir)
    protocol = load_protocol(output)
    if (output / "predictions.jsonl").exists():
        raise FileExistsError("Predictions already exist; use summarize, not re-forward")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; check sandbox device access")
    torch.set_num_threads(4)
    dataset, _, baseline = load_crop_manifest(protocol["dataset_dir"])
    grouped = defaultdict(list)
    for row in baseline:
        if row["sample_key"] in protocol["sample_metadata"]:
            grouped[row["sample_key"]].append(row)
    vision, head = load_detector(torch.device("cuda"))
    ocr, _ = load_ctc(protocol["ocr"]["checkpoint"], torch.device("cpu"))
    before = model_digest((vision, head, ocr))
    crop_dir = output / "crops"
    crop_dir.mkdir(exist_ok=True)
    rows, pending = [], []

    def consume(samples):
        pixels = torch.stack([image_tensor(sample["image"]) for sample in samples]).cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = head(vision.forward_stroke_resolution(pixels))
            replay = head(vision.forward_stroke_resolution(pixels))
        if any(not prediction[key].equal(replay[key]) for key in prediction):
            raise ValueError("Independent detector replay differs")
        objects = decode_text_objects(
            prediction, threshold=protocol["detector"]["threshold"],
            maximum_objects=protocol["detector"]["maximum_objects"],
        )
        for sample, detected in zip(samples, objects):
            key = sample["key"]
            for baseline_row in grouped[key]:
                box = baseline_row["bbox"]
                rows.append({
                    "sample_key": key, "arm": ARMS[0],
                    "object_key": f"A:{baseline_row['object_key']}",
                    "bbox": [box[name] for name in ("x0", "y0", "x1", "y1")],
                    "score": baseline_row["role_score"],
                    "crop_path": str(dataset / baseline_row["crop"]["path"]),
                    "crop_sha256": baseline_row["crop"]["sha256"],
                })
            for index, item in enumerate(detected):
                box = [768 * (item["bbox_x"] - item["half_width"]),
                       768 * (item["bbox_y"] - item["half_height"]),
                       768 * (item["bbox_x"] + item["half_width"]),
                       768 * (item["bbox_y"] + item["half_height"])]
                crop_path = crop_dir / f"{key}-b{index:03d}.png"
                crop = bbox_crop(sample["image"].convert("RGB"), box, context=4.0)
                crop.save(crop_path)
                rows.append({
                    "sample_key": key, "arm": ARMS[1],
                    "object_key": f"B:{key}:{index:03d}", "bbox": box,
                    "score": item["score"], "crop_path": str(crop_path),
                    "crop_sha256": contract.sha256_file(crop_path),
                })

    for sample in selected_images(protocol):
        pending.append(sample)
        if len(pending) == protocol["detector"]["batch_size"]:
            consume(pending)
            pending = []
    if pending:
        consume(pending)
    print(f"detected_images={len(protocol['sample_metadata'])} crops={len(rows)}", flush=True)
    rows.sort(key=lambda row: row["object_key"])
    batch_size = protocol["ocr"]["batch_size"]
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        crops = []
        for row in batch:
            contract.validate_frozen_file(row["crop_path"], row["crop_sha256"], "crop")
            with Image.open(row["crop_path"]) as crop:
                if crop.size != tuple(protocol["ocr"]["crop_size"]):
                    raise ValueError("OCR crop size differs")
                crops.append(image_tensor(crop))
        with torch.inference_mode():
            logits = ocr(torch.stack(crops))
        for row, sequence in zip(batch, logits):
            row["text"] = decode_ctc(sequence.argmax(-1))
        if start % (batch_size * 10) == 0:
            print(f"recognized={min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    after = model_digest((vision, head, ocr))
    if before != after:
        raise ValueError("Frozen model state changed")
    contract.write_new_jsonl(output / "predictions.jsonl", rows)
    receipt = {
        "protocol_sha256": contract.sha256_file(output / "protocol.json"),
        "predictions_sha256": contract.sha256_file(output / "predictions.jsonl"),
        "rows": len(rows), "sample_keys": sorted(protocol["sample_metadata"]),
        "detector_repeat_exact": True, "models_before_sha256": before,
        "models_after_sha256": after, "finished_utc": contract.utc_now(),
        "scoring_truth_used_for_proposals": False, "training_performed": False,
        "cuda_device": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
    }
    contract.write_new_json(output / "prediction_receipt.json", receipt)
    return receipt
