"""CPU frozen-OCR crop intervention and local visual review artifacts."""

from collections import defaultdict
from html import escape
from pathlib import Path

import chemvector.external_ocr_r0_contracts as contract


def crop_intervention(output, failures, protocol):
    import torch
    from PIL import Image, ImageDraw
    from chemvector.ctc_ocr import bbox_crop, decode_ctc
    from chemvector.evaluate_direct_atom_ocr import load_ctc
    from chemvector.image_io import image_tensor
    from chemvector.text_entry_predictions import model_digest, selected_images

    output = Path(output)
    visuals = output / "visuals"
    visuals.mkdir()
    crops = output / "oracle-crops"
    crops.mkdir()
    grouped = defaultdict(list)
    for row in failures:
        grouped[row["sample_key"]].append(row)
    jobs, previews = [], {}
    for sample in selected_images(protocol):
        image = sample["image"].convert("RGB")
        for row in grouped[sample["key"]]:
            key, box = row["gt_key"], row["gt_bbox"]
            proposal = row["associated_prediction"]
            x0, y0, x1, y1 = box
            if proposal:
                px0, py0, px1, py1 = proposal["bbox"]
                x0, y0, x1, y1 = min(x0, px0), min(y0, py0), max(x1, px1), max(y1, py1)
            bounds = [int(x0) - 28, int(y0) - 28, int(x1) + 29, int(y1) + 29]
            preview = image.crop(bounds)
            draw = ImageDraw.Draw(preview)
            for current, color in [(box, "green")] + ([(proposal["bbox"], "red")] if proposal else []):
                draw.rectangle([current[0] - bounds[0], current[1] - bounds[1],
                                current[2] - bounds[0], current[3] - bounds[1]], outline=color, width=1)
            relative = Path("visuals") / f"{key}.png"
            preview.resize((preview.width * 3, preview.height * 3)).save(output / relative)
            previews[key] = str(relative)
            if row["category"] == "matched_ocr_error":
                oracle_path = crops / f"{key}.png"
                bbox_crop(image, box, context=4.0).save(oracle_path)
                jobs.append({"gt_key": key, "truth": row["truth"],
                             "baseline_path": proposal["crop_path"],
                             "baseline_sha256": proposal["crop_sha256"],
                             "oracle_path": str(oracle_path),
                             "oracle_sha256": contract.sha256_file(oracle_path)})
    if set(previews) != {row["gt_key"] for row in failures}:
        raise ValueError("Missing visual review samples")
    # Freeze the exact intervention crops before calling the recognizer.
    jobs.sort(key=lambda row: row["gt_key"])
    contract.write_new_jsonl(output / "crop_interventions.jsonl", jobs)
    torch.set_num_threads(4)
    model, _ = load_ctc(protocol["ocr"]["checkpoint"], torch.device("cpu"))
    before = model_digest((model,))
    tasks = [(job, arm) for job in jobs for arm in ("baseline", "oracle")]
    for start in range(0, len(tasks), 32):
        batch = tasks[start:start + 32]
        tensors = []
        for job, arm in batch:
            contract.validate_frozen_file(job[f"{arm}_path"], job[f"{arm}_sha256"], arm)
            with Image.open(job[f"{arm}_path"]) as crop:
                if crop.size != (192, 48):
                    raise ValueError("Unexpected crop size")
                tensors.append(image_tensor(crop))
        with torch.inference_mode():
            logits = model(torch.stack(tensors))
        for (job, arm), sequence in zip(batch, logits):
            job[f"{arm}_text"] = decode_ctc(sequence.argmax(-1))
    after = model_digest((model,))
    if before != after:
        raise ValueError("OCR model state changed")
    contract.write_new_jsonl(output / "ocr_counterfactual.jsonl", jobs)
    return ({job["gt_key"]: job for job in jobs}, previews,
            {"model_before_sha256": before, "model_after_sha256": after,
             "device": "cpu", "precision": "float32", "decoder": "greedy",
             "training_performed": False, "torch_version": torch.__version__})


def write_gallery(output, rows, oracle, previews):
    """Escape all dataset text; no scripts or remote resources in the review page."""
    output = Path(output)
    sections = []
    for row in sorted(rows, key=lambda r: (r["category"], r["gt_key"])):
        key = row["gt_key"]
        prediction = row["associated_prediction"]
        details = f"truth={row['truth']!r}; baseline={row['baseline']['predicted_text']!r}"
        images = f'<img src="{escape(previews[key], quote=True)}" alt="local context">'
        if prediction:
            # Copy of the frozen input for an independently browsable gallery.
            from shutil import copyfile
            path = Path("visuals") / f"{key}-baseline.png"
            copyfile(prediction["crop_path"], output / path)
            images += f'<img src="{escape(str(path), quote=True)}" alt="baseline OCR input">'
        if key in oracle:
            details += f"; oracle={oracle[key]['oracle_text']!r}"
            path = Path("oracle-crops") / f"{key}.png"
            images += f'<img src="{escape(str(path), quote=True)}" alt="GT-box OCR input">'
        sections.append(f'<article><h3>{escape(row["category"])} — {escape(key)}</h3>'
                        f'<p>{escape(details)}</p>{images}</article>')
    with (output / "review.html").open("x", encoding="utf-8") as handle:
        handle.write('<!doctype html><meta charset="utf-8"><title>Text failure review</title>'
                     '<style>body{font:16px sans-serif;margin:24px}article{border-top:1px solid #ccc;'
                     'padding:16px 0}img{max-width:540px;margin:8px;vertical-align:middle}</style>'
                     '<h1>Frozen text failure review</h1><p>Green: GT bbox. Red: associated A bbox. '
                     'OCR inputs follow context image: baseline, then GT-box intervention if present. '
                     'GT crops are diagnostic only. Missing anchor candidates are not drawn.</p>'
                     + "".join(sections))
