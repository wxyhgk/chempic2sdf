"""Create a deterministic source-raster review sheet; does not mark labels reviewed."""

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from Data.sdf.train_visible_text_report import digest, write_json


def build(review_dir, output_dir):
    source, output = Path(review_dir), Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Contact-sheet output must be empty")
    report = json.loads((source / "report.json").read_text())
    records_path = source / "records.jsonl"
    if digest(records_path.read_bytes()) != report["artifacts"]["records.jsonl"]:
        raise ValueError("Review records differ")
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    chosen = {}
    for row in sorted(records, key=lambda item: item["object_key"]):
        if row["diagnostic_plain_c_disagreement"]:
            chosen.setdefault(row["molecule_id"], row)
    selected = sorted(chosen.values(), key=lambda item: item["object_key"])
    sheet = Image.new("RGB", (1200, max(1, math.ceil(len(selected) / 3)) * 240), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        left, top = (index % 3) * 400, (index // 3) * 240
        draw.text((left + 8, top + 6), f"{index + 1}: {row['object_key']}", fill="black")
        draw.text((left + 8, top + 22), f"{row['molecule_id']} layer={row['layer_text']}", fill="black")
        original = source / row["preview"]["source_png"]
        if digest(original.read_bytes()) != row["source_hashes"]["png"]:
            raise ValueError("Original source PNG differs")
        for field, offset, bounds in (("crop", 8, (150, 170)), ("context", 168, (220, 185))):
            path = source / row["preview"][field]
            if digest(path.read_bytes()) != report["artifacts"][row["preview"][field]]:
                raise ValueError("Preview hash differs")
            with Image.open(path) as image:
                image.thumbnail(bounds)
                sheet.paste(image, (left + offset, top + 45))
    output.mkdir(parents=True, exist_ok=True)
    sheet.save(output / "contact-sheet.png")
    write_json(output / "selection.json", {
        "selection": "lexically first flagged object per molecule; no review decisions assigned",
        "source_report_sha256": digest((source / "report.json").read_bytes()),
        "records": selected, "contact_sheet_sha256": digest((output / "contact-sheet.png").read_bytes()),
        "revision_authorized": False,
    })
    return {"objects": len(selected), "sheet": str(output / "contact-sheet.png")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.review_dir, args.output_dir)))
