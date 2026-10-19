"""Visible-carbon review records and presentation, without dataset/model access."""

from collections import Counter
from html import escape
import hashlib
import json
import math
from pathlib import Path


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def neutral_visible_carbons(layers, chemmol):
    chemical = {a["id"]: a for a in chemmol["atoms"]}
    if len(chemical) != len(chemmol["atoms"]):
        raise ValueError("Duplicate chemical atom identity")
    layer_ids = [a["atom_id"] for a in layers["atoms"]]
    if len(set(layer_ids)) != len(layer_ids) or set(layer_ids) != set(chemical):
        raise ValueError("Chemical/layer atom identities differ")
    result = []
    for layer in layers["atoms"]:
        atom = chemical[layer["atom_id"]]
        if (atom["element"] != "C" or atom.get("formal_charge", 0)
                or atom.get("isotope", 0) or atom.get("radical_electrons", 0)
                or not layer["visible_text"]):
            continue
        if layer["element"] != atom["element"]:
            raise ValueError("Chemical/layer element differs")
        box = layer["bbox"]
        if (not isinstance(box, list) or len(box) != 4
                or not all(type(v) in (float, int) and math.isfinite(v) for v in box)
                or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError("Visible carbon requires a finite positive bbox")
        if not isinstance(layer["text"], str) or not layer["text"].strip():
            raise ValueError("Visible carbon requires a transcription")
        result.append((layer, atom))
    return result


def candidate_record(key, molecule, variant, layer, atom, diagnostic, source_hashes):
    tokens = diagnostic["labels"].get(layer["atom_id"], [])
    if diagnostic["evidence_kind"] != "nofreetype_diagnostic" or diagnostic["revision_authorized"]:
        raise ValueError("Expected diagnostic-only renderer evidence")
    return {
        "object_key": f"{key}:a{layer['atom_id']:04d}", "sample_key": key,
        "molecule_id": molecule, "variant": variant, "atom_id": layer["atom_id"],
        "layer_text": layer["text"], "bbox": layer["bbox"], "anchor": layer["center"],
        "implicit_h_count": atom.get("implicit_h_count", 0),
        "explicit_h_count": atom.get("explicit_h_count", 0),
        "diagnostic_tokens": tokens,
        "diagnostic_error": diagnostic.get("error"),
        "diagnostic_plain_c_disagreement": tokens == ["C"] and layer["text"] != "C",
        "source_hashes": source_hashes, "review_status": "unreviewed",
        "evidence_kind": "nofreetype_diagnostic", "same_source_raster": False,
        "revision_authorized": False,
    }


def save_previews(output, image, png_payload, rows):
    """Keep exact source PNG, a clean magnified crop, and contextual bbox view."""
    from PIL import ImageDraw

    if not rows:
        return
    output = Path(output)
    sample = rows[0]["sample_key"]
    original = Path("source-images") / f"{sample}.png"
    (output / original).parent.mkdir(exist_ok=True)
    with (output / original).open("xb") as handle:
        handle.write(png_payload)
    (output / "crops").mkdir(exist_ok=True)
    for row in rows:
        x0, y0, x1, y1 = row["bbox"]
        crop = image.crop((math.floor(x0) - 4, math.floor(y0) - 4,
                           math.ceil(x1) + 4, math.ceil(y1) + 4))
        crop_path = Path("crops") / f"{row['object_key']}.png"
        crop.resize((crop.width * 4, crop.height * 4)).save(output / crop_path)
        left, top = math.floor(x0) - 28, math.floor(y0) - 28
        context = image.crop((left, top, math.ceil(x1) + 28, math.ceil(y1) + 28))
        ImageDraw.Draw(context).rectangle([x0 - left, y0 - top, x1 - left, y1 - top],
                                          outline="red", width=1)
        context_path = Path("crops") / f"{row['object_key']}-context.png"
        context.resize((context.width * 3, context.height * 3)).save(output / context_path)
        row["preview"] = {"source_png": str(original), "crop": str(crop_path),
                          "context": str(context_path), "crop_context_px": 4,
                          "crop_display_scale": 4}


def summarize(records, sample_counts):
    flagged = [r for r in records if r["diagnostic_plain_c_disagreement"]]
    return {
        "images": len(sample_counts),
        "molecules": len({s["molecule_id"] for s in sample_counts}),
        "visible_text_objects": sum(s["visible_text_objects"] for s in sample_counts),
        "visible_neutral_carbons": len(records),
        "images_with_visible_neutral_carbon": len({r["sample_key"] for r in records}),
        "diagnostic_plain_c_disagreements": len(flagged),
        "disagreement_images": len({r["sample_key"] for r in flagged}),
        "disagreement_molecules": len({r["molecule_id"] for r in flagged}),
        "unresolved_diagnostic_objects": sum(not r["diagnostic_tokens"] or bool(r["diagnostic_error"])
                                             for r in records),
        "layer_text_counts": dict(sorted(Counter(r["layer_text"] for r in records).items())),
        "flagged_layer_text_counts": dict(sorted(Counter(r["layer_text"] for r in flagged).items())),
        "flagged_variants": dict(sorted(Counter(r["variant"] for r in flagged).items())),
        "review_status": "awaiting_source_raster_review",
        "training_performed": False, "source_data_modified": False,
        "revisions_applied": 0, "revision_authorized": False,
        "limitations": ["sampled training population, not whole corpus census",
                        "renderer disagreement is not an automatic label correction",
                        "neutral visible carbon only; other label classes unassessed",
                        "four variants are correlated within each molecule",
                        "no estimate of checkpoint-specific gradient exposure"],
    }


def write_gallery(output, records):
    sections = []
    for row in sorted(records, key=lambda r: (not r["diagnostic_plain_c_disagreement"], r["object_key"])):
        preview = row["preview"]
        text = (f"layer={row['layer_text']!r}; diagnostic={row['diagnostic_tokens']!r}; "
                f"review={row['review_status']}")
        sections.append(
            f'<article><h3>{escape(row["object_key"])}</h3><p>{escape(text)}</p>'
            f'<a href="{escape(preview["source_png"], quote=True)}">Original PNG</a><br>'
            f'<img src="{escape(preview["crop"], quote=True)}" alt="clean crop">'
            f'<img src="{escape(preview["context"], quote=True)}" alt="context"></article>')
    with (Path(output) / "review.html").open("x", encoding="utf-8") as handle:
        handle.write('<!doctype html><meta charset="utf-8"><title>Train visible text review</title>'
                     '<style>body{font:16px sans-serif;margin:24px}article{border-top:1px solid #bbb;'
                     'padding:16px}img{max-width:550px;vertical-align:middle;margin:8px}</style>'
                     '<h1>Train-only visible carbon review</h1><p>Original rasters are authoritative. '
                     'Alternate-renderer tokens are diagnostic only. No revisions applied.</p>'
                     + "".join(sections))
