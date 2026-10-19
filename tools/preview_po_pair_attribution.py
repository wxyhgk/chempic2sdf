"""Human-only endpoint overlays on saved images; no model or graph modification."""

import json
from pathlib import Path

from PIL import Image, ImageDraw
import torch

from Data.sdf.chemvector_curriculum_v0_1.gallery import _font
from tools import diagnose_po_pair_attribution as probe
from tools import review_structure_expansion_identity as fixed

OUTPUT = fixed.REPO / "outputs/reviews/chemvector-po-pair-attribution-20260924"


def run():
    fixed.require(not OUTPUT.exists(), "Fresh review directory required")
    report = json.loads(fixed.read(probe.OUTPUT/"report.json"))
    audit = json.loads(fixed.read(probe.OUTPUT/"postcheck.json"))
    plan = json.loads(fixed.read(probe.OUTPUT/"plan.json"))
    fixed.require(report["complete"] and audit["status"] == "passed_saved_aggregate_postcheck"
        and audit["source_report_sha256"] == fixed.digest(probe.OUTPUT/"report.json")
        and report["plan_sha256"] == fixed.digest(probe.OUTPUT/"plan.json"), "Audited complete probe required")
    # Show all four structures, not the best recovery. Fixed Liberation/40/3.2
    # pair differs ONLY in font size (1011 versus 1111).
    mids = sorted({row["molecule_id"] for row in plan["samples"]})
    fixed.require(len(mids) == 4, "Four structures required")
    bindings = {str(path): fixed.digest(path) for path in (probe.OUTPUT/"report.json",
        probe.OUTPUT/"postcheck.json", probe.OUTPUT/"plan.json", Path(__file__).absolute(),
        fixed.REPO/"Data/sdf/chemvector_curriculum_v0_1/gallery.py")}
    panels = []
    for mid in mids:
        for profile in ("po1011", "oa22"):
            key = mid+"__"+profile
            path = probe.OUTPUT/"cases"/(key+".json")
            bindings[str(path)] = report["case_sha256"][key]
            original_path = probe.source.OUTPUT/"cases"/(key+".json")
            image_path = probe.source.DATA/"samples"/(key+".png")
            layers_path = probe.source.DATA/"samples"/(key+".layers.json")
            for source_path in (original_path, image_path, layers_path):
                bindings[str(source_path)] = plan["input_sha256"][str(source_path)]
            panels.append((json.loads(fixed.read(path)), json.loads(fixed.read(original_path)),
                           json.loads(fixed.read(layers_path)), image_path))
    fixed.verify_bindings(bindings)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    pages = []
    for start in (0, 4):
        canvas = Image.new("RGB", (1080, 1060), "white"); draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), "P-O endpoint sensitivity / HUMAN DIAGNOSTIC ONLY", font=_font(21), fill="black")
        draw.text((12, 38), "Blue ring: quantized target. Red cross: prediction. Same image/model; oracle moves sampling anchors only.",
                  font=_font(13), fill="#475569")
        for i, (case, original, layers, path) in enumerate(panels[start:start+4]):
            x, y = i % 2*540, 68+i//2*490
            draw.rectangle((x, y, x+539, y+489), outline="#cbd5e1")
            font_size = 24 if case["bits"][1] else 12
            draw.text((x+12, y+8), f"{case['molecule_id']} / font {font_size}", font=_font(16), fill="black")
            draw.text((x+12, y+31), "Liberation Sans / length cap 40 / stroke 3.2 / 2x crop", font=_font(13), fill="#475569")
            atoms = {r["atom_id"]: r for r in layers["atoms"]}
            phosphorus = next(r for r in atoms.values() if r["element"] == "P")
            left, top = [round(v)-80 for v in phosphorus["center"]]
            fixed.require(0 <= left <= 608 and 0 <= top <= 608, "Crop outside source")
            with Image.open(path) as image:
                crop = image.convert("RGB").crop((left, top, left+160, top+160)).resize((320, 320), Image.Resampling.NEAREST)
            pen = ImageDraw.Draw(crop)
            candidates = {c["candidate"]: c for c in original["candidates"]["candidates"]}
            mapping = {candidate: aid for b in original["po_bonds"]
                       for candidate, aid in zip(b["candidates"], b["atom_ids"])}
            for candidate, aid in mapping.items():
                target = [float(torch.tensor(round(v/767*1023))/1023)*767 for v in atoms[aid]["center"]]
                predicted = [v*767 for v in candidates[candidate]["geometry"][:2]]
                tx, ty = [(v-origin)*2 for v, origin in zip(target, (left, top))]
                px, py = [(v-origin)*2 for v, origin in zip(predicted, (left, top))]
                pen.line((tx, ty, px, py), fill="#64748b", width=2)
                pen.ellipse((tx-5, ty-5, tx+5, ty+5), outline="#2563eb", width=2)
                pen.line((px-4, py-4, px+4, py+4), fill="#dc2626", width=2)
                pen.line((px-4, py+4, px+4, py-4), fill="#dc2626", width=2)
            canvas.paste(crop, (x+110, y+58))
            singles = [b for b in case["bonds"] if b["expected_type"] == "SINGLE"]
            for j, bond in enumerate(singles):
                text = f"P-O {bond['bond_id']}: p {bond['baseline']['presence_probability']:.4f} -> {bond['oracle']['presence_probability']:.4f} (oracle)"
                draw.text((x+14, y+389+j*24), text, font=_font(16), fill="black")
            draw.text((x+14, y+446), "Threshold 0.85; ground-truth intervention, not a deployed fix.", font=_font(13), fill="#475569")
        name = f"endpoints-{start//4+1:02d}.png"
        canvas.save(OUTPUT/name); pages.append(name)
    fixed.verify_bindings(bindings)
    result = dict(status="human_review_only", structures=4, images=8, model_forwards=0,
        selection="all four structures; fixed font-only 1011/1111 pair", coordinate_display="normalized*767",
        original_images_modified=False, training_authorized=False, input_sha256=bindings,
        pages={name: fixed.digest(OUTPUT/name) for name in pages})
    fixed.write_new(OUTPUT/"report.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in run().items() if k in ("status", "images", "pages")}))
