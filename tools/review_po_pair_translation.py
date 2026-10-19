"""Post-hoc saved-response review only: no inference, threshold search or repair."""

from collections import Counter
import json
import math
from pathlib import Path
import statistics

from PIL import Image, ImageDraw

from Data.sdf.chemvector_curriculum_v0_1.gallery import _font
from tools import diagnose_po_pair_translation as experiment
from tools import review_structure_expansion_identity as fixed

OUTPUT = fixed.REPO/"outputs/reviews/chemvector-po-pair-translation-20260924"


def probability(logit):
    return 1/(1+math.exp(-logit)) if logit >= 0 else math.exp(logit)/(1+math.exp(logit))


def run():
    fixed.require(not OUTPUT.exists(), "Fresh review directory required")
    report = json.loads(fixed.read(experiment.OUTPUT/"report.json"))
    plan = json.loads(fixed.read(experiment.OUTPUT/"plan.json"))
    audit = json.loads(fixed.read(experiment.OUTPUT/"postcheck.json"))
    fixed.require(report["complete"] and audit["status"] == "passed_saved_translation_postcheck"
        and audit["source_report_sha256"] == fixed.digest(experiment.OUTPUT/"report.json")
        and report["plan_sha256"] == fixed.digest(experiment.OUTPUT/"plan.json"), "Completed audited run required")
    bindings = {str(p): fixed.digest(p) for p in (experiment.OUTPUT/"report.json", experiment.OUTPUT/"plan.json",
        experiment.OUTPUT/"postcheck.json", Path(__file__).absolute(),
        fixed.REPO/"Data/sdf/chemvector_curriculum_v0_1/gallery.py")}
    cases = []
    for key, value in sorted(report["case_sha256"].items()):
        path = experiment.OUTPUT/"cases"/(key+".json"); bindings[str(path)] = value
        cases.append(json.loads(fixed.read(path)))
    fixed.verify_bindings(bindings)
    threshold = plan["thresholds"]["pair_threshold"]
    positive_hist, po_hist, shifted_losses = Counter(), Counter(), Counter()
    direction_groups = {"transverse_le_1px": [], "transverse_ge_3px": []}
    false_edge_events, panels = [], []
    original_correct = known_negative_count = 0
    for case in cases:
        pairs = [tuple(p) for p in case["pair_predictions"]["pairs"]]
        indices = {p: i for i, p in enumerate(pairs)}
        bonds = case["baseline_metrics"]["bonds"]
        positives = {tuple(sorted(b["candidates"])) for b in bonds}
        matched = {i for pair in positives for i in pair}
        # This review uses the already-proven all-reachable, no-isolated-node panel.
        # Never generalize positive-endpoint membership to arbitrary source graphs.
        fixed.require(len(matched) == case["baseline_metrics"]["coverage"]["selected_matched"]
            and len(positives) == case["baseline_metrics"]["pair_contract"]["reachable_positive_pairs"],
            "Complete matched-node coverage required")
        candidates = {r["candidate"]: r["geometry"][:2] for r in case["candidates"]["candidates"]}
        probabilities = [[probability(v) for v in row] for row in case["views"]["presence_logits"]]
        local_negatives = 0
        for pair, index in indices.items():
            if pair in positives or not all(i in matched for i in pair): continue
            local_negatives += 1
            for view in range(1, 9):
                if probabilities[view][index] >= threshold:
                    false_edge_events.append(dict(sample_key=case["sample_key"], pair=list(pair),
                        offset_pixels=plan["offsets_pixels"][view], probability=probabilities[view][index],
                        original_probability=probabilities[0][index]))
        fixed.require(local_negatives == case["baseline_metrics"]["pair_contract"]["known_negative_pairs"],
            "Nonedge correspondence reconstruction changed")
        known_negative_count += local_negatives
        for bond in bonds:
            index = indices[tuple(sorted(bond["candidates"]))]
            values = [row[index] for row in probabilities]
            above = sum(v >= threshold for v in values)
            positive_hist[above] += 1
            if bond["bond_id"] in case["po_bond_ids"] and bond["expected_type"] == "SINGLE":
                po_hist[(12 if case["bits"][1] == 0 else 24, above)] += 1
                # Fixed all four old oa22 controls; no best/worst case selection.
                if case["bits"] == [1, 1, 1, 1]:
                    mixed = next(b for b in case["metrics"]["bonds"] if b["bond_id"] == bond["bond_id"])
                    panels.append(dict(sample_key=case["sample_key"], bond_id=bond["bond_id"],
                        probabilities=values, mean_probability=mixed["presence_probability"]))
            if bond["stage"] != "reachable_correct": continue
            original_correct += 1
            first, second = (candidates[i] for i in bond["candidates"])
            dx, dy = second[0]-first[0], second[1]-first[1]
            length = math.hypot(dx, dy)
            fixed.require(length > 0, "Distinct predicted anchors required")
            for view, (x, y) in enumerate(plan["offsets_pixels"][1:], 1):
                shifted_losses[view] += values[view] < threshold
                perpendicular = abs(-dy*x+dx*y)/length
                if perpendicular <= 1: direction_groups["transverse_le_1px"].append(values[view])
                if perpendicular >= 3: direction_groups["transverse_ge_3px"].append(values[view])
    fixed.require(len(panels) == 8 and original_correct == report["summary"]["baseline"]["stages"]["reachable_correct"],
        "Review scope changed")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    canvas = Image.new("RGB", (1040, 1260), "white"); draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), "Saved P-O responses to sampling translations / ALL four oa22 controls", font=_font(19), fill="black")
    draw.text((12, 36), "Rows: dy pixels. Columns: dx pixels. Blue border: original sampling. No best-offset selection or inference.",
        font=_font(13), fill="#475569")
    offset_index = {tuple(offset): i for i, offset in enumerate(plan["offsets_pixels"])}
    for i, panel in enumerate(panels):
        x, y = i % 2*520, 70+i//2*295
        draw.rectangle((x, y, x+519, y+294), outline="#cbd5e1")
        draw.text((x+12, y+8), panel["sample_key"].split("__")[0]+f" / P-O {panel['bond_id']}", font=_font(16), fill="black")
        for col, dx in enumerate((-4, 0, 4)):
            draw.text((x+112+col*124, y+36), f"dx {dx:+d}", font=_font(13), fill="#475569")
        for row, dy in enumerate((-4, 0, 4)):
            draw.text((x+12, y+77+row*54), f"dy {dy:+d}", font=_font(13), fill="#475569")
            for col, dx in enumerate((-4, 0, 4)):
                value = panel["probabilities"][offset_index[(dx, dy)]]
                cx, cy = x+92+col*124, y+60+row*54
                draw.rectangle((cx, cy, cx+117, cy+49), fill="#dcfce7" if value >= threshold else "#fff7ed",
                    outline="#2563eb" if dx == dy == 0 else "#e2e8f0", width=3 if dx == dy == 0 else 1)
                draw.text((cx+18, cy+13), f"{value:.4f}", font=_font(18), fill="black")
        draw.text((x+12, y+240), f"Nine-view mean: {panel['mean_probability']:.4f} / threshold {threshold}", font=_font(15), fill="black")
        draw.text((x+12, y+266), "Presence probabilities only; not whole-molecule accuracy.", font=_font(12), fill="#475569")
    canvas.save(OUTPUT/"response-grid.png")
    result = dict(status="posthoc_saved_response_review", model_forwards=0, model_loads=0, training_authorized=False,
        input_sha256=bindings, original_correct_bonds=original_correct, known_nonedges=known_negative_count,
        positive_number_of_views_present=dict(sorted(positive_hist.items())),
        po_number_of_views_present=[dict(font_size=font, views_present=count, bonds=number)
            for (font, count), number in sorted(po_hist.items())],
        shifted_presence_losses=[dict(offset_pixels=plan["offsets_pixels"][i], lost_presence=shifted_losses[i]) for i in range(1, 9)],
        directional_groups={name: dict(observations=len(v), mean_probability=statistics.mean(v),
            below_threshold=sum(p < threshold for p in v)) for name, v in direction_groups.items()},
        directional_scope="Post-hoc geometry bins of repeated original-correct bonds; not matched causal contrasts or tuning",
        shifted_false_edge_events=false_edge_events,
        affected_nonedges=len({(r["sample_key"], tuple(r["pair"])) for r in false_edge_events}),
        gallery_selection="all four oa22 images, both P-O singles", gallery=panels,
        artifacts={"response-grid.png": fixed.digest(OUTPUT/"response-grid.png")})
    fixed.verify_bindings(bindings)
    fixed.write_new(OUTPUT/"report.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in run().items() if k in ("status", "original_correct_bonds", "affected_nonedges", "directional_groups")}))
