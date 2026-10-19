"""Two frozen neural passes (predicted/GT geometry), then scene-level factorial controls."""

from collections import defaultdict
from copy import deepcopy
import json
import math

from rdkit import Chem
import torch
from torch.nn import functional as F

from Data.sdf.chemscene_v0_2 import QUANTIZATION_MAX
from chemvector.ctc_ocr import bbox_crop, decode_ctc
from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from chemvector.direct_scene_inference import apply_atom_ocr, direct_scene_tensors, load_direct_model
from chemvector.evaluate_direct_atom_ocr import load_role, load_bbox, load_ctc
from chemvector.external_ocr_r0_contracts import (
    ROLE_CHECKPOINT, BBOX_CHECKPOINT, CTC_CHECKPOINT, SOURCE_CHECKPOINT, GROUNDED_GRAPH_CHECKPOINT,
    read_jsonl, sha256_file, write_new_json, write_new_jsonl,
)
from chemvector.image_io import image_tensor
from chemvector.placeholder_scene import placeholder_scene
from chemvector.predicted_incidence import PredictedProposalConfig, decode_predicted_atom_proposals, decode_predicted_bond_proposals
from chemvector.shadow_chemscene_v02_oracle import build_oracle_chemistry_shadow_scene
from chemvector.text_entry_predictions import model_digest
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_spatial_incidence import forward_frozen_bond_head
from chemvector.evaluation.audits.ocr.set_rejected_text.checks import check_deployed
from chemvector.evaluation.audits.ocr.set_rejected_text.inputs import selected_images
from chemvector.evaluation.audits.ocr.set_rejected_text.runtime import text_forward
from .controls import compile_and_score, intervene
from .protocol import BASELINE, DATASET, DIRECT, targets, verify


def oracle_atoms(output, samples, truth):
    batch, capacity = len(samples), 128
    anchors = torch.zeros(batch, capacity, 2, device="cuda", dtype=torch.float32)
    mask = torch.zeros(batch, capacity, device="cuda", dtype=torch.bool)
    for i, sample in enumerate(samples):
        objects = truth[sample["key"]]["scene"]["atom_objects"]
        if not 0 < len(objects) <= capacity:
            raise ValueError("GT atom set exceeds fixed capacity or is empty")
        anchors[i, :len(objects)] = torch.tensor([a["anchor"] for a in objects], device="cuda") / QUANTIZATION_MAX
        mask[i, :len(objects)] = True

    def sample(field):
        # This oracle changes the sampling locations, never injects GT features/text.
        with torch.autocast("cuda", enabled=False):
            return F.grid_sample(field.float(), (anchors * 2 - 1)[:, :, None], mode="bilinear",
                                 padding_mode="border", align_corners=False).squeeze(-1).transpose(1, 2)

    geometry = torch.cat((anchors, sample(output["role_logits"].softmax(1)),
                          sample(output["text_centers"]), sample(output["text_extents"])), dim=-1)
    return {"atom_geometry": geometry, "atom_features": sample(output["object_features"]),
            "atom_scores": sample(output["heatmap_logits"].sigmoid()).squeeze(-1), "atom_mask": mask}


def compile_prediction(atoms, pairs, predicted, ocr, threshold):
    tensors = direct_scene_tensors(atoms, pairs, predicted)
    scene, diagnostics = build_oracle_chemistry_shadow_scene(
        molecule_id="diagnostic", gt_scene=placeholder_scene("diagnostic", 768, 768),
        atom_geometry=atoms["atom_geometry"][0].float().cpu(), atom_mask=atoms["atom_mask"][0].cpu(),
        atom_presence_logits=tensors["atom_presence_logits"].float().cpu(),
        bond_types=tensors["bond_types"].float().cpu(), bond_mask=tensors["bond_mask"].cpu(),
        bond_presence_logits=tensors["bond_presence_logits"].float().cpu(),
        incidence_logits=tensors["incidence_logits"].float().cpu(), atom_threshold=0., bond_threshold=threshold)
    diagnostics["ocr"] = apply_atom_ocr(scene, diagnostics["selected_atom_candidate_indices"], ocr)
    return scene, diagnostics


def summarize(rows, config):
    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in config["arms"]}
    if any(len(group) != 200 or len({r['sample_key'] for r in group}) != 200 for group in by_arm.values()):
        raise ValueError("Incomplete factorial denominator")
    summaries = {arm: {name: sum(r[name] for r in group) for name in (
        "sdf_valid", "core_graph_exact", "stereo_graph_exact", "carbon_fallback")} for arm, group in by_arm.items()}
    lookup = {(r["sample_key"], r["arm"]): r for r in rows}
    baseline = summaries["A0B0C0"]["core_graph_exact"]
    transitions = {}
    # Full conditional effects expose interactions, not only single-axis gains.
    for arm in config["arms"]:
        for offset, factor in ((1, "A"), (3, "B"), (5, "C")):
            if arm[offset] != "0":
                continue
            changed = arm[:offset] + "1" + arm[offset + 1:]
            pairs = [(lookup[k, arm]["core_graph_exact"], lookup[k, changed]["core_graph_exact"])
                     for k in config["sample_order"]]
            transitions[f"{arm}->{changed}"] = {"factor": factor,
                "rescued": sum(not a and b for a, b in pairs), "harmed": sum(a and not b for a, b in pairs)}
    passed = summaries["A1B1C1"]["core_graph_exact"] == config["full_oracle_required_primary"]
    return {"images": 200, "arms": summaries, "baseline_core_exact": baseline,
            "conditional_transitions": transitions, "full_oracle_control_passed": passed,
            "architecture_ranking_authorized": passed, "promotion_authorized": False,
            "limitations": [config["scope"], "B combines atom identity/charge/isotope/H/stereo attributes, not pure OCR",
                            "A changes feature sampling distribution; its harm is not evidence that accurate anchors are undesirable",
                            "primary excludes stereo; secondary includes stereo", "oracle gains are not achievable deployment gains"]}


def run(output, config):
    truth = targets(config)
    direct, direct_payload = load_direct_model(str(DIRECT), torch.device("cuda"))
    if (direct_payload["training_args"]["source_checkpoint"] != str(SOURCE_CHECKPOINT)
            or direct_payload["training_args"]["grounded_graph_checkpoint"] != str(GROUNDED_GRAPH_CHECKPOINT)):
        raise ValueError("Direct model dependency differs")
    vision, atom, bond, selector, source = load_components(str(SOURCE_CHECKPOINT), str(GROUNDED_GRAPH_CHECKPOINT), torch.device("cuda"))
    role, role_payload = load_role(str(ROLE_CHECKPOINT), torch.device("cuda"))
    bbox, _ = load_bbox(str(BBOX_CHECKPOINT), torch.device("cuda"))
    ctc, _ = load_ctc(str(CTC_CHECKPOINT), torch.device("cpu"))
    models = (vision, atom, bond, selector, direct, role, bbox, ctc)
    for model in models:
        model.eval().requires_grad_(False)
    before = model_digest(models)
    proposals = PredictedProposalConfig(**source["proposal_config"])
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    role_threshold = float(role_payload["validation"]["threshold"])
    probability = float(direct_payload["validation"]["threshold"])
    bond_threshold = math.log(probability / (1 - probability))
    baseline_ocr, deployed = defaultdict(dict), defaultdict(list)
    for row in read_jsonl(BASELINE / "predictions.jsonl"):
        if row["arm"] != "atom_conditioned":
            continue
        x0, y0, x1, y1 = row["bbox"]
        index = int(row["object_key"].rsplit(":p", 1)[1])
        baseline_ocr[row["sample_key"]][index] = {"text": row["text"], "score": row["score"],
            "center": [(x0 + x1) / 1536, (y0 + y1) / 1536], "half_extent": [(x1 - x0) / 1536, (y1 - y0) / 1536]}
    for row in read_jsonl(DATASET / "atom-objects.jsonl"):
        if row["sample_key"] in config["sample_metadata"]:
            deployed[row["sample_key"]].append(row)
    (output / "sdf").mkdir()
    rows, scenes, neural_records = [], [], []

    def consume(samples):
        pixels = torch.stack([image_tensor(s["image"]) for s in samples]).cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
            atom_output = atom(features)
            atoms = decode_predicted_atom_proposals(atom_output, proposals)
            bonds = decode_predicted_bond_proposals(forward_frozen_bond_head(vision, bond, features), proposals)
            selected = selector(atoms["atom_features"], atoms["atom_geometry"], atoms["atom_scores"], atoms["atom_mask"],
                                bonds["bond_features"], bonds["bond_endpoints"], bonds["bond_types"], bonds["bond_scores"], bonds["bond_mask"])
            atoms["atom_mask"] = atoms["atom_mask"] & (selected["atom_presence_logits"] >= atom_threshold)
            for i, sample in enumerate(samples):
                check_deployed(sample["key"], atoms["atom_geometry"][i].float().cpu().numpy(),
                               atoms["atom_mask"][i].cpu().numpy(), deployed[sample["key"]])
            oracle = oracle_atoms(atom_output, samples, truth)
        text_boxes, _ = text_forward(role, bbox, pixels, oracle, oracle["atom_mask"], role_threshold)
        oracle_ocr = defaultdict(dict)
        for start in range(0, len(text_boxes), 32):
            batch = text_boxes[start:start + 32]
            crops = [image_tensor(bbox_crop(samples[r["batch_index"]]["image"], r["bbox"], context=4.)) for r in batch]
            with torch.inference_mode():
                logits = ctc(torch.stack(crops))
            for item, sequence in zip(batch, logits):
                x0, y0, x1, y1 = item["bbox"]
                oracle_ocr[item["batch_index"]][item["proposal_index"]] = {"text": decode_ctc(sequence.argmax(-1)), "score": item["score"],
                    "center": [(x0 + x1) / 1536, (y0 + y1) / 1536], "half_extent": [(x1 - x0) / 1536, (y1 - y0) / 1536]}
        for i, sample in enumerate(samples):
            key = sample["key"]
            for a, current in enumerate((atoms, oracle)):
                local = {name: value[i:i + 1] for name, value in current.items()}
                pairs, indices = enumerate_unordered_atom_pairs(local["atom_mask"])
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = direct(features[i:i + 1], local["atom_features"], local["atom_geometry"][..., :2],
                                       local["atom_scores"], pairs, indices)
                ocr = baseline_ocr[key] if not a else oracle_ocr[i]
                scene, diagnostics = compile_prediction(local, pairs, predicted, ocr, bond_threshold)
                neural_records.append({"sample_key": key, "A": a, "ocr": ocr, "diagnostics": diagnostics})
                protected = json.dumps(scene, sort_keys=True)
                for b in (0, 1):
                    for c in (0, 1):
                        arm = f"A{a}B{b}C{c}"
                        changed, control = intervene(scene, truth[key]["scene"], attributes=bool(b), bonds=bool(c))
                        metrics, compiled, block = compile_and_score(changed, truth[key]["mol"])
                        rows.append({"sample_key": key, "variant": config["sample_metadata"][key]["variant"], "arm": arm, **metrics, **control})
                        scenes.append({"sample_key": key, "arm": arm, "scene": compiled})
                        if block is not None:
                            with (output / "sdf" / f"{key}-{arm}.sdf").open("x") as stream:
                                stream.write(block + "\n$$$$\n")
                if json.dumps(scene, sort_keys=True) != protected:
                    raise ValueError("An oracle arm mutated its baseline scene")
        print(f"images={len(rows)//8}/200", flush=True)

    pending = []
    for sample in selected_images(config):
        pending.append(sample)
        if len(pending) == 6:
            consume(pending)
            pending = []
    if pending:
        consume(pending)
    if model_digest(models) != before:
        raise ValueError("Frozen model state changed")
    verify(output)
    write_new_jsonl(output / "samples.jsonl", rows)
    write_new_jsonl(output / "scenes.jsonl", scenes)
    write_new_jsonl(output / "neural_records.jsonl", neural_records)
    report = summarize(rows, config)
    report["protocol_sha256"] = sha256_file(output / "protocol.json")
    report["frozen_models_unchanged"] = True
    report["baseline_deployed_replay_exact"] = True
    report["thresholds"] = {"atom": atom_threshold, "bond": probability, "role": role_threshold}
    report["artifacts"] = {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_new_json(output / "report.json", report)
    return report
