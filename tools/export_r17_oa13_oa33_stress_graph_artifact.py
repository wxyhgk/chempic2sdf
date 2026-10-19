"""Export a target-free oa13/oa33 owner+NMS graph artifact.

This is an integration receipt, not training.  It freezes the production
checkpoint, applies only target-free 16 px selected-owner exclusion and 8 px
score NMS to sidecar candidates, then records the resulting atom mask and
top-192 direct-pair outputs.  Public target graphs are intentionally not read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from chemvector.components.graph.joint.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.audit_r17_selector_graph_replay import SelectorSidecar, score_nms


MANIFEST = ROOT / "outputs/audits/chemvector-r17-oa13-oa33-known-clean-stress-manifest-v2/manifest.json"
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923/samples"
STATE = ROOT / "outputs/audits/chemvector-r17-selector-retention-m128-v4/sidecar-state.pt"
OUT = ROOT / "outputs/audits/chemvector-r17-oa13-oa33-stress-graph-artifact-nms8-v1"
SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
THRESHOLD = 0.95
OWNER_RADIUS = 16.0
NMS_RADIUS = 8.0
MAX_NEW = 16
MAX_BONDS = 192


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serial(tensor: torch.Tensor):
    return tensor.detach().cpu().tolist()


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(INITIAL) != SHA:
        raise RuntimeError("initial checkpoint SHA changed")
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))["rows"]
    state = torch.load(STATE, map_location="cpu", weights_only=False)
    if state.get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("sidecar/checkpoint binding mismatch")
    model, config, payload = load_bundle_for_probe(INITIAL, SHA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval().requires_grad_(False)
    scorer = SelectorSidecar().to(device)
    scorer.load_state_dict(state["state_dict"])
    scorer.eval()
    mean = state["mean"].to(device)
    std = state["std"].to(device)
    records = []
    owner_excluded = 0
    nms_removed = 0
    with torch.inference_mode():
        for row in rows:
            key = row["sample_key"]
            pixels = image_tensor(Image.open(SOURCE / f"{key}.png").convert("RGB"))[None].to(device)
            outputs = structure_forward(model, pixels, config, payload["provenance"]["atom_threshold"])
            vision, _, _, raw, _, setout, base = outputs
            raw_mask = raw["atom_mask"][0].bool()
            base_mask = base["atom_mask"][0].bool()
            raw_slots = torch.where(raw_mask)[0]
            raw_features = raw["atom_features"][0, raw_slots].float()
            xy = raw["atom_geometry"][0, raw_slots, :2].float()
            scores = raw["atom_scores"][0, raw_slots].float()
            presence = setout["atom_presence_logits"][0, raw_slots].float()
            pre_presence = setout.get("atom_pre_presence_logits", setout["atom_presence_logits"])[0, raw_slots].float()
            features = torch.cat((raw_features, scores[:, None], xy,
                                  presence[:, None], pre_presence[:, None]), -1)
            sidecar_probability = scorer((features - mean) / std).sigmoid()
            eligible = (sidecar_probability >= THRESHOLD) & (~base_mask[raw_slots])
            candidate_pre_owner = torch.where(eligible)[0]
            if int(base_mask.sum()) > 0 and len(candidate_pre_owner):
                selected_points = raw["atom_geometry"][0, torch.where(base_mask)[0], :2]
                distance = torch.cdist(xy[candidate_pre_owner], selected_points).min(-1).values * 768.0
                owner_keep = distance > OWNER_RADIUS
                owner_excluded += int((~owner_keep).sum())
                candidate_post_owner = candidate_pre_owner[owner_keep]
            else:
                candidate_post_owner = candidate_pre_owner
            candidate = score_nms(candidate_post_owner, xy, sidecar_probability, NMS_RADIUS)
            nms_removed += int(len(candidate_post_owner) - len(candidate))
            if len(candidate) > MAX_NEW:
                candidate = candidate[torch.argsort(sidecar_probability[candidate], descending=True)[:MAX_NEW]]
            augmented_mask = base_mask.clone()
            augmented_mask[raw_slots[candidate]] = True
            pairs, sample_indices = enumerate_unordered_atom_pairs(augmented_mask[None])
            direct = model.pair(vision, raw["atom_features"], raw["atom_geometry"][..., :2], raw["atom_scores"], pairs, sample_indices)
            probabilities = direct["presence_logits"].sigmoid()
            if len(pairs) > MAX_BONDS:
                keep = torch.argsort(probabilities, descending=True)[:MAX_BONDS]
                pairs = pairs[keep]
                probabilities = probabilities[keep]
                chemical = direct["chemical_logits"][keep]
            else:
                chemical = direct["chemical_logits"]
            endpoints = raw["atom_geometry"][0, pairs, :2]
            mask = torch.ones(len(pairs), dtype=torch.bool, device=device)
            if len(pairs) < MAX_BONDS:
                pad = MAX_BONDS - len(pairs)
                endpoints = torch.cat((endpoints, endpoints.new_zeros((pad, 2, 2))))
                probabilities = torch.cat((probabilities, probabilities.new_zeros(pad)))
                chemical = torch.cat((chemical, chemical.new_zeros((pad, chemical.size(-1)))))
                mask = torch.cat((mask, torch.zeros(pad, dtype=torch.bool, device=device)))
            records.append({
                "sample_key": key, "split": row["split"], "profile_id": row["profile_id"],
                "atom_points": serial(raw["atom_geometry"][0, :, :2]),
                "atom_keep_mask": serial(augmented_mask), "atom_raw_mask": serial(raw_mask),
                "bond_endpoints": serial(endpoints), "bond_presence_logits": serial(probabilities),
                "bond_type_ids": serial(chemical.argmax(-1)), "bond_type_logits": serial(chemical),
                "bond_mask": serial(mask),
            })
    OUT.mkdir(parents=True, exist_ok=False)
    artifact = OUT / "graph-artifact.jsonl"
    with artifact.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    report = {
        "schema_version": "chemvector-r17-oa13-oa33-stress-graph-artifact-nms8-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "gpu_used": device.type == "cuda", "device": str(device),
        "sealed_test_read": False, "production_checkpoint_written": False,
        "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": digest(INITIAL),
        "manifest_sha256": digest(MANIFEST), "sidecar_sha256": digest(STATE),
        "artifact_sha256": digest(artifact), "rows": len(records),
        "target_free": True, "owner_radius_px": OWNER_RADIUS, "nms_radius_px": NMS_RADIUS,
        "owner_excluded": owner_excluded, "nms_removed": nms_removed,
        "safe_compile_applied": False,
        "gate": {"checkpoint_unchanged": digest(INITIAL) == SHA,
                 "artifact_complete": len(records) == len(rows), "target_free": True,
                 "production_go": False},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "report": report}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
