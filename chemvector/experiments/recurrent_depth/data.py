"""Exact train-shard allowlist and truth-free frozen candidate feature cache."""

import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch

from chemvector.external_ocr_r0_contracts import sha256_bytes, sha256_file, validate_frozen_file

ROOT = Path("Data/sdf/chemvector_joint_v1/style_train_100k")
SHARD = ROOT / "shards/train-000000.tar"


def selection(train_molecules=256, validation_molecules=64):
    groups = {}
    with (ROOT / "samples.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["output_shard"] != "shards/train-000000.tar":
                continue
            if row["split"] != "train":
                raise ValueError("Only training corpus is allowed")
            groups.setdefault(row["inchikey"], []).append(row)
    ordered = sorted(groups, key=lambda key: hashlib.sha256(("rd-pilot-20260913:" + key).encode()).hexdigest())
    if len(ordered) < train_molecules + validation_molecules:
        raise ValueError("Insufficient molecule groups")
    selected = {}
    for index, identity in enumerate(ordered[:train_molecules + validation_molecules]):
        rows = groups[identity]
        if len(rows) != 4 or {r["variant"] for r in rows} != {"original", "rdkit", "coordgen", "stress"}:
            raise ValueError("Require four variants per molecule")
        for row in rows:
            selected[row["sample_key"]] = {**row, "pilot_split": "train" if index < train_molecules else "validation"}
    train_ids = {r["molecule_id"] for r in selected.values() if r["pilot_split"] == "train"}
    validation_ids = {r["molecule_id"] for r in selected.values() if r["pilot_split"] == "validation"}
    if train_ids & validation_ids:
        raise ValueError("Molecule identity crosses pilot split")
    return selected


def selected_records(selected):
    with tarfile.open(SHARD, "r:") as archive:
        members = {member.name: member for member in archive if member.isfile()}
        for key, row in sorted(selected.items()):
            payloads = {}
            for suffix, hash_field in (("png", "png_sha256"), ("chemmol.json", "chemmol_sha256"),
                                       ("render.json", "render_sha256")):
                with archive.extractfile(members[f"{key}.{suffix}"]) as stream:
                    payload = stream.read()
                if sha256_bytes(payload) != row[hash_field]:
                    raise ValueError(f"Selected member hash differs: {key}.{suffix}")
                payloads[suffix] = payload
            with Image.open(io.BytesIO(payloads["png"])) as image:
                if image.size != (768, 768):
                    raise ValueError("Unexpected image coordinate frame")
                pixels = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32).copy()).permute(2, 0, 1) / 127.5 - 1
            chemical = json.loads(payloads["chemmol.json"])
            render = json.loads(payloads["render.json"])
            if chemical["molecule_id"] != row["molecule_id"] or render["variant"] != row["variant"]:
                raise ValueError("Molecule/variant identity changed")
            if {a["id"] for a in chemical["atoms"]} != {a["atom_id"] for a in render["atoms"]}:
                raise ValueError("Chemical/render atom IDs differ")
            targets = torch.tensor([[a["image_x"], a["image_y"]] for a in render["atoms"]], dtype=torch.float32)
            yield key, pixels, targets


def assign(anchors, mask, targets, radius=6.0):
    """Maximum-cardinality geometric matching, with distance tie-breaking."""
    valid = np.flatnonzero(mask)
    if not len(valid) or not len(targets):
        return np.empty(0, dtype=int)
    distances = np.linalg.norm(anchors[valid, None] - targets[None], axis=-1)
    penalty = (min(distances.shape) + 1) * (radius + 1)
    costs = np.where(distances <= radius, distances, penalty)
    rows, cols = linear_sum_assignment(costs)
    return valid[rows[distances[rows, cols] <= radius]]


def build_cache(output, protocol):
    from chemvector.predicted_incidence import PredictedProposalConfig, decode_predicted_atom_proposals, decode_predicted_bond_proposals
    from chemvector.train_bond_presence_calibration import load_components
    from chemvector.train_spatial_incidence import forward_frozen_bond_head
    from chemvector.text_entry_predictions import model_digest

    if (output / "cache.pt").exists():
        raise FileExistsError("Cache already exists")
    vision, atom, bond, selector, payload = load_components(protocol["source"], protocol["graph"], torch.device("cuda"))
    models = (vision, atom, bond, selector)
    for model in models:
        model.eval().requires_grad_(False)
    before = model_digest(models)
    config = PredictedProposalConfig(**payload["proposal_config"])
    threshold = float(payload["training_args"]["atom_decision_threshold"])
    if threshold != -.25 or config.maximum_atoms != 128:
        raise ValueError("Frozen candidate configuration differs")
    records, pending = [], []

    def consume(batch):
        # Model receives ONLY pixels; target coordinates are used after inference.
        pixels = torch.stack([r[1] for r in batch]).cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            features = vision.forward_stroke_resolution(pixels)
            atoms = decode_predicted_atom_proposals(atom(features), config)
            bonds = decode_predicted_bond_proposals(forward_frozen_bond_head(vision, bond, features), config)
            result = selector(atoms["atom_features"], atoms["atom_geometry"], atoms["atom_scores"], atoms["atom_mask"],
                              bonds["bond_features"], bonds["bond_endpoints"], bonds["bond_types"], bonds["bond_scores"], bonds["bond_mask"])
            evidence = torch.cat((atoms["atom_features"], atoms["atom_geometry"], atoms["atom_scores"][..., None],
                                  result["atom_tokens"], result["atom_presence_logits"].sigmoid()[..., None]), dim=-1).float().cpu()
        for i, (key, _, targets) in enumerate(batch):
            mask = atoms["atom_mask"][i].cpu()
            anchors = atoms["atom_geometry"][i, :, :2].float().cpu() * 768
            labels = torch.zeros(mask.shape, dtype=torch.float32)
            labels[assign(anchors.numpy(), mask.numpy(), targets.numpy())] = 1
            records.append({"key": key, "x": evidence[i].masked_fill(~mask[:, None], 0), "mask": mask,
                            "base": result["atom_presence_logits"][i].float().cpu() - threshold,
                            "anchors": anchors, "targets": targets, "labels": labels,
                            "split": protocol["selection"][key]["pilot_split"]})
        if len(records) % 96 == 0:
            print(f"cache images={len(records)}/{len(protocol['selection'])}", flush=True)

    for record in selected_records(protocol["selection"]):
        pending.append(record)
        if len(pending) == 6:
            consume(pending)
            pending = []
    if pending:
        consume(pending)
    if before != model_digest(models):
        raise ValueError("Frozen feature producer changed")
    if {r["key"] for r in records} != set(protocol["selection"]):
        raise ValueError("Incomplete feature cache")
    torch.save({"records": records, "producer_state_sha256": before}, output / "cache.pt")
    return {"images": len(records), "sha256": sha256_file(output / "cache.pt"), "producer_state_sha256": before}
