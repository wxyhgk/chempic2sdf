"""Read-only gap attribution on the ten cov13_rot0 failure identities.

Renders three isolated profile variants in memory and runs the frozen initial
model.  No image payloads are persisted and no optimizer/model state changes.
"""

from collections import Counter, defaultdict
from hashlib import sha256
import io
import json
from pathlib import Path
import math

from PIL import Image
import torch

from Data.sdf.case_diversity_v1.rendering import render
from Data.sdf.chemvector_curriculum_v0_1.rendering import Profile
from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator, load_sample
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/datasets/chemvector-oa13-oa33-coverage-panel-20261017-r2"
OUTPUT = ROOT / "outputs/audits/chemvector-cov13-cap-stroke-gap-attribution-20261017/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_sample(row, profile):
    if profile.profile_id == "cov13_base":
        # Reuse the already accepted quarantine payload; this avoids making a
        # second renderer decision for the control cell.
        return load_sample(SOURCE, row["molecule_id"] + "__cov13_base", purpose="diagnostic")
    entry = {"molecule_id": row["molecule_id"], "canonical_smiles": row["canonical_smiles"],
             "split": "quarantine", "training_authorized": False,
             "external_overlap_status": "unchecked"}
    files, meta = render(entry, profile)
    graph = json.loads(files[".graph.json"])
    layers = json.loads(files[".layers.json"])
    records = json.loads(files[".text-records.json"])
    with Image.open(io.BytesIO(files[".png"])) as image:
        image = image.copy()
    return {"key": meta["sample_key"], "image": image, "graph": graph,
            "layers": layers, "text_records": records, "meta": meta,
            "purpose": "diagnostic"}


def distance_bucket(value):
    if value < 20:
        return "lt20"
    if value < 30:
        return "20_30"
    if value < 40:
        return "30_40"
    return "ge40"


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows = [row for row in json.loads((SOURCE / "rejections.json").read_bytes())
            if row["profile_id"] == "cov13_rot0"]
    if len(rows) != 10:
        raise ValueError("expected exactly ten cov13_rot0 failures")
    base = dict(width=768, height=768, font_size=32, bond_length_cap=28.0,
                bond_line_width=3.2, rotation_degrees=0.0, font_family="DejaVu Serif")
    variants = {
        "cov13_base": Profile(profile_id="cov13_base", **{**base, "rotation_degrees": 180.0}),
        "cap52": Profile(profile_id="probe_cap52", **{**base, "bond_length_cap": 52.0}),
        "stroke16": Profile(profile_id="probe_stroke16", **{**base, "bond_line_width": 1.6}),
    }
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    if digest(INITIAL) != INITIAL_SHA or not torch.cuda.is_available():
        raise RuntimeError("initial binding or CUDA unavailable")
    device = torch.device("cuda:0")
    model.to(device).eval().requires_grad_(False)
    before = state_digest(model)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
                                                       bond_queries=config.maximum_bonds), purpose="diagnostic")
    result = {}
    for name, profile in variants.items():
        samples, render_rejections = [], Counter()
        for row in rows:
            try:
                samples.append(make_sample(row, profile))
            except ValueError as error:
                render_rejections[str(error)] += 1
        if not samples:
            result[name] = {"attempted_images": len(rows), "accepted_images": 0,
                            "render_rejections": dict(sorted(render_rejections.items())),
                            "target_atoms": 0, "raw_gap": None, "raw_reachability": None,
                            "buckets": {}}
            continue
        batch = collator(samples)
        tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        with torch.inference_mode():
            _, _, _, raw, _, _, _ = structure_forward(model, tensors["pixel_values"], config,
                                                       payload["provenance"]["atom_threshold"])
        raw_map, _ = match_atom_targets(raw, tensors, config)
        block = Counter()
        for index, sample in enumerate(samples):
            graph = sample["graph"]
            layers = {int(atom["atom_id"]): atom for atom in sample["layers"]["atoms"]}
            atom_ids = [int(value) for value in batch["atom_ids"][index]]
            raw_ids = {atom_ids[int(value)] for value in raw_map[index].tolist()
                       if 0 <= int(value) < len(atom_ids)}
            distances = defaultdict(list)
            centers = {int(atom["atom_id"]): tuple(atom["center"]) for atom in sample["layers"]["atoms"]}
            for bond in graph["bonds"]:
                a, b = int(bond["begin_atom"]), int(bond["end_atom"])
                distance = math.dist(centers[a], centers[b])
                distances[a].append(distance); distances[b].append(distance)
            for atom in graph["atoms"]:
                atom_id = int(atom["atom_id"]); info = layers[atom_id]
                if info["visible_text"] and info["element"] in {"O", "N"}:
                    kind = "visible_O_N"
                elif not info["visible_text"] and info["element"] == "C":
                    kind = "implicit_C"
                elif info["visible_text"]:
                    kind = "other_visible"
                else:
                    kind = "other_implicit"
                nearest = min(distances[atom_id], default=0.0)
                dist = distance_bucket(nearest)
                status = "gap" if atom_id not in raw_ids else "matched"
                block[f"{status}:{kind}:{dist}"] += 1
                block[f"{status}:{kind}"] += 1
        target = sum(value for key, value in block.items() if key.startswith("matched:") and key.count(":") == 1) + sum(value for key, value in block.items() if key.startswith("gap:") and key.count(":") == 1)
        gap = sum(value for key, value in block.items() if key.startswith("gap:") and key.count(":") == 1)
        result[name] = {"attempted_images": len(rows), "accepted_images": len(samples),
                        "render_rejections": dict(sorted(render_rejections.items())),
                        "target_atoms": target, "raw_gap": gap,
                        "raw_reachability": (target-gap) / max(target, 1),
                        "buckets": dict(sorted(block.items()))}
    after = state_digest(model)
    if before != after or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("frozen initial state changed")
    report = {"schema_version": "chemvector-cov13-cap-stroke-gap-attribution-0.1",
              "complete": True, "read_only": True, "training_performed": False,
              "gpu_used": True, "model_forwards": 3, "sealed_test_read": False,
              "initial_sha256": INITIAL_SHA, "initial_sha256_after": digest(INITIAL),
              "source_rejections_sha256": digest(SOURCE / "rejections.json"),
              "identities": len(rows), "variants": result,
              "state_before": before, "state_after": after,
              "next": "cap-only variant can be considered as a new profile; do not relabel cov13_rot0 or train from this probe"}
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
