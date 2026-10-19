"""Count actual, target-compatible unique pixels; never turn quarantine into training."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import io
import json
from pathlib import Path

from PIL import Image

from .expansion import write_json
from .expansion_plan import MINIMUM_IMAGES
from .full_molecule_data import file_hash


def unique_candidates(records):
    """One representative per pixel identity; conflicting chemical labels stay out."""
    by_pixels = defaultdict(list)
    for row in records:
        by_pixels[row["pixel_sha256"]].append(row)
    selected, duplicates, conflicts = [], [], []
    for pixel_hash, rows in sorted(by_pixels.items()):
        if len({r["canonical_smiles"] for r in rows}) != 1:
            conflicts.append({"pixel_sha256": pixel_hash, "sample_keys": sorted(r["sample_key"] for r in rows)})
            continue
        compatible = [r for r in rows if r["target_compatible"]]
        if not compatible:
            continue
        chosen = min(compatible, key=lambda r: (r["profile_id"] != "reference", r["sample_key"]))
        selected.append(chosen)
        duplicates.extend(r["sample_key"] for r in compatible if r is not chosen)
    return selected, sorted(duplicates), conflicts


def run(expansion_root, preflight_root, output):
    expansion_root, preflight_root, output = map(Path, (expansion_root, preflight_root, output))
    if output.exists():
        raise FileExistsError("Use a new expansion inventory directory")
    root = expansion_root / "dataset"
    inputs = {}
    def read(path):
        payload = path.read_bytes()
        inputs[str(path.resolve())] = sha256(payload).hexdigest()
        return payload
    plan = json.loads(read(expansion_root / "plan.json"))
    generation = json.loads(read(expansion_root / "report.json"))
    if generation["plan_sha256"] != file_hash(expansion_root / "plan.json"):
        raise ValueError("Generation plan binding changed")
    preflight = json.loads(read(preflight_root / "report.json"))
    target_rows = json.loads(read(preflight_root / "samples.json"))
    raw = {name: read(root / name) for name in ("summary.json", "samples.jsonl", "sample-sha256.json", "molecules.json")}
    if any(sha256(raw[name]).hexdigest() != expected for name, expected in preflight["input_sha256"].items()):
        raise ValueError("Target audit is not bound to these generated samples")
    summary, hashes, molecules = (json.loads(raw[n]) for n in ("summary.json", "sample-sha256.json", "molecules.json"))
    rows = [json.loads(line) for line in raw["samples.jsonl"].splitlines()]
    targets = {r["sample_key"]: r for r in target_rows}
    if (len(targets) != len(target_rows) or set(targets) != {r["sample_key"] for r in rows}
            or len(rows) != summary["accepted_images"] or len(rows) != preflight["checked_images"]
            or summary["training_authorized"] is not False or preflight["training_authorized"] is not False):
        raise ValueError("Missing/duplicate or non-quarantine expansion audit rows")
    chemistry = {r["molecule_id"]: r["canonical_smiles"] for r in molecules}
    records, by_profile = [], defaultdict(Counter)
    for number, row in enumerate(rows, 1):
        key = row["sample_key"]
        if key not in targets or row["molecule_id"] not in chemistry:
            raise ValueError("Unknown expansion sample identity")
        path = "samples/" + key + ".png"
        data = (root / path).read_bytes()
        if sha256(data).hexdigest() != hashes[path]:
            raise ValueError("Generated image changed after target checks")
        with Image.open(io.BytesIO(data)) as image:
            if image.mode != "RGB" or image.size != (768, 768):
                raise ValueError("Unexpected image shape or mode")
            pixel_hash = sha256(image.tobytes()).hexdigest()
        target = targets[key]
        records.append({"sample_key": key, "molecule_id": row["molecule_id"], "profile_id": row["profile_id"],
            "canonical_smiles": chemistry[row["molecule_id"]], "target_compatible": target["compatible"],
            "pixel_sha256": pixel_hash})
        bucket = by_profile[row["profile_id"]]
        bucket["render_accepted"] += 1
        bucket["target_compatible"] += bool(target["compatible"])
        if target["compatible"]:
            bucket["text_atom_targets"] += target["text_atoms"]
            bucket["implicit_atom_targets"] += target["atoms"] - target["text_atoms"]
        if number % 1000 == 0 or number == len(rows):
            print(f"Pixel inventory: {number}/{len(rows)}", flush=True)
    chosen, duplicates, conflicts = unique_candidates(records)
    for row in chosen:
        by_profile[row["profile_id"]]["unique_compatible_representatives"] += 1
    failed_keys = [r["sample_key"] for r in target_rows if not r["compatible"]]
    result = {"schema_version": "chemvector-expansion-inventory-0.1", "input_sha256": inputs,
        "planned_images": plan["planned_images"], "render_accepted_images": len(records),
        "chemistry_excluded_images": generation["chemistry_excluded_images"],
        "vector_excluded_images": generation.get("vector_excluded_images", 0),
        "render_rejected_images": summary["rejected_images"], "target_incompatible_images": len(failed_keys),
        "unique_pixel_images": len({r["pixel_sha256"] for r in records}),
        "compatible_unique_candidate_images": len(chosen), "compatible_duplicate_images_not_counted": len(duplicates),
        "conflicting_pixel_groups": len(conflicts), "candidate_molecules_with_unique_images": len({r["molecule_id"] for r in chosen}),
        "minimum_candidate_count_reached": len(chosen) >= MINIMUM_IMAGES, "minimum_admitted_train_images": MINIMUM_IMAGES,
        "admitted_new_train_images": 0, "training_authorized": False, "gpu_used": False,
        "images_by_profile": {k: dict(v) for k, v in sorted(by_profile.items())},
        "failed_target_sample_keys": failed_keys, "duplicate_sample_keys": duplicates,
        "conflicts": conflicts, "model_loaded": False, "training_performed": False,
        "limitations": ["Unique and compatible candidates are not an admitted training split.",
            "Private identity review, current public clearance and exact-release source-use review remain separate.",
            "Pixel deduplication does not prove chemical-environment diversity or model accuracy."]}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", result)
    write_json(output / "unique-candidates.json", chosen)
    write_json(output / "all-pixel-records.json", records)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expansion-root", "preflight-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.expansion_root, args.preflight_root, args.output)
    print(json.dumps({k: result[k] for k in ("compatible_unique_candidate_images", "target_incompatible_images",
        "conflicting_pixel_groups", "minimum_candidate_count_reached", "admitted_new_train_images")}))
    return 0 if result["minimum_candidate_count_reached"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
