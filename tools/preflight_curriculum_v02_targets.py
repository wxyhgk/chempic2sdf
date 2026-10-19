"""CPU-only target preflight of the exact v0.2 internal quality handoff.

No checkpoint, model forward, optimizer, identity index, or split is opened.
The prior independent auditor and quality handoff remain immutable. All raw
PNG conflict/dedup decisions are revalidated; only retained images are collated.
An incomplete journal is never a passed preflight or a training manifest.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import inspect
import json
import os
from pathlib import Path
import signal
import time

import torch

from chemvector.atom_bond_scene import ATOM_ROLES, AtomBondSceneConfig
from chemvector.bond_object import BOND_TYPE_TO_ID, DIRECTION_TO_ID
from chemvector.training.curriculum import data, pixels
from Data.sdf.chemscene_v0_2 import _clip_port_to_text_bbox
from tools import prepare_curriculum_v02_quality_handoff as quality

SCHEMA = "chemvector-curriculum-v0.2-selected-target-preflight"
read, decode, digest, require = quality.read, quality.decode, quality.digest, quality.require


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encoded(value))


def source_paths():
    return {Path(__file__).absolute(), Path(quality.__file__).absolute(), quality.AUDITOR,
            *(quality.REPO / name for name in quality.AUDITOR_DEPENDENCIES),
            Path(data.__file__).absolute(), Path(pixels.__file__).absolute(),
            Path(inspect.getfile(AtomBondSceneConfig)).absolute(),
            Path(inspect.getfile(_clip_port_to_text_bbox)).absolute(),
            quality.REPO / "chemvector/components/bonds/objects/bond_object.py"}


def snapshot(paths):
    return {str(path): digest(read(path)) for path in sorted(set(paths))}


def verify_bindings(bindings):
    for path, expected in bindings.items():
        require(digest(read(path)) == expected, f"Input changed: {path}")


def check_handoff(handoff):
    require(handoff.get("schema_version") == quality.SCHEMA
            and handoff.get("status") == "proposed_internal_quality_handoff_only",
            "Expected completed internal quality handoff")
    require(handoff.get("training_authorized") is False
            and handoff.get("external_overlap_status") == "unchecked"
            and handoff.get("formal_training_admission") == "not_granted"
            and handoff.get("external_identity_queries") == 0
            and handoff.get("split_created") is False
            and handoff.get("train_manifest_published") is False,
            "Handoff scope must remain quarantine-only")
    require(handoff.get("handoff_source_sha256") == digest(read(Path(quality.__file__).absolute())),
            "Quality handoff implementation changed")
    # Reuse the unchanged, tested conservative policy instead of implementing
    # a second selector that could accidentally retain conflicting identities.
    expected = quality.prepare_proposal(handoff["dataset"], handoff["audit"]["path"])
    require(all(handoff.get(key) == value for key, value in expected.items()),
            "Handoff differs from independently recomputed fixed selection")
    require(handoff["statistics"]["retained_distinct_decoded_rgb"] == len(handoff["retained_candidates"]),
            "Retained candidate RGB duplicates need a separate quality decision")
    return handoff["retained_candidates"]


def verify_sample(root, row, hashes):
    key = row["sample_key"]
    sample = data.load_sample(root, key, purpose="diagnostic")
    for suffix in (*data.PAYLOADS, ".meta.json"):
        name = f"samples/{key}{suffix}"
        require(digest(read(root / name)) == hashes.get(name), f"Corpus payload SHA mismatch: {name}")
    meta = sample["meta"]
    require(meta.get("training_authorized") is False and meta.get("split") == "quarantine"
            and meta.get("external_overlap_status") == "unchecked", "Sample is not quarantined")
    require(all(meta[field] == row[field] for field in ("sample_key", "molecule_id", "profile_id")),
            "Selected sample identity mismatch")
    require(meta["sha256"][".png"] == row["png_sha256"]
            and digest(sample["image"].tobytes()) == row["decoded_rgb_sha256"],
            "Selected pixel binding mismatch")
    # Verify both entry points still refuse quarantine, even if the caller
    # changes only the in-memory diagnostic purpose flag.
    try:
        data.load_sample(root, key, purpose="training")
    except ValueError as exc:
        require("not admitted for training" in str(exc), "Unexpected training loader rejection")
    else:
        raise ValueError("Quarantine accepted by training loader")
    try:
        data.CurriculumCollator(purpose="training")([{**sample, "purpose": "training"}])
    except ValueError as exc:
        require("Training admission is required" in str(exc), "Unexpected training collator rejection")
    else:
        raise ValueError("Quarantine accepted by training collator")
    return sample


def validate_targets(sample, batch, config):
    """Independent shape/mask/class/semantic checks of one real collated image."""
    na, nb = config.atom_queries, config.bond_queries
    shapes = {
        "pixel_values": ((1, 3, 768, 768), torch.float32),
        "atom_anchors": ((1, na, 2), torch.float32),
        "atom_roles": ((1, na), torch.int64),
        "atom_text_center": ((1, na, 2), torch.float32),
        "atom_text_bbox": ((1, na, 4), torch.float32),
        "atom_text_mask": ((1, na), torch.bool), "atom_mask": ((1, na), torch.bool),
        "bond_ports": ((1, nb, 2, 2), torch.float32),
        "bond_display": ((1, nb), torch.int64), "bond_chemical": ((1, nb), torch.int64),
        "bond_direction": ((1, nb), torch.int64), "bond_atom_refs": ((1, nb, 2), torch.int64),
        "bond_mask": ((1, nb), torch.bool),
    }
    for name, (shape, dtype) in shapes.items():
        tensor = batch[name]
        require(tuple(tensor.shape) == shape and tensor.dtype == dtype and tensor.device.type == "cpu",
                f"Target shape/dtype/device mismatch: {name}")
        if dtype == torch.float32:
            require(bool(torch.isfinite(tensor).all()), f"Nonfinite target: {name}")
            low = -1 if name == "pixel_values" else 0
            require(bool(((tensor >= low) & (tensor <= 1)).all()), f"Target out of range: {name}")
    require(batch["purpose"] == "diagnostic" and batch["training_authorized"] is False
            and batch["target_schema"] == "chemvector-curriculum-targets-0.1", "Wrong target scope/schema")
    graph, layers = sample["graph"], sample["layers"]
    atoms = sorted(graph["atoms"], key=lambda item: item["atom_id"])
    bonds = sorted(graph["bonds"], key=lambda item: item["bond_id"])
    visuals = {a["atom_id"]: a for a in layers["atoms"]}
    drawings = {b["bond_id"]: b["drawing"] for b in layers["bonds"]}
    atom_ids, bond_ids = [a["atom_id"] for a in atoms], [b["bond_id"] for b in bonds]
    require(batch["keys"] == [sample["key"]] and batch["atom_ids"] == [atom_ids]
            and batch["bond_ids"] == [bond_ids] and batch["chemical_graphs"] == [graph],
            "Semantic identity/chemical graph not preserved")
    require(batch["atom_mask"][0].tolist() == [True] * len(atoms) + [False] * (na - len(atoms))
            and batch["bond_mask"][0].tolist() == [True] * len(bonds) + [False] * (nb - len(bonds)),
            "Target mask mismatch")
    quantized = {key: [round(v / 767 * 1023) for v in visual["center"]]
                 for key, visual in visuals.items()}
    cells = Counter(tuple(min(191, int(v / 1023 * 192)) for v in point) for point in quantized.values())
    collisions = sum(n * (n - 1) // 2 for n in cells.values())
    require(batch["spatial_cell_collision_pairs"] == [collisions], "Collision count mismatch")
    require(collisions == 0, "Existing spatial target has colliding atom cells")
    for index, atom in enumerate(atoms):
        aid = atom["atom_id"]
        visual = visuals[aid]
        expected = torch.tensor(quantized[aid]) / 1023
        require(torch.equal(batch["atom_anchors"][0, index], expected), "Atom anchor mismatch")
        require(batch["atom_roles"][0, index].item() == ATOM_ROLES.index(
                    "TEXT" if visual["visible_text"] else "IMPLICIT")
                and batch["atom_text_mask"][0, index].item() == visual["visible_text"],
                "Visible text role/mask mismatch")
    require(batch["visible_text"] == [[visuals[aid]["text"] for aid in atom_ids]], "Visible text changed")
    for index, bond in enumerate(bonds):
        drawing = drawings[bond["bond_id"]]
        refs = batch["bond_atom_refs"][0, index].tolist()
        require(all(0 <= ref < len(atoms) for ref in refs), "Bond endpoint points outside active atoms")
        require([atom_ids[ref] for ref in refs] == [drawing["begin_atom"], drawing["end_atom"]]
                and {atom_ids[ref] for ref in refs} == {bond["begin_atom"], bond["end_atom"]},
                "Bond endpoint identity/drawing direction mismatch")
        for field, expected in (("bond_chemical", BOND_TYPE_TO_ID[bond["chemical_type"]]),
                                ("bond_display", BOND_TYPE_TO_ID[drawing["bond_type"]]),
                                ("bond_direction", DIRECTION_TO_ID[drawing["direction"]])):
            require(batch[field][0, index].item() == expected, f"Bond class mismatch: {field}")
    return {"atoms": len(atoms), "bonds": len(bonds), "text_atoms": int(batch["atom_text_mask"].sum()),
            "spatial_cell_collision_pairs": collisions,
            "target_sha256": {name: digest(batch[name].contiguous().numpy().tobytes())
                              for name in shapes if name != "pixel_values"}}


def run(handoff_path, output, *, max_seconds=900):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only: set CUDA_VISIBLE_DEVICES to empty")
    require(type(max_seconds) is int and 0 < max_seconds <= 1800, "Invalid bounded CPU duration")
    torch.set_num_threads(1)
    handoff_path, output = Path(handoff_path).absolute(), Path(output).absolute()
    require(output.resolve() == output, "Symlinked output")
    handoff_raw = read(handoff_path)
    handoff = decode(handoff_raw)
    root = Path(handoff["dataset"])
    require(root.is_absolute() and root.resolve() == root, "Invalid dataset path")
    require(not output.is_relative_to(root), "Output must not modify the dataset")
    bindings = snapshot(source_paths() | {handoff_path, Path(handoff["audit"]["path"]),
                                          *(root / name for name in quality.MANIFESTS)})
    require(bindings[str(handoff_path)] == digest(handoff_raw), "Handoff changed at startup")
    output.mkdir(parents=True, exist_ok=False)
    config = AtomBondSceneConfig()
    start = time.monotonic()
    report = dict(schema_version=SCHEMA, status="incomplete", complete=False, dataset=str(root),
                  handoff_path=str(handoff_path), input_sha256=bindings, config=asdict(config),
                  candidate_selection="exact_unchanged_internal_quality_handoff",
                  selected_images=len(handoff["retained_candidates"]), checked_images=0,
                  compatible_images=0, failed_images=0, max_seconds=max_seconds,
                  training_authorized=False, training_performed=False, gpu_used=False,
                  model_forwards=0, optimizer_steps=0, external_identity_queries=0,
                  external_overlap_status="unchecked", split_created=False,
                  predicted_candidate_reachability="not_measured",
                  not_supervised_by_adapter=["OCR transcription", "element/charge/isotope classification",
                                             "functional-group classification"],
                  limitations=["Data target compatibility is not model accuracy or predicted pair reachability.",
                               "Drawing wedge/dash is separate from chemical bond order.",
                               "Identity admission, grouped release and new training budget remain separate."])
    write_new(output / "plan.json", report)
    totals, elements, chemical, display, directions, profiles = (Counter() for _ in range(6))
    identities = set()
    maxima = {"atoms": 0, "bonds": 0}
    failure_examples = []

    def deadline():
        if time.monotonic() - start >= max_seconds:
            raise TimeoutError("CPU target preflight deadline reached")

    def terminate(signum, frame):
        raise InterruptedError(f"Preflight interrupted by signal {signum}")

    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        print("Revalidating fixed selection and original PNG conflict groups", flush=True)
        candidates = check_handoff(handoff)
        verify_bindings(bindings)
        hashes = decode(read(root / "sample-sha256.json"))
        deadline()
        collator = data.CurriculumCollator(config, purpose="diagnostic")
        with (output / "samples.jsonl").open("x", encoding="utf-8") as journal:
            for row in candidates:
                deadline()
                result = {"sample_key": row["sample_key"], "molecule_id": row["molecule_id"], "compatible": False}
                try:
                    sample = verify_sample(root, row, hashes)
                except Exception as exc:
                    result.update(error_type=type(exc).__name__, error=str(exc), artifact_integrity_failure=True)
                    journal.write(json.dumps(result, sort_keys=True) + "\n")
                    journal.flush()
                    raise  # Corruption must not turn into a quiet candidate exclusion.
                try:
                    result.update(validate_targets(sample, collator([sample]), config), compatible=True)
                except (ValueError, KeyError, IndexError, RuntimeError) as exc:
                    result.update(error_type=type(exc).__name__, error=str(exc))
                    if len(failure_examples) < 20:
                        failure_examples.append(result.copy())
                journal.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                journal.flush()
                report["checked_images"] += 1
                report["compatible_images"] += int(result["compatible"])
                report["failed_images"] += int(not result["compatible"])
                if result["compatible"]:
                    totals.update({name: result[name] for name in ("atoms", "bonds", "text_atoms", "spatial_cell_collision_pairs")})
                    for name in maxima:
                        maxima[name] = max(maxima[name], result[name])
                    elements.update(a["element"] for a in sample["graph"]["atoms"])
                    chemical.update(b["chemical_type"] for b in sample["graph"]["bonds"])
                    display.update(b["drawing"]["bond_type"] for b in sample["layers"]["bonds"])
                    directions.update(b["drawing"]["direction"] for b in sample["layers"]["bonds"])
                identities.add(row["molecule_id"])
                profiles.update([row["profile_id"]])
                if report["checked_images"] % 250 == 0 or report["checked_images"] == len(candidates):
                    os.fsync(journal.fileno())
                    print(json.dumps({"checked": report["checked_images"], "total": len(candidates),
                                      "failed": report["failed_images"], "elapsed_seconds": round(time.monotonic() - start, 2)}), flush=True)
        verify_bindings(bindings)
        deadline()
        require(report["checked_images"] == report["selected_images"], "Incomplete selected denominator")
        report.update(complete=True, status="passed_target_compatibility_only" if report["failed_images"] == 0
                      else "failed_target_compatibility", journal_sha256=digest(read(output / "samples.jsonl")))
    except BaseException as exc:
        report.update(status="failed_or_incomplete", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        report.update(elapsed_seconds=time.monotonic() - start, failure_examples=failure_examples,
                      checked_molecules=len(identities), compatible_totals=dict(totals), max_objects=maxima,
                      elements=dict(elements), chemical_bonds=dict(chemical), drawing_bonds=dict(display),
                      drawing_directions=dict(directions), checked_profiles=dict(profiles),
                      training_loader_rejections=report["checked_images"],
                      training_collator_rejections=report["checked_images"])
        write_new(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    result = run(args.handoff, args.output, max_seconds=args.max_seconds)
    print(json.dumps({key: result[key] for key in ("status", "checked_images", "compatible_images", "failed_images")}))
    return 0 if result["status"] == "passed_target_compatibility_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
