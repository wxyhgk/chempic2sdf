"""Investigate a failed oracle preflight without retrying or changing its protocol."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import tarfile

from rdkit import Chem, RDLogger

from Data.sdf.chemscene_v0_2 import build_chemscene_v0_2
from chemvector.external_ocr_r0_contracts import sha256_file, write_new_json, write_new_jsonl
from .controls import compile_and_score
from .protocol import verify


def inspect(output):
    config = verify(output)
    destination = output / "control_investigation"
    destination.mkdir(exist_ok=False)
    records = []
    archives = {path: tarfile.open(path, "r:") for path in config["shards"]}
    try:
        for key in config["sample_order"]:
            archive = archives[config["image_index"][key]["shard"]]
            values = {}
            for suffix in ("chemmol.json", "render.json", "layers.json", "sdf"):
                with archive.extractfile(f"{key}.{suffix}") as stream:
                    values[suffix] = stream.read()
            chemical, render, layers = [json.loads(values[s]) for s in ("chemmol.json", "render.json", "layers.json")]
            target = Chem.MolFromMolBlock(values["sdf"].decode(), sanitize=True, removeHs=True)
            coordinates = defaultdict(list)
            for atom in render["atoms"]:
                coordinates[atom["quantized_x"], atom["quantized_y"]].append(atom)
            collisions = []
            for position, atoms in coordinates.items():
                if len(atoms) < 2:
                    continue
                collisions.append({"quantized": list(position), "atom_ids": [a["atom_id"] for a in atoms],
                    "pixels": [[a["image_x"], a["image_y"]] for a in atoms],
                    "maximum_pair_distance_px": max(math.hypot(a["image_x"] - b["image_x"], a["image_y"] - b["image_y"])
                                                    for a in atoms for b in atoms)})
            row = {"sample_key": key, "molecule_id": chemical["molecule_id"], "variant": render["variant"],
                   "source_sdf_valid": target is not None, "coordinate_collisions": collisions,
                   "scene_constructed": False, "core_graph_exact": False, "stereo_graph_exact": False}
            try:
                scene = build_chemscene_v0_2(chemical, render, layers)
                row["scene_constructed"] = True
                metrics, _, _ = compile_and_score(scene, target)
                row.update({name: metrics[name] for name in ("core_graph_exact", "stereo_graph_exact", "errors")})
            except Exception as error:
                row["errors"] = [type(error).__name__ + ": " + str(error)]
            records.append(row)
    finally:
        for archive in archives.values():
            archive.close()
    write_new_jsonl(destination / "samples.jsonl", records)
    report = {"protocol_sha256": sha256_file(output / "protocol.json"),
              "investigator_source_sha256": sha256_file(Path(__file__)), "images": len(records),
              "source_sdf_valid": sum(r["source_sdf_valid"] for r in records),
              "scene_constructed": sum(r["scene_constructed"] for r in records),
              "oracle_core_graph_exact": sum(r["core_graph_exact"] for r in records),
              "oracle_stereo_graph_exact": sum(r["stereo_graph_exact"] for r in records),
              "errors": dict(Counter(e for r in records for e in r["errors"])),
              "failed_keys": [r["sample_key"] for r in records if not r["core_graph_exact"]],
              "samples_sha256": sha256_file(destination / "samples.jsonl"),
              "neural_inference_run": False, "training_performed": False, "original_failure_preserved": True,
              "architecture_bottleneck_ranking_available": False}
    write_new_json(destination / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    print(json.dumps(inspect(Path(args.output_dir)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
