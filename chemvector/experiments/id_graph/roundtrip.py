"""Frozen, truth-only compiler control on the already exposed 200 dev examples.

Run from the repository root: python -m chemvector.experiments.id_graph.roundtrip
No model, image decoding, legacy scene conversion, or production changes.
"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import tarfile

from rdkit import Chem, rdBase

from .graph import from_chemmol, to_molecule, to_sdf

SELECTION = Path("outputs/experiments/e2e-bottleneck-factorial-20260913/protocol.json")
SELECTION_SHA = "d7c000dd18372cb2c1449d60bd51d6d3fae73ecc580635f2748525981c087c0c"
DEFAULT_OUTPUT = Path("outputs/experiments/id-graph-roundtrip-20260913")
METRICS = ("direct_core_exact", "direct_stereo_exact", "sdf_core_exact", "sdf_stereo_exact",
           "relabel_invariant", "observation_independent", "inputs_unchanged")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def require_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Frozen file changed: {path}")


def freeze(output):
    if output.exists():
        raise FileExistsError("Use a new output directory; do not overwrite results")
    require_hash(SELECTION, SELECTION_SHA)
    old = json.loads(SELECTION.read_text())
    order = old["sample_order"]
    if len(order) != 200 or len(set(order)) != 200:
        raise ValueError("Require the fixed 200 distinct development samples")
    allowed = {f"Data/sdf/chemvector_joint_v1/fresh_eval_12k/shards/fresh_dev-{i:06d}.tar" for i in (0, 1)}
    if set(old["shards"]) != allowed:
        raise ValueError("Unexpected development shards")
    for path, digest in old["shards"].items():
        require_hash(path, digest)
    for key in order:
        if old["image_index"][key]["shard"] not in allowed or "/" in key:
            raise ValueError("Unexpected selected member")
    sources = sorted(Path(__file__).parent.glob("*.py")) + [Path("tests/chemvector/test_id_graph.py")]
    config = {"schema": "id-graph-roundtrip-control-0.1",
              "frozen_utc": datetime.now(timezone.utc).isoformat(),
              "inputs": {str(SELECTION): SELECTION_SHA},
              "sources": {str(p): sha256(p) for p in sources},
              "shards": old["shards"], "sample_order": order,
              "sample_metadata": old["sample_metadata"], "image_index": old["image_index"],
              "rdkit_version": rdBase.rdkitVersion,
              "compiler_inputs": ["chemmol.json chemical attributes and ID connectivity", "render.json observations only"],
              "scorer_only": "source SDF; never used to construct semantic graphs",
              "denominator": 200, "required_all_checks": 200, "metrics": list(METRICS),
              "core_definition": "canonical sanitized graph without stereo; retain isotope/charge/radicals",
              "stereo_definition": "canonical isomeric SMILES after SDF readback or direct compilation",
              "relabel_seeds": [0, 17, 2026],
              "observation_tests": ["remove all observations", "collapse all coordinates to (0, 0)"],
              "failures": "retain all rows; count failures in denominator; no exclusions or retries",
              "scope": "truth-only compiler control, not image recognition or architecture comparison",
              "training_performed": False, "promotion_authorized": False}
    output.mkdir(parents=True)
    write_json(output / "protocol.json", config)
    return config


def signature(mol, stereo=True):
    if mol is None:
        raise ValueError("Invalid SDF molecule")
    copy = Chem.RemoveHs(Chem.Mol(mol))
    if not stereo:
        Chem.RemoveStereochemistry(copy)
    # isomericSmiles remains True even for core scoring, to retain isotopes.
    return Chem.MolToSmiles(copy, canonical=True, isomericSmiles=True)


def relabel(graph, seed):
    """Independently permute IDs and arrays, including all stereo references."""
    changed = deepcopy(graph)
    rng = random.Random(seed)
    atom_ids = [a["id"] for a in changed["atoms"]]
    labels = [100 + 7 * i for i in range(len(atom_ids))]
    rng.shuffle(labels)
    mapping = dict(zip(atom_ids, labels))
    bond_labels = [1000 + 11 * i for i in range(len(changed["bonds"]))]
    rng.shuffle(bond_labels)
    for atom in changed["atoms"]:
        atom["id"] = mapping[atom["id"]]
        atom["stereo"]["neighbor_ids"] = [mapping[i] for i in atom["stereo"]["neighbor_ids"]]
    for bond, label in zip(changed["bonds"], bond_labels):
        bond["id"] = label
        bond["atom_ids"] = [mapping[i] for i in bond["atom_ids"]]
        bond["stereo_atom_ids"] = [mapping[i] for i in bond["stereo_atom_ids"]]
    for position in changed.get("observations", {}).get("atom_positions", []):
        position["atom_id"] = mapping[position["atom_id"]]
    rng.shuffle(changed["atoms"])
    rng.shuffle(changed["bonds"])
    return changed


def check_sample(chemical, render, source_sdf, seeds):
    original = deepcopy((chemical, render))
    graph = from_chemmol(chemical, render)
    before = deepcopy(graph)
    direct = to_molecule(graph)
    sdf = to_sdf(graph)
    readback = Chem.MolFromMolBlock(sdf, sanitize=True, removeHs=True)
    # The compiler and serialization have completed before scorer-only truth is parsed.
    truth = Chem.MolFromMolBlock(source_sdf, sanitize=True, removeHs=True)
    expected = {name: signature(truth, stereo) for name, stereo in (("core", False), ("stereo", True))}
    row = {"expected": expected, "actual": {}, "relabel_checks": []}
    for name, mol in (("direct", direct), ("sdf", readback)):
        row["actual"][name] = {}
        for kind, stereo in (("core", False), ("stereo", True)):
            actual = signature(mol, stereo)
            row["actual"][name][kind] = actual
            row[f"{name}_{kind}_exact"] = actual == expected[kind]
    baseline = signature(direct)
    for seed in seeds:
        changed = relabel(graph, seed)
        row["relabel_checks"].append({"seed": seed,
            "direct": signature(to_molecule(changed)) == baseline,
            "sdf": signature(Chem.MolFromMolBlock(to_sdf(changed))) == baseline})
    row["relabel_invariant"] = all(c["direct"] and c["sdf"] for c in row["relabel_checks"])
    row["observation_independent"] = True
    for remove in (True, False):
        changed = deepcopy(graph)
        if remove:
            changed.pop("observations")
        else:
            for p in changed["observations"]["atom_positions"]:
                p["xy"] = [0., 0.]
        row["observation_independent"] &= (signature(to_molecule(changed)) == baseline and to_sdf(changed) == sdf)
    row["inputs_unchanged"] = graph == before and (chemical, render) == original
    positions = {}
    for p in graph["observations"]["atom_positions"]:
        positions.setdefault(tuple(p["xy"]), []).append(p["atom_id"])
    row["overlap_groups"] = [{"xy": list(xy), "atom_ids": ids} for xy, ids in positions.items() if len(ids) > 1]
    row["atom_count"], row["bond_count"] = len(graph["atoms"]), len(graph["bonds"])
    row["all_checks_pass"] = all(row[m] for m in METRICS)
    return row, graph, sdf


def run(output):
    config = freeze(output)
    for path, digest in config["sources"].items():
        require_hash(path, digest)
    rows, artifacts = [], {}
    (output / "samples").mkdir()
    with ExitStack() as stack:
        archives = {p: stack.enter_context(tarfile.open(p, "r:")) for p in config["shards"]}
        for key in config["sample_order"]:
            row = {"sample_key": key, "molecule_id": config["sample_metadata"][key]["molecule_id"],
                   **{m: False for m in METRICS}, "all_checks_pass": False}
            try:
                archive = archives[config["image_index"][key]["shard"]]
                payload = {}
                for suffix in ("chemmol.json", "render.json", "sdf"):
                    member = archive.getmember(f"{key}.{suffix}")
                    if not member.isfile():
                        raise ValueError("Selected payload is not a regular file")
                    with archive.extractfile(member) as stream:
                        payload[suffix] = stream.read()
                row["input_sha256"] = {s: hashlib.sha256(b).hexdigest() for s, b in payload.items()}
                chemical, render = [json.loads(payload[s]) for s in ("chemmol.json", "render.json")]
                if (chemical["molecule_id"] != row["molecule_id"]
                        or render["variant"] != config["sample_metadata"][key]["variant"]):
                    raise ValueError("Selected sample identity mismatch")
                result, graph, sdf = check_sample(chemical, render, payload["sdf"].decode(), config["relabel_seeds"])
                row.update(result)
                graph_path, sdf_path = output / "samples" / f"{key}.graph.json", output / "samples" / f"{key}.sdf"
                write_json(graph_path, graph)
                with sdf_path.open("x", encoding="utf-8") as stream:
                    stream.write(sdf)
                for path in (graph_path, sdf_path):
                    artifacts[str(path.relative_to(output))] = sha256(path)
            except Exception as error:
                row["all_checks_pass"] = False
                row["error"] = {"type": type(error).__name__, "message": str(error)}
            rows.append(row)
            if len(rows) % 25 == 0:
                print(f"Checked {len(rows)}/200; all-check passes: {sum(r['all_checks_pass'] for r in rows)}", flush=True)
    for group in ("sources", "inputs", "shards"):
        for path, digest in config[group].items():
            require_hash(path, digest)
    with (output / "samples.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    for name in ("protocol.json", "samples.jsonl"):
        artifacts[name] = sha256(output / name)
    report = {"schema": config["schema"], "denominator": 200,
              "molecule_count": len({r["molecule_id"] for r in rows}),
              "counts": {m: sum(r[m] for r in rows) for m in (*METRICS, "all_checks_pass")},
              "error_count": sum("error" in r for r in rows),
              "failed_keys": [r["sample_key"] for r in rows if not r["all_checks_pass"]],
              "overlap_keys": [r["sample_key"] for r in rows if r.get("overlap_groups")],
              "compiler_control_pass": len(rows) == 200 and all(r["all_checks_pass"] for r in rows),
              "scope": config["scope"], "artifacts": artifacts}
    write_json(output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "artifacts"}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.output)
    return 0 if report["compiler_control_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
