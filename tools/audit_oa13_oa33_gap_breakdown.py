"""GPU read-only breakdown of raw atom gaps for oa13 and oa33.

Uses the already frozen 10k replay manifest and the immutable initial bundle.
It records no predictions or images, only aggregate target-atom buckets, so it
cannot become a training or evaluation-data side channel.
"""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator, load_sample
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "outputs/audits/chemvector-structure-expansion-10k-release-plan-20261016-r1/manifest.json"
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
OUTPUT = ROOT / "outputs/audits/chemvector-oa13-oa33-gap-breakdown-20261017/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
BATCH_SIZE = 8


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bucket_counts():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    rows = [row for row in plan["rows"] if row["profile_id"] in {"oa13", "oa33"}]
    if len(rows) != 2520:
        raise ValueError("expected 1260 rows per difficult profile")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    if digest(INITIAL) != INITIAL_SHA or not torch.cuda.is_available():
        raise RuntimeError("initial binding or CUDA unavailable")
    device = torch.device("cuda:0")
    model.to(device).eval().requires_grad_(False)
    before = state_digest(model)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
                                                       bond_queries=config.maximum_bonds), purpose="diagnostic")
    counts = defaultdict(Counter)
    pending = []

    def flush(items):
        samples = [load_sample(SOURCE, row["sample_key"], purpose="diagnostic") for row in items]
        batch = collator(samples)
        tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        with torch.inference_mode():
            _, _, _, raw, _, _, _ = structure_forward(model, tensors["pixel_values"], config,
                                                       payload["provenance"]["atom_threshold"])
        raw_map, _ = match_atom_targets(raw, tensors, config)
        for index, row in enumerate(items):
            sample = samples[index]
            graph = sample["graph"]
            layers = {int(atom["atom_id"]): atom for atom in sample["layers"]["atoms"]}
            degree = Counter()
            for bond in graph["bonds"]:
                degree[int(bond["begin_atom"])] += 1
                degree[int(bond["end_atom"])] += 1
            atom_ids = [int(value) for value in batch["atom_ids"][index]]
            raw_ids = {atom_ids[int(value)] for value in raw_map[index].tolist()
                       if 0 <= int(value) < len(atom_ids)}
            block = counts[row["profile_id"]]
            block["images"] += 1
            block["target_atoms"] += len(graph["atoms"])
            block["raw_matched"] += len(raw_ids & set(layers))
            for atom in graph["atoms"]:
                atom_id = int(atom["atom_id"])
                info = layers[atom_id]
                key = ("visible_text" if info["visible_text"] else "implicit",
                       atom["element"], f"degree_{degree[atom_id]}")
                block["target:" + "|".join(key)] += 1
                if atom_id not in raw_ids:
                    block["gap:" + "|".join(key)] += 1

    for row in rows:
        pending.append(row)
        if len(pending) == BATCH_SIZE:
            flush(pending); pending = []
    if pending:
        flush(pending)
    after = state_digest(model)
    if before != after or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("frozen initial state changed")
    result = {}
    for profile, block in sorted(counts.items()):
        target = block["target_atoms"]
        result[profile] = {
            "images": block["images"],
            "target_atoms": target,
            "raw_matched": block["raw_matched"],
            "raw_gap": target - block["raw_matched"],
            "raw_reachability": block["raw_matched"] / max(target, 1),
            "buckets": {key: value for key, value in sorted(block.items())
                        if key.startswith("target:") or key.startswith("gap:")},
        }
    report = {
        "schema_version": "chemvector-oa13-oa33-gap-breakdown-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "gpu_used": True, "device": str(device), "sealed_test_read": False,
        "initial_sha256": INITIAL_SHA, "initial_sha256_after": digest(INITIAL),
        "plan_sha256": digest(PLAN), "source_summary_sha256": digest(SOURCE / "summary.json"),
        "profiles": result, "batch_size": BATCH_SIZE,
        "state_before": before, "state_after": after,
        "next": "diagnose candidate/visual features; do not train or lower profile gates",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    bucket_counts()
