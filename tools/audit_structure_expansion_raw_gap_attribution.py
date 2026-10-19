"""Attribute frozen-initial raw atom gaps by target atom semantics.

Diagnostic only: reads the quarantine 10k plan and the frozen initial checkpoint,
does not read sealed evaluation data, train, sweep thresholds, or write a
checkpoint.  The output is intended to separate visible-text failures from
implicit-carbon/geometry failures in oa13 and oa33.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib, json, os
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
OUT = Path(os.environ.get("CHEMVECTOR_RAW_GAP_ATTRIBUTION_OUTPUT", str(ROOT / "outputs/audits/chemvector-structure-expansion-raw-gap-attribution-20261017-r1/report.json")))
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def run() -> dict:
    if OUT.exists(): raise FileExistsError(OUT)
    plan = json.loads(PLAN.read_text())
    rows = [r for r in plan["rows"] if r["profile_id"] in {"oa00", "oa13", "oa33"}]
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    if digest(INITIAL) != INITIAL_SHA or not torch.cuda.is_available(): raise RuntimeError("initial/CUDA binding failed")
    device = torch.device("cuda:0"); model.to(device).eval().requires_grad_(False)
    before = state_digest(model)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds), purpose="diagnostic")
    by = defaultdict(Counter); examples = defaultdict(list); total = Counter()
    for row in rows:
        sample = load_sample(SOURCE, row["sample_key"], purpose="diagnostic")
        batch = collator([sample]); tensors = {k:v.to(device) for k,v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.inference_mode():
            _, _, _, raw, _, _, _ = structure_forward(model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"])
        _, matched = match_atom_targets(raw, tensors, config)
        graph_atoms = {int(a["atom_id"]): a for a in sample["graph"]["atoms"]}
        atom_ids = batch["atom_ids"][0]
        texts = batch["visible_text"][0]
        profile = row["profile_id"]
        target_n = int(tensors["atom_mask"][0].sum())
        total["target_atoms"] += target_n; by[profile]["target_atoms"] += target_n; total["images"] += 1; by[profile]["images"] += 1
        for tid in range(target_n):
            aid = int(atom_ids[tid]); atom = graph_atoms[aid]
            # Explicit labels are supplied by the renderer contract; implicit
            # carbon has an empty visible-text slot.
            visible = bool(texts[tid])
            key = f"{atom['element']}|{'visible' if visible else 'implicit'}|{'aromatic' if atom.get('aromatic') else 'aliphatic'}|h{int(atom.get('implicit_h_count',0))}"
            bucket = by[profile]
            bucket["matched_atoms" if bool(matched[0, tid]) else "raw_gap_atoms"] += 1
            bucket[("matched:" if bool(matched[0, tid]) else "gap:") + key] += 1
            total["matched_atoms" if bool(matched[0, tid]) else "raw_gap_atoms"] += 1
            if not bool(matched[0, tid]) and len(examples[(profile,key)]) < 5:
                examples[(profile,key)].append({"sample_key": row["sample_key"], "atom_id": aid, "text": texts[tid], "element": atom["element"], "aromatic": bool(atom.get("aromatic")), "implicit_h_count": int(atom.get("implicit_h_count",0))})
    after = state_digest(model)
    if before != after or any(p.grad is not None for p in model.parameters()): raise RuntimeError("frozen state changed")
    report = {"schema_version":"chemvector-structure-expansion-raw-gap-attribution-0.1", "complete":True, "read_only":True, "training_performed":False, "sealed_test_read":False, "gpu_used":True, "device":str(device), "initial_sha256":INITIAL_SHA, "initial_sha256_after":digest(INITIAL), "state_before":before, "state_after":after, "rows":len(rows), "totals":dict(total), "by_profile":{p:dict(c) for p,c in sorted(by.items())}, "gap_examples":{f"{p}|{k}":v for (p,k),v in sorted(examples.items())}, "interpretation": {"visible_gap_fraction":{}, "implicit_gap_fraction":{}}}
    for p,c in by.items():
        gaps = c["raw_gap_atoms"]
        vis = sum(v for k,v in c.items() if k.startswith("gap:") and "|visible|" in k)
        imp = gaps - vis
        report["interpretation"]["visible_gap_fraction"][p] = vis / max(gaps,1)
        report["interpretation"]["implicit_gap_fraction"][p] = imp / max(gaps,1)
    OUT.parent.mkdir(parents=True, exist_ok=False); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({"output":str(OUT),"totals":dict(total),"visible_gap_fraction":report["interpretation"]["visible_gap_fraction"]}, indent=2))
    return report


if __name__ == "__main__": run()
