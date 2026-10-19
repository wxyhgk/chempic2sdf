#!/usr/bin/env python3
"""Target-free CPU probe for aromatic transaction safety.

The probe consumes only saved scene/promotion artifacts from the prior CPU
transaction replay.  It intentionally ignores expected identities, target
SMILES, and accuracy fields.  It checks whether the existing transaction and
local compiler guards can safely contain aromatic topology failures.  No
model, GPU, sealed data, training, or production inference is used.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Data.sdf.chemscene_v0_2 import mol_from_chemscene_v0_2
from chemvector.components.atoms.chemistry.aromatic_transactions import scene_digest, transaction_groups

SOURCE = ROOT / "outputs/audits/chemvector-aromatic-transactions-20260922"
OUT = ROOT / "outputs/audits/chemvector-aromatic-transaction-target-free-cpu-20261018/report.json"
REPORT_SHA = "40f6e02cc8e0ac7f54a7ad2643f7e73b058b97205d44d20c238de8b13999a0dd"


def digest(path: Path) -> str:
    h = sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def compile_ok(scene: dict) -> tuple[bool, str | None]:
    try:
        molecule = mol_from_chemscene_v0_2(deepcopy(scene))
        if molecule is None:
            return False, "compiler returned no molecule"
        return True, None
    except (ValueError, RuntimeError) as error:
        return False, str(error)


def reconstruct_raw(guarded: dict, changed_bonds: list[dict]) -> dict:
    raw = deepcopy(guarded)
    by_id = {int(bond["id"]): bond for bond in raw["bond_objects"]}
    for change in changed_bonds:
        bond = by_id[int(change["bond_id"])]
        bond.update(change["before"])
    return raw


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(SOURCE / "report.json") != REPORT_SHA:
        raise RuntimeError("bound transaction replay report changed")
    # This worker is intentionally CPU-only.  Do not initialize torch/CUDA.
    paths = sorted(SOURCE.glob("initial--*.json")) + sorted(SOURCE.glob("control50--*.json")) + sorted(SOURCE.glob("replay50--*.json"))
    if len(paths) != 174:
        raise RuntimeError(f"expected 174 saved transaction cases, found {len(paths)}")
    totals = Counter()
    failures: list[dict] = []
    status = Counter()
    transition = Counter()
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        # Do not access `expected_smiles`, `sdf_exact`, or any target identity.
        guarded = row.get("guarded_scene")
        journal = row.get("journal")
        changed = row.get("changed_bonds", [])
        proposals = row.get("proposals", [])
        if not isinstance(guarded, dict) or not isinstance(journal, dict):
            failures.append({"file": path.name, "reason": "missing guarded scene/journal"})
            continue
        totals["cases"] += 1
        status[str(journal.get("status"))] += 1
        totals["transactions"] += len(journal.get("transactions", []))
        for transaction in journal.get("transactions", []):
            action = str(transaction.get("action"))
            totals[f"transaction_{action}"] += 1
        if journal.get("local_rollback"):
            totals["rollback_cases"] += 1
        if journal.get("use_legacy"):
            totals["use_legacy_cases"] += 1
        # Reconstruct the pre-promotion scene without consulting identity.
        raw = reconstruct_raw(guarded, changed)
        raw_hash = scene_digest(raw)
        if raw_hash != row.get("legacy_scene_sha256"):
            failures.append({"file": path.name, "reason": "raw scene digest mismatch"})
        ok_raw, raw_error = compile_ok(raw)
        ok_guarded, guarded_error = compile_ok(guarded)
        totals["raw_compile_ok"] += int(ok_raw)
        totals["guarded_compile_ok"] += int(ok_guarded)
        if raw_error:
            totals["raw_compile_error"] += 1
        if guarded_error:
            totals["guarded_compile_error"] += 1
        # The replay stores fallback booleans from target-free decoder output;
        # no identity comparison is performed here.
        legacy = row.get("legacy", {}); guarded_summary = row.get("guarded", {})
        if bool(legacy.get("fallback")) and not bool(guarded_summary.get("fallback")):
            transition["fallback_removed"] += 1
        if not bool(legacy.get("fallback")) and bool(guarded_summary.get("fallback")):
            transition["fallback_introduced"] += 1
        # Validate proposal groups and transactional edit boundaries.
        try:
            groups = transaction_groups(raw, proposals)
            totals["valid_proposal_groups"] += len(groups)
        except (ValueError, TypeError, KeyError) as error:
            failures.append({"file": path.name, "reason": f"proposal validation: {error}"})
        raw_atoms = raw.get("atom_objects", []); guarded_atoms = guarded.get("atom_objects", [])
        if raw_atoms != guarded_atoms:
            failures.append({"file": path.name, "reason": "atom objects changed"})
        raw_bonds = {int(bond["id"]): bond for bond in raw.get("bond_objects", [])}
        guarded_bonds = {int(bond["id"]): bond for bond in guarded.get("bond_objects", [])}
        changed_ids = {int(item["bond_id"]) for item in changed}
        if set(raw_bonds) != set(guarded_bonds):
            failures.append({"file": path.name, "reason": "bond ids changed"})
        for bond_id in raw_bonds.keys() & guarded_bonds.keys():
            delta = {key for key in raw_bonds[bond_id].keys() | guarded_bonds[bond_id].keys() if raw_bonds[bond_id].get(key) != guarded_bonds[bond_id].get(key)}
            if delta and bond_id not in changed_ids:
                failures.append({"file": path.name, "reason": "unjournaled bond edit", "bond_id": bond_id, "delta": sorted(delta)})
            if delta - {"chemical_type", "display_type"}:
                failures.append({"file": path.name, "reason": "non-chemical transaction edit", "bond_id": bond_id, "delta": sorted(delta)})
        if scene_digest(guarded) != row.get("guarded_scene_sha256"):
            failures.append({"file": path.name, "reason": "guarded scene digest mismatch"})
    report = {
        "schema_version": "chemvector-aromatic-transaction-target-free-cpu-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "production_modified": False,
        "source_report_sha256": REPORT_SHA,
        "scope": {"saved_cases": 174, "arms": ["initial", "control50", "replay50"], "identity_fields_read": False},
        "counts": dict(totals),
        "statuses": dict(status),
        "fallback_transition": dict(transition),
        "invariant_failures": failures,
        "interpretation": {
            "transaction_safety": "target-free compiler/scene safety can be checked independently of molecular identity",
            "identity_limit": "a compile-safe promotion is not evidence that the image identity is correct",
            "training_signal": "the transaction layer cannot supply a visual training signal; aromatic probability/edge semantics remain upstream",
        },
        "gate": {
            "all_saved_cases_checked": totals["cases"] == 174,
            "raw_digest_reconstruction_pass": not any(item["reason"] == "raw scene digest mismatch" for item in failures),
            "transaction_invariants_pass": not any(item["reason"] in {"atom objects changed", "bond ids changed", "non-chemical transaction edit", "unjournaled bond edit"} for item in failures),
            "target_free_cpu_only": True,
            "training_admissible": False,
            "production_go": False,
            "worth_training_transaction_layer": False,
            "next": "if training is revisited, target upstream aromatic/bond semantic logits with fixed ring safety objective; retain transactions only as opt-in safety diagnostics",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "counts": report["counts"], "fallback_transition": report["fallback_transition"], "failures": len(failures), "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
