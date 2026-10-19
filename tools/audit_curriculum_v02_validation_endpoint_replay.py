"""Frozen-initial endpoint coverage replay on the proposed v0.2 validation panel.

This is an evaluation-only audit.  The proposed v0.2 rows are still quarantine
records, therefore they are loaded with the diagnostic data path and are never
passed to a training collator.  Ground-truth targets are used only after the
frozen model has produced raw and selected proposals for matching.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator, load_sample
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/curriculum-v0.2-expanded-20260919"
PREPARATION = ROOT / "outputs/audits/curriculum-v02-release-preparation-20260919"
ROWS_PATH = PREPARATION / "sample-split-proposal.json"
PREPARATION_REPORT = PREPARATION / "report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
ROWS_SHA = "f033e48b2194be5c41f3a0050509edd4684a285e05b39000f5a5513802d5cb28"
PREPARATION_REPORT_SHA = "39e66a3a4b0ba17651a5834fe3ac65fa8661fef247df98f58195f9e6fe496310"
DEFAULT_OUTPUT = ROOT / "outputs/audits/chemvector-curriculum-v02-validation-endpoint-replay-20261006"


def _reject_sealed(path: Path) -> Path:
    path = path.absolute()
    if "sealed_test" in path.parts or path.is_symlink():
        raise ValueError(f"sealed/symlinked path is forbidden: {path}")
    return path


def _sha(path: Path) -> str:
    path = _reject_sealed(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _bucket(value: str) -> dict:
    return {"target_atoms": 0, "raw_matched": 0, "selected_matched": 0,
            "raw_gap": 0, "selector_drop": 0}


def _add(bucket: dict, *, matched_raw: bool, matched_selected: bool) -> None:
    bucket["target_atoms"] += 1
    bucket["raw_matched"] += int(matched_raw)
    bucket["selected_matched"] += int(matched_selected)
    bucket["raw_gap"] += int(not matched_raw)
    bucket["selector_drop"] += int(matched_raw and not matched_selected)


def run(*, device: str = "auto", limit: int | None = None, output: Path = DEFAULT_OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if _sha(INITIAL) != INITIAL_SHA:
        raise ValueError("Initial checkpoint SHA changed")
    for path, expected in ((ROWS_PATH, ROWS_SHA), (PREPARATION_REPORT, PREPARATION_REPORT_SHA)):
        _reject_sealed(path)
        if _sha(path) != expected:
            raise ValueError(f"Fixed replay input changed: {path}")
    rows = json.loads(ROWS_PATH.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Malformed fixed split proposal")
    rows = sorted((r for r in rows if r.get("proposed_split") == "validation"),
                  key=lambda r: r["sample_key"])
    if len(rows) != 2619:
        raise ValueError(f"Expected 2,619 proposed validation rows, found {len(rows)}")
    if limit is not None:
        if not 1 <= limit <= len(rows):
            raise ValueError("limit must be between 1 and the validation size")
        rows = rows[:limit]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cuda", "cpu"} or device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device is unavailable")

    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model = model.to(device).eval().requires_grad_(False)
    state_before = state_digest(model)
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds),
        purpose="diagnostic",
    )
    threshold = float(payload["provenance"]["atom_threshold"])
    totals = _bucket("all")
    by_profile: dict[str, dict] = defaultdict(lambda: _bucket("profile"))
    by_role_element: dict[str, dict] = defaultdict(lambda: _bucket("role_element"))
    by_incident_bond_type: dict[str, dict] = defaultdict(lambda: _bucket("bond_type"))
    images = 0
    with torch.no_grad():
        for index, row in enumerate(rows, 1):
            key = row["sample_key"]
            sample = load_sample(DATA, key, purpose="diagnostic")
            meta = sample["meta"]
            if meta.get("split") != "quarantine" or meta.get("training_authorized") is not False:
                raise ValueError(f"Validation row is not quarantine-only: {key}")
            meta_path = DATA / "samples" / f"{key}.meta.json"
            if _sha(meta_path) != row.get("meta_sha256"):
                raise ValueError(f"Metadata binding mismatch: {key}")
            batch = collator([sample])
            tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            _, _, _, raw, _, set_output, selected = structure_forward(
                model, tensors["pixel_values"], config, threshold)
            raw_mapping, _ = match_atom_targets(raw, tensors, config)
            selected_mapping, _ = match_atom_targets(selected, tensors, config)
            target_atom_ids = [int(v) for v in batch["atom_ids"][0]]
            raw_targets = {target_atom_ids[int(v)] for v in raw_mapping[0].detach().cpu().tolist()
                           if 0 <= int(v) < len(target_atom_ids)}
            selected_targets = {target_atom_ids[int(v)] for v in selected_mapping[0].detach().cpu().tolist()
                                if 0 <= int(v) < len(target_atom_ids)}
            atoms = {int(a["atom_id"]): a for a in sample["layers"]["atoms"]}
            bonds = sample["layers"]["bonds"]
            incident: dict[int, set[str]] = defaultdict(set)
            for bond in bonds:
                drawing = bond["drawing"]
                kind = str(bond.get("chemical_type", drawing.get("bond_type", "UNKNOWN")))
                for endpoint in (int(drawing["begin_atom"]), int(drawing["end_atom"])):
                    incident[endpoint].add(kind)
            profile = str(sample["meta"].get("profile_id", row.get("profile_id", "unknown")))
            for target_id, atom in atoms.items():
                raw_ok, selected_ok = target_id in raw_targets, target_id in selected_targets
                role = "visible" if bool(atom.get("visible_text")) else "implicit"
                element = str(atom.get("element", "UNKNOWN"))
                for bucket in (totals, by_profile[profile], by_role_element[f"{role}:{element}"]):
                    _add(bucket, matched_raw=raw_ok, matched_selected=selected_ok)
                for kind in incident.get(target_id, {"ISOLATED"}):
                    _add(by_incident_bond_type[kind], matched_raw=raw_ok, matched_selected=selected_ok)
            images += 1
            if index % 200 == 0:
                print(json.dumps({"images": index, "raw_gap": totals["raw_gap"],
                                  "selector_drop": totals["selector_drop"]}), flush=True)
    state_after = state_digest(model)
    if state_before != state_after or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Frozen replay changed model state or produced gradients")
    result = {
        "schema_version": "chemvector-curriculum-v02-validation-endpoint-replay-0.1",
        "complete": limit is None,
        "read_only": True, "training_performed": False, "optimizer_steps": 0,
        "gpu_used": device == "cuda", "device": device,
        "sealed_test_read": False, "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA, "initial_state_before_sha256": state_before,
        "initial_state_after_sha256": state_after,
        "validation_images": images, "validation_panel_size": 2619,
        "source": {"data_root": str(DATA.relative_to(ROOT)),
                   "rows": str(ROWS_PATH.relative_to(ROOT)),
                   "rows_sha256": _sha(ROWS_PATH),
                   "preparation_report_sha256": _sha(PREPARATION_REPORT)},
        "totals": totals,
        "by_profile": {k: dict(v) for k, v in sorted(by_profile.items())},
        "by_role_element": {k: dict(v) for k, v in sorted(by_role_element.items())},
        "by_incident_bond_type": {k: dict(v) for k, v in sorted(by_incident_bond_type.items())},
        "limitations": ["Endpoint coverage and selector retention are not molecule accuracy.",
                        "Quarantine validation rows are read diagnostically and are not training-admitted.",
                        "Ground truth is used only for post-forward matching and stratification."],
    }
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "images": images, "totals": totals}, ensure_ascii=False))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--limit", type=int, default=None, help="CPU/GPU smoke prefix; formal run omits this")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    run(device=args.device, limit=args.limit, output=args.output)


if __name__ == "__main__":
    main()
