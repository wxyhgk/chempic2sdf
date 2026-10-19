"""Read-only aromatic safety audit for the chemical-head probe.

The audit replays the ten fixed development images whose outcome changed at
the final chemical-head checkpoint.  It compares initial/400/800/1200
chemical heads, checks saved-prediction parity, and records aromatic margin
crossings.  It never trains, reads sealed data, or changes production files.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import os
import json
from pathlib import Path
import time

import networkx as nx
import torch

from chemvector.training.curriculum.bundle import load_bundle
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.chemical_head_probe import _presence_signature
from chemvector.training.curriculum.evaluation import FrozenOCRPredictor
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.validation_truth import canonical_molblock
from tools.run_pair_scale_trace_diagnostic import (
    compare_prediction,
    pair_rows,
    promotion_summary,
    scene_summary,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get(
    "CHEMVECTOR_AROMATIC_SAFETY_RUN",
    str(ROOT / "outputs/experiments/chemvector-chemical-head-probe-20260929-r2"),
))
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
MANIFEST = ROOT / "outputs/datasets/chemvector-expanded-release-20260915/manifest.json"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_AROMATIC_SAFETY_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-chemical-head-aromatic-safety-20260929-r2"),
))
STEPS = (0, 400, 800, 1200)
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
MANIFEST_SHA = "67ccb1307f7af3ca19f16ee6fd900d1a2e8c3c62dd0b9c0ec28704ddd93aa05e"


def _sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _read_records(path: Path) -> dict[str, dict]:
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    result = {row["key"]: row for row in records}
    if len(result) != len(records):
        raise ValueError(f"Duplicate validation key: {path}")
    return result


def _fixed_cases() -> tuple[dict[str, dict], dict[str, dict]]:
    baseline = _read_records(RUN / "baseline-summary.json")
    final = _read_records(RUN / "validation-001200.json")
    if baseline.keys() != final.keys() or len(baseline) != 944:
        raise ValueError("Chemical-head development cohort changed")
    changed = {
        key: baseline[key]
        for key in baseline
        if any(baseline[key][field] != final[key][field] for field in ("exact", "valid", "fallback"))
    }
    if not changed:
        raise ValueError("Chemical-head replay has no changed development witness")
    return baseline, {key: changed[key] for key in sorted(changed)}


def _load_states() -> dict[int, dict[str, torch.Tensor]]:
    model, _, _ = load_bundle(INITIAL, expected_sha256=INITIAL_SHA)
    initial = {name: value.detach().cpu().clone() for name, value in model.pair.chemical.state_dict().items()}
    states = {0: initial}
    for step in STEPS[1:]:
        payload = torch.load(RUN / f"chemical-{step:06d}.pt", map_location="cpu", weights_only=True)
        if payload.get("initial_sha256") != INITIAL_SHA or payload.get("deployment_authorized") is not False:
            raise ValueError(f"Checkpoint binding changed: {step}")
        states[step] = {name: value.detach().cpu().clone() for name, value in payload["chemical_state"].items()}
    return states


def _edge_changes(before: list[dict], after: list[dict]) -> list[dict]:
    old = {tuple(row["atom_candidates"]): row for row in before if row["selected_at_pair_threshold"]}
    new = {tuple(row["atom_candidates"]): row for row in after if row["selected_at_pair_threshold"]}
    if old.keys() != new.keys():
        raise ValueError("Selected pair set changed during chemical-head replay")
    result = []
    for pair in sorted(old):
        left, right = old[pair], new[pair]
        if left["raw_chemical"] != right["raw_chemical"] or left["final_chemical"] != right["final_chemical"]:
            result.append({
                "atom_candidates": list(pair),
                "row": left["row"],
                "before": {key: left[key] for key in ("raw_chemical", "final_chemical", "aromatic_probability")},
                "after": {key: right[key] for key in ("raw_chemical", "final_chemical", "aromatic_probability")},
            })
    return result


def _cycle_margin_crossings(rows_by_step: dict[int, list[dict]]) -> list[dict]:
    """Return only cycles whose production promotion decision changes."""
    selected_by_step = {
        step: {tuple(row["atom_candidates"]): row for row in rows if row["selected_at_pair_threshold"]}
        for step, rows in rows_by_step.items()
    }
    initial_pairs = set(selected_by_step[0])
    if any(set(rows) != initial_pairs for rows in selected_by_step.values()):
        raise ValueError("Selected pair set changed during cycle replay")
    graph = nx.Graph()
    graph.add_edges_from(initial_pairs)
    cycles = []
    for cycle in nx.chordless_cycles(graph, length_bound=6):
        if len(cycle) not in {5, 6}:
            continue
        nodes = tuple(sorted(cycle))
        edges = tuple(sorted(
            tuple(sorted((cycle[index], cycle[(index + 1) % len(cycle)])))
            for index in range(len(cycle))
        ))
        margins = {}
        for step in STEPS:
            values = [selected_by_step[step][edge]["aromatic_probability"] for edge in edges]
            mean = sum(values) / len(values)
            minimum = min(values)
            margins[str(step)] = {
                "mean": mean,
                "minimum": minimum,
                "accepted": mean >= .40 and minimum >= .05,
            }
        if len({item["accepted"] for item in margins.values()}) > 1:
            cycles.append({"nodes": list(nodes), "edges": [list(edge) for edge in edges], "margins": margins})
    return sorted(cycles, key=lambda item: (len(item["nodes"]), item["nodes"]))


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {output}")
    if file_digest(INITIAL) != INITIAL_SHA or file_digest(MANIFEST) != MANIFEST_SHA:
        raise ValueError("Initial checkpoint or release manifest changed")
    baseline, changed = _fixed_cases()
    states = _load_states()
    saved_by_step = {
        0: baseline,
        **{step: _read_records(RUN / f"validation-{step:06d}.json") for step in STEPS[1:]},
    }
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    loader = RecordLoader(manifest)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("This bounded replay requires CUDA; no CPU fallback")
    torch.set_num_threads(2)
    torch.manual_seed(20260925)
    torch.cuda.manual_seed_all(20260925)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model, config, payload = load_bundle(INITIAL, expected_sha256=INITIAL_SHA)
    model.to(device).eval().requires_grad_(False)
    predictor = FrozenOCRPredictor(payload["provenance"], device)
    results = []
    started = time.monotonic()
    with torch.no_grad():
        for key in changed:
            row = next(item for item in manifest["samples"] if item["source"] + ":" + item["sample_key"] == key)
            if row["split"] != "validation":
                raise ValueError("Witness is not validation-only")
            sample = loader(manifest, row)
            if sample.get("purpose") != "diagnostic":
                raise ValueError("Witness is not diagnostic-only")
            pixels = image_tensor(sample["image"]).unsqueeze(0).to(device)
            arms = {}
            baseline_presence = None
            baseline_role = None
            baseline_ocr = None
            pair_rows_by_step = {}
            for step in STEPS:
                model.pair.chemical.load_state_dict(states[step], strict=True)
                structure = predictor.predict_structure(model, config, pixels)
                trace = {}
                prediction = predictor.from_structure(
                    structure, model.role, model.bbox, pixels, sample["image"], trace=trace
                )
                saved = saved_by_step[step][key]
                if not compare_prediction(prediction, saved)["matches_saved"]:
                    raise ValueError(f"Saved prediction parity failed: {step} {key}")
                presence = _presence_signature(structure["output"], structure["pairs"], .85)
                role = _sha(trace.get("roles", []))
                ocr = _sha(trace.get("ocr", []))
                if baseline_presence is None:
                    baseline_presence, baseline_role, baseline_ocr = presence, role, ocr
                elif (presence, role, ocr) != (baseline_presence, baseline_role, baseline_ocr):
                    raise ValueError(f"Frozen upstream/OCR parity failed: {step} {key}")
                rows = pair_rows(structure)
                pair_rows_by_step[step] = rows
                promotion = promotion_summary(structure)
                promotion.pop("accepted_cycle_details", None)
                arms[str(step)] = {
                    "exact": bool(saved["exact"]),
                    "valid": bool(saved["valid"]),
                    "fallback": bool(saved["fallback"]),
                    "presence_signature": presence,
                    "accepted_cycles": promotion["accepted_cycles"],
                    "candidate_cycles": promotion["candidate_cycles"],
                    "final_aromatic_selected": promotion["final_aromatic_selected"],
                    "scene": scene_summary(trace["scene_before_fallback"]),
                    "sanitize_error": trace.get("sanitize_error"),
                    "edge_changes_from_initial": [],
                }
            for step in STEPS[1:]:
                arms[str(step)]["edge_changes_from_initial"] = _edge_changes(
                    pair_rows_by_step[0], pair_rows_by_step[step]
                )
            results.append({
                "key": key,
                "domain": changed[key]["domain"],
                "arms": arms,
                "cycle_margin_crossings": _cycle_margin_crossings(pair_rows_by_step),
            })
            del pixels

    aggregate = {
        "cases": len(results),
        "new_fallback_cases": sum(
            not case["arms"]["0"]["fallback"] and case["arms"]["1200"]["fallback"] for case in results
        ),
        "removed_fallback_cases": sum(
            case["arms"]["0"]["fallback"] and not case["arms"]["1200"]["fallback"] for case in results
        ),
        "rescued_exact_cases": sum(
            not case["arms"]["0"]["exact"] and case["arms"]["1200"]["exact"] for case in results
        ),
        "presence_parity": all(
            len({case["arms"][str(step)]["presence_signature"] for step in STEPS}) == 1
            for case in results
        ),
        "total_edge_changes_1200": sum(
            len(case["arms"]["1200"]["edge_changes_from_initial"]) for case in results
        ),
        "cases_with_cycle_margin_crossing": sum(bool(case["cycle_margin_crossings"]) for case in results),
        "new_fallback_sanitize_errors": Counter(
            case["arms"]["1200"]["sanitize_error"]
            for case in results
            if not case["arms"]["0"]["fallback"] and case["arms"]["1200"]["fallback"]
        ),
    }
    # Counter is not JSON-native and the report must be deterministic.
    aggregate["new_fallback_sanitize_errors"] = dict(sorted(aggregate["new_fallback_sanitize_errors"].items()))
    report = {
        "schema_version": "chemvector-chemical-head-aromatic-safety-0.1",
        "complete": True,
        "training_performed": False,
        "gpu_used": True,
        "model_forwards": len(results) * len(STEPS),
        "sealed_test_read": False,
        "deployment_authorized": False,
        "production_modified": False,
        "scope": {"fixed_changed_development_images": len(results), "steps": list(STEPS)},
        "source_sha256": {
            "initial.pt": file_digest(INITIAL),
            "manifest.json": file_digest(MANIFEST),
            "probe_report.json": file_digest(RUN / "report.json"),
        },
        "aggregate": aggregate,
        "results": results,
        "elapsed_seconds": time.monotonic() - started,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": "completed", "output": str(OUTPUT), "model_forwards": result["model_forwards"]}, ensure_ascii=False))
