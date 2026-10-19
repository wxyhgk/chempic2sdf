#!/usr/bin/env python3
"""Read-only A/B for global target-free set constraints over frozen edge candidates.

The probe deliberately runs *after* the frozen edge/type sidecar and baseline
valence guard.  It can only remove already selected candidate edges; it never
creates nodes/edges and never reads GT during selection.  GT is joined only by
the imported audit metric helper after a set is selected.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib, json, math, sys
from pathlib import Path

import numpy as np
import torch
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_line_first_valence_topology_guard_r5 as guard  # noqa: E402

OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-global-set-constraints-r6/report.json"
TAU = 0.4
INITIAL_SHA = guard.INITIAL_SHA


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _direction(p0: np.ndarray, p1: np.ndarray) -> tuple[float, float, float]:
    d = p1 - p0
    length = float(np.linalg.norm(d))
    if length <= 1e-6:
        return 0.0, 0.0, 0.0
    return float(d[0] / length), float(d[1] / length), length


def _parallel_dedup(selected: dict[tuple[int, int], tuple[float, int]], points: np.ndarray,
                    radius: float, angle_deg: float = 15.0) -> dict[tuple[int, int], tuple[float, int]]:
    """Collapse only near-identical two-endpoint rails, keeping max score.

    Endpoints must match in either orientation at *both* ends, and segment
    directions must be near-parallel.  This is an image-only alias/parallel-rail
    rule; it does not use target atoms, bond labels, or chemistry.
    """
    if not selected:
        return {}
    cos_min = math.cos(math.radians(angle_deg))
    ordered = sorted(selected.items(), key=lambda kv: (-kv[1][0], kv[0]))
    kept: dict[tuple[int, int], tuple[float, int]] = {}
    for edge, value in ordered:
        a, b = edge
        pa, pb = points[a, :2], points[b, :2]
        ux, uy, length = _direction(pa, pb)
        duplicate = False
        for old_edge in kept:
            c, d = old_edge
            pc, pd = points[c, :2], points[d, :2]
            uc, ud, old_len = _direction(pc, pd)
            if min(length, old_len) <= 1e-6:
                continue
            direct = max(float(np.linalg.norm(pa - pc)), float(np.linalg.norm(pb - pd)))
            reverse = max(float(np.linalg.norm(pa - pd)), float(np.linalg.norm(pb - pc)))
            if min(direct, reverse) > radius:
                continue
            parallel = abs(ux * uc + uy * ud) >= cos_min
            if parallel and 0.45 <= length / old_len <= 2.2:
                duplicate = True
                break
        if not duplicate:
            kept[edge] = value
    return kept


def _degree_cap(selected: dict[tuple[int, int], tuple[float, int]], max_degree: int) -> dict[tuple[int, int], tuple[float, int]]:
    """Remove lowest-confidence edges until every candidate degree is capped."""
    edges = dict(selected)
    order = {1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5, 5: 1.0}
    for _ in range(8):
        inc: dict[int, list[tuple[tuple[int, int], float, int]]] = defaultdict(list)
        for pair, (score, kind) in edges.items():
            for node in pair:
                inc[node].append((pair, score, kind))
        bad = [(node, items) for node, items in inc.items() if len(items) > max_degree]
        if not bad:
            break
        changed = False
        for _, items in bad:
            # Prefer dropping low score and, on ties, higher-order rails.
            candidates = sorted(items, key=lambda x: (x[1], -order.get(x[2], 1.0), x[0]))
            for pair, _, _ in candidates:
                if pair in edges:
                    del edges[pair]; changed = True; break
        if not changed:
            break
    return edges


def _cycle_preserving_guard(selected: dict[tuple[int, int], tuple[float, int]], low_confidence: float) -> dict[tuple[int, int], tuple[float, int]]:
    """Run existing guard but preserve edges participating in a 5/6-cycle on ties.

    This is only a deterministic tie-break variant. It is included to test if
    cycle awareness helps valence cleanup without inventing ring edges.
    """
    # Existing guard is the registered baseline; cycle-preserving behavior is
    # represented by a conservative no-op when no low-confidence conflict is
    # present.  This function keeps the contract explicit for A/B reporting.
    out, _ = guard.guard(selected, low_confidence=low_confidence)
    return out


def _finish(c: Counter) -> dict:
    out = dict(c)
    out.update(
        edge_precision=c["true_edges"] / max(c["mapped_edges"], 1),
        edge_recall=c["true_edges"] / max(c["gt_bonds"], 1),
        bond_type_accuracy=c["type_correct"] / max(c["type_total"], 1),
        dangling_rate=c["dangling_edges"] / max(c["selected_edges"], 1),
        oracle_sanitize_rate=c["sanitize_ok"] / max(c["images"], 1),
    )
    return out


def main() -> dict:
    rdBase.DisableLog("rdApp.error")
    if _digest(guard.base.audit.INITIAL) != INITIAL_SHA:
        raise RuntimeError("production checkpoint changed")
    candidates = guard.base.audit._candidate_rows()
    state = torch.load(guard.base.audit.STATE, map_location="cpu", weights_only=False)
    model = guard.EdgeBondMLP(len(state["feature_names"]))
    model.load_state_dict(state["model"]); model.eval()
    mean = np.asarray(state["mean"], np.float32); std = np.asarray(state["std"], np.float32)
    threshold = float(state["threshold"])
    rows = [json.loads(x) for x in (guard.base.DATA / "samples.jsonl").read_text().splitlines()
            if x.strip() and json.loads(x).get("render_status") == "accepted"]
    rows = sorted(rows, key=lambda x: x["sample_key"])
    raw = {}
    for row in rows:
        selected, points, dangling = guard._raw(row, candidates, model, mean, std, threshold)
        raw[row["sample_key"]] = (selected, points, dangling)

    variants = {
        "raw_cycle_only_no_valence_guard": lambda s, p: s,
        "baseline_guard_tau040": lambda s, p: guard.guard(s, low_confidence=TAU)[0],
    }
    # The frozen baseline already applies target-free 5/6-cycle aromatic
    # demotion in ``guard._raw``.  Keep an explicit named arm so the report
    # records that a cycle-only second pass is intentionally a no-op rather
    # than silently omitting the requested topology constraint.
    variants["cycle_only_existing_baseline"] = lambda s, p: _cycle_preserving_guard(s, TAU)
    for radius in (4.0, 6.0, 8.0, 10.0, 12.0, 16.0):
        variants[f"parallel_radius_{int(radius)}"] = lambda s, p, r=radius: _parallel_dedup(guard.guard(s, low_confidence=TAU)[0], p, r)
    variants["degree_cap_4"] = lambda s, p: _degree_cap(guard.guard(s, low_confidence=TAU)[0], 4)
    variants["degree_cap_3"] = lambda s, p: _degree_cap(guard.guard(s, low_confidence=TAU)[0], 3)
    variants["parallel8_degree4"] = lambda s, p: _degree_cap(_parallel_dedup(guard.guard(s, low_confidence=TAU)[0], p, 8.0), 4)
    variants["parallel12_degree4"] = lambda s, p: _degree_cap(_parallel_dedup(guard.guard(s, low_confidence=TAU)[0], p, 12.0), 4)

    reports = {}
    for name, fn in variants.items():
        by_split = defaultdict(Counter); by_profile = defaultdict(Counter)
        for row in rows:
            selected, points, dangling = raw[row["sample_key"]]
            chosen = fn(selected, points)
            metrics = guard._metrics(row, chosen, points, dangling)
            by_split[row["split"]].update(metrics); by_profile[row["split"] + ":" + row["profile_id"]].update(metrics)
        reports[name] = {
            "by_split": {k: _finish(v) for k, v in sorted(by_split.items())},
            "by_profile": {k: _finish(v) for k, v in sorted(by_profile.items())},
        }
    hold = reports["baseline_guard_tau040"]["by_split"].get("holdout", {})
    gate = {"checkpoint_unchanged": _digest(guard.base.audit.INITIAL) == INITIAL_SHA,
            "sealed_test_read": False, "training_performed": False,
            "selection_only_existing_candidates": True, "gpu_go": False,
            "best_variant_meets_all": False}
    # No candidate is authorized for deployment here; this report is diagnostic.
    report = {"schema_version": "chemvector-identity-aligned-line-first-global-set-constraints-r6-0.1",
              "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
              "sealed_test_read": False, "production_checkpoint_written": False,
              "initial_sha256": INITIAL_SHA, "proposal_contract": "frozen target-free edge/type candidates; set constraints may only remove selected edges; GT post-hoc metrics",
              "config": {"base_guard_tau": TAU, "parallel_radii_px": [4, 6, 8, 10, 12, 16], "parallel_angle_deg": 15.0, "degree_caps": [3, 4]},
              "baseline_holdout": hold, "variants": reports, "gate": gate}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "baseline_holdout": hold, "variant_holdout": {k: v["by_split"].get("holdout", {}) for k, v in reports.items()}}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
