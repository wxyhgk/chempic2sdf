#!/usr/bin/env python3
"""CPU-only calibration shadow for the r17 pair-incidence sidecar.

This audit never changes the production checkpoint or the frozen pair set.  It
uses the already-trained residual sidecar and calibrates *only its routing* on
the sidecar train split.  The trust-region rule is deliberately conservative:
every frozen baseline edge is retained, while a new edge must satisfy a
residual margin and a calibrated score threshold.  Thresholds are selected on
train molecules and measured once on the disjoint holdout molecules.

This is an audit/diagnostic, not a deployment or GPU-training command.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.bond_object import BOND_TYPES
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from tools.train_r17_pair_incidence_known_clean_cpu import (
    HOLDOUT_MOLECULES,
    PAIR_THRESHOLD,
    SHA,
    TRAIN_MOLECULES,
    choose,
    digest,
    extract,
)
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe


OUT = ROOT / os.environ.get(
    "R17_PAIR_CALIB_OUT",
    "outputs/audits/chemvector-r17-pair-incidence-calibration-disjoint-cpu-v1",
)
CACHE = ROOT / os.environ.get("R17_PAIR_CALIB_CACHE", str(OUT.relative_to(ROOT) / "features-cache.pt"))
COHORT = ROOT / os.environ.get(
    "R17_PAIR_CALIB_COHORT",
    os.environ.get("R17_PAIR_COHORT", "outputs/audits/chemvector-r17-selector-slice80-disjoint-shadow-v1/manifest.json"))
STATE = ROOT / os.environ.get(
    "R17_PAIR_CALIB_STATE",
    "outputs/audits/chemvector-r17-pair-incidence-slice80-disjoint-cpu-v1/sidecar-state.pt",
)
BASE_LOGIT = math.log(PAIR_THRESHOLD / (1.0 - PAIR_THRESHOLD))
THRESHOLDS = tuple(round(v, 2) for v in [0.72 + i * 0.01 for i in range(28)])
ALPHAS = (0.25, 0.5, 0.75, 1.0)
CLIPS = (0.5, 1.0, 2.0, 4.0)
MARGINS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_PROFILES = ("oa00", "oa02", "oa11", "oa13", "oa20", "oa22", "oa31", "oa33")

# Feature extraction is the expensive part of this diagnostic (the actual
# threshold sweep is tiny).  Keep the default conservative, but permit a
# bounded CPU worker pool.  Workers load the same frozen checkpoint and never
# write model state; this is still a read-only calibration audit.
EXTRACTION_WORKERS = max(1, int(os.environ.get("R17_PAIR_CALIB_WORKERS", "1")))
EXTRACTION_TIMEOUT_SECONDS = max(30, int(os.environ.get("R17_PAIR_CALIB_EXTRACTION_TIMEOUT", "600")))

_WORKER_MODEL = None
_WORKER_CONFIG = None


def _init_extraction_worker(checkpoint: str, checkpoint_sha: str) -> None:
    """Load one frozen inference bundle in a CPU worker process."""
    global _WORKER_MODEL, _WORKER_CONFIG
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, config, _ = load_bundle_for_probe(Path(checkpoint), checkpoint_sha)
    model.eval().requires_grad_(False)
    _WORKER_MODEL, _WORKER_CONFIG = model, config


def _extract_worker(row: dict) -> dict:
    if _WORKER_MODEL is None:
        raise RuntimeError("calibration extraction worker was not initialized")
    return extract(row, _WORKER_MODEL, _WORKER_CONFIG)


def _extract_rows(rows, model, config):
    """Extract with an explicit wall-clock bound and no partial cache writes."""
    if EXTRACTION_WORKERS <= 1:
        # The sequential path is useful for tiny cohorts and remains the
        # deterministic baseline.  The deadline prevents another unbounded
        # CPU run if a renderer/model regression makes extraction hang.
        import time
        start = time.monotonic()
        out = []
        for row in rows:
            out.append(extract(row, model, config))
            if time.monotonic() - start > EXTRACTION_TIMEOUT_SECONDS:
                raise TimeoutError(
                    f"calibration feature extraction exceeded {EXTRACTION_TIMEOUT_SECONDS}s"
                )
        return out
    import time
    start = time.monotonic()
    # Do not pickle the model into every task.  Each worker loads the frozen
    # bundle once and returns only CPU feature tensors.
    del model, config
    with ProcessPoolExecutor(
        max_workers=EXTRACTION_WORKERS,
        initializer=_init_extraction_worker,
        initargs=(str(INITIAL), SHA),
    ) as pool:
        futures = [pool.submit(_extract_worker, row) for row in rows]
        out = []
        for future in futures:
            remaining = EXTRACTION_TIMEOUT_SECONDS - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError(
                    f"calibration feature extraction exceeded {EXTRACTION_TIMEOUT_SECONDS}s"
                )
            out.append(future.result(timeout=remaining))
        return out


def _finish(c: dict) -> dict:
    return {
        "images": c["images"],
        "predicted_edges": c["predicted"],
        "mapped_edges": c["predicted"],
        "true_edges": c["true"],
        "typed_true_edges": c["typed"],
        "target_edges": c["target"],
        "edge_precision": c["true"] / max(c["predicted"], 1),
        "edge_recall": c["true"] / max(c["target"], 1),
        "typed_edge_precision": c["typed"] / max(c["predicted"], 1),
        "typed_edge_recall": c["typed"] / max(c["target"], 1),
    }


def _empty() -> defaultdict:
    return defaultdict(int)


def evaluate(items, head, params, thresholds, decisions: bool = False):
    """Evaluate baseline-union trust-region selection.

    ``thresholds`` is either a scalar or a profile -> threshold mapping.  The
    return includes optional target-free selection decisions for a digest; the
    digest itself does not use labels.
    """
    alpha, clip, margin = params
    total = _empty()
    by = defaultdict(_empty)
    selected_digest = []
    for item in items:
        p = item["profile"]
        threshold = thresholds[p] if isinstance(thresholds, dict) else thresholds
        # Explicit None means abstain from all additions in this profile.  The
        # baseline is still retained; using .72 here would *not* be fallback.
        threshold_logit = float("inf") if threshold is None else math.log(threshold / (1.0 - threshold))
        base = item["base_logits"]
        # The residual head is frozen during calibration.  Cache its output
        # per image so the thousands of threshold candidates do not rerun the
        # same tiny network (the previous implementation made the sweep the
        # dominant CPU cost after feature extraction).
        residual = item.get("_residual")
        if residual is None:
            residual = head(item["features"])
        adjusted = base + alpha * residual.clamp(-clip, clip)
        base_sel = base >= BASE_LOGIT
        additions = (~base_sel) & (residual >= margin) & (adjusted >= threshold_logit)
        selected = base_sel | additions
        if decisions:
            selected_digest.append(
                {"sample_key": item["key"], "indices": torch.where(selected)[0].tolist()}
            )
        idxs = torch.where(selected)[0].tolist()
        c = by[p]
        c["images"] += 1
        total["images"] += 1
        ntrue = ntyped = 0
        for i in idxs:
            ntrue += int(item["labels"][i] > 0.5)
            if item["labels"][i] > 0.5 and item["type_labels"][i] is not None:
                ntyped += int(BOND_TYPES[int(item["chemical_logits"][i].argmax())] == item["type_labels"][i])
        target = int(item["labels"].sum())
        for bucket, value in (("predicted", len(idxs)), ("true", ntrue),
                              ("typed", ntyped), ("target", target)):
            c[bucket] += value
            total[bucket] += value
    result = _finish(total), {p: _finish(by[p]) for p in sorted(by)}
    if decisions:
        canonical = json.dumps(sorted(selected_digest, key=lambda x: x["sample_key"]), separators=(",", ":"))
        result += (hashlib.sha256(canonical.encode()).hexdigest(),)
    return result


def _baseline_metrics(items, head):
    # A scalar threshold equal to the frozen threshold is exactly the baseline
    # routing (the head is ignored by evaluate's baseline-union rule).
    return evaluate(items, head, (0.0, 1.0, float("inf")), PAIR_THRESHOLD)


def _qualifies(candidate: dict, baseline: dict, floor_drop: float = 0.0) -> bool:
    # Require strict train non-degradation for both precision measures.  This
    # is intentionally stricter than the holdout 1pp gate: the sidecar only
    # gets to add edges when the calibration split gives no precision budget.
    return (
        candidate["edge_precision"] >= baseline["edge_precision"] - floor_drop
        and candidate["typed_edge_precision"] >= baseline["typed_edge_precision"] - floor_drop
    )


def _select_global(train, head, train_base):
    best = None
    tried = 0
    for alpha in ALPHAS:
        for clip in CLIPS:
            for margin in MARGINS:
                for threshold in THRESHOLDS:
                    scored, _ = evaluate(train, head, (alpha, clip, margin), threshold)
                    tried += 1
                    if not _qualifies(scored, train_base):
                        continue
                    key = (scored["typed_edge_recall"], scored["edge_recall"], -threshold, -margin, alpha)
                    if best is None or key > best["key"]:
                        best = {"key": key, "params": (alpha, clip, margin), "threshold": threshold, "metrics": scored}
    return best, tried


def _select_profile(train, head, train_base, profiles, params):
    chosen = {}
    diagnostics = {}
    # Profile gates are selected independently, but only on train rows of the
    # same profile.  This is routing calibration, not a profile-conditioned
    # production model.
    for p in profiles:
        rows = [x for x in train if x["profile"] == p]
        base = train_base[p]
        best = None
        tried = 0
        alpha, clip, margin = params
        for threshold in THRESHOLDS:
            scored, _ = evaluate(rows, head, params, threshold)
            tried += 1
            if not _qualifies(scored, base):
                continue
            key = (scored["typed_edge_recall"], scored["edge_recall"], -threshold)
            if best is None or key > best["key"]:
                best = {"key": key, "params": params, "threshold": threshold, "metrics": scored}
        # If calibration cannot admit any new edge under the strict train
        # floor, explicit baseline fallback is safer than extrapolation.
        if best is None:
            # A fallback must be baseline-only.  Threshold 1.0 makes the
            # additions predicate false while preserving all baseline edges.
            # Use None rather than 1.0 because logit(1.0) is undefined.
            best = {"key": None, "params": params, "threshold": None, "metrics": base, "fallback": True}
        chosen[p] = best
        diagnostics[p] = {"tried": tried, **best}
    # Every profile shares the train-selected residual transform.  Only the
    # scalar score threshold is profile-calibrated; this prevents a reported
    # profile route from silently applying a different transform than the one
    # audited above.
    params = tuple(params)
    thresholds = {p: chosen[p]["threshold"] for p in profiles}
    return params, thresholds, diagnostics


def _deltas(scored, base):
    return {p: {k: scored[p][k] - base[p][k] for k in ("edge_recall", "typed_edge_recall", "edge_precision", "typed_edge_precision")} for p in base}


def run():
    if digest(INITIAL) != SHA:
        raise RuntimeError("production checkpoint changed before calibration")
    if not STATE.exists():
        raise FileNotFoundError(STATE)
    # A sidecar can only be calibrated on the cohort from which it was fitted.
    # It is not enough for the cache keys to match the requested manifest.
    train_report_path = STATE.parent / "report.json"
    train_report = json.loads(train_report_path.read_text())
    training_cohort = (ROOT / train_report["cohort"]).resolve()
    if training_cohort != COHORT.resolve():
        raise RuntimeError(f"sidecar/cohort mismatch: {training_cohort} != {COHORT.resolve()}")
    manifest = json.loads(COHORT.read_text())
    # ``choose`` historically used the trainer's small-cohort defaults.  That
    # is unsafe for a larger source: calibration can silently run on a prefix
    # while the sidecar was fitted on the full cohort.  Require explicit
    # counts whenever the manifest contains more molecules than those
    # defaults, while retaining compatibility with genuinely small cohorts.
    available = defaultdict(set)
    for row in manifest["rows"]:
        available[str(row["split"])].add(str(row["molecule_id"]))
    available_train = len(available.get("train", ()))
    available_holdout = len(available.get("holdout", ()))
    explicit_counts = {
        "train": "R17_PAIR_TRAIN_MOLECULES" in os.environ,
        "holdout": "R17_PAIR_HOLDOUT_MOLECULES" in os.environ,
    }
    if available_train > TRAIN_MOLECULES and not explicit_counts["train"]:
        raise RuntimeError(
            "calibration molecule scope is ambiguous: set R17_PAIR_TRAIN_MOLECULES "
            f"(manifest has {available_train}, default is {TRAIN_MOLECULES})"
        )
    if available_holdout > HOLDOUT_MOLECULES and not explicit_counts["holdout"]:
        raise RuntimeError(
            "calibration molecule scope is ambiguous: set R17_PAIR_HOLDOUT_MOLECULES "
            f"(manifest has {available_holdout}, default is {HOLDOUT_MOLECULES})"
        )
    train_rows, hold_rows = choose(manifest)
    if {r["molecule_id"] for r in train_rows} & {r["molecule_id"] for r in hold_rows}:
        raise RuntimeError("identity overlap")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    provenance = {
        "initial_checkpoint_sha256": SHA,
        "state_sha256": digest(STATE),
        "training_report_sha256": digest(train_report_path),
        "manifest_sha256": digest(COHORT),
        "cohort": str(COHORT.resolve()),
    }
    if CACHE.exists():
        cached = torch.load(CACHE, map_location="cpu")
        train, hold = cached["train"], cached["hold"]
        if cached.get("train_keys") != [r["sample_key"] for r in train_rows] or cached.get("hold_keys") != [r["sample_key"] for r in hold_rows]:
            raise RuntimeError("feature cache cohort mismatch")
        if cached.get("provenance") != provenance:
            raise RuntimeError("feature cache provenance missing or mismatched; no silent reuse")
    else:
        # Load the frozen bundle only for extraction.  With >1 workers the
        # helper loads one isolated copy per worker, avoiding repeated model
        # pickling and giving this diagnostic a predictable CPU bound.
        model, config, _ = load_bundle_for_probe(INITIAL, SHA)
        model.eval().requires_grad_(False)
        train = _extract_rows(train_rows, model, config)
        hold = _extract_rows(hold_rows, model, config)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": "r17-pair-calibration-features-0.2", "provenance": provenance, "train": train, "hold": hold,
                    "train_keys": [r["sample_key"] for r in train_rows], "hold_keys": [r["sample_key"] for r in hold_rows]}, CACHE)
    payload = torch.load(STATE, map_location="cpu")
    if payload.get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("sidecar was not trained from the frozen checkpoint")
    # Importing the class avoids duplicating model architecture while keeping
    # all inference frozen.
    from tools.train_r17_pair_incidence_known_clean_cpu import Residual
    head = Residual(262)
    head.load_state_dict(payload["state_dict"])
    head.eval()

    with torch.inference_mode():
        for item in train + hold:
            item["_residual"] = head(item["features"]).float().cpu()

    train_base, train_base_profiles = _baseline_metrics(train, head)
    hold_base, hold_base_profiles = _baseline_metrics(hold, head)
    global_choice, tried = _select_global(train, head, train_base)
    if global_choice is None:
        global_choice = {"params": (0.0, 1.0, float("inf")), "threshold": PAIR_THRESHOLD, "metrics": train_base, "fallback": True}
    gparams = global_choice["params"]
    gthreshold = global_choice["threshold"]
    gtrain, gtrain_profiles = evaluate(train, head, gparams, gthreshold)
    ghold, ghold_profiles, gdigest = evaluate(hold, head, gparams, gthreshold, decisions=True)
    profiles = tuple(sorted({x["profile"] for x in train} & {x["profile"] for x in hold}))
    pparams, pthresholds, pdiag = _select_profile(train, head, train_base_profiles, profiles, gparams)
    ptrain, ptrain_profiles = evaluate(train, head, pparams, pthresholds)
    phold, phold_profiles, pdigest = evaluate(hold, head, pparams, pthresholds, decisions=True)
    gd = _deltas(ghold_profiles, hold_base_profiles)
    pd = _deltas(phold_profiles, hold_base_profiles)
    gate = {
        "checkpoint_unchanged": digest(INITIAL) == SHA,
        "sealed_test_read": False,
        "gpu_used": False,
        "production_checkpoint_written": False,
        "edge_set_frozen": True,
        "global_each_profile_precision_drop_le_0.01": all(v["edge_precision"] >= -0.01 for v in gd.values()),
        "global_each_profile_typed_precision_drop_le_0.01": all(v["typed_edge_precision"] >= -0.01 for v in gd.values()),
        "profile_each_profile_precision_drop_le_0.01": all(v["edge_precision"] >= -0.01 for v in pd.values()),
        "profile_each_profile_typed_precision_drop_le_0.01": all(v["typed_edge_precision"] >= -0.01 for v in pd.values()),
        "profile_oa13_typed_recall_gain": pd.get("oa13", {}).get("typed_edge_recall", 0.0) > 0.0,
        "profile_oa33_typed_recall_gain": pd.get("oa33", {}).get("typed_edge_recall", 0.0) > 0.0,
        "production_go": False,
    }
    result = {
        "schema_version": "chemvector-r17-pair-incidence-calibration-disjoint-cpu-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "calibration_performed": True,
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "cohort": str(COHORT.relative_to(ROOT)),
        "cohort_sha256": digest(COHORT),
        "cache_sha256": digest(CACHE) if CACHE.exists() else None,
        "state_sha256": digest(STATE),
        "provenance": provenance,
        "cache_sha256": digest(CACHE),
        "profile_routing_scope": "metadata-only diagnostic; profile_id is not a production-observable router",
        "state": str(STATE.relative_to(ROOT)),
        "train_images": len(train), "holdout_images": len(hold),
        "train_molecules": len({r["molecule_id"] for r in train_rows}),
        "holdout_molecules": len({r["molecule_id"] for r in hold_rows}),
        "available_train_molecules": available_train,
        "available_holdout_molecules": available_holdout,
        "explicit_molecule_counts": explicit_counts,
        "base_threshold": PAIR_THRESHOLD,
        "train_baseline": train_base, "holdout_baseline": hold_base,
        "global": {"tried": tried, "choice": global_choice, "train": gtrain, "holdout": ghold,
                    "holdout_profiles": ghold_profiles, "deltas": gd, "decision_digest": gdigest},
        "profile": {"params": pparams, "thresholds": pthresholds, "selection": pdiag,
                    "train": ptrain, "holdout": phold, "holdout_profiles": phold_profiles,
                    "deltas": pd, "decision_digest": pdigest},
        "gate": gate,
        "initial_checkpoint_sha256": SHA,
        "initial_checkpoint_sha256_after": digest(INITIAL),
    }
    # The feature cache is intentionally stored below OUT, so CACHE.parent may
    # have created OUT before the report is written.  Refuse only to overwrite
    # an existing report, while allowing a resumable cache-only diagnostic.
    report_path = OUT / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    OUT.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "global": result["global"], "profile": result["profile"], "gate": gate}, indent=2))


if __name__ == "__main__":
    run()
