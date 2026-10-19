"""Independent float64 analytic check of cached-logit atom-loss derivatives.

No PyTorch/model imports, optimization, or edited predictions. The two output
counts describe a loss conflict at a saved point, not parameter-update effects.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "outputs/audits/curriculum-v02-bond-diagnostic32-corrected-20260919"
AUDIT = REPO / "outputs/audits/curriculum-v02-selector-matching-replay-20260919/report.json"
OUTPUT = AUDIT.parent / "analytic-postcheck.json"


def analytic_derivatives(logits, positive):
    """Existing per-image gamma=2 focal, .5 ranking, and .1 SmoothL1 count."""
    z, positive = np.asarray(logits, dtype=np.float64), np.asarray(positive, dtype=bool)
    if z.ndim != 1 or z.shape != positive.shape or not np.isfinite(z).all() or not len(z):
        raise ValueError("Expected finite nonempty one-dimensional cached logits/labels")
    p = np.exp(-np.logaddexp(0, -z))
    log_p, log_not_p = -np.logaddexp(0, -z), -np.logaddexp(0, z)
    weight = np.clip(max(1, (~positive).sum()) / max(1, positive.sum()), 1, 8)
    focal = np.where(positive, weight * (2 * p * (1-p)**2 * log_p - (1-p)**3),
                     -2 * p**2 * (1-p) * log_not_p + p**3) / len(z)
    count = .1 * np.clip(p.sum() - positive.sum(), -1, 1) * p * (1-p)
    ranking = np.zeros_like(z)
    if positive.any() and (~positive).any() and 1 - z[positive].min() + z[~positive].max() > 0:
        low = positive & (z == z[positive].min())
        high = ~positive & (z == z[~positive].max())
        ranking[low], ranking[high] = -.5 / low.sum(), .5 / high.sum()
    return dict(focal=focal, weighted_ranking=ranking, weighted_count=count,
                atom_only_total=focal + ranking + count)


def run():
    if OUTPUT.exists():
        raise FileExistsError("Never overwrite a saved postcheck")
    payload = AUDIT.read_bytes()
    audit = json.loads(payload)
    stats = Counter()
    fragile, max_error = [], 0.
    threshold = audit["thresholds"]["atom_threshold"]
    for case in audit["cases"]:
        path = SOURCE / (case["sample_key"] + ".json")
        saved = path.read_bytes()
        if sha256(saved).hexdigest() != audit["input_sha256"][str(path)]:
            raise ValueError("Saved prediction changed")
        rows = json.loads(saved)["candidates"]["candidates"]
        positions = {row["candidate"]: i for i, row in enumerate(rows)}
        positives = {d["raw_assigned_candidate"] for d in case["target_details"]
                     if d["raw_assigned_candidate"] is not None}
        result = analytic_derivatives([r["selection_logit"] for r in rows],
                                      [r["candidate"] in positives for r in rows])
        counted_positives = 0
        for row in case["relevant_candidates"]:
            i = positions[row["candidate"]]
            if row["raw_positive"] != (row["candidate"] in positives):
                raise ValueError("Positive-label replay inconsistent")
            for key, value in row["atom_only_logit_derivatives"].items():
                error = abs(result[key][i] - value)
                max_error = max(max_error, error)
                if error > 2e-7:
                    raise ValueError("Analytic derivative disagrees with autograd")
            if not row["raw_positive"]:
                continue
            counted_positives += 1
            stats["raw_positive"] += 1
            down = result["atom_only_total"][i] > 0
            stats["positive_downward_atom_only_pressure"] += int(down)
            stats["retained_positive"] += int(row["selected"])
            stats["rejected_positive"] += int(not row["selected"])
            stats["rejected_positive_upward_atom_only_pressure"] += int(not row["selected"] and not down)
            if down and not (result["focal"][i] < 0 and result["weighted_count"][i] > 0
                             and result["weighted_ranking"][i] == 0):
                raise ValueError("Positive downward pressure is not explained by count dominance")
            if down and threshold <= row["selector_logit"] < 0:
                fragile.append(dict(sample_key=case["sample_key"], canonical_smiles=case["canonical_smiles"],
                    candidate=row["candidate"], selector_logit=row["selector_logit"],
                    margin_above_threshold=row["selector_logit"] - threshold,
                    derivatives=row["atom_only_logit_derivatives"]))
        if counted_positives != len(positives):
            raise ValueError("Relevant rows omit positive candidates")
    if stats["raw_positive"] != audit["summary"]["counts"]["raw_positive"]:
        raise ValueError("Aggregate positive denominator changed")
    result = dict(complete=True, method="independent numpy float64 closed-form derivatives; tolerance 2e-7",
        report_sha256=sha256(payload).hexdigest(), checker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        max_absolute_derivative_error=max_error, counts=dict(stats),
        retained_positive_downward_pressure_between_selector_threshold_and_zero=fragile,
        model_forwards=0, optimizer_steps=0, predictions_modified=False, production_loss_modified=False,
        limitation="Only per-image atom-logit partial loss derivatives, not full model/training-mode gradients; no causal training-regression claim.")
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    result = run()
    print(json.dumps({k: result[k] for k in ("counts", "max_absolute_derivative_error")}, sort_keys=True))


if __name__ == "__main__":
    main()
