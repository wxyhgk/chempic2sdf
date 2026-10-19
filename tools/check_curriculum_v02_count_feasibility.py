"""Saved-gradient algebra only: can one count multiplier meet fixed signs?

No model, optimizer, new predictions or numerical coefficient search. Restrict
to alpha in [0,1], scaling BOTH existing .1 atom and .1 bond count terms. The
sign test is local, per-image and Euclidean; it is not a training impossibility
theorem and does not apply to independent weights or other optimizers.
"""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "outputs/audits/curriculum-v02-selector-loss-terms4-20260919"
REPORT_SHA = "a47b66a9946ec7bca3cce733127ecaa136e976d905f59485268b42b962bc7fc9"
COUNTS = ("atom_count_positive", "atom_count_negative", "bond_count")


def feasible_interval(constraints):
    """Intersect affine half-lines: positive requires >=0, negative <=0."""
    lower, upper = 0., 1.
    lower_witness = upper_witness = None
    contradictions = []
    for row in constraints:
        a, b = row["intercept"], row["slope"]
        if not math.isfinite(a) or not math.isfinite(b) or type(row["positive"]) is not bool:
            raise ValueError("Invalid affine direction constraint")
        sign = 1 if row["positive"] else -1
        a, b = sign * a, sign * b
        if b == 0:
            if a < 0:
                contradictions.append(row)
        elif b > 0 and -a / b > lower:
            lower, lower_witness = -a / b, row
        elif b < 0 and -a / b < upper:
            upper, upper_witness = -a / b, row
    return dict(feasible=not contradictions and lower <= upper, required_lower=lower, required_upper=upper,
                lower_witness=lower_witness, upper_witness=upper_witness, constant_contradictions=contradictions)


def run(output=None):
    output = SOURCE / "count-feasibility.json" if output is None else Path(output)
    if output.exists():
        raise FileExistsError("Never overwrite a saved algebra check")
    payload = (SOURCE / "report.json").read_bytes()
    if sha256(payload).hexdigest() != REPORT_SHA:
        raise ValueError("Saved decomposition report changed")
    report = json.loads(payload)
    if not report["complete"] or report["checked_images"] != 4:
        raise ValueError("Incomplete four-witness diagnostic")
    constraints = []
    for key, expected in report["case_sha256"].items():
        if Path(key).name != key:
            raise ValueError("Unsafe saved sample key")
        saved = (SOURCE / (key + ".json")).read_bytes()
        if sha256(saved).hexdigest() != expected:
            raise ValueError("Saved directional decomposition changed")
        case = json.loads(saved)
        for row in case["rows"]:
            parts = row["retention"]["components"]
            slope = sum(parts[name] for name in COUNTS)
            total = row["retention"]["total"]
            if not math.isclose(sum(parts.values()), total, rel_tol=1e-4, abs_tol=1e-4):
                raise ValueError("Saved parts do not add to original direction")
            constraints.append(dict(sample_key=key, canonical_smiles=case["canonical_smiles"],
                candidate=row["candidate"], positive=row["positive"], intercept=total - slope, slope=slope))
    result = dict(schema_version="chemvector-count-multiplier-local-feasibility-0.1", complete=True,
        report_sha256=REPORT_SHA, checker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        coefficient_family="alpha in [0,1] times BOTH original atom_count and bond_count coefficients",
        criterion="Every probed raw positive has nonnegative retention derivative; every hardest raw negative has nonpositive derivative",
        constraints=constraints, interval=feasible_interval(constraints),
        normalization="Original positive full-gradient norm is fixed per image; rescaled gradient norm cannot change signs",
        model_forwards=0, optimizer_steps=0, coefficients_applied=False, numerical_grid_search=False,
        limitations=["Initial-point per-image first-order sign feasibility, not actual predictions or accuracy.",
            "Not a claim about finite training, warm AdamW, mixed batches, separate count weights or other loss changes.",
            "Hardest negatives already below the selector cutoff also retain a suppression constraint; this is conservative."])
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    result = run()
    print(json.dumps(result["interval"], sort_keys=True))


if __name__ == "__main__":
    main()
