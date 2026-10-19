# r17 profile-conditioned bias calibration (CPU)

This is a second quarantine-only probe after the global class calibration. It
fits only four bond-class biases and one temperature independently for each of
the eight t4b rendering profiles. Selection uses t4b train rows only, with a
per-profile 1pp typed-precision floor; evaluation is on the disjoint t4b
holdout. The candidate edge set and production checkpoint remain frozen.

Command:

```bash
python tools/train_r17_profile_bias_calibration_cpu.py
```

## Result

Holdout typed deltas (percentage points) were:

```text
oa00  +1.25       oa02  +1.35       oa11   0.00       oa13  +2.50
oa20   0.00       oa22   0.00       oa31   0.00       oa33   0.00
```

Edge precision and recall were unchanged because calibration cannot create
edges. The all-profile recall-gain gate failed: five profiles had no gain, and
the apparent gains are only one or two discrete decisions per profile. The
result is therefore not evidence that a profile-conditioned bias is a useful
production component; it is compatible with sparse finite-sample type flips.

The experiment is useful diagnostically: even with eight separate low-capacity
calibrators, the aromatic/type errors do not move consistently across profiles.
The next experiment should change the upstream visual representation or add
style-diverse training examples, not add another type-sidecar parameterization.

Authoritative report:

`outputs/audits/chemvector-r17-profile-bias-calibration-cpu-20261021-v1/report.json`
