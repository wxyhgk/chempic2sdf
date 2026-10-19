# Structure-expansion targeted split plan (2026-10-06)

This is a metadata-only plan. It does not publish the quarantine source, run a
model, read sealed data, or authorize training. The bound report is
`outputs/audits/chemvector-structure-expansion-targeted-split-plan-20261006-r4/report.json`.

## Source and exclusion

The source is the already-rendered
`outputs/datasets/structure-expansion-batch-12272-20260923` quarantine batch.
It contains 1,534 candidate molecules and 12,184 accepted images in eight
fixed profiles. The planner excludes the 44 molecule identities used by the
previous targeted replay, then collapses remaining stereo/isotope variants by
`connection_identity_without_isotope_stereo` before splitting. It excludes
connection groups represented by the previous 44-ID replay and by either the
admitted release or v0.2 registry. Canonical-SMILES overlap against the two
registries is zero for the selected slice.

## Fixed slice

The plan contains 64 molecules and 128 images, using only `oa11` and `oa13`
(serif/large-style views):

| split | molecules | images | O-aromatic | O-double | O-single | N-aromatic | N-single |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 48 | 96 | 23 | 37 | 26 | 41 | 34 |
| holdout | 16 | 32 | 12 | 11 | 11 | 14 | 11 |

The count is by molecule containing a visible element and incident chemical
bond cell, not by duplicated atom target. N-double and O-triple are absent in
the source chemistry and are not fabricated. N-triple was the prior 44-ID
diagnostic stratum and is excluded from this follow-up slice.

## Admission order

1. Perform exact-byte layer/graph/SDF/PNG identity and release admission; the
   source remains quarantine until that receipt exists.
2. Run frozen-initial forward-only replay on all 128 planned rows, reporting
   raw and selected endpoint recall by style and cell.
3. Require zero train/holdout identity or connection-group overlap, at least
   50/14 strict selector-drop positives in train/holdout, baseline prefix
   retention of 100%, and strict residual precision of at least 0.85.
4. Only if all gates pass, authorize one fixed bounded shadow training run.
   No threshold sweep, automatic retry, or sealed-test access is part of this
   plan.
