# Selector residual sidecar trial (2026-10-06)

After raw-dense field failure, a separate bounded selector trial was run from
the immutable `initial.pt` checkpoint.  The cohort was split by group, molecule,
and canonical SMILES into 480 train and 720 holdout images with zero overlap.

The scorer saw only frozen raw candidates that were not already in the
selected prefix.  Existing selected candidates were never rescored or
removed.  Confirmed non-edge rows were used as negatives; ambiguous rows were
excluded from loss and reported as unknown.  Only a 4,321-parameter sidecar was
optimized for 200 in-memory steps.

## Result

| split | candidate rows | known positives | unknown | train-calibrated result |
|---|---:|---:|---:|---|
| train | 44,570 | 104 | 291 | no threshold reached conservative precision 0.85 |
| holdout | 68,050 | 163 | 455 | no admissible threshold; no deployment candidate |

At the highest grid threshold (0.95), the train sidecar still selected 57
rows: 27 positives, 16 confirmed negatives, and 14 unknowns.  The
conservative precision was only 0.474, so the pre-registered gate failed.

Decision: **NO-GO**.  Do not tune the threshold, add steps, enlarge the MLP,
or connect it to the set-head.  The production checkpoint remains unchanged;
sealed test identities were not read.

Machine report:
`outputs/audits/chemvector-selector-residual-sidecar-20261006-r1/report.json`.
