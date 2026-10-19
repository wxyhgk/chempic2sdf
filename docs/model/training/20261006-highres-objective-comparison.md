# High-resolution branch objective comparison (2026-10-06)

All variants froze the production checkpoint and the 192×192 field, used the
same 120/1,080 train/holdout split, and trained only the 384×384 sidecar in
memory for 200 virtual steps.  Candidate evaluation used the assignment-safe
contract (top 16 peaks, at least 12 px from baseline-selected points).  No
selector, scorer, checkpoint, or sealed test was touched.

| objective | train candidates/positive | train AUPRC | holdout candidates/positive | holdout AUPRC | train precision max on grid | gate |
|---|---:|---:|---:|---:|---:|---|
| local margin | 59/7 | 0.226 | 605/39 | 0.263 | 0.286 | NO-GO |
| focal heatmap | 56/7 | 0.326 | 588/36 | 0.264 | 0.333 | NO-GO |
| focal + hard negatives | 50/6 | 0.438 | 561/39 | 0.369 | 0.500 | NO-GO |

The hard-negative objective is a real improvement in ranking quality, but it
does not meet the pre-registered train-only precision floor of 0.85.  Its best
holdout grid point is 8/15 (53.3%) at τ=0.60; that threshold is not selected
from holdout, and the corresponding train point is only 2/4 (50%).  Therefore
it is evidence for a better branch objective, not evidence that a scorer or
selector can be trained safely.

An independent locked-assignment selector shadow with the same objective found
39 new holdout targets, but the ordinary fused set head lost 8 old selected
IDs.  Its gate was therefore **NO-GO** even though independent fill did not
lower the aggregate matched count.  This confirms that the field objective
must remain separate from the production selector.

Expanding the fixed-cohort raw-gap slice from 120 to 180 images produced only
10 train-sidecar positives and still had no train threshold meeting the 0.85
floor (train AUPRC 0.424; holdout AUPRC 0.428).  A 240-image request was
rejected because that fixed cohort has fewer than 240 eligible raw-gap images;
the request did not alter data or model state.

The untrained feature ablation identifies the remaining bottleneck: on the
180-image focal-hard run, holdout top-1 precision was 14.7% for raw score,
11.8% for distance, and 16.2% for the fixed score+distance+darkness composite.
Therefore a small hand-designed scorer cannot meet the gate either; the
candidate pool still needs better geometry/prominence or more positive
coverage.

Reports:

- `outputs/audits/chemvector-rescue-scorer-readiness-20261006-r3/report.json`
- `outputs/audits/chemvector-highres-focal-readiness-20261006-r1/report.json`
- `outputs/audits/chemvector-highres-focal-hard-readiness-20261006-r1/report.json`
- `outputs/audits/chemvector-highres-focal-hard-readiness-20261006-r3/report.json`
- `outputs/audits/chemvector-highres-selector-retention-20261006-r6/report.json`

## Decision

Keep the focal+hard-negative objective as the leading shadow candidate, but do
not integrate it or train RescueScorer yet.  The next useful experiment is to
increase train-only coverage of positive rescue cases (stratified by visible
O/N text and implicit carbon) and rerun the same fixed threshold audit.  Only
if the enlarged train slice reaches the precision floor with holdout support
should a scorer trial be authorized.
