# Raw proposal capacity ladder (2026-10-06)

This was a read-only counterfactual on the fixed 1,200-image training-only
cohort bound to `initial.pt`.  It reran the existing image-only structure
forward with `maximum_atoms` set to 128, 192, and 256.  No optimizer step,
checkpoint write, selector/threshold change, or sealed-test read occurred.

| raw capacity | raw proposals | raw matched targets | selected proposals | selected matched targets | images at raw capacity |
|---:|---:|---:|---:|---:|---:|
| 128 | 149,127 | 36,072 | 36,184 | 35,799 | 961 |
| 192 | 192,126 | 36,084 | 36,172 | 35,791 | 399 |
| 256 | 206,274 | 36,087 | 36,174 | 35,794 | 117 |

The target population is 36,304 atoms.  Raising capacity from 128 to 256
therefore recovers only 15 additional raw matches (0.04 percentage points of
target recall), while selected recall does not improve and is slightly lower.
The additional candidates are mostly unmatched/competing peaks rather than
missing target endpoints.

## Decision

Do not spend GPU budget on a capacity-only change and do not alter the
production capacity.  The missing endpoint problem is primarily dense peak
placement/offset and candidate discrimination, not the number of returned
slots.  Any next experiment must be a separately shadowed offset/dense-field
or selector-discrimination intervention, with per-ID retention and fixed
valid/exact/SDF gates.

## Binding

- checkpoint: `outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt`
- cohort: `outputs/audits/chemvector-edge-label-reachability-20261005/report.json`
- images: 1,200 training-only rows
- `training_performed`: false
- `sealed_test_read`: false
- `deployment_authorized`: false
