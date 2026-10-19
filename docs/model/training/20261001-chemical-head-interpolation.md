# 2026-10-01 chemical-head trust-region replay

The initial chemical head and the r3 1200-step chemical head were linearly
interpolated at α = 0, .25, .5, .75, and 1.0. This was a read-only replay on
the fixed 944-image development cohort (4,720 forwards; no training or
deployment change).

Report: `outputs/audits/chemvector-chemical-head-interpolation-20261001-full/report.json`.

| α | exact | valid | fallback | new fallback vs initial |
|---:|---:|---:|---:|---:|
| 0.00 | 233 | 690 | 51 | 0 |
| 0.25 | 233 | 690 | 51 | 1 |
| 0.50 | 233 | 690 | 49 | 1 |
| 0.75 | 235 | 690 | 49 | 1 |
| 1.00 | 235 | 690 | 51 | 3 |

The only new fallback at α ≤ .75 is consistently `001401_coordgen`; it is
introduced by even a small movement away from initial. α=.75 retains r3's two
exact rescues and removes two fallbacks, but still fails the hard no-new-
fallback gate. Therefore interpolation is not a deployable candidate and no
fine-grid search is justified.

Decision: keep initial as the production baseline. The next training change
must explicitly protect non-regression of chemically valid aromatic/non-
aromatic boundary edges; a generic smaller learning rate or post-hoc weight
blend is insufficient.
