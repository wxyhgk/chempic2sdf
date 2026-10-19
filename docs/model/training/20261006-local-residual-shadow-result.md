# Local heatmap residual shadow (2026-10-06)

This was a bounded, in-memory shadow on the fixed 1,200-image training cohort.
The production `initial.pt` was verified by SHA-256 before loading.  A new
3x3 depthwise local residual (3,457 parameters) was zero-initialized, so its
step-zero output was exactly the production atom heatmap.  Only this residual
was optimized for 40 virtual steps; no checkpoint was written and no
validation or sealed test was read.

## Result

The 12-image slice contained four actual raw-gap images from each of
`font_large`, `serif`, and `line_thick`:

| gate | result |
|---|---:|
| baseline raw targets lost | 0 |
| raw targets recovered | 0 |
| target-cell local peaks decreased | 0 images |
| decision | **NO-GO** |

The objective decreased during the in-memory update, but did not move any
missing target into the decoded raw proposal set.  A lower local-margin loss
alone is therefore not evidence that the residual improves the deployed
decoder.  Keep `initial.pt`; do not extend this run or promote the residual.

Machine report: `outputs/audits/chemvector-atom-local-residual-shadow-20261006/report.json`.
