# oa13/oa33 coverage panel render result

The metadata-only panel was rendered with the existing font-capable renderer.
No model was loaded, no sealed evaluation content was read, and no training or
checkpoint write occurred.

## Result

| item | value |
|---|---:|
| planned images | 2,560 |
| accepted images | 2,550 |
| rejected images | 10 |
| distinct accepted PNGs | 2,496 |
| renderer profiles | 8 |

All ten rejections are in `cov13_rot0` (DejaVu Serif, 32 px, 3.2 px stroke,
28 px bond cap, 0° rotation):

- 8 `Typography clearance would erase a bond segment`
- 2 `Degenerate auxiliary junction`

The other seven cells have 320/320 accepted images.  The rejected rows are
retained in `rejections.json`; they are not replaced or moved between splits.

## Gate decision

NO-GO for frozen initial replay.  The panel contract requires a complete
profile denominator and per-cell raw reachability gates.  Treating the 10
rejections as if they did not exist would bias the cell comparison.

Artifacts:

- Dataset: `outputs/datasets/chemvector-oa13-oa33-coverage-panel-20261017-r2/`
- Summary: `outputs/datasets/chemvector-oa13-oa33-coverage-panel-20261017-r2/summary.json`
- Rejection inventory: `outputs/datasets/chemvector-oa13-oa33-coverage-panel-20261017-r2/rejections.json`
- Gate report: `outputs/audits/chemvector-oa13-oa33-coverage-panel-quality-20261017-r1/report.json`

Next action is a renderer-only fix or a replacement single-factor control for
`cov13_rot0`, followed by a fresh quality audit.  Do not train the selector,
change OCR, lower the raw-reachability threshold, or alter `initial.pt`.
