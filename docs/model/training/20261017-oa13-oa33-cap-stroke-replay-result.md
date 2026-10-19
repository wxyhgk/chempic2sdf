# oa13 cap/stroke isolation replay

The fail-branch panel was rendered and replayed against the immutable initial
checkpoint.  The panel was quality-clean (480/480 accepted, no rejections),
but none of the controls recovered the model's raw candidate reachability.

| profile | raw reachability | selected reachability |
|---|---:|---:|
| `cov13_base` | 0.93794 | 0.87447 |
| `cov13_cap52` | 0.93982 | 0.85708 |
| `cov13_stroke16` | 0.93559 | 0.86836 |

The aggregate raw reachability is 0.93778 (6,381 target atoms; 397 raw
gaps).  The per-cell 0.98 gate fails for all three cells.  The initial SHA and
model state are unchanged; no selector, OCR, graph module, or checkpoint was
modified.

The follow-up attribution shows that changing cap or stroke does not remove a
single semantic failure class: each cell retains substantial visible O/N and
implicit-carbon gaps, with only small profile-to-profile changes.  This is
consistent with a candidate-generation/representation limitation, not a
bond-rendering-only defect.

This rules out bond length cap and line width as sufficient single-factor
repairs.  Do not admit this panel for training.  The next experiment is a
pixel/geometry diagnostic of visible serif O/N glyphs and their candidate
heatmap/anchor response.  Training remains unauthorized until a future panel
passes renderer quality and per-cell raw reachability gates.

Artifacts:

- Rendered quarantine panel: `outputs/datasets/chemvector-oa13-oa33-cap-stroke-panel-20261017-r1/`
- Replay report: `outputs/audits/chemvector-oa13-oa33-cap-stroke-frozen-replay-20261017-r1/report.json`
- Metadata plan: `outputs/audits/chemvector-oa13-oa33-cap-stroke-panel-plan-20261017-r1/`
- Gap attribution: `outputs/audits/chemvector-oa13-oa33-cap-stroke-gap-attribution-20261017-r1/report.json`
