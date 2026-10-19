# Local peak-margin trial (2026-10-06)

After the dense peak audit identified cell/peak placement as the bottleneck, a
bounded in-memory trial was run on 12 actual raw-gap images (four each from
`font_large`, `serif`, and `line_thick`).  Only the 129-parameter atom heatmap
readout was updated.  The objective was the existing focal heatmap loss plus
`0.5 * atom_local_peak_margin_loss(margin=0.75, kernel=3)`, which explicitly
raises each target cell above nearby non-target peaks.  Offset, vision,
selector, pair, OCR, and all other parameters were frozen.

## Result

- 100 virtual SGD steps completed;
- raw-gap targets recovered: **2**;
- baseline raw targets lost: **1**;
- selected matched targets: one unchanged, one +1, one unchanged on the
  affected rows;
- gate: **NO-GO** because any baseline raw loss is disqualifying;
- no checkpoint or optimizer state was persisted and `initial.pt` remains the
  only usable model.

The objective can move a dense peak, but the current 129-parameter global
readout moves old peaks as well.  This is not safe as a standalone update.
Do not extend the run or deploy the in-memory state.  A future dense proposal
would need a spatial/contextual head or replay-preserving constraint, not just
more steps or a larger margin weight.

Artifact: [machine-readable trial report](../../../outputs/experiments/chemvector-atom-peak-margin-trial-20261006/report.json).
