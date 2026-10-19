# Dense peak gap diagnosis (2026-10-06)

This is a read-only replay of the same 1,200 training-only images and
`initial.pt`.  For every target that was unmatched by the production top-128
raw proposals, it searched **all** local maxima of the 192x192 atom heatmap
and measured the nearest predicted anchor.  No optimizer step, checkpoint
write, threshold change, or sealed-test read occurred.

## Result

There are 232 raw-gap targets.  Only 26 (11.2%) have any dense local peak
within the production 6 px matching radius.  The nearest peak is already in
the top-128 for 210/232 targets, so increasing top-K cannot recover most gaps.

| role | targets | nearest peak within 6 px | median nearest distance |
|---|---:|---:|---:|
| visible text | 193 | 23 (11.9%) | 6.25 px |
| implicit | 39 | 3 (7.7%) | 7.05 px |

The dominant elements are O (112 gaps, 10 within 6 px) and N (64 gaps, 13
within 6 px).  `font_large` contributes 105 gaps but only 21 have a reachable
dense peak; `serif` contributes 79 gaps but only 4 are reachable.  The
nearest-peak distance, rather than candidate capacity, is the limiting factor.

## Decision

The evidence now rules out three cheap interventions:

1. raising raw capacity/top-K (128→256 recovered only 15 targets);
2. heatmap-only readout training (historical 129-parameter trial, 0/10 gaps);
3. offset-only training (new 258-parameter, gap-aware 10-step shadow, 0 gaps,
   with no baseline loss).

The next model work must alter the dense field's **cell/peak evidence** (or
the upstream feature representation), not just final offset, threshold, or
capacity.  A new trainable route should first prove that a different dense
cell objective can move the nearest peak into 6 px without disturbing existing
peaks; otherwise keep `initial.pt`.

Artifact: [machine-readable report](../../../outputs/audits/chemvector-dense-peak-gap-20261006/report.json).
