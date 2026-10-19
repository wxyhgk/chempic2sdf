# Anchor-offset-only shadow (2026-10-06)

## Purpose

After the raw-capacity ladder showed that increasing top-K from 128 to 256
recovered only 15/36,304 targets, we tested a different intervention: update
only `atom.anchor_offset.weight` and `atom.anchor_offset.bias`.  Heatmap,
vision/context, selector, pair, role, bbox, and OCR remained frozen.

The first 12-image style slice was replaced by a deterministic slice containing
actual raw-gap images from the same fixed 1,200 training-only cohort: four
`font_large`, four `serif`, and four `line_thick` rows.  Ten in-memory SGD
virtual steps were run.  No checkpoint or optimizer state was persisted.

## Result

The gap-aware slice contained 13 raw-missing targets.  After 10 virtual steps:

- raw-missing targets recovered: **0**;
- baseline raw targets lost: **0**;
- total raw matched targets: unchanged on all 12 images;
- all gradients were finite and the 258-parameter scope was respected.

This is a failed shadow gate, not permission for a 100-step trial.  The lack of
recovery despite directly optimizing anchor offsets suggests the missing O/N
cases are primarily wrong dense peak/cell selection or upstream feature
evidence, not merely the final sub-cell offset.

## Decision

Do not spend another GPU trial on offset-only training.  Keep `initial.pt` as
the production baseline.  The next useful work is a read-only dense peak/cell
diagnosis for the O/N `font_large`/`serif` gaps (which cell is selected, local
peak margin, and whether a neighboring cell is within 6 px), followed by a new
proposal only if that diagnosis identifies a trainable upstream signal.

Artifacts:

- [shadow report](../../../outputs/audits/chemvector-atom-offset-shadow-20261006-r2/report.json)
- [shadow implementation](../../../tools/shadow_atom_offset_trial.py)

Safety flags: `training_performed=false`, `optimizer_steps=0`,
`sealed_test_read=false`, `deployment_authorized=false`.
