# R5 text-occlusion / implicit-endpoint proposal sidecar

## Why this is not the residual-only route

The old residual-only heatmap added arbitrary atom candidates to the full image. It
achieved residual precision `0.6111`, recall `0.9429`, and one-residual-bond coverage
only `0.3727`; it is therefore not a safe base for another GPU run. The new sidecar
must be structurally conditioned and additive:

* the frozen baseline candidate prefix is copied byte-for-byte and retained at 100%;
* a suffix is allowed only where a frozen text/glyph region and line evidence jointly
  support a bond-to-text seam;
* an implicit-carbon suffix is allowed only at a provisional junction with at least
  two incident line supports;
* no arbitrary heatmap residual, atom identity, OCR text token, GT bond, or target
  coordinate enters inference.

Each suffix candidate carries `source`, center, local offset, support score and a
collision flag. The sidecar predicts presence and local `(dx,dy)` from frozen local
features (glyph-mask/bbox geometry, line endpoint distance/angle/ink support,
baseline candidate score and frozen visual crop), with at most about 3k trainable
parameters. It never changes edge/type logits.

If a glyph mask is unavailable, a no-training ceiling may use only raster evidence:
threshold dark pixels, subtract a 2–3 px dilated line-support mask, form connected
components, and retain a component only when a line ray enters its expanded box (or
ends within 8 px). The suffix anchor is the deterministic ray/box projection. This
approximation must be evaluated as a candidate ceiling first; it is not a text label
or an authorization to train.

## CPU ceiling before training

The fixed train/holdout replay must report visible-text and implicit-junction strata
separately. Current evidence shows visible-text raw recall is `.937` overall but only
`.777` on OA13 and `.716` on OA33, while implicit-C raw recall is about `.998`; this
justifies targeting text seams rather than another generic atom head.

Admission requires all of:

1. suffix strict precision `>=.85`, AUPRC `>=.50`, at least 14 new holdout matches,
   hard-collision rate `<=.10`, and every profile reported;
2. exact baseline-prefix retention (`1.0`) and no duplicate ownership ambiguity;
3. baseline+suffix oracle graph union precision `>=.965917`, recall strictly above
   `.850031`, sanitize `>=.951477`, with every profile recall `>=.75` and sanitize
   `>=.90`.

If the union ceiling fails, no scorer training is justified. If it passes, fit only
the frozen-feature sidecar on CPU for at most 500 steps; a 200-step full-holdout
checkpoint without strict graph improvement is an immediate NO-GO. Only then may one
isolated `<=800`-step GPU shadow be considered; it remains outside production.

The existing residual-only, offset-geometry and heatmap text-gap trials did not pass
their admission gates, so this plan does not authorize a retry of those routes.

## First additive ceiling result

A raster/text seam suffix ceiling was replayed without training. It proposed 5,243
suffix edges on the 474-image holdout, with suffix precision `0.7344` and recall
`0.5053`. The union graph reached P/R/sanitize
`0.8305/.9103/.3186`, with dangling rate `0.7814`. This fails precision,
sanitize, and dangling gates; the suffix is over-proposing and is not eligible
for a CPU scorer or GPU shadow.

A tightened seam rule (radius 18 px, cosine gate 0.75) improved holdout graph
P/R to `0.9673/.8680` and suffix P/R to `0.9457/.4576`, but oracle sanitize
fell to `0.5422` and dangling rate remained `0.8778`; every profile missed the
sanitize floor. Higher recall therefore came from structurally unsafe edges,
so this variant is also NO-GO for training.

A small deterministic threshold check confirmed the failure is structural rather
than a loose threshold: the best observed P/R was about `.9705/.8745`, but
sanitize was only `.5633` and dangling was `.894`. Tightening the seam gate did
not recover chemistry validity, so further threshold-only sweeps are not useful.

A later glyph-center ownership reroute reduced the suffix set but dropped graph
recall to `0.7795` (P `0.9809`, sanitize `0.9937`) and left 4,092 unresolved
endpoints. It therefore also fails the recall and ownership gates; no scorer
training is authorized.
