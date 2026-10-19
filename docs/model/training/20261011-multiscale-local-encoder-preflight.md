# Multi-scale local encoder CPU preflight (2026-10-07)

## Scope

This was a bounded CPU-only diagnostic after the 32 px/16 px linear patch
probe failed to recover strict candidates. It does not change the production
checkpoint and does not authorize a GPU run.

The new probe keeps three fixed candidate-centred raster windows (24, 48, and
96 source pixels), resizes each to 24 px, applies one small convolutional
branch per scale, and joins the pooled features with a 64-unit MLP. The
training set contains only explicit strict positives and confirmed far
negatives from coordinate-release r2. Unknown members and near-unknown
members are masked from the loss. Sampling is fixed 1:1 positive/negative,
seed `20261011`, 400 AdamW steps, batch size 512, and a single pre-registered
logit gate of `-2.75`; no threshold sweep is allowed.

## Result

Machine-readable evidence is in:

`outputs/audits/chemvector-targeted-selector-multiscale-patch-cpu-preflight-20261007-r1/report.json`

| split | selected | strict precision (conservative) | strict recall | hard negatives selected |
|---|---:|---:|---:|---:|
| train | 1,722 | 0.208 | 1.000 | 1,057 |
| holdout | 978 | 0.269 | 0.920 | 554 |

The precision gates fail by a large margin. This is a NO-GO for GPU training,
deployment, or threshold adjustment. The result indicates that raster-only
local appearance cannot distinguish atom candidates from nearby bond/ink
hard negatives in this release. The coordinate candidate gate remains the
current CPU-ready route; a future visual branch must be a late-fusion or
post-hoc abstention component using frozen geometry, with a new independently
registered preflight.
