# Frozen-backbone feature-probe smoke (2026-10-13)

A CPU smoke pass was run for the proposed late-fusion probe. The immutable
initial vision backbone produced stride-4 features `[96,192,192]` and stride-8
features `[192,96,96]`; candidate coordinates were bilinearly sampled from
both maps and concatenated with only `raw_score`, `presence_logit`, `x_norm`,
and `y_norm`. `nearest_distance_px`, target ids, layers, and graphs were not
read as model inputs.

The smoke processed 104 images (batch granularity) in 27.34 seconds on CPU,
with 219 train rows and 1,928 diagnostic rows assembled. No optimizer,
backward pass, GPU, checkpoint, or sealed data was used. The run is marked
`complete=false` because the explicit `--max-samples 100` cap stopped the scan;
it is a pipeline/cost check, not an accuracy result.

Artifact:

`outputs/audits/chemvector-style-factorial-frozen-feature-cpu-preflight-20261010-smoke100/report.json`

At this measured rate, a complete 5,770-image CPU scan would take roughly 25
minutes, so a full probe requires a separate bounded decision.
