# r5 line-first edge/bond-type shadow (2026-10-18)

This is one isolated GPU shadow following the CPU line-first feature gate. It
does not modify `initial.pt`, the production graph, selector, OCR, or bond
heads. The fixed 160/80 identity-aligned r5 slice is used; no sealed test is
read.

## Route

Raster LSD segments are clustered into provisional endpoint-free nodes and
deduplicated into unordered edges. The sidecar consumes only 17 features from
segment geometry, endpoint-cluster geometry/counts, local angle evidence, and
raster ink support. A small MLP has an edge-presence head and a bond-type head.
Labels are created in memory only after the target-independent proposal field
is frozen. The edge threshold is calibrated on train only.

## Result

Receipt: `outputs/audits/chemvector-identity-aligned-line-first-edge-bond-shadow-r5/report.json`

| holdout metric | value | gate |
|---|---:|---:|
| candidate edges | 526 | denominator |
| target bonds | 550 | denominator |
| selected edges | 525 | — |
| edge precision | 0.9524 | >= 0.85 pass |
| edge recall (against target bonds) | 0.9091 | >= 0.50 pass |
| edge AUPRC | 0.9929 | diagnostic |
| conditional bond-type accuracy | 0.9820 (491/500) | >= 0.75 pass |

The sidecar ran 800 steps on `cuda:0` in under one second. The state is
isolated at `.../sidecar-state.pt`; `production_go=false` and
`training_admissible=false` remain explicit. This is a strong route signal,
not a promotion authorization.

An independent target-free graph integration replay then produced 513 mapped
selected edges, 500 true edges, and 550 target bonds on holdout: edge precision
0.9747, edge recall 0.9091, conditional bond-type accuracy 0.9820, and 46/80
images with all bonds recovered. Its receipt is
`outputs/audits/chemvector-identity-aligned-line-first-edge-bond-integration-r5/report.json`.
The graph assembler and production path remain untouched.

## Comparison

The residual-only vertex route produced 270 holdout residual candidates with
precision 0.6111 and required suppressing 76/105 false candidates (72.4%) to
reach 0.85 while retaining its true hits. The endpoint-free line-first route
therefore receives the next experiment budget; do not spend another 800-step
run on scalar residual hard-negative tuning before line-first graph assembly
is evaluated.
