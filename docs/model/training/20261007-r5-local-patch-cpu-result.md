# r5 local bond-patch residual CPU feasibility

This is an isolated feasibility probe.  It does not modify the production
checkpoint, production graph, OCR, or selector, and it never reads sealed
test data.

## Contract

The proposal is frozen LSD endpoint clusters plus frozen glyph aliases.  The
scorer sees only local raster support along each already-proposed segment,
length/orientation/position, glyph distance, and aggregate multiplicity.  GT
coordinates and bond identity are joined only after proposal construction for
loss and reporting.  No new node or edge can be invented by the scorer.

The model is an 8-input local patch summary (max/mean) with a 24-unit MLP:
745 trainable parameters, 300 CPU steps.  Threshold selection uses train only;
the 599-image fresh holdout is evaluated once.

## Result

Artifact: `outputs/audits/chemvector-identity-aligned-line-first-local-patch-cpu-r5/report.json`.

| metric | fresh holdout |
|---|---:|
| pair precision / reachable recall | 0.95783 / 0.78880 |
| strict frozen-proposal recall ceiling | 0.79591 |
| graph-roundtrip edge precision / recall | 1.00000 / 0.62256 |
| dangling endpoint rate | 0.04217 |
| sanitize rate (single-bond diagnostic) | 1.00000 |
| type accuracy (multiplicity proxy) | 0.00025 |

Per-profile recall is only `0.548–0.651`; the worst dangling profile is
`cov13_cap52_r45` at `0.12443`.  Thus a local residual scorer cannot repair the
identified endpoint-reachability deficit.  It fails the edge recall, profile,
and type gates; no GPU shadow or production integration is authorized.
