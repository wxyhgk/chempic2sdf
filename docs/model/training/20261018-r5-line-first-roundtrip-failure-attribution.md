# r5 line-first roundtrip failure attribution and targeted data plan

The full strict replay is diagnostic only:

```text
outputs/audits/chemvector-identity-aligned-line-first-graph-chemistry-roundtrip-r5-v2/report.json
```

It reaches edge precision `0.9598`, edge recall `0.7981`, and conditional bond
type accuracy `0.9667`, but fails roundtrip: dangling `5.22%`, candidate-only
sanitize `0.7595`, oracle-element sanitize `0.8038`. No GPU or production state
was changed.

## Failure localization

| profile | edge P/R | dangling | oracle sanitize | interpretation |
|---|---:|---:|---:|---|
| cov13_base48 | .939/.716 | 8.07% | .696 | worst topology/aromatic profile |
| cov13_cap52 | .972/.785 | 5.74% | .835 | longer cap improves reachability |
| cov13_stroke16_48 | .928/.718 | 7.22% | .747 | thin stroke loses line endpoints |
| cov33_base | .973/.857 | 4.28% | .861 | stronger baseline |
| cov33_cap32 | .967/.861 | 1.61% | .823 | best edge recovery |
| cov33_stroke32 | .975/.852 | 4.89% | .861 | strong precision/recovery |

There are no self-loops. The high conditional bond-type score means bond-order
classification is not the first bottleneck. The dominant failures are endpoint
coverage/ownership and ring/aromatic graph sanitization. The cov13 family uses
180-degree rotation and DejaVu Serif; its line-width/cap combinations are much
more fragile than cov33/Nimbus Roman. Oracle-element sanitize remains low, so
the problem is not only O/N recognition: aromatic flags, ring closure, and
topology need explicit handling.

## Targeted data plan (data/preflight only; no training yet)

1. **Cov13 endpoint panel.** Render a balanced panel of the same molecules
   with cov13 base48, cap52, and stroke16_48, stratified by aromatic ring,
   degree-3/4 junction, visible O/N, and bond-order (single/double/triple/
   aromatic). Record endpoint-to-cluster distance, unresolved endpoint rate,
   line-node count, and dangling-edge rate per stratum.
2. **Aromatic closure cases.** Add explicit six-member and fused-ring cases
   with alternating aromatic strokes, ring junctions, and double-line spacing;
   retain the cov33 profiles as controls. The target is to reduce oracle
   aromatic/kekulization failures before changing any scorer.
3. **Thin-stroke stress.** Oversample the 1.6 px stroke profiles and add
   controlled anti-alias/rotation variants. Keep line length and font fixed so
   the endpoint effect is isolated; do not mix this with a new model head.
4. **Ownership hard negatives.** For each cov13 image, retain local candidates
   at the line endpoint, 6--20 px along the stroke, and at ring crossings as
   target-free hard-negative strata. The next CPU gate must report per-profile
   candidate ownership and dangling ceilings before any GPU budget.
5. **Roundtrip gate.** Re-run the full 474-image holdout roundtrip after the
   data panel. Require edge recall >=.75, edge precision >=.90, type accuracy
   >=.90, dangling <=.05, no self-loops, and oracle-element sanitize >=.95;
   keep cov13 profile floors visible. Do not authorize a production graph
   merge or another node GPU shadow until these full-split gates pass.

This plan changes coverage and diagnostics only. It does not authorize sealed
test access, checkpoint changes, or training.
