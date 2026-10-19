# r17 m128 local edge-raster feature shadow (2026-10-07)

This was a bounded CPU-only sidecar experiment on the completed r17 m128
cache.  The production `initial.pt`, selected AtomObject/AtomPair outputs, and
candidate edge set were frozen.  For each already-selected pair, a target-free
24-dimensional raster summary was extracted from the source PNG: centre-line
ink continuity, normal offsets, centre/side contrast, endpoint ink, and edge
length.  The 526-dimensional frozen pair/context feature was concatenated
with that summary and a 64-unit MLP was trained for 500 CPU steps.  Labels
were joined only after feature extraction.  No edge was added or removed.

Command:

```bash
python tools/shadow_r17_local_edge_feature_cpu.py
```

The sidecar used the identity-disjoint r17 m128 panel (512/256/512 images;
7,204/3,536/7,135 candidate edges).  The sealed test was not read and the
production checkpoint was not written.  A target-perturbation sentinel
flipped all training labels after feature construction; feature digests and
forward logits remained byte-identical.

| split | frozen precision / recall | local-raster precision / recall |
|---|---:|---:|
| dev | .994113 / .927248 | .992625 / .923816 |
| fresh | .989063 / .939730 | .990128 / .937998 |

The local branch passes the coarse `.90/.90` diagnostic floor, but fails the
strict no-decrease gate on both dev and fresh recall.  Candidate-edge identity
is preserved, but there is no useful generalization gain.  Decision:
**NO-GO for GPU, production, or larger training**.  Keep this as a diagnostic
that local raster evidence alone does not improve the frozen edge proposal.

Machine report:
`outputs/audits/chemvector-r17-local-edge-feature-shadow-m128-v2/report.json`.
