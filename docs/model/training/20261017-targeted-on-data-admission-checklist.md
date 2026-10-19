# Targeted O/N expansion: minimum admission checklist

The v0.2 quarantine catalog contains 3,484 distinct hand-authored/RDKit-
sanitized structures: 1,191 with O, 933 with N, and 431 with both.  Its six
styles produce 3,447 visible O and 1,841 visible N labels per style.  This is
enough for a diagnostic panel, but its existing large-serif control is only
`font_large` (DejaVu Sans, 32 px) and `serif` (DejaVu Serif, 24 px); it does not
cover the failing oa13/oa33 tuples.

## Required O/N strata

Select identities (before rendering) so each train/holdout split contains:

* terminal degree-1 and internal degree-2 O/N labels, plus branched degree≥3;
* adjacent single, double, triple, and aromatic bonds;
* N–N, N–O, O–P, O–S heteroatom bonds;
* charged `[NH3+]`, `[N+]`, `[O-]`, isotope N/O/C labels, and aromatic N/O;
* ring/fused and branch contexts;
* visible C/halogen/S/P/Si and implicit-C controls to detect regressions.

The current catalog counts are useful for planning (terminal labels 1,162;
internal 868; branched 129; single/double/triple/aromatic 2,306/814/377/565;
heteroatom-heteroatom 132; charge labels 38/56; isotopes 4; ring contexts
1,014), but counts are not an admission decision.

## Renderer and split gates

Render every selected identity under the exact difficult profile tuples and
their single-factor controls.  Measure visible labels from `layers.json`, not
from SMILES or template names.  Require complete profile denominators, zero
quality rejects, byte/RGB uniqueness, SVG/layers/graph/SDF identity
round-trip, and no cross-identity image duplicates.  Split by molecule,
connection identity, and scaffold; no identity may cross train/holdout.

## Authenticity boundary

v0.2 is deterministic synthetic template data.  RDKit sanitization proves
representation/valence consistency, not experimental stability, real-world
publication style, or OCR accuracy.  Keep it quarantine/diagnostic-only.  A
training release requires an independent, licensed real-structure/real-scan
comparison set, identity-overlap review, and profile-level frozen-initial raw
reachability ≥0.98 on an identity-disjoint holdout.  Aggregate clean-synthetic
coverage must never be used as evidence of deployment accuracy.
