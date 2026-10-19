# r7 local visual endpoint hypothesis result

This was a CPU-only deterministic proposal ceiling on the 95-image r6
`fresh_holdout` (901 target atoms, 826 target bonds). The production
checkpoint was verified unchanged:

`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`.

The probe keeps LSD line segments and endpoint clusters, then uses only image
geometry of visible glyph rectangles. For each rectangle it compares its
centre against a local dark-pixel (ink) centroid. Text labels, atom IDs, target
coordinates, and graph edges are not used to construct aliases or edges; they
are joined only after proposal construction for scoring.

## Result

The best variant is the local visual route (`ink_centroid`, margin 8/12/16
all equivalent for the selected rows):

| metric | value |
|---|---:|
| node precision / recall | 0.9132 / 0.9578 |
| edge precision / recall | 0.8606 / 0.7772 |
| dangling edge rate | 0.0779 |
| mapped true edges | 642 / 826 |

The complete variant sweep is in
`outputs/audits/chemvector-identity-aligned-line-first-local-visual-endpoint-r7/report.json`.

An independent anchor-distance diagnostic found that the ink centroid lowers
the mean visible-anchor error from 5.89 px (box centre) to 5.14 px and is
closer in 98.8% of visible nodes. This local improvement does not restore
segments whose endpoint is absent from the candidate set, and the full graph
proposal remains below the strict gate (`P >= .965917`, `R > .850031`,
`dangling <= .05`).

## Decision

NO-GO for CPU scorer or GPU training. The hypothesis is useful as an image
feature for a future explicit multi-hypothesis endpoint representation, but
replacing a single glyph centre with an ink centroid is insufficient by itself.
Production state and checkpoint remain untouched.
