# r6 geometry-port proposal probe

This is a read-only proposal-capacity probe on all 95 accepted `fresh_holdout`
images in `chemvector-identity-aligned-line-first-targeted-diversity-r6`.
It does not read sealed test data, train a model, use GPU, or modify the
production checkpoint.

The proposal route uses raw LSD endpoint clusters, image-only glyph bounding
boxes (four-side nearest boundary projections), and incident-direction
offsets for implicit clusters. Text-record atom IDs, labels, and graph truth
are ignored during proposal construction; graph truth is joined only for
post-hoc metrics.

Using `box_margin=12 px` and implicit offsets `{-4,+4} px`, the aggregate was:

| metric | result |
|---|---:|
| images | 95 |
| unique matched-node recall | 0.9922 |
| edge precision | 0.8850 |
| edge recall | 0.7264 |
| dangling rate | 0.2196 |

The required proposal gate (`precision >= .965917`, `recall > .850031`,
`dangling <= .05`, and every profile recall `>= .75`) fails. Profile edge
recalls were `.8517 / .8134 / .6281 / .6077`; therefore this route is NO-GO
for sidecar training or GPU shadow. It is useful only as evidence that
geometry offsets improve node reachability while leaving line ownership and
duplicate/dangling edges as the bottleneck.

Machine-readable report:
[`report.json`](../../../outputs/audits/chemvector-identity-aligned-line-first-geometry-ports-r6/report.json)
