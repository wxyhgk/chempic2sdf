# r7 segment-port / endpoint-to-glyph assignment screen

## Scope

This is a 50-image, fresh-holdout CPU proposal screen on the accepted r6
targeted panel.  It does not load a checkpoint, alter the selector, train a
sidecar, read sealed identities, or write production state.  LSD segment
endpoints are clustered; glyph rectangles are consumed only as image geometry
(their labels and atom IDs are ignored while proposing).  Ground-truth graph
identity is joined only for final metrics.

The implementation is
`tools/probe_line_first_segment_port_bipartite_r7.py`; its machine-readable
report is
`outputs/audits/chemvector-identity-aligned-line-first-segment-port-bipartite-r7/report.json`.

## A/B results

| reducer | edge precision | edge recall | dangling | selected/mapped/true |
|---|---:|---:|---:|---:|
| many-to-one nearest glyph | 0.8924 | 0.7589 | 0.1496 | 448 / 381 / 340 |
| greedy one-to-one | 0.8839 | 0.6964 | 0.3132 | 514 / 353 / 312 |
| directional glyph gate | 0.9288 | 0.6406 | 0.3988 | 514 / 309 / 287 |

The directional gate improves precision relative to many-to-one, but drops
recall and increases unresolved/dangling edges.  Greedy one-to-one matching is
worse on every graph measure because a visible glyph legitimately receives
multiple bond ports at junctions; a one-to-one constraint is therefore the
wrong topology.  No reducer is close to the fixed proposal gate
(`P >= .965917`, `R > .850031`, `dangling <= .05`).

## Decision

`NO-GO`: do not train an endpoint scorer or launch GPU from this representation.
The experiment changes the assignment representation but cannot recover the
missing raw segment endpoint identities.  Further radius/offset sweeps would
be a parameter search over the same failed ceiling.  Any subsequent proposal
route must generate additional endpoint evidence (for example a learned
local raster endpoint detector) and prove a target-free ceiling before fitting
an edge/type/objectness head.
