# Graph-assignment veto probe (2026-10-15)

The existing `residual_bond_attachment.py` and
`protected_residual_merge.py` provide the right ownership primitives:
residual points attach to unoccupied predicted bond endpoints under a fixed
radius, and protected baseline rows remain an immutable prefix. The existing
bond-context CPU preflight, however, still trains a pointwise scalar head and
therefore does not test graph assignment itself; its small-slice core gate is
NO-GO.

As a cheap target-independent check, a full-cache CPU probe was run with a
12 px coordinate component, a 6 px protected duplicate veto, one residual per
component, and a fixed frozen `presence_logit >= 0` score gate. No GT distance
or target id was used to form groups; labels were consulted only after
assignment.

| dev style | accepted | strict selected | hard selected | near selected | conservative precision |
|---|---:|---:|---:|---:|---:|
| coordgen | 70 | 3 | 29 | 35 | 0.043 |
| original | 94 | 7 | 33 | 50 | 0.074 |
| rdkit | 87 | 6 | 33 | 43 | 0.069 |
| stress | 142 | 19 | 70 | 51 | 0.134 |

This coordinate-component assignment is clearly insufficient. It is not a
bond-endpoint result: the next legitimate experiment would replay the frozen
model's predicted `bond_endpoints` and call
`attach_residual_candidates(radius_px=12)` with occupied protected endpoints,
then evaluate one-to-one graph ownership. Do not turn this veto into a new
pointwise gate or authorize GPU training.

Artifact:

`outputs/audits/chemvector-style-factorial-graph-assignment-veto-cpu-20261015-r1/report.json`
