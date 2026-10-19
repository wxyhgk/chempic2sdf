# Graph-consistency sidecar CPU preflight (2026-10-16)

We tested the materially different frozen-baseline graph sidecar proposed in
`20261011-graph-consistency-sidecar-proposal.md`: two 32-dimensional message
passing layers over candidate proximity, with frozen predicted-bond endpoint
context. The initial model, OCR, selector, and graph outputs were frozen; only
the small sidecar was optimized for 300 CPU steps. No checkpoint was written.

| style | conservative precision | strict recall | hard-negative survival |
|---|---:|---:|---:|
| coordgen | 0.0054 | 0.571 | 0.364 |
| original | 0.0041 | 0.750 | 0.363 |
| rdkit | 0.0062 | 0.750 | 0.365 |
| stress | 0.0191 | 0.423 | 0.429 |

Protected retention was 1.0, but every core style failed the precision and
hard-negative gates by a large margin. The graph message passing amplified
hard negatives rather than resolving them. This is a terminal CPU NO-GO for
this architecture; do not tune the gate, add steps, or authorize a GPU run.

Artifact:

`outputs/audits/chemvector-style-factorial-graph-consistency-cpu-preflight-20261007-r1/report.json`

## Current route

Freeze all selector/rescue/graph-consistency training routes. The next actual
training candidate must come from a new identity-audited data release, using
the existing 12,184-image quarantine structure-expansion batch as the first
source instead of blindly rendering another 100k-derived pool. Its release,
prior-use identity ledger, frozen replay, and strict-only CPU gate are still
required before GPU authorization.
