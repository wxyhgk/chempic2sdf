# r17 target-free rollback audit (2026-10-18)

This is an isolated CPU audit of the predicted-element r17 compile path. It
does not modify the production checkpoint, read sealed data, train a model, or
authorize GPU/production integration.

## Contract

The baseline is the existing `candidate_context_elements` route: frozen
AtomObject/AtomPair candidates, r17 graph-context aromatic head with the
strict alternate-path cycle set, r17 type residual, and r17 element-context
head. A graph that already sanitizes is copied unchanged. For an invalid graph,
the audit sorts the remaining edges by a target-free confidence

`min(endpoint element confidence) * type confidence * aromatic confidence`

and tries a one-edge removal on a deep copy. Every attempt calls the compiler;
the first successful transaction is committed locally. No successful removal
is found within 64 attempts, the graph is left invalid and marked `abstain`.
At most eight transactions are allowed per image. Ground-truth graph joins
occur only after all decisions, for reporting.

## Results (256 images per split)

| split | baseline sanitize | rollback sanitize | delta | accepted edge removals | abstain images |
|---|---:|---:|---:|---:|---:|
| train | 0.847656 | 0.875000 | +0.027344 | 7 | 32 (0.1250) |
| dev | 0.871094 | 0.949219 | +0.078125 | 20 | 13 (0.0508) |
| fresh holdout | 0.921875 | 0.953125 | +0.031250 | 8 | 12 (0.0469) |

Fresh profile sanitize changes were `oa00 .890625→.921875`,
`oa11 .953125→.968750`, `oa20 .859375→.921875`, and
`oa33 .984375→1.000000`; no profile dropped and every abstain rate was below
0.30. Fresh all-edge precision/recall changed from `.997742/.943170` to
`.998869/.942903` (five edges removed, one true-positive edge lost). Aromatic
edge precision/recall remained `1.000000/.869973`.

The predeclared CPU candidate gate (fresh sanitize gain ≥ .03, dev sanitize ≥
.90, no fresh profile drop > .02, and abstain ≤ .30) passes. This is only
evidence for a later shadow replay; it is not a production-go decision.

## Target-free sentinel

The same audit was rerun after replacing every in-memory target element,
aromatic target, mapping, and type label with empty/zero values. Routing,
compiler outcomes, attempt counts, and transaction journals were byte-for-byte
identical for all 768 rows (`decisions_equal=true`). The sentinel report is
`outputs/audits/chemvector-r17-target-free-rollback-m64-v3-perturb/report.json`.

Normal report: `outputs/audits/chemvector-r17-target-free-rollback-m64-v2/report.json`.

Next step, if desired, is a separately authorized CPU shadow integration with
the rollback journal exposed as an explicit abstention signal. Do not merge or
start GPU training from this audit alone.

The read-only shadow receipt is available at
`outputs/audits/chemvector-r17-rollback-shadow-receipt-m64-v2/report.json`.
It verifies the 768-row target-perturbation decision equality and records
per-profile graph/sanitize deltas without changing model weights.
