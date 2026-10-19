# r17 fresh-source shadow pair→graph result

This is a frozen CPU shadow only.  It does not alter the production
checkpoint (`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`),
does not read sealed test data, and performs no GPU training.

## Cohort and caveat

The metadata-only cohort is
`outputs/audits/chemvector-r17-oa13-oa33-fresh-holdout-all-history-v1/`.
It contains 64 molecules × oa13/oa33 (128 images), with zero overlap against
the authoritative r17 manifests, r17/graph-context feature caches, and r17
graph artifacts across molecule, canonical, connection, and scaffold identity.
All rows originate from the public release `train` split, so this is a
`fresh_holdout` role for frozen evaluation only, not an independent release
holdout.  Formal release holdout availability is zero under the current
prior-use ledger.

## Frozen safe-compile result

Receipt:
`outputs/audits/chemvector-r17-fresh-holdout-pair-graph-integration-cpu-v1/report.json`.

Baseline → calibrated pair-presence safe-compile:

- edge precision/recall: `0.988616/0.884244 → 0.988159/0.894427`
- typed precision/recall: `0.869982/0.778135 → 0.883363/0.799571`
- sanitize: `0.8828125 → 0.8828125`
- exact typed topology: `0.1171875 → 0.140625`

The per-profile gates also pass.  The calibration cohort differs from the
evaluation cohort and is explicitly recorded as cross-cohort; therefore this
result is diagnostic evidence only and `production_go=false`.

Target perturbation receipt:
`outputs/audits/chemvector-r17-fresh-holdout-pair-graph-integration-cpu-v1-target-perturb/report.json`.
The target-free decision digest is identical to the normal run:
`1ab34646bf01c021fac2e6be2e228808fdfdb7a124c82962b2e5ea49a16471df`.

## Decision

The pair→graph route is worth retaining as a shadow candidate, but there is
not enough admissible evidence for GPU training or production integration.
The next admissible action is a genuinely new source batch with an
append-only prior-use ledger, followed by metadata admission and frozen CPU
replay.  Existing release-train rows must not be relabeled as holdout.
