# Identity-aligned visible O/N panel: r4 result

This round is quarantine-only evidence. It is not a training release and does
not change the production checkpoint.

## Render and CPU gates

- Planned: 1,200 images from 120 train and 80 holdout molecules.
- Accepted: 1,187; rejected: 13.
- Every rejection was the fixed renderer quality reason
  `Typography clearance would erase a bond segment`; no retry or threshold
  tuning was applied.
- Accepted split: 715 train / 472 holdout; molecule overlap: 0.
- All six profiles are represented; accepted artifacts pass loader roundtrip.
- The plan declared a 48 px target-anchor separation, but the observed minimum
  is 42 px. The declared geometry contract therefore fails, although the
  16 px ownership radius still has a >20 px nearest-other margin.

Receipts:

- `outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r4/`
- `outputs/audits/chemvector-identity-aligned-visible-text-render-20261017-r4/report.json`
- `outputs/audits/chemvector-identity-aligned-visible-text-20261017-r4-cpu-audit/report.json`

## Frozen initial raw replay

The production `initial.pt` was used read-only with the fixed raw proposal
path. No Hungarian or nearest-target assignment was used; a target was counted
only when ownership was unique within 16 px, the candidate was farther than
20 px from every other target, and the 4 px margin held.

- Visible O/N targets: 3,179.
- Raw candidates: 127,018.
- Unique raw O/N ownership: 73 (2.30%).
- Strict (≤6 px): 71; refinable (6–16 px): 2.
- Ambiguous/zero target ownership: 3,106.
- Initial checkpoint digest and model state were unchanged.

Receipt: `outputs/audits/chemvector-identity-aligned-raw-reachability-20261017-r4/report.json`.

## Decision

`NO-GO` for atom-head, offset, residual, or selector training. The raw
proposal path does not reach the visible targets, and the geometry contract is
also not met. The next productive step is to redesign the renderer/data
contract (enforce the intended separation and remove mixed molecule-level
rejections), then rerun CPU integrity and a small frozen raw replay. Do not
adjust thresholds or train on this r4 panel.
