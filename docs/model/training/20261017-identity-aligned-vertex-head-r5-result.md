# r5 independent dense vertex proposal shadow

This is the first bounded training result after the r4 data-contract failure.
The new head is an isolated shadow; it is not merged into the production
ChemVector path and no checkpoint was written.

## Data contract

- 1,200 planned rows from the r5 identity-disjoint plan.
- 1,170 accepted rows: 195 rows in each of six profiles.
- 30 rows rejected as five whole molecules (six profiles each) because the
  typography quality rule failed. No partial molecule was admitted.
- 696 train / 474 holdout rows; molecule and scaffold overlap are zero.
- Connection identities are non-empty deterministic de-stereo keys.
- Observed minimum anchor separation is 48 px within floating-point tolerance.
- DejaVu Serif is used for cov13; Nimbus Roman is used for cov33, with exact
  font-match provenance.

CPU receipt:
`outputs/audits/chemvector-identity-aligned-visible-text-render-20261017-r5/report.json`

## Shadow recipe

The run used a new `ChemVertexDetector` (92,678 parameters) and the final
stride-4 vision block (79,200 parameters). Existing atom, bond, set, pair,
role, bbox, OCR, and production proposal modules were frozen. It ran exactly
800 steps, batch size 8, head learning rate `2e-4`, block learning rate `1e-6`.
The output stayed in memory and only a metrics receipt was written.

## Results

| split | recall | precision |
|---|---:|---:|
| train | 0.9994 | 0.5146 |
| holdout | 0.9897 | 0.4575 |
| old frozen holdout baseline | 0.2010 | 0.0512 |

Holdout visible O recall is 1.0000, visible N recall is 0.9839, and implicit-C
recall is 0.9866. The six profile recalls range from 0.9854 to 0.9951.

Receipts:

- Shadow: `outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v2/report.json`
- Same-decoder baseline: `outputs/audits/chemvector-identity-aligned-vertex-baseline-20261017-r5-v2/report.json`

## Decision

The proposal-source hypothesis is supported: a compact independent dense head
can recover atom centers on this clean, contract-valid panel, while the frozen
production raw field cannot. This does **not** authorize deployment or full
training. The next experiment must be an integration shadow that keeps the
new field separate, tests bond attachment/graph compatibility, and preserves
the production prefix and checkpoint byte-for-byte. No threshold sweep,
selector rescue, or production replacement is authorized by this receipt.
