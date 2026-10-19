# Text-center versus atom-anchor evidence (2026-10-06)

The current dense target places heatmap positives at the chemical atom anchor.
For visible text this point is often on the letter edge where the bond attaches,
not at the ink mass. A forward-only replay tested the alternative evidence
location on the 193 visible-text raw-gap targets.

| evidence location | median heatmap rank | rank ≤128 |
|---|---:|---:|
| atom-anchor cell | 665 | 22 / 193 |
| rendered text-center cell | 203 | 89 / 193 |

The improvement is strongest for large-font O/N cases. However, the predicted
anchor at the text-center cell is still only within 6px for 74/193 targets
(median distance 6.17px), so changing the objectness cell alone is not enough;
the sidecar must learn a center-to-attachment offset as well.

Artifact: `outputs/audits/chemvector-text-center-reachability-20261006-r2/report.json`.

This motivates one distinct trial: a frozen-feature text-center objectness head
with a small center-to-anchor offset head. It must remain residual-only and
must pass the same strict precision and baseline ownership gates.
