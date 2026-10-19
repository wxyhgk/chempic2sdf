# t4b split-complexity audit

The t4b identity split is disjoint, but it is not complexity matched:

| split | molecules | mean atoms | mean bonds | aromatic molecules |
|---|---:|---:|---:|---:|
| train | 24 | 8.54 | 8.58 | 8 |
| holdout | 12 | 7.58 | 6.83 | 1 |

The aromatic fraction is therefore 33.3% in train versus 8.3% in holdout (a
25pp gap). The gate correctly fails `aromatic_fraction_gap_le_0.20` and marks
the split as not complexity matched. This explains why a sidecar can gain
4–9pp on train yet regress on a future harder profile: the current holdout is
too easy to validate generalization.

Before another type-side training run, rebuild a larger identity-disjoint
holdout stratified by aromatic/ring status and atom/bond complexity. Keep the
production checkpoint frozen until that gate passes.

Authoritative audit:

`outputs/audits/chemvector-r17-t4b-split-complexity-audit-v2/report.json`
