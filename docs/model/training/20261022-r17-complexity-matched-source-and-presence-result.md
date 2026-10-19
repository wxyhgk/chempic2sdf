# r17 complexity-matched source and presence CPU trial

## Source admission

The new quarantine source is:

`outputs/datasets/chemvector-r17-complexity-matched-source-20261022-v2`

It contains 48 identity-disjoint molecules (32 train, 16 holdout), rendered in
the eight `oa00, oa02, oa11, oa13, oa20, oa22, oa31, oa33` profiles (384
images). The metadata-only admission receipt passed ledger coverage, history
identity disjointness, unique sample/image bytes, and endpoint quotas. The
complexity audit passed: train and holdout each contain three aromatic/ring
molecules, with a 9.4pp aromatic-fraction gap and mean atom gap below one.

Authoritative receipts:

- `outputs/audits/chemvector-r17-complexity-matched-admission-20261022-v2/report.json`
- `outputs/audits/chemvector-r17-complexity-matched-endpoint-tags-20261022-v2/report.json`
- `outputs/audits/chemvector-r17-complexity-matched-complexity-20261022-v3/report.json`

## Presence trial

The frozen initial model plus a 300-step CPU pair-incidence residual was
evaluated on the 16-molecule holdout. Aggregate edge recall rose from 94.77%
to 100.00%, but precision fell from 99.77% to 99.45%. The per-profile gate
failed, most visibly:

```text
oa13 edge precision -1.67 pp, typed precision -3.38 pp
oa33 typed precision -2.23 pp
```

The candidate edge set therefore becomes too permissive for a downstream type
head. No type training or GPU run was started from this sidecar. The next
permitted experiment is train-only threshold/abstention calibration on the
presence scores; production remains frozen.

Authoritative report:

`outputs/audits/chemvector-r17-complexity-matched-presence-cpu-20261022-v2/report.json`
