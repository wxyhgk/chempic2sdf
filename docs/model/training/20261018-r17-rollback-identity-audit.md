# r17 rollback identity audit (2026-10-18)

The rollback shadow improves compiler validity, but it is not identity
preserving.  Across 768 rows (192 molecule groups × 4 profiles), source graph
joins and profile identities were exact, and every final edge set was a
subset of the baseline.  Aromatic edges were never removed.

However, 35 accepted transactions contained:

| class | count |
|---|---:|
| mapped false-positive edge | 22 |
| mapped true non-aromatic target edge | 5 |
| unmapped edge | 8 |

The five true-edge deletions reduce edge recall by `0.001067` on dev and
`0.000267` on fresh.  Thus rollback may remain a fail-closed CPU shadow, but
it cannot be described as exact graph repair or integrated into production
without an explicit non-aromatic edge-loss policy.

Artifact:

```text
outputs/audits/chemvector-r17-rollback-identity-audit-m64-v1/report.json
```
