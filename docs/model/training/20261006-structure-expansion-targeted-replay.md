# Structure-expansion targeted replay (2026-10-06)

The v0.2 validation replay showed that the endpoint field is largely
reachable, while selector retention is weakest for text-heavy styles and
triple-bond contexts.  The existing `case_diversity_v1` catalog could not
provide a fresh identity-disjoint pool after excluding the v0.2 quarantine
bundle, so this read-only probe used the already identity-audited
`structure-expansion-batch-12272-20260923` quarantine source.

The probe selected 44 identity-disjoint molecules (all available molecules
with visible O/N and a triple bond), alternating profiles `oa11` and `oa13`.
It is a diagnostic, not a train/holdout release: no optimizer step, no model
write, and no sealed-data access occurred.

Results (`outputs/audits/chemvector-structure-expansion-targeted-replay-20261006-r2/report.json`):

- 424 target atoms; raw matched 405 (95.52%); selected matched 375 (88.44%).
- 19 raw gaps and 30 selector drops.
- `oa13` visible N: 28/37 raw and 14/37 selected.
- `oa11` visible N: 41/41 raw and 38/41 selected.
- implicit C contributes 20 selector drops, but is not the dominant raw gap.

The available targeted pool is only 44 molecules, below the planned 64-case
minimum.  This confirms a style-dependent selector failure, especially in the
large serif profile, but it is still insufficient to authorize training.  The
next data action is to obtain or generate a larger identity-disjoint
`font_large/serif × visible N/O × triple` pool, with fixed train/holdout
groups; no threshold sweep or selector sidecar is justified.
