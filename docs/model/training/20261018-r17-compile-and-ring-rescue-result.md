# r17 predicted-element compile and complete-ring rescue (2026-10-18)

The r17 element head was connected only to a read-only CPU compile audit; the
production checkpoint and sealed test were not touched.  On 64 molecules per
split (all four profiles), the strict target-free cycle transaction reached:

| split | predicted sanitize | oracle-element sanitize | aromatic P/R |
|---|---:|---:|---:|
| train | .9063 | .8281 | 1.000/.9304 |
| dev | .8125 | .8281 | 1.000/.7403 |
| fresh | .8281 | .8906 | 1.000/.8318 |

Fresh residual failures were 9 valence and 2 aromatic compile errors.  The
candidate/oracle gap is therefore not explained by aromatic scoring alone;
predicted element/type/topology errors remain material.

The follow-up complete-ring transaction enumerated only existing frozen
AtomPair 5/6-cycles.  A cycle could promote at most one below-threshold edge,
and every fused/overlapping proposal was committed atomically or rolled back
after a target-free compiler check.  No candidate edges were added and no
target labels entered selection.

The fresh aromatic recall rose from `.8318` to `.8799` at the declared mean
threshold `.85` (and `.9069` in a lower-threshold diagnostic), while precision
and sanitize rate did not meet a joint release gate: dev precision was `.9760`
for the rescue arm and fresh sanitize remained `.8281`.  This is useful as a
bounded diagnostic but is **NO-GO for GPU or production integration**.

Artifacts:

```text
outputs/audits/chemvector-graph-context-aromatic-compile-r17-all-m16-v5/report.json
outputs/audits/chemvector-aromatic-ring-rescue-r17-all-m16/report.json
```

Next work is valence/type error attribution or a target-free abstention audit;
do not lower thresholds and do not start GPU training from this result.
