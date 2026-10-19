# Aromatic atom probe (2026-10-16)

The r15 element-conditioned repair passed the fixed fresh holdout, but that
holdout contains no aromatic atom positives.  Its perfect sanitize score is
therefore not evidence for aromatic generalization.

An isolated 257-parameter binary head (`LayerNorm(128) -> Linear(128, 1)`) was
trained on frozen AtomObject features from the fixed 96-image train split. It
did not modify the production checkpoint.

| split | atom rows | aromatic positives | accuracy | positive recall |
|---|---:|---:|---:|---:|
| dev (96) | 882 | 340 | .79932 | .79412 |
| fresh holdout (95) | 884 | 0 | .77036 | n/a |

The dev errors are concentrated in aromatic carbon and nitrogen.  The fresh
split has no positive aromatic labels and cannot serve as an aromatic gate.
The four dev molecules that caused the r15 candidate aromatic failures repeat
across all four style profiles, so this is a structural coverage/attribute
problem rather than a style-specific regression.

Artifact:

```text
outputs/audits/chemvector-atom-aromatic-head-cpu-r16/report.json
outputs/audits/chemvector-atom-aromatic-head-cpu-r16/head-state.pt
```

Decision: do not start the r15 GPU type shadow.  First add a targeted aromatic
split (aromatic C/N/S, fused and heteroaromatic rings, plus DOUBLE/TRIPLE
controls), then evaluate on an identity-disjoint aromatic holdout.  Production
remains unchanged.
