# Dense-cell multiplicity / peak contract audit

The cap/stroke replay was followed by a frozen forward-only audit of dense
heatmap peaks.  At the production 6 px radius, target coverage remained
0.936--0.940.  Taking the top-128 local peaks yields only about 10.2% target
precision, far below the 0.85 hard-negative requirement for an appended
candidate head.

The audit therefore does not authorize a GPU shadow.  A safe extra-heatmap
head would need a candidate-level manifest containing cell IDs, scores,
protected baseline ownership, collision groups, and hard negatives; those
fields are not present in the current panel artifact.  The current
multi-slot/multiplicity contract remains CPU-NO-GO until that manifest and
100% protected selected retention are demonstrated.

Artifacts:

- Forward audit: `outputs/audits/chemvector-oa13-cap-stroke-multislot-admission-20261017-r1/report.json`
- Contract audit: `outputs/audits/chemvector-dense-cell-multiplicity-contract-20261017-r1/report.json`
- Protocol: `outputs/audits/chemvector-dense-cell-multiplicity-contract-20261017-r1/protocol.json`
