# Style-factorial candidate strata replay (2026-10-10)

The immutable `initial.pt` was replayed over the proposed core train/dev and
stress-dev manifests. This was inference only: no gradients, optimizer,
checkpoint, production file, or sealed test was touched. The replay used
one-to-one matching at the existing 6 px radius and recorded candidate-level
JSONL for a later CPU gate.

The 6,400 requested samples produced 6,296 replayable samples; 104 samples
(26 molecules × four variants) were skipped by the committed coordinate
collision rule. The candidate cache contains 798,294 valid JSONL records and
is byte-bound by SHA-256:

`73bf419dd64b45782f62aa573682317f521bebd9af2e13b47d91f938ade8553a`

| split/style | strict | protected | near_unknown | hard_negative |
|---|---:|---:|---:|---:|
| train/coordgen | 342 | 47,775 | 4,928 | 145,369 |
| train/original | 470 | 47,610 | 4,710 | 146,167 |
| train/rdkit | 432 | 47,706 | 4,993 | 146,211 |
| dev/coordgen | 88 | 12,387 | 1,285 | 36,458 |
| dev/original | 105 | 12,367 | 1,234 | 36,652 |
| dev/rdkit | 102 | 12,418 | 1,330 | 36,578 |
| stress-dev/stress | 728 | 11,700 | 1,480 | 36,606 |

`strict` means raw matched within 6 px but selected matching failed;
`protected` is already retained and is not a loss row; `near_unknown` means
unmatched but within 10 px and has loss weight zero; only `hard_negative` is a
confirmed negative. The stress row remains diagnostic and cannot enter core
training.

Artifacts:

- report: `outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/report.json`
- candidate cache: `outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/candidates.jsonl`

The cache is an input to a CPU-only visual-gate experiment, not a training
authorization. The gate remains closed until precision ≥0.85 and recall ≥0.50
hold independently for each of the three core styles.
