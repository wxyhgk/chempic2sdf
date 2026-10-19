# Frozen backbone feature probe

The complete CPU probe sampled frozen initial stride-4 and stride-8 vision
features at candidate coordinates, concatenated them with raw score,
presence-logit, and normalized coordinates, and trained a small CPU head.  It
excluded nearest-distance/GT atom IDs/layers/graph fields, masked
near-unknowns, and used a fixed zero-logit gate.  No GPU, backward pass, or
checkpoint was used.

The run processed 5,770 replayable samples and 152,646 evaluation candidates.
Core dev results were:

| variant | precision | recall |
| --- | ---: | ---: |
| coordgen | 0.085 | 0.886 |
| original | 0.099 | 0.933 |
| rdkit | 0.091 | 0.804 |

Stress was 0.177 precision / 0.871 recall and remained a separate holdout.

Report: `outputs/audits/chemvector-style-factorial-frozen-feature-cpu-preflight-20261010-r1/report.json`.

The frozen backbone representation improves recall but still admits many
line-like hard negatives.  It does not meet the core gate and does not justify
GPU adaptation.
