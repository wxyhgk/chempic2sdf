# Predicted bond-port residual attachment smoke (2026-10-15)

The frozen initial model's predicted bond endpoints were used directly with
`attach_residual_candidates(radius_px=12)`. Baseline occupancy came only from
the model-visible `selected_kept` flag; protected/strict labels were read only
after assignment. A 6 px duplicate veto and one-to-one endpoint ownership
were enforced. No pointwise scorer, optimizer, GPU, or production checkpoint
was used.

On 20 images per style, dev conservative strict precision was:

| style | strict selected / attached | precision |
|---|---:|---:|
| coordgen | 2 / 1,018 | 0.0020 |
| original | 3 / 944 | 0.0032 |
| rdkit | 3 / 974 | 0.0031 |
| stress | 40 / 1,126 | 0.0355 |

The predicted bond ports are not sufficiently aligned to identify strict
residual atoms. This closes the endpoint-attachment rescue branch for the
current initial checkpoint; repeating pointwise gates or changing the radius
would not be a justified next step.

Artifact:

`outputs/audits/chemvector-style-factorial-predicted-port-attachment-cpu-20261015-r1/report.json`
