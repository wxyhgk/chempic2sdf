# Within-group pairwise rescue CPU preflight

This bounded probe ranked dropped atom candidates only within their original
candidate group. Strict-positive rows were paired against four deterministic
hard negatives per positive. Protected baseline rows were preserved and
excluded from rescue ranking; near-unknown and selected-unmatched rows were
masked from the pairwise loss. Features were frozen candidate scores/ranks
and candidate-to-candidate geometry. No GT distance or target ID was read as
an input.

The fixed slice was 120 train samples per core style and 40 dev samples per
core/stress style. Training used 300 AdamW steps and margin 0.5. Evaluation
used fixed top-1 and top-3 group ranking, with no threshold sweep.

Machine-readable result:

`outputs/audits/chemvector-style-factorial-group-pairwise-rescue-cpu-preflight-20261007-r1/report.json`

| dev style | top-1 strict group recall | top-3 strict group recall | top-1 strict precision over all groups |
|---|---:|---:|---:|
| coordgen | 0.222 | 0.556 | 0.050 |
| original | 0.000 | 0.500 | 0.000 |
| rdkit | 0.750 | 0.875 | 0.150 |
| stress (separate) | 0.364 | 0.500 | 0.200 |

Top-3 recall has some signal, but top-1 precision is far below the deployment
gate and the three core styles do not pass together. This is a diagnostic
NO-GO for GPU training; protected baseline retention remains the safe path.
