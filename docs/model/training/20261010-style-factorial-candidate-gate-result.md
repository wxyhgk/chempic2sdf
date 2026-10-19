# Style-factorial candidate replay and CPU gate

The frozen initial model was replayed once over the committed style-factorial
pool.  The replay used forward inference and one-to-one Hungarian matching
only; it did not read sealed data, call backward, update an optimizer, or write
a checkpoint.

The replay covered 6,296 of 6,400 manifest images.  The remaining 104 were
the already recorded coordinate-collision cases.  The candidate cache contains
798,294 valid JSONL records and is byte-for-byte consistent with the first
aggregate replay:

`outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/`

For the three core dev variants, the candidate strata were:

| variant | strict | near-unknown | hard negative |
| --- | ---: | ---: | ---: |
| coordgen | 88 | 1,285 | 36,458 |
| original | 105 | 1,234 | 36,652 |
| rdkit | 102 | 1,330 | 36,578 |

Near-unknown candidates remain masked and are not negatives.  Stress remains a
separate holdout.

## CPU gate

A single fixed scalar gate was fit on train-core strict positives and hard
negatives using only frozen-model score/presence/coordinate features.  The
zero-logit decision was evaluated once, without threshold scanning.  It failed
decisively:

- train conservative precision 0.307, recall 0.047;
- core dev precision 0.004–0.009, recall 0.019–0.045;
- stress was reported separately (precision 0.089, recall 0.102).

Report: `outputs/audits/chemvector-style-factorial-scalar-cpu-gate-20261010-r1/report.json`.

## Decision

The style pool is large enough in raw image count, but strict selector errors
are too sparse and scalar features do not separate them from hard negatives.
The core strict precision/recall gate is not met.  Therefore no GPU adaptation
is authorized.  The pure local-raster branch is also frozen after its separate
multiscale CPU probe failed.  The next viable experiment must add geometry
late-fusion (coordinate-refiner output plus local pixels) and pass a new CPU
gate before any bounded GPU shadow; it must not retry the failed scalar or
pure-visual objectives.
