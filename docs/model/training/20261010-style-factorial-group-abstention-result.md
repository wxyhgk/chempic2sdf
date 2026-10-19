# Style-factorial group abstention CPU probe

A second structural alternative was tested after pointwise scalar and
geometry-plus-pixel gates failed.  Candidates were grouped by frozen proposal
competition; the scorer saw frozen scores, candidate geometry competition, and
top-8 local summaries.  It did not receive ground-truth distances.  Protected
and near-unknown members were excluded from negative loss.  Group sampling,
seed, training steps, and the prevalence-derived gate were fixed in advance;
there was no threshold sweep, GPU, or checkpoint.

Core dev results:

| variant | group precision | group recall | strict candidate recall after gate |
| --- | ---: | ---: | ---: |
| coordgen | 0.731 | 0.333 | 0.432 |
| original | 0.593 | 0.242 | 0.343 |
| rdkit | 0.591 | 0.191 | 0.235 |

Stress was evaluated separately (group precision 0.904, recall 0.425).

Report: `outputs/audits/chemvector-style-factorial-group-abstention-cpu-preflight-20261007-r1/report.json`.

## Decision

Group competition improves conservative precision relative to pointwise
late-fusion, but still misses the required 0.85 precision / 0.50 recall on all
core styles.  It is not sufficient to authorize GPU adaptation.  The current
evidence rules out scalar-only, pointwise local fusion, and this lightweight
group scorer.  Any further experiment needs a materially stronger pretrained
visual representation or explicit graph/bond context, with a new CPU gate
before GPU use.
