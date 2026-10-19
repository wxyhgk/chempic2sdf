# Style-factorial group abstention CPU preflight

This experiment tested whether a frozen candidate group can be rejected or
passed as a whole before a more expensive local/chemical re-observation.
Groups were built from the candidate JSONL replay, with train-core groups as
the only fitting data. Features used frozen candidate scores, candidate-to-
candidate coordinate competition, and image-only crops around the eight
highest-presence candidates. Ground-truth distance was not an input.
Protected rows were context only; neither protected nor near-unknown rows
were treated as negative labels. Samples absent from replay due to coordinate
collisions were excluded fail-closed.

The protocol was fixed: seed `20261013`, 300 AdamW steps, 1:1 positive and
negative group sampling, and one prior-derived logit gate (the logit of the
train positive-group prevalence). No threshold sweep, GPU, checkpoint, or
sealed test was used.

Machine-readable result:

`outputs/audits/chemvector-style-factorial-group-abstention-cpu-preflight-20261007-r1/report.json`

| dev variant | group precision | group recall | strict candidate recall after group gate |
|---|---:|---:|---:|
| coordgen | 0.731 | 0.333 | 0.432 |
| original | 0.593 | 0.242 | 0.343 |
| rdkit | 0.591 | 0.191 | 0.235 |
| stress (separate) | 0.904 | 0.425 | 0.663 |

All three core variants fail the precision/recall gate. Group-level abstention
is therefore not a better route than the existing coordinate candidate gate;
it should not receive GPU authorization. The stress result is reported only
as a separate diagnostic and cannot be used to justify core training.
