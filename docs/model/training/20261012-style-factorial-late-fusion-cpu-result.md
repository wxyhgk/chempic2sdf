# Style-factorial geometry + pixel late-fusion CPU preflight (2026-10-12)

The candidate-level cache was used to fit a small CPU-only late-fusion head:
four candidate-observable geometry values (`raw_score`, `presence_logit`,
`x_norm`, `y_norm`) plus a 32 px grayscale crop area-pooled to 16×16. The
GT-derived `nearest_distance_px`, atom ids, layers, and graph fields were
explicitly excluded. Training used only strict positives and deterministic
hard-negative samples (at most 8× strict per core style); near-unknown,
protected, and unresolved selected rows were not loss rows. Seed 20261012,
300 AdamW steps, batch 512, and a fixed zero-logit gate were used without
threshold search.

| dev style | conservative precision | strict recall | selected | strict selected |
|---|---:|---:|---:|---:|
| coordgen | 0.091 | 0.909 | 878 | 80 |
| original | 0.098 | 0.886 | 946 | 93 |
| rdkit | 0.095 | 0.853 | 913 | 87 |
| stress (separate) | 0.195 | 0.852 | 3,180 | 620 |

All core styles pass the provisional recall floor but fail the precision floor
of 0.85 by a wide margin. Near-unknowns were masked during training but remain
part of the conservative review denominator. This confirms that adding a
small local crop to the frozen geometry scores does not resolve the
bond/ink hard-negative ambiguity in the current candidate pool.

Decision: **NO-GO for GPU or production**. Do not tune the gate, retry this
run, or reclassify near-unknowns as negatives. The strict candidate gate
remains closed; the next useful work is to improve proposal/endpoint evidence
or obtain reviewed hard-negative labels, then repeat the same fixed CPU gate.

Artifact:

`outputs/audits/chemvector-style-factorial-late-fusion-cpu-preflight-20261010-r1/report.json`
