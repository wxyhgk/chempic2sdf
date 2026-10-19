# Strict local-patch CPU preflight (2026-10-07)

## Protocol

The frozen r2 coordinate release was used without reading sealed data. A
candidate-centred 32 px grayscale crop was resized to 16×16 and passed to a
small 2-layer CPU head. Training rows were only explicit strict members and
confirmed far negatives; near-unknown rows were masked from the loss and
counted as conservative false selections. The fixed zero-logit gate was used
without threshold search. No checkpoint or GPU run was authorized.

## Result

| split | conservative strict precision | strict recall | near-unknown selected | hard-negative selected |
|---|---:|---:|---:|---:|
| train | 0.746 | 0.287 | 35 | 0 |
| holdout | 0.926 | 0.262 | 6 | 0 |

The crop improves precision relative to the scalar strict probe (holdout
0.279), but it misses most strict positives and fails the predeclared recall
floor of 0.50. The high holdout precision is therefore not a deployment
result; it is the low-recall edge of a tiny 359-positive training pool.

## Decision

**NO-GO for GPU or production.** The result supports the diagnosis that local
pixels contain useful evidence, but the current 480-image release does not
contain enough style/geometry variation to train a strict selector. Do not
lower the gate or increase steps. The next admissible work is an
identity-disjoint style-factorial release focused on font-large/serif visible
N/O, implicit-C endpoint placement, rare bond types, and reviewed 10--16 px
boundary negatives. A multi-scale crop encoder can be revisited only after a
new CPU strict preflight reaches precision ≥0.85 and recall ≥0.50 in every
declared style stratum.

Artifact:

`outputs/audits/chemvector-targeted-selector-strict-patch-cpu-preflight-20261007-r1/report.json`
