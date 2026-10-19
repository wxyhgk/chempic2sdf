# Coordinate refiner-only shadow (2026-10-07, r2)

## Decision

The corrected coordinate release and a frozen CPU candidate gate passed their
readiness checks. One new, bounded GPU shadow was run with the candidate gate
frozen; only the coordinate refiner was trainable. No production checkpoint was
written and no sealed test content was read.

## Protocol

- release: `chemvector-targeted-selector-coordinate-release-20261007-r2`
- candidate gate: 300-step CPU head, seed `20261006`, fixed logit gate `0.0`
- refiner: 100 GPU steps, seed `20261008`, one card, 1,800 s bound
- train-only refinement targets; holdout was evaluation-only
- strict-positive-group attached unknown members excluded from both losses
- old joint candidate+refiner shadow remains terminal and is not retried

## Result

| split | refinable precision | refinable recall | selected refined <=6 px | assignment precision | strict recovered |
|---|---:|---:|---:|---:|---:|
| train | 0.872 | 0.553 | 0.863 | 0.975 | 252 |
| holdout | 0.893 | 0.579 | 0.895 | 0.984 | 214 |

The frozen candidate gate metrics are unchanged by refiner training (holdout
precision `0.893`, recall `0.579`). The assignment figures are diagnostics after
one-to-one matching and must not be reported as strict selector accuracy.

## Status and next boundary

The shadow gate passed, so the refiner may remain a local, opt-in sidecar for
further public-style evaluation. It is not a production model: strict-only
precision is still unresolved, deployment remains unauthorized, and the
initial production checkpoint is byte-identical. Any production proposal needs
a separate strict-selector contract and an independent authorization.

Artifacts:

- authorization: `outputs/audits/chemvector-targeted-selector-coordinate-refiner-run-authorization-20261007-r2/receipt.json`
- report: `outputs/audits/chemvector-targeted-selector-coordinate-refiner-shadow-20261007-r2/report.json`
- frozen gate: `outputs/audits/chemvector-targeted-selector-frozen-coordinate-gate-20261007-r2/report.json`
- sidecar: `outputs/audits/chemvector-targeted-selector-coordinate-sidecar-20261007-r2/sidecar.json`
