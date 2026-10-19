# Port-conditioned endpoint-objectness sidecar result

## Protocol

We ran the preregistered 800-step trial twice. `initial.pt`, the bond detector,
and the old set-head were frozen; residual rows never entered the old set-head.
The split was 480/720 and group-, molecule-, and canonical-SMILES-disjoint.
Thresholds were selected from train only and use the same 16-claims-per-image
capacity as evaluation. No sealed-test data or production checkpoint was used.

## Result

| feature mode | split | candidates | positives | AUPRC | threshold result |
|---|---|---:|---:|---:|---|
| full (142D) | train | 117,519 | 98 | 0.592 | no threshold reached 0.85 precision (best 0.607) |
| full (142D) | holdout | 178,997 | 188 | 0.0535 | no admissible train threshold |
| explicit (8D) | train | 117,519 | 98 | 0.261 | no threshold reached 0.85 precision (best 0.406) |
| explicit (8D) | holdout | 178,997 | 188 | 0.258 | no admissible train threshold |

Ownership and identity gates passed: prefix immutable, old selected retention
1.0, and all three split overlaps were zero. The deployment gate is still
`admissible=false`.

## Interpretation

The full feature sidecar overfits the frozen 128D bond embedding: its train
ranking is high but holdout ranking collapses. Removing that embedding makes
train and holdout ranking agree, but the resulting precision is still far
below the deployment gate. This is not a GPU, step-count, or threshold problem.
The frozen port route is also incomplete: the reachability audit found only
about 62.8% of unmatched holdout targets near a free predicted port, and strict
bond-identity compatibility was only 47 cases. The current nearest-atom label
is weaker than strict bond-identity supervision and can mark wrong-bond
positives. We will not spend another sweep on this scorer.

Artifact:

`outputs/audits/chemvector-port-endpoint-sidecar-20261008-r1/report.json`

Explicit-feature artifact:

`outputs/audits/chemvector-port-endpoint-sidecar-20261009-explicit/report.json`

Decision: **NO-GO** for scorer training/deployment. Do not continue with
threshold, K, MLP-capacity, or step-count sweeps, and do not promote either
sidecar cache or weights. The next experiment should be a bounded local
atom+bond joint candidate/paired-endpoint representation, with strict bond
identity labels and a reachability audit before any training.
