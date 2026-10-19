# r17 t4b full-profile type-correction CPU result

## Data and admission

`outputs/datasets/chemvector-r17-residual-density-source-20261021-t4b` has 24
train and 12 holdout molecules rendered in all eight `oa00, oa02, oa11, oa13,
oa20, oa22, oa31, oa33` profiles (288 images).  The train split contains 8
independent aromatic and 9 ring molecules; the holdout contains one aromatic
and one ring molecule.  The source manifest, append-only ledger, identity
admission, endpoint quotas, and graph strata receipts are all complete; the
source remains quarantine-only and `training_admissible=false`.

## Experiments

The type residual uses the contract-matched objective
`chemical_logits + type_residual(features)` for both training and replay.  The
candidate edge set and presence route are frozen.  The feature cache is bound
to the checkpoint, manifest, source files, extractor, and exact train/holdout
keys.

At 300 CPU steps, weight 1 (`.../chemvector-r17-t4b-type-correction-cpu-contract3`),
several profiles improved, but oa00 typed precision fell 3.75 percentage
points.  A hard-example weight of 4 (`.../chemvector-r17-t4b-type-correction-cpu-hard4`)
made the regression worse: oa00 −8.75pp, oa22 −3.85pp, and oa31 −2.44pp,
despite small gains on oa11/oa13/oa20/oa33.

## Decision

Both runs fail the per-profile precision gate.  The failure is style/profile
conditional, not evidence that more global type loss is safe.  Do not start a
GPU run or merge a type sidecar.  The next useful experiment is a profile-
conditioned calibration/abstention probe selected on an independent source;
otherwise stop this route and focus on upstream visual features for the
aromatic-vs-double distinction.

## Fair t4b replay

The t4b presence residual was then trained from the same bound t4b feature
cache (`outputs/audits/chemvector-r17-t4b-presence-cpu/report.json`) and the
type residual was replayed with that sidecar
(`outputs/audits/chemvector-r17-t4b-type-correction-cpu-fair/report.json`).
This removes the earlier t3f-presence mismatch.  The result is unchanged in
the important respect: oa00 typed precision falls 3.75pp while oa02/oa11/oa13/
oa20/oa33 gain 1.25–2.60pp.  The all-profile gate remains false.

A profile-conditioned head was also run with the same fair t4b presence
sidecar (`.../chemvector-r17-t4b-profile-type-cpu-fair`).  Conditioning removes
cross-profile interference for several profiles, but oa00 still falls 3.75pp;
there is no all-profile-safe router.  This is diagnostic evidence, not a
release candidate.
