# r17 new-source v6: pair incidence → graph compile

This run is a bounded CPU shadow.  It does not modify the production
checkpoint (`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`),
does not read sealed data, and does not use GPU training.

## Data admission

Source:
`outputs/datasets/chemvector-r17-new-source-20261008-v6/`.

The cohort has 64 synthetic molecules (40 train, 24 holdout), each rendered in
oa13 and oa33 (128 images).  The admission audit also scans dataset-side
`molecules.json`/`manifest.json`, not only audit receipts.  It verifies zero
overlap on molecule, canonical, connection, and scaffold identities, unique
image hashes, train/holdout disjointness, complete provenance, and exact
ledger rows:

`outputs/audits/chemvector-r17-new-source-admission-20261008-v6-visible-inventory/report.json`.

## CPU sidecar and calibration

The frozen pair-presence residual was trained for 600 CPU steps:

`outputs/audits/chemvector-r17-new-source-pair-incidence-cpu-v6/report.json`.

On 48 holdout images, baseline → sidecar was:

- edge precision/recall: `0.98603/0.91214 → 0.98972/0.99483`
- typed precision/recall: `0.78492/0.72610 → 0.78663/0.79070`
- atom recall unchanged at `0.9875`

Train-only calibration passed all profile precision gates:

`outputs/audits/chemvector-r17-new-source-pair-calibration-cpu-v6/report.json`.

## Safe graph integration

`outputs/audits/chemvector-r17-new-source-pair-graph-integration-cpu-v6/report.json`

Holdout baseline → augmented (48 images, 24 molecules):

- edge precision/recall: `0.98603/0.89141 → 0.98663/0.93182`
- typed precision/recall: `0.75698/0.68434 → 0.75401/0.71212`
- sanitize: `0.8125 → 0.8333`
- exact typed topology: `0.2917 → 0.4167`

All per-profile gates passed.  The target perturbation receipt produced the
same decision digest as the normal run:

`3b14f7fbf58fc0993b205387294cc1483402ee31eedec7c9dc4205f2dc068446`.

## Decision

This validates the pair-incidence → safe-compile direction on a genuinely
new, ledger-bound source.  It is still only 24 holdout molecules and two
profiles, so `production_go=false` and GPU training remains NO-GO.  The next
experiment must expand the same source protocol to at least 128 train / 64
holdout molecules across the full eight profiles, then repeat the CPU gates
before any 30-minute GPU shadow.
