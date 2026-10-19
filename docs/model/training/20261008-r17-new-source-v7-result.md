# r17 new-source v7 pair-incidence route (2026-10-08)

## Scope

This is a frozen-checkpoint CPU shadow on the disjoint v7 source cohort.  The
production checkpoint, atom proposals, unordered pair set, type heads, OCR,
and sealed test remain untouched.  Only a residual pair-presence sidecar is
trained and routed through the safe graph compiler.

Source and admission:

- `outputs/datasets/chemvector-r17-new-source-20261008-v7/manifest.json`
- `outputs/audits/chemvector-r17-new-source-admission-20261008-v7/report.json`
- 128 train / 64 holdout molecules, 8 profiles, 1024 / 512 images
- molecule, canonical, connection, scaffold, image, ledger, and split gates pass

## Corrected calibration scope

The first v7 calibration report used the trainer defaults (32/16 molecules),
which silently calibrated only a prefix of the v7 cohort.  That report is not
used for decisions.  Calibration was rerun with explicit 128/64 molecule
counts and written independently:

`outputs/audits/chemvector-r17-new-source-pair-calibration-cpu-v8/report.json`

The profile route uses parameters `[0.75, 0.5, 0.0]`.  The difficult profiles
both gain typed recall on the disjoint holdout:

| profile | typed recall | typed precision |
|---|---:|---:|
| oa13 | 0.7406 → 0.7478 (+0.720pp) | 0.7981 → 0.7948 |
| oa33 | 0.6774 → 0.6847 (+0.730pp) | 0.7557 → 0.7577 |

All profile precision-drop gates pass; no holdout labels are used to choose the
route.

## Safe graph integration

`outputs/audits/chemvector-r17-new-source-pair-graph-integration-cpu-v8/report.json`

On the 512-image holdout:

- edge recall: `0.95425 → 0.95767`
- typed edge recall: `0.75180 → 0.75594`
- exact typed topology: `0.36523 → 0.37891`
- edge and typed precision remain within the 1pp guard
- sanitize: `0.73242 → 0.73242`

The target-free routing digest is:

`ba799fb26c6b48d409ed880e52f00f4b67d469c555ee24db17f161d482fd89aa`

The independent target-perturb sentinel produces the exact same digest:

`outputs/audits/chemvector-r17-new-source-pair-graph-integration-cpu-v8-target-perturb/report.json`

## Decision

This is the first full-profile, larger-cohort CPU evidence that pair-incidence
residual routing plus safe graph compilation has a reproducible positive
direction.  It is not a production merge and does not authorize a GPU run:

- production checkpoint remains
  `09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`
- `production_go=false`
- GPU used: false
- sealed test read: false
- the v7 strata audit still reports insufficient rare visible O/N subtype
  coverage for a training release

The next admissible action is to improve the source strata (especially rare
visible O/N bond subtypes), then repeat the same CPU contract.  Do not start
full production training or relax the precision gates based on this shadow.
