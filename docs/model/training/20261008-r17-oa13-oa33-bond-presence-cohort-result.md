# r17 oa13/oa33 bond-presence cohort and strict CPU replay (2026-10-08)

## What was added

`tools/prepare_r17_oa13_oa33_bond_presence_manifest.py` creates a
metadata-only cohort from the public structure-expansion release.  It excludes
the prior selector cache, known-clean graph artifact, slice80 shadow, and the
earlier oa13/oa33 stress manifest by molecule/canonical/connection/scaffold
identity.  The manifest contains sample keys and identity metadata only; no
layers, graph objects, coordinates, target IDs, or nearest-target distances
are exposed to a model.

The resulting manifest is
`outputs/audits/chemvector-r17-oa13-oa33-bond-presence-manifest-v1/manifest.json`.
It has 32 train and 16 holdout molecules, with both `oa13` and `oa33` views
(64/32 images).  All four identity overlap checks are zero.  The holdout has
54 visible O/N atoms per profile, including 16 degree-1 O, 6 degree-2+ O,
4 degree-1 N, and 28 degree-2+ N.  Its visible O/N-incident target bonds
contain 32 single, 15 double, 38 aromatic, and 1 triple bonds per profile.
These are post-hoc audit counts and are not model inputs.

## Frozen replay

The frozen selector/graph replay used the production checkpoint only as a
read-only inference source and used the existing sidecar state.  It ran with
16 px ownership, 8 px target-free NMS, and safe compiler mode:

`outputs/audits/chemvector-r17-oa13-oa33-bond-presence-replay-v1/report.json`

On the 32-image holdout:

| profile | edge recall (base → shadow) | strict typed recall (base → shadow) |
|---|---:|---:|
| oa13 | 0.9478 → 0.9478 | 0.8918 → 0.8918 |
| oa33 | 0.8657 → 0.9067 | 0.7090 → 0.7500 |

Pooled recall was `0.9067 → 0.9272`; strict typed recall was
`0.8004 → 0.8209`.  Precision stayed above 0.99 at edge level and above
0.88 under strict typed attribution; sanitize did not drop.  The checkpoint
SHA remained
`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`, and no
production checkpoint was written.

The target-perturbation sentinel produced the same candidate-selection
decision digest (`11c3f14d73cad2697da2dc9c577ea9b11384106eb1cb747959f38154cc99eded`),
confirming that routing and candidate selection do not depend on target data.

## Bounded CPU pair-incidence shadow

The same cohort was then used for one 600-step CPU-only residual presence
shadow (`outputs/audits/chemvector-r17-oa13-oa33-bond-presence-cpu-v1/report.json`).
The production pair set and all upstream heads remained frozen.  Holdout
pooled edge recall improved `0.9257 -> 1.0000` and strict typed recall
`0.8895 -> 0.9543`; oa13/oa33 typed recall gains were `+4.85pp/+8.17pp`.
However, oa13 typed precision fell `1.15pp` and oa33 `1.22pp` (edge precision
oa13 fell `1.05pp`), exceeding the per-profile one-point guard.  Thus the
shadow is useful evidence for the residual objective but is **not admissible
for deployment**.

## Decision

This is evidence for a narrow oa33 pair-incidence residual, not authorization
for GPU training or production merge.  The next admissible experiment is a
bounded CPU calibration/trust-region shadow on this manifest, with independent
oa13/oa33 precision floors and protected-prefix retention.  A GPU run remains
NO-GO until that shadow passes and a larger disjoint holdout is available.
