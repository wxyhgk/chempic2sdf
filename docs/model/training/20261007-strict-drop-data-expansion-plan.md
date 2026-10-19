# Strict-drop data expansion and CPU gate (proposal only)

## Decision boundary

This is a data and CPU-readiness proposal. It does not authorize GPU training,
does not change `initial.pt`, and does not promote the raw-objectness shadow to
production. The graph-veto and group-rescue branches remain frozen.

The frozen style-factorial replay has 4,800 train-core images (1,600
molecules), 1,200 dev-core images (400 molecules), and 400 stress images whose
molecule identities are the same as dev-core. Strict selector drops are only
342/470/432 by train style and 88/105/102 by dev style. The failed graph audit
also shows that endpoint evidence cannot safely recover these drops.

## One bounded expansion

Prefer first reusing the existing quarantine structure-expansion batch: it has
12,184 distinct accepted images from 1,534 molecules across eight fixed
profiles and unusually broad N/O/S/P/Si/B, aromatic/double/triple, and stereo
coverage. It is **not** admitted and must pass the full identity/release and
frozen-initial replay gates below. If its profile gate fails, render the
targeted supplement described here. For a fresh supplement, render 12,000
images and require at least 10,000 eligible, unique images after the
chemistry/spatial/byte checks. Molecules must be approved by a complete
prior-use identity ledger (prefer a separately admitted source;
`style_train_100k` rows are eligible only after excluding every molecule
already used by prior train/confirm/route data):

For the existing batch, a practical first partition is 1,200 molecules for
train and 300 for holdout, stratified by scaffold/connection, with the 34
remaining molecules reserved for failed-render/profile replacement. Count
actual bytes rather than assuming eight images per molecule: `oa13` currently
has 1,446 rows while the other seven profiles have 1,534 each.

| split | new molecules | images/molecule | images | target purpose |
|---|---:|---:|---:|---|
| train | 2,400 | 4 | 9,600 | fit-only strict-drop evidence |
| holdout | 600 | 4 | 2,400 | identity-disjoint CPU gate |

The four renders of each molecule are fixed targeted profiles, not random
resampling. Use a balanced 24-cell style panel (three font classes × two size
classes × two bond-offset classes × two colour modes):

* three font classes: serif/roman, sans-narrow/condensed, and bold/mixed; use
  six fixed families (`dejavu-serif`, `gnu-free-serif`, `liberation-serif`,
  `nimbus-roman`, `urw-bookman`, `yrsa`) plus sans-narrow as a diagnostic;
* small/large (include a small extreme-font slice);
* thin/thick lines;
* wide/extra-wide multiple-bond offsets (retain a smaller tight control);
* near-zero/negative/positive rotation;
* black-white plus grayscale/limited-color.

Allocate approximately 50% of cells to serif/roman + small/large +
wide/extra-wide, 25% to sans-narrow/condensed controls, and 25% to stress
combinations (thin or thick, extreme/small, limited-color, near-zero). The
current replay has the highest ordinary strict-drop rates in small and
extra-wide cells, while dev failures concentrate in serif/large and
wide/extra-wide cells; the stress replay is not an identity holdout and must
not replace the 600-molecule holdout.

For each cell, alternate `thin`/`thick` line widths and
`near_zero`/`negative`/`positive` rotations across molecules using a fixed
seed; these are balanced factors, not an unrecorded fifth profile choice. A
cell assignment must be written into the manifest before rendering, so a
failed render cannot silently change its target cell.

Molecule composition should be fixed before rendering: 30% N/O text-bearing,
20% S/P/Se/Si/B/halogen-bearing, 20% mixed heteroatom/charged or aromatic
structures, and 30% carbon-rich implicit-C/line-ambiguity structures. Reserve
at least 10% for triple/aromatic or other non-single bond cases and 10% for
branching/ring junctions. This prevents a text-only expansion from hiding
vertex/line errors. Keep ring, multiple-bond, and branching strata in each
split.

## Identity and admission requirements

1. New train and holdout sets have disjoint molecule IDs, InChIKeys,
   connection identities, and scaffolds. Both are disjoint from every existing
   train/dev/confirm/route identity used by this project.
2. Each molecule has one canonical structure and exactly four image rows;
   no image-level split is allowed.
3. Exact-byte manifest, source SDF, renderer/font IDs, profile tuple, and image
   hashes are committed. Quarantine data is not readable by a training job
   before admission.
4. Every profile cell has at least 80 train molecules and 20 holdout
   molecules; every core style and each major element/size stratum has at least
   50 strict positives in train and 20 in holdout after frozen replay.
5. A fixed replay must find at least 1,000 train strict positives and 250
   holdout strict positives (plus at least 200 separately reported stress
   strict positives). If the counts are lower, expand or redesign the data;
   never rebalance labels or lower the gate.

## Frozen-initial CPU gate (one run, no sweep)

Run the exact production-initial forward on all eligible images (target 10,000
or more after the 12,000-image quality filter). Labels are
formed only after forward:

* strict positive: raw candidate within 6 px and selector dropped it;
* hard negative: raw unmatched and nearest target distance greater than 10 px;
* near/unknown: 6--10 px or selected-unmatched ambiguity, masked from loss.

The cache must retain distance bins 6--10, 10--16, and greater than 16 px in
the audit. The 10--16 px boundary negatives are especially important for
measuring whether a local crop is merely learning proximity; they may not be
discarded as ordinary easy negatives.

The proposed strict-rescue sidecar may consume frozen candidate-observable
features only; target IDs, target distances, graph layers, and profile labels
are audit labels, never inputs. Use one fixed CPU preflight (train split only,
fixed seed/steps/normalization, no threshold or architecture sweep), then
evaluate holdout per style and per profile.

The preflight is admissible only if all of these hold:

* raw candidate reachability (before selector filtering) is at least 0.98 per
  core style; report the all-target ceiling separately so a selector score is
  never credited for atoms that had no candidate;
* core-style conservative precision >= 0.85 and strict recall >= 0.50 for
  coordgen, original, and rdkit independently;
* holdout strict precision >= 0.85 and recall >= 0.50 in every major
  element/size stratum with the minimum counts above;
* protected baseline prefix retention is exactly 100%; near/unknown rows are
  never promoted as positives;
* old style-factorial replay is unchanged, initial SHA is unchanged, and no
  sealed identity is read;
* raw-atomness metrics remain reported separately from strict rescue. A good
  raw-atomness score cannot satisfy the strict-rescue gate.

Any failed style, profile, identity, or prefix gate is a NO-GO. Do not start
GPU training from a failed CPU result. Only after one complete CPU gate passes
should a separate authorization be considered for a single bounded sidecar
shadow; it must write no production checkpoint.
