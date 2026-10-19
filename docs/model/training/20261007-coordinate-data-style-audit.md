# Coordinate release r2: data/style and label-strata audit

This is a CPU/read-only audit of the public r2 coordinate release. It does
not read sealed test data, write a checkpoint, or authorize a GPU run.

## Source and counts

Source: `outputs/datasets/chemvector-targeted-selector-coordinate-release-20261007-r2`

| split | images | profiles | strict | near-unknown | confirmed hard | total candidate rows |
|---|---:|---|---:|---:|---:|---:|
| train | 320 | 160 oa11 + 160 oa13 | 359 | 1,121 | 27,305 | 28,785 |
| holdout | 160 | 80 oa11 + 80 oa13 | 286 | 655 | 13,985 | 14,926 |

`near-unknown` means the nearest atom is at most 10 px but the candidate is
not a strict-positive member. It must not be placed in a confirmed-negative
loss. Hard negatives are members explicitly marked confirmed-negative and
more than 10 px from every rendered atom.

## Style coverage

The profile split is numerically balanced, but style support is narrow:

- only DejaVu Serif;
- font sizes 18 and 32 px, rotation 0 and 180 degrees;
- bond length caps 64 and 28 px, mostly line widths 2.4 and 3.2 px;
- no independent font family, anti-aliasing, scale, stroke, or rotation
  factors are represented.

The two profiles have sharply different candidate strata:

| split/profile | strict | near-unknown | hard | near+strict fraction |
|---|---:|---:|---:|---:|
| train / oa11 | 145 | 264 | 14,565 | 2.73% |
| train / oa13 | 214 | 857 | 12,740 | 7.75% |
| holdout / oa11 | 111 | 178 | 7,586 | 3.67% |
| holdout / oa13 | 175 | 477 | 6,399 | 9.25% |

Here oa13 is the 32 px, 180-degree, short-bond, thicker-line profile. This
is a style-conditioned geometry shift, not a sample-count imbalance.

Other coverage gaps are structural: rare elements are represented by only
2--20 molecules (P=4, F=6, B=12, Si=20 images), and common functional-group
counts dominate (amine=180 image occurrences). Phenol has only 2, while
nitrile and sulfonamide have 12 each.

Raw-distance tails also differ by profile. In train, hard rows in 10--12 px
are 325 (oa11) and 427 (oa13); in holdout they are 183 and 189. These are
the first reviewed hard-negative boundary cases to add, because they are
closest to a candidate acceptance error.

## Low-cost CPU checks

The existing fixed 300-step near-target gate remains the only admissible
candidate gate: holdout refinable precision 0.893 and recall 0.579. Training
separate heads per profile gives holdout P/R of 0.922/0.533 (oa11) and
0.893/0.664 (oa13); a global head gives 0.893/0.579. Thus a profile bit does
not solve the problem, although profile-stratified reporting is required.

A strict-only diagnostic trained on strict positives versus hard negatives,
with near-unknown rows ignored, gives holdout known precision 0.892 and
strict recall 0.517 at a fixed zero-logit gate. This is diagnostic only:
near-unknown rows remain unlabelled, so the number cannot be reported as
strict deployment precision.

## Recommendation

Do not add more copies of the current two profiles and do not increase hard
negative volume indiscriminately. Build the next identity-disjoint release
with a factorial style grid (at least three font/size/rotation/stroke
variants, including both oa11-like and oa13-like conditions), and pair each
new molecule with every style. For each style, deliberately include strict
positives (<6 px), near-unknowns (6--10 px), and reviewed boundary negatives
(10--16 px). Oversample rare atoms, double/triple/aromatic bonds, and rare
functional groups rather than adding more amine/heteroaromatic examples.

Before any GPU authorization, run a CPU-only, fixed-gate audit stratified by
style and by distance band. Required metrics are strict-only precision,
near/refinable precision, strict recall, and abstention rate; no threshold
scan and no sealed test access. A new GPU run is justified only if every
style stratum reaches the predeclared precision/recall floors.
