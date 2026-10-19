# Visible O/N data-first pilot (2026-10-17)

## Why this is the next step

The offset-aware residual gate was replayed on the frozen initial checkpoint.
It retained every protected baseline candidate (`1.0`) and selected no strict
positive residual (`0 / 1,056`).  That is a valid CPU negative result, not a
reason to spend more GPU on threshold tuning or another selector.  The
remaining evidence points to a training-data and anchor-supervision problem:
the model sees visible glyph ink, but its heatmap target is the chemical
attachment anchor and the two are often separated by several pixels.

The pilot therefore changes the data distribution first and keeps the model
change deliberately small.  It is not a new decoder, OCR system, MoE route,
or coordinate-refiner.

## Frozen data contract

`tools/prepare_visible_on_data_first_pilot.py` reads only the public
`chemvision_layers_v0_2/font_pilot_10k` rendered shards.  It writes a manifest,
not a new checkpoint or model input.  All four views of a molecule remain in
the same side of the split.  The 1k manifest was produced at:

`outputs/audits/chemvector-visible-on-data-first-pilot-20261017-r1/manifest.json`

The manifest has 200 molecule groups / 800 train images and 50 molecule groups
/ 200 holdout images.  The molecule identities are disjoint, and every selected
sample is accepted by the production v0.4 target serializer.  A CPU audit is
recorded at `outputs/audits/chemvector-visible-on-data-first-pilot-20261017-r1/audit.json`.
It contains:

* train visible O/N: 9,952 / 5,792;
* holdout visible O/N: 1,652 / 892;
* train implicit-C replay count: 29,450;
* 12 font families in both sides, all four size buckets, and negative,
  near-zero, and positive rotation buckets;
* equal exposure of the four renderer variants per molecule.

The manifest contains provenance and audit summaries only.  Target IDs,
coordinates, graph objects, and nearest-target distances are not provided to
the model.

## 1k pilot (the only GPU candidate now)

1. Run a read-only target reachability audit on the manifest and the frozen
   initial checkpoint.  Report `oa13` and `oa33` separately, then split O/N by
   element, degree, font family, size bucket, rotation bucket, and variant.
2. Keep the existing atom decoder and one-to-one anchor ownership.  Do not
   turn text-center into an atom candidate.
3. Add only an optional, low-weight attachment consistency term to the atom
   supervision.  The chemical anchor remains the primary heatmap positive;
   the visible text center is auxiliary geometry.
4. Freeze the vision trunk.  At most the atom head and the final one or two
   vision blocks may update; use atom-head learning rate `1e-5`--`2e-5` and
   trunk rate at most `1e-6`.
5. Cap the run at 400--800 steps.  Evaluate the same fixed holdout every 100
   steps; no checkpoint or threshold selection is allowed.

## Promotion gates

The 1k result is admissible only if all of the following hold:

* visible O/N raw-gap recovery improves over initial on holdout and on both
  `oa13` and `oa33`;
* no core O/N stratum regresses;
* implicit-carbon recall drops by at most 0.5--1.0 percentage points;
* protected baseline retention remains exactly `1.0`;
* the fixed holdout, not merely train, passes;
* stress profiles are reported separately and cannot hide a core failure.

Any failure stops the route.  We do not increase steps, alter thresholds, or
open a 5k run after a failed gate.

## 5k expansion (only after 1k passes)

The same script supports 1,000 train molecule groups / 250 holdout groups,
which yields 4,000 / 1,000 images.  The 5k manifest must preserve identity
disjointness and the same style strata; it is an expansion of diversity, not
four copies of one SVG.  Its target minimum is 2,000 visible O and 2,000
visible N in train, and 750 of each in holdout.  Only after the 1k gates pass
may a 1,200--2,000 step atom-head pilot be considered.

## Current decision

The 1k manifest is ready.  No production checkpoint has been written and no
GPU training has been authorized by this document.  The next concrete action
is the frozen read-only reachability audit; training starts only if that audit
confirms the manifest is consumable and the baseline retention contract is
intact.
