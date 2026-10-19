# Offset-aware geometry route after multiplicity shadow failure

## Decision

The dense-cell multiplicity shadow is a **NO-GO** for integration: holdout
precision was `0.03785` (required `>=0.85`) despite recovering 17 collision
targets.  The failure is caused by globally appending extra heatmap peaks;
the shadow has no text-role, extent, ink, or attachment-offset admission and
therefore admits mostly line/background peaks.

Do not retry the extra heatmap, sweep its threshold/capacity, or run the old
targeted-selector coordinate refiner.  The latter operates on a different
release and a raster-centroid correction contract, not the current cap/stroke
collision panel.

## Smallest viable route

Keep the frozen baseline candidate prefix byte-for-byte.  Only inspect a
small, fixed collision-cell pool and admit a residual candidate when all of
the following frozen fields pass one pre-registered geometry gate:

1. predicted text-role probability `>= 0.20` (one registered probe; the
   frozen dense field has p(TEXT) median about `3e-4`, so `0.55` produces an
   empty residual set and is not a useful gate);
2. local heatmap peak prominence `>= 0.02`;
3. raw-pixel ink fraction inside the predicted text bbox `>= 0.05`;
4. predicted text extents in `[2, 96]` px on both axes;
5. normalized attachment offset
   `max(abs(anchor - text_center) / (text_extent + 2 px))` in `[0.20, 1.75]`;
6. 6 px predicted-anchor duplicate suppression and at most 16 residuals per
   image.

These conditions are target-independent.  The candidate's target identity,
GT distance, and graph labels are used only after selection for an audit.
Existing model outputs already expose the necessary frozen geometry:
`anchors`, `text_centers`, `text_extents`, `role_logits`, and heatmap logits in
`SpatialAtomObjectDetector`.  `protected_residual_merge.py` supplies the
immutable-prefix merge contract.  No new head or training is needed for this
preflight.

Implementation:

- policy and selector:
  `chemvector/components/graph/proposals/offset_geometry_gate.py`;
- CPU-only evaluator:
  `tools/audit_offset_geometry_cpu_gate.py`.

The evaluator intentionally fails closed until a new frozen replay exports a
feature cache.  Required cache fields are the frozen geometry/pixel features
listed above plus `sample_key`, `split`, `profile_id`, `candidate_id`,
`protected`, and predicted anchor coordinates.  Forbidden fields include
`target_id`, any GT distance, layers, graph, or OCR text.

## Fixed CPU gate

Run one cache, one policy, and one split (no threshold search).  Report each
`cov13/cov33 × base/cap/stroke × train/holdout` stratum and aggregate:

- conservative precision (protected + strict-positive / all selected)
  `>= 0.85`;
- hard-negative survival `<= 0.15`;
- exact protected retention `1.0`;
- holdout strict-positive support `> 0`;
- at least one holdout strict-positive is actually selected and the residual
  admission is non-empty (an empty set is an automatic failure);
- residual count `<=16` per image and no baseline loss;
- strict collision recovery compared with the zero-residual baseline.

Failure of any stratum is an immediate NO-GO.  Passing this CPU gate still
only authorizes a separately reviewed, bounded in-memory shadow; it does not
authorize selector, bond, graph, or production training.
