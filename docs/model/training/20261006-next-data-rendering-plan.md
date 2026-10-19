# Next route after endpoint/selector NO-GO (2026-10-06)

The frozen production endpoint path has now failed two independent bounded
trials, and the bond-supported selector attribution has only 24 strict train
positives.  Repeating heatmap capacity, offset, threshold, or selector-MLP
experiments is therefore not justified.

The next work unit is data admission and read-only replay.  It should not be
mixed into a GPU run:

1. Reuse the existing v0.2 quarantine bundle first: it already contains
   20,904 images and 3,484 molecules.  Its style audit shows abundant
   `font_large`/`serif` O/N and implicit-C coverage; only O-aromatic and
   N-double are sparse.  Do not create another generic 3,600-image batch.
2. If a sparse cell must be filled, generate only a small quarantined pilot
   with identity-disjoint molecules and the existing renderer contracts.
   Prioritize visible O/N labels and the sparse bond cells; keep wedge/dash as
   a separate stress stratum.
3. Before training, validate every PNG/layer/SDF identity and report exact
   counts by `(style, role, element, bond type, functional group)`.  Do not
   promote the slice or expose sealed identities.
4. Run a frozen-initial forward-only replay on a new identity-disjoint split.
   Admission requires raw/selected endpoint recall improvement, no baseline
   prefix loss, and a strict residual precision proxy of at least 0.85.  Only
   then is one bounded training trial allowed.

The replay was completed on the fixed 2,619-row proposed validation panel.
It found 19,724/19,926 raw endpoints (98.985%) and 19,292/19,926 selected
endpoints (96.817%); 432 raw-matched endpoints were dropped by the selector.
Loss is concentrated in `font_large`/`serif`, implicit carbon, and triple-bond
incident endpoints.  This is diagnostic evidence only, not a training release.

An identity-disjoint planner check also found that the current
`case_diversity_v1` catalog has only four molecules left after excluding both
the admitted corpus and the v0.2 quarantine identities, versus the 225 needed
for the proposed targeted pool.  It therefore emits an explicit NO-GO plan
instead of reusing quarantine molecules.  A new targeted pool requires a new
identity source/catalog; no GPU training is justified until that pool passes
the same split and label checks.

The existing 12,239-image case-diversity bundle is broad (768 structures and
16 style variants), but its audit explicitly did not measure predicted
candidate reachability.  More generic images alone are therefore not the
next experiment; the targeted strata above must be measured against the
endpoint failure taxonomy first.
