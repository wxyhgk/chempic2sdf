# r17 pair-incidence strict shadow (2026-10-08)

This CPU-only shadow used the new 32-molecule train / 16-molecule holdout
cohort after excluding the r17 plan, materialized caches, and slice80.  Train
and holdout are disjoint by canonical, connection, and scaffold identity.

The frozen pair candidate set was retained and only a 600-step incidence
residual was fitted.  Holdout edge recall improved `.95349 -> .99459` and
typed recall `.93456 -> .97296`; however typed precision fell `.97517 ->
.96825`.  The per-profile precision gate also failed (oa13 and oa22 exceed the
allowed 1pp drop).  Therefore this is evidence that oa13/oa33 are presence
recall bottlenecks, not an integration candidate.  Next work, if any, is
train-only precision calibration/trust-region clipping with the same strict
per-profile gates; no GPU or production merge is authorized.
