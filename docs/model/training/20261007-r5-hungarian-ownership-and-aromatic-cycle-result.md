# r5 ownership assignment and aromatic cycle probe

## Ownership policy

The frozen nearest-distance ownership policy was compared with a target-free
one-to-one Hungarian assignment between provisional LSD nodes and frozen
vertex candidates.  The cost is `distance - 8 * candidate_score`, with a 24px
radius.  The configuration was selected on train only.

On all 474 fixed holdout images it improves edge recall from `0.7981` to
`0.8503` at precision `0.9559`; cov13 base/stroke recall rises to `0.8033` and
`0.8052`.  It also reduces dangling edges to `0.00686`.

## Chemistry round-trip

The raw Hungarian graph still has oracle-element sanitize `0.7911`, so it is
not a training or production release by itself.  A target-free cycle probe was
then applied: an edge predicted AROMATIC is demoted to SINGLE when it is not
part of a 5/6-member graph cycle.

This keeps edge P/R at `0.9585/0.8503`, raises bond-type accuracy to `0.9822`,
and raises aggregate oracle sanitize to `0.9219` with zero aromatic sanitize
errors.  Weak cov13 profiles reach `0.9241` (`base48`) and `0.9114`
(`stroke16_48`).  The stronger `cov13_cap52` and `cov33_base` profiles are
just below the 0.90 per-profile floor (`0.8987`), so the rule is still a
shadow-only candidate.

The demotion rule is a diagnostic postprocessor only.  It does not alter the
production graph, selector, checkpoint, or sealed test data.

The aggregate sanitize improvement is real but not yet a release gate: the
full holdout reaches `0.9219`, while `cov13_cap52` and `cov33_base` are each
`0.8987` at the profile level.  The remaining failures are valence errors,
not aromatic-kekulization errors.  A learned critic should therefore be
considered only after a CPU preflight proves it improves those profile floors;
another node-keep or ownership training run is not justified.

Authoritative reports:

- `outputs/audits/chemvector-identity-aligned-line-first-ownership-assignment-r5/report.json`
- `outputs/audits/chemvector-identity-aligned-line-first-ownership-roundtrip-r5/report.json`
- `outputs/audits/chemvector-identity-aligned-line-first-aromatic-cycle-demote-r5/report.json`

Next bounded experiment: CPU preflight an aromatic type-refiner using frozen
edge features/logits plus target-free cycle/degree features.  Only if it
improves the full holdout and profile floors should a single <=800-step GPU
shadow be attempted.

That CPU preflight has now run for 500 steps.  It preserves edge P/R and
passes the fixed 80-image slice, but on the full 474 holdout its oracle
sanitize is only `0.7996` (cycle demotion alone was `0.9219`).  The full-split
gate therefore fails and no second GPU shadow is authorized.  The learned
critic is rejected; retain the deterministic cycle rule as a diagnostic only.
