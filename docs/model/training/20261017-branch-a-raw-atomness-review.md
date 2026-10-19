# Branch A raw-atomness sidecar review (2026-10-17)

## Decision

Branch A remains a **frozen, independent sidecar candidate**.  It is not a
replacement for the atom/text path and it is not currently admissible for a
new training run.  The existing bounded shadow is sufficient to show that the
sidecar can separate confirmed raw atom proposals from confirmed hard
negatives.  It does not show that the sidecar can recover the current
large-serif visible-text gap or insert candidates safely into a molecular
graph.

No training, checkpoint write, deployment, or sealed-test read is authorized
by this review.

## Evidence already available

The fixed raw-atomness shadow used only frozen scalar features
`raw_score`, `presence_logit`, `x_norm`, and `y_norm`.  Protected candidates
were immutable positives; `hard_negative` was the only negative; unknown
strata were masked.

| split | conservative precision | raw-atom recall |
|---|---:|---:|
| coordgen | 0.9809 | 0.9903 |
| original | 0.9804 | 0.9883 |
| rdkit | 0.9810 | 0.9903 |
| stress (separate) | 0.9291 | 0.9692 |

The initial checkpoint was unchanged and no checkpoint was written.  These
numbers validate raw proposal atomness only; they do not validate selector
rescue, text recognition, bond endpoint assignment, or graph validity.

The latest downstream contract is explicitly incomplete because no aligned
frozen bond/pair artifact was supplied.  Consequently graph-level metrics
have not been measured and `graph_downstream_training_ready` is false.

## Consistency with the current gap attribution

The latest frozen replay finds 2,280 raw atom gaps over 3,780 images.  The
oa13 and oa33 stress profiles account for 1,012 and 1,241 gaps respectively;
96.0% and 99.7% are visible-text gaps.  The dominant cases are large-serif
visible O/N (especially O degree 1/2 and N degree 2), while implicit carbon
is nearly reachable.

Branch A is therefore consistent only at the upstream candidate-visibility
level.  Its cache labels are not stratified by the missing atom's element,
visible-text status, font family, or oa13/oa33 profile.  In particular, the
high precision is dominated by protected raw candidates and cannot be read as
evidence for recovery of missing visible O/N atoms.  A sidecar that scores
existing raw proposals also cannot create a candidate when no usable raw
proposal is present.

## Minimum coverage gate before any future training decision

Run a read-only replay with the same frozen initial and a separately aligned
bond/pair artifact.  The report must include, independently for oa13 and oa33
and for large/serif strata:

1. raw-gap recovery for strict visible-text O/N targets (not aggregate atom
   recall);
2. conservative precision at least 0.85 and hard-negative survival at most
   0.15;
3. exact protected retention (`1.0`), with no baseline candidate replacement;
4. raw and selected atom precision/recall and selected-count delta;
5. frozen bond-endpoint reachability and valid/sanitize/connected graph rates;
6. a disjoint style/identity holdout, with stress reported separately.

Ground-truth distances and target IDs may be used only for post-hoc labels,
never as sidecar inputs.  `near_unknown` and `selected_unmatched` remain
masked and protected candidates are never negatives.

The sidecar may proceed to a bounded CPU admission-head experiment only if
the coverage gate passes.  A graph-level failure, protected-retention
failure, or either oa13/oa33 visible O/N recovery failure is an immediate
NO-GO and does not justify GPU training.

