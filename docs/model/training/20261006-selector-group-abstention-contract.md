# Group selector v1: existence, ranking, abstention

## Scope

This is a CPU mathematical contract only. It does not load a checkpoint or a
dataset, read sealed evaluation data, fit parameters, run an optimizer, or
authorize training. The existing exact-one prototype remains unchanged and is
still rejected by its open-neighborhood counterexample.

Implementation: `tools/selector_group_abstention.py`.

## Why the previous shadow was invalid as a selector

The bounded shadow trained one candidate scorer and then always took an
`argmax` for every non-protected group. A confirmed-negative group therefore
had to emit a winner whenever its largest sigmoid crossed 0.5. On the fixed
130-image release this produced 1,096 confirmed-negative selections and
holdout known precision 0.0126 (the production checkpoint was unchanged).

## Contract

For group `g`, candidates have logits `s_gi` and a separate group existence
logit `e_g`.

* Strict-positive group: `y_g=1`; existence contributes
  `BCEWithLogits(e_g, 1)`. Reviewed positive candidates may be ranked against
  explicitly reviewed negative candidates with
  `softplus(m + max(s_negative) - s_positive)`. Unknown alternatives are not
  silently converted to negatives.
* Confirmed-negative group: `y_g=0`; it contributes only
  `BCEWithLogits(e_g, 0)`. It has no ranking term and no winner target.
* Protected baseline group: hard locked; no rescoring and no loss.
* Excluded group: requires an auditable reason and contributes no loss.

Inference first checks `e_g >= existence_threshold`. If this fails, it returns
`abstain_existence` before evaluating `argmax`. Otherwise it requires the
winner score and winner-vs-runner-up margin gates. Thus a negative group cannot
be forced to choose its highest candidate, and ties abstain rather than use a
hidden pseudo-label.

The implementation returns `training_authorized=False` and performs no update.
The synthetic contract tests cover negative-group no-winner behavior,
unknown-member zero gradient, protected-prefix locking, margin abstention,
input validation, and absence of file/model/optimizer access.

## Current recommendation

Do **not** start another GPU run from this contract. The old cache has 51 train
and 16 holdout strict positives but 8,635 / 2,880 confirmed-negative groups;
the cache audit reports the best existing per-member AUC around 0.85, not a
group-level calibrated existence head. The next admissible experiment would
require a new CPU readiness replay that supplies a dedicated existence head,
uses train-only threshold calibration, and proves holdout false-positive and
abstention gates before any bounded training. The 100-step pointwise shadow
already failed this requirement, so it must not be retried with threshold
sweeps.
