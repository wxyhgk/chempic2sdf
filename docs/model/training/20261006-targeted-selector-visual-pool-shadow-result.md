# 2026-10-06 targeted selector visual-pool shadow

## Decision

**NO-GO.** The run was a single authorized, in-memory 100-step shadow on one
GPU. It did not write a checkpoint, modify the frozen initial model, or read
sealed data. The production selector remains unchanged.

## Fixed protocol

- release: 320 train images / 160 holdout images;
- strict-positive groups: 359 train / 286 holdout;
- frozen initial checkpoint: `09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`;
- trainable parameters: a 609-parameter group existence head only;
- inputs: frozen candidate features plus two raw-PNG crop features;
- protected baseline prefix: hard locked;
- unknown members: excluded from the loss;
- max steps: 100; wall limit: 1,800 seconds; actual wall time: about 25 seconds;
- existence threshold: fixed at logit 0; no threshold sweep or retry.

## Result

At step 100 on the fixed holdout:

| metric | value |
|---|---:|
| strict groups available | 286 |
| strict groups recovered | 262 |
| strict recall | 0.9161 |
| groups selected | 3,206 |
| confirmed-negative selections | 2,936 |
| unknown selections | 8 |
| known precision | 0.0819 |
| conservative precision | 0.0817 |
| baseline prefix retention | 1.0 |

The final gate failed both precision requirements. The initial checkpoint
hash was unchanged and all other safety gates passed.

## Train-only calibration check

A separate CPU-only diagnostic applied one predeclared rule to the saved
step-100 scores: maximize train strict recall subject to conservative
precision ≥ 0.85, with no holdout fitting. It found **zero feasible train
thresholds**. The top-scored groups were already dominated by confirmed
negatives, so threshold calibration cannot rescue this head; no holdout
threshold was applied.

## Interpretation

This is not evidence that the PNG crop signal is useless. The preceding
read-only combined audit measured holdout AUC 0.9346 for the fixed
`atom_score + largest_component` combination. The failure is stronger than a
simple threshold choice: the learned existence head did not rank strict
positives above confirmed negatives even on train at the required precision.
The roughly 1.3% positive group rate and large positive weight made the head
recall-oriented, while the fixed zero-logit gate emitted thousands of
confirmed-negative groups. A high separability AUC from a different scalar
diagnostic therefore did not translate to a learned group head.

## Consequence

Do not promote, retry, extend, or threshold-sweep this run. The next admissible
work is a redesign review, not another GPU run: use the positive/negative
imbalance and group construction to define a rank-aware, class-prior-aware
objective, then run a new read-only CPU preflight before requesting any new
authorization.
