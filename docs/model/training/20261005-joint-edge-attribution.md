# 2026-10-05 joint-edge failure attribution

This is a read-only attribution of the first rejected joint-edge probe. It
does not read the sealed test set, does not update a checkpoint, and does not
change the production baseline.

## 1. The broken development case

The only broken exact case at the 100-step checkpoint is
`fresh_dev_001966_coordgen`. The initial model is exact and valid; the
100-step research delta is invalid with:

`Explicit valence for atom #40 O, 3, is greater than permitted`.

The upstream prediction is unchanged:

- selected atoms: 68;
- pair candidates: 2,278;
- OCR/role/box path: unchanged;
- the presence threshold mask changes by exactly one pair.

The responsible pair is candidate slots `[41, 65]`:

| quantity | initial | joint-100 |
|---|---:|---:|
| presence probability | 0.847902 | 0.851724 |
| presence decision (`>= 0.85`) | false | true |
| chemical class | AROMATIC | AROMATIC |

The compiler maps this newly accepted relation to scene atoms `(62, 40)`.
Atom 40 is the OCR-resolved `OH` atom (`element=O`, explicit H=1), and it
already has one single bond to atom 48. The new aromatic bond gives O a third
valence contribution and RDKit rejects the graph. The initial scene contains
only `(48, 40)` at that location and sanitizes successfully.

As a controlled replay, removing only the newly added `(62, 40)` edge (and
its two incident-port records) makes the research graph valid and restores the
same canonical SMILES as the expected molecule. This confirms that the single
edge is sufficient to explain the regression; it is not merely correlated
with an unrelated compiler failure.

The six other chemical-argmax changes occur on pairs whose presence remains
far below threshold (approximately `2e-5` to `1e-3`), so they cannot explain
the invalid graph. The failure is therefore a threshold-crossing false edge
plus missing deterministic valence protection, not OCR drift, atom selection
drift, or aromatic-cycle promotion drift.

Inputs used for this replay:

- initial checkpoint SHA-256: `09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`;
- research delta SHA-256: `2a56923109ce30a2cf011279d16a6961041c931d85ef03e444e76d24fb2986ab`;
- validation cohort only; sealed-test reads: 0.

## 2. The separate training-label reachability problem

The fixed 1,200-image edge-label audit reports:

| quantity | count |
|---|---:|
| candidate pairs | 636,354 |
| confirmed positive pairs | 38,583 |
| confirmed non-edge pairs | 584,119 |
| unknown pairs (excluded from both losses) | 13,652 |
| unreachable target bonds | 793 |

Of the 793 unreachable bonds, 783 are in `full_molecule` and 10 in the
curriculum subset. The examples show an endpoint-coverage failure rather than
a chemical-head-label shortage:

- `curr_0c1e83af2dca5083__line_thick`: 42 selected of 50 target atoms,
  21 unreachable bonds (14 aromatic, 7 single);
- `curr_f92f2304ee73e156__reference`: 12 selected of 20 target atoms,
  12 unreachable bonds, despite zero unmatched selected atoms;
- `curr_c6e1739df549e34a__font_large`: 58 selected of 57 target atoms,
  7 unreachable bonds, mostly C-N/C-O;
- `curr_f0dd6cb41b7c4ec7__serif`: 65 selected of 60 target atoms,
  369 unknown pairs but only one unreachable bond.

Thus “unknown-pair ambiguity” and “missing atom endpoint” must not be fixed
by the same pair-head update. The existing masked objective correctly avoids
using unknown pairs as negatives, but it cannot create a target bond whose
endpoint was never selected.

## Decision

The joint presence+chemical recipe remains **NO-GO**. Do not extend it to 400
steps, change its threshold, or unfreeze the shared trunk. The production
baseline remains the unchanged initial checkpoint.

The next engineering experiment should be a CPU/read-only replay of a
deterministic valence-safe edge acceptance rule on the fixed public validation
cohort. Separately, endpoint coverage should be measured and improved at the
atom selector; it should not be mixed into another chemical-head trial.
