# r17 t3f type-correction CPU result (2026-10-20)

## Scope

The t3f quarantine source contains 24 train and 12 holdout molecules, with
independent aromatic/ring scaffolds on both sides.  The production checkpoint
is the frozen initial checkpoint
`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`.

The experiment trains only a small four-class bond-type residual.  Presence
selection, atom proposals, candidate pairs, and the production model remain
frozen.  The residual sees target-free pair features; labels are used only for
the train-side loss.  No sealed test payload is read and no production
checkpoint is written.

## Results (corrected contract)

Corrected report: `outputs/audits/chemvector-r17-t3f-type-correction-cpu-v2/report.json`.

The first implementation reported a small gain, but audit found that its
training loss used the new head alone (`head(x) + chemical_logits * 0`) while
inference added frozen `chemical_logits` to that head. That was an invalid train/inference contract;
the first report must not be used as evidence.

The corrected implementation uses exactly
`chemical_logits + type_residual(features)` for both loss and replay.  At 300
CPU steps and presence threshold 0.95, the disjoint holdout is:

| profile | typed precision (base → residual) | typed recall (base → residual) | edge P/R |
|---|---:|---:|---:|
| oa13 | 0.8913 → 0.8913 | 0.9011 → 0.9011 | unchanged 0.9674 / 0.9780 |
| oa33 | 0.9540 → 0.9540 | 0.9121 → 0.9121 | unchanged 1.0000 / 0.9560 |

The corrected result is therefore a no-gain shadow.  A cross-source replay on
the independent t3e holdout is mixed (oa13 typed precision −1.01pp, oa33
+4.35pp), so it also fails the per-profile precision gate. The earlier
positive reports are historical artifacts, not executable regression tests.

## Decision

The narrow route is not yet validated.  It remains a possible research
direction only after a larger, scaffold-stratified fresh source and a fully
contract-matched objective produce a simultaneous per-profile gain.  No GPU
trial or production merge is justified.  Global valence projection and greedy
budget repair are rejected because their independent probes reduced typed
accuracy.
