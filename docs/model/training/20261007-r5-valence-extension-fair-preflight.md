# r5 valence extension: fair CPU preflight

## Scope

This is a quarantine-only A/B of the identity-disjoint valence/topology
extension.  It does not read the sealed test, write a production checkpoint,
or modify the selector, OCR, bond head, or graph route.  The production
initial checkpoint remains SHA256
`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`.

The comparison uses the exact r5 `EdgeBondMLP`, feature contract, random seed,
Adam learning rate, 800 optimizer steps, 160-image fixed train slice, and
train-only threshold calibration in both arms.  The old arm has the fixed
160-image slice; the second arm appends 216 identity-disjoint extension
images.  Both arms are replayed on the same 80-image diagnostic slice and the
complete 474-image fixed holdout.

## Result

| arm | full-holdout edge precision | graph-bond recall | typed accuracy |
|---|---:|---:|---:|
| old-only | 0.924724 | 0.881262 | 0.961053 |
| old + extension | 0.922355 | 0.881571 | 0.992283 |
| delta | -0.002369 | +0.000309 | +0.031231 |

The recall delta is one additional recovered bond out of 3,234.  Precision
falls because the augmented model's calibrated threshold selects essentially
all candidate edges (3,091/3,091) on the fixed holdout.  This is consistent
with the extension's positive-candidate rate (0.9627) being higher than the
holdout rate (0.9224), and with the fixed line proposal ceiling: no classifier
can recover a bond absent from the target-free proposal set.  Weak profiles
remain the same or lower in precision.

## Decision

`gpu_training_admissible = false`.  Do not spend the next 800-step GPU shadow
on this extension/edge-classifier recipe.  Keep the extension in quarantine
for future topology/ownership work.  The next admissible experiment must
change proposal ownership or endpoint pairing and first pass a CPU preflight
with strict full-holdout and profile-level improvements.  The deterministic
valence/topology guard is a separate CPU post-processing candidate; it should
not be conflated with this failed training A/B.

Evidence:

- [fair A/B report](../../outputs/audits/chemvector-identity-aligned-line-first-valence-extension-fair-r5/report.json)
- [runner](../../../tools/preflight_line_first_valence_extension_fair_r5.py)

