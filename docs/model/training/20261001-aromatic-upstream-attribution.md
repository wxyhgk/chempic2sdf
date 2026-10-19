# 2026-10-01 r3 upstream aromatic attribution

The three new r3 fallback witnesses were replayed on the fixed initial and
1200-step chemical-head arms. Saved prediction parity passed for all six
forwards. No backward pass, optimizer step, deployment change, or sealed-test
read occurred.

Report: `outputs/audits/chemvector-aromatic-upstream-attribution-20261001/report.json`.

Findings:

- Pair presence is not the cause: selected-pair signatures are identical in
  all three cases.
- `001401_coordgen`: one selected edge changed `AROMATIC → DOUBLE` and the
  molecule became non-kekulizable.
- `001866_stress`: one selected edge changed `AROMATIC → DOUBLE`; another
  remained aromatic but its probability moved down. The molecule became
  non-kekulizable.
- `001833_original`: three edges lost aromatic promotion, so the accepted
  cycle count changed `1 → 0`; this also caused a kekulization failure.

The regression is therefore in chemical bond semantics / aromatic promotion,
not atom selection or pair presence. Since the local one-edge repair replay
found no identity-correct candidate for any of these three images, the next
experiment should be a bounded chemical-logit attribution or decoder-policy
diagnostic—not another full training run and not production fallback wiring.
