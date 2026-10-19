# 2026-10-02 chemical-head trust-region objective

The r3 replay showed that pair presence stayed fixed while chemical logits
crossed aromatic/non-aromatic decoder boundaries. A new opt-in objective is
implemented in `chemical_head_stage.py`: probability-space KL distillation
from the initial chemical head, applied to the sampled positive and negative
rows. It is added only when `trust_region_weight > 0`; the default remains
zero, so existing behavior and the completed r3 receipt are unchanged.

The probe wiring snapshots the initial chemical linear head and can pass its
reference logits to the loss. The current constant is deliberately
`TRUST_REGION_WEIGHT = 0.0`; no new training run has been launched.

The objective is a local constraint, not an identity oracle: it cannot prove a
candidate molecule correct and must still be judged on the fixed 944-image
development gate (including zero new fallback and no exact regressions).

Tests: `8 passed` for the chemical-head stage/probe subset. A future run must
use a new preparation/authorization receipt binding the nonzero weight and
compare against initial on the same fixed cohort.
