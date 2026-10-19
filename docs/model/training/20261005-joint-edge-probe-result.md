# 2026-10-05 conservative joint edge probe

The approved 400-attempt bounded trial was launched from the unchanged
`initial.pt` checkpoint. It trained only the two linear pair readouts:

- `pair.presence.weight/bias`
- `pair.chemical.weight/bias`

The pair shared trunk, vision/atom/bond/selector path, OCR, role and bbox
parameters were frozen. The run used FP32 CUDA, AdamW, learning rate `1e-5`,
the fixed released training order, and the masked objective with replay and
aromatic protection. No sealed-test content was read and no default artifact
was changed.

## Early-stop result

The fixed 100-attempt checkpoint failed the hard regression gate, so the worker
stopped immediately instead of spending the remaining 300 attempts:

| cohort | initial exact | after 100 | initial valid | after 100 | fallback |
|---|---:|---:|---:|---:|---:|
| full molecule (800) | 170 | 169 | 552 | 551 | 51 → 51 |
| curriculum (144) | 63 | 64 | 138 | 138 | 0 → 0 |

There was one broken exact case (`fresh_dev_001966_coordgen`) and one rescued
curriculum case. New fallbacks were zero, but the full-molecule valid count
dropped by one, which is sufficient for **NO-GO**.

The production decision is therefore **keep initial**. The 100-step delta is a
research artifact only and is not deployable or resumable.

## Integrity

- Training attempts/updates: 100/100
- Evaluation: fixed 944-image development cohort
- Frozen non-head digest: unchanged before/after
- Sealed-test reads: 0
- Deployment authorization: false
- Runtime: approximately 125 seconds

Run artifacts:

`outputs/experiments/chemvector-joint-edge-probe-20261005-400/`

The result rules out this exact two-head recipe at its first checkpoint. Do not
retry with more steps, change the threshold, or unfreeze the shared trunk based
on this run. The next useful work is diagnosis of the broken full-molecule
case and atom-endpoint coverage, not another blind training pass.
