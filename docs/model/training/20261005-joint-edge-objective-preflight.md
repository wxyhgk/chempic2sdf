# 2026-10-05 conservative joint edge objective preflight

Following the edge-label reachability audit, a standalone objective and a
read-only gradient preflight were added. They are not wired into a training
launcher and cannot change a checkpoint.

## Contract

`chemvector/training/curriculum/joint_edge_objective.py` enforces four rules:

1. confirmed non-edge rows supervise only the presence logit;
2. confirmed positive rows supervise chemical type;
3. unknown rows supervise neither output; and
4. initial replay and the asymmetric initial-aromatic anchor are explicit
   protected terms; the target-ring floor remains a separate safety term.

The implementation rejects malformed masks, non-finite logits, and missing
replay/context tensors before producing a loss.

## Preflight

Tool: `tools/audit_joint_edge_witness_gradients.py`  
Report: `outputs/audits/chemvector-joint-edge-witness-gradients-20261005-r2/report.json`

The check used three fixed released training rows (not the sealed test set):

| witness | known pairs | positive | confirmed non-edge | loss |
|---|---:|---:|---:|---:|
| `curr_9b54d301940543f8__font_large` | 435 | 34 | 401 | 0.1813 |
| `curr_75deb03a71d9a1d3__font_small` | 1,378 | 59 | 1,319 | 0.2423 |
| `curr_e6354f1e02efa8bd__reference` | 1,326 | 58 | 1,268 | 0.3972 |

This last witness also contains 52 unknown pairs; both unknown gradient maxima
are exactly zero, confirming that the mask is active rather than vacuously
passing on an all-known batch.

All gradients were finite. Unknown-gradient maxima were exactly zero for both
heads on every witness. No optimizer, checkpoint write, or parameter update
occurred.

The three aromatic fallback witnesses (`001401_coordgen`, `001833_original`,
`001866_stress`) remain validation-only and carry no training graph labels in
the release interface. They are therefore not converted into pseudo-labels;
they remain fixed regression witnesses for the 944-image gate.

## Decision

The objective is technically reachable and mask-safe, so a future bounded trial
may be prepared. This is not yet a training authorization or GO decision. The
trial must first bind the exact train-row cache, freeze the intended parameter
set, and retain the existing gate: no full exact/valid regression, no new
fallback, and no curriculum exact regression.
