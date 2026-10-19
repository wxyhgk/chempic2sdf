# Port-conditioned endpoint objectness sidecar (bounded trial)

This is the next trainable experiment after the 2026-10-08 bond-port
reachability audit.  It is intentionally isolated from the production
selector: `initial.pt` and its bond detector are frozen, and residual rows are
never passed into the old set head.  A sidecar row represents one endpoint of
an already predicted frozen bond.  The merge may claim an unoccupied port only
once, with at most 16 accepted claims per image.

The launcher is `tools/train_port_endpoint_objectness_sidecar.py`.

## Fixed protocol

- cohort: the public 1,200-image initial cohort only;
- split: `split_group_disjoint(rows, 480)`, requiring exactly 480 train and
  720 holdout rows and zero group/molecule/canonical-SMILES overlap;
- labels: residual endpoint objectness within 6 px of an unmatched atom;
  target atom IDs are used only to construct labels, never as features;
- features: frozen bond feature/type/score, endpoint coordinate and side,
  and distance to the nearest baseline atom;
- sidecar: MLP with fewer than 50,000 parameters, maximum 800 optimizer steps;
- threshold: fixed grid selected from train labels only;
- outputs: CPU candidate cache and audit report only.  No production
  checkpoint is written and sealed-test content is not read.

## Admission gate

The trial is admissible only if train positives ≥50, train AUPRC ≥0.50,
holdout precision ≥0.85, holdout new matches ≥14, baseline prefix retention is
100%, and all split/ownership invariants pass.  Failure is a NO-GO; threshold
or MLP-capacity sweeps are not authorized.
