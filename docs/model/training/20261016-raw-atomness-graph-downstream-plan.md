# Raw atomness sidecar → frozen graph downstream plan (2026-10-16)

## Decision

The raw atomness sidecar remains an isolated experiment.  It may rank or
abstain on non-protected raw candidates, but it cannot create an atom, modify
the frozen selector, or alter the production checkpoint.  The candidate cache
contains only four scalar proposal features; it does **not** contain the
per-image frozen bond/pair outputs needed to claim graph improvement.

The read-only contract audit is:

```text
tools/audit_raw_atomness_graph_downstream_contract.py
```

Its current report must remain `graph_downstream_training_ready=false` until a
separate replay artifact is produced.

## Required interface

For every `sample_key`, a frozen replay must provide:

- atom points and the immutable protected/selected mask;
- bond endpoint coordinates, bond presence logits, bond type ids, and bond
  validity mask;
- an explicit binding to the frozen `initial.pt` SHA and candidate-cache SHA.

The sidecar consumes only `raw_score`, `presence_logit`, `x_norm`, and
`y_norm`.  `nearest_distance_px`, target atom ids, layers, graph labels, and
any target-derived assignment are audit-only and forbidden as inputs.

## Evaluation gate

The first graph report must be independent of training and report, per core
style and separately on stress: raw atom precision/recall, selected atom
count, bond-endpoint reachability, connectedness, RDKit sanitize rate, exact
graph/component recovery, and abstention counts.  A good atomness score alone
is insufficient.  No graph sidecar training, checkpoint save, threshold sweep,
or deployment is admitted before this report exists.

## Next action

Replay the frozen bond/pair outputs once, aligned by `sample_key`, and run the
contract audit with `--graph-artifact`.  If the schema and identity bindings
pass, write one graph-level dev/stress report.  Otherwise stop and fix the
interface; do not spend GPU on another rescue architecture.
