# r5 line-first valence/topology guard probe

This CPU-only probe evaluates a target-free guard after the frozen Hungarian
ownership and line/bond sidecar. The proposal stage sees only LSD geometry,
candidate score, edge score, bond type, and graph topology. Element and bond
ground truth are joined only after proposal construction for the RDKit audit.
The production checkpoint and sealed test are untouched.

The guard first applies the existing 5/6-cycle aromatic demotion. It then
removes an edge only when a generic bond-order budget is exceeded or a
low-confidence single edge is attached to a multiple-bond junction. No edge
with confidence above the fixed sweep value is edited.

On the complete 474-image holdout, the cycle-only baseline has oracle sanitize
rate `0.9219`, edge precision/recall `0.9585/0.8503`, and 37 valence failures.
At low-confidence threshold `0.4`, the guard reaches sanitize `0.9451`, edge
precision/recall `0.9659/0.8500`, bond-type accuracy `0.9822`, and 26 valence
failures. Threshold `0.8` reaches sanitize `0.9515` but lowers recall to
`0.8491`; it is not selected as the default. Profile-level cov13 weak styles
remain below the proposed `0.90` floor, so this is a diagnostic candidate and
not a production or GPU-training authorization.

The complete sweep and per-profile strata are recorded in:

`outputs/audits/chemvector-identity-aligned-line-first-valence-topology-guard-r5/report.json`

