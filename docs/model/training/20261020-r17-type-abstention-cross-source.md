# r17 type-confidence abstention cross-source result

The probe `tools/probe_r17_type_abstention_cross_source_cpu.py` tested a
target-free policy that drops selected candidate edges whose frozen chemical
type margin is below a threshold. The margin was selected on the independent
t3e holdout and evaluated on t3f; no labels are used by routing.

The selection batch found no nonzero margin satisfying the edge-recall gate
for both profiles. It selected margin 0.0 (the identity policy). On t3f the
policy therefore made no change. Nonzero margins did raise precision on some
slices, but dropped edge recall by 1.0–16.7 percentage points in the selection
batch, so they are not safe repairs.

Decision: abstention by raw chemical-type margin alone is NO-GO. A future
abstention policy would need a calibrated risk model with an explicit recall
budget and an independent calibration split; do not add a hand threshold to
production.
