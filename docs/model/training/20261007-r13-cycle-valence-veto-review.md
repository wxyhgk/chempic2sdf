# r13 cycle/valence veto review

`tools/diagnose_aromatic_cycle_veto_r13.py` keeps target joins inside the
post-hoc audit functions.  The actual `veto_item` decision uses only frozen
presence logits, predicted pair slots, profile-conditioned type residual
outputs, and the selected-pair adjacency; it does not read target mapping,
elements, bond labels, or GT chemistry.  The edge set is therefore unchanged
by the veto.  The report's `one()` and `type_stats()` functions use targets only
to calculate audit metrics.

The cycle-only diagnostic improved oracle sanitize to `90/95 = .9474`, but the
serif30 r30 profile remained `20/23 = .8696`, below the per-profile `.90` floor.
The optional valence pass improved oracle sanitize to `94/95 = .9895` (profile
rates `1.0, 1.0, .9565, 1.0`), but it assumes every atom has a carbon-like
maximum bond order of four.  It can therefore demote valid N/O/P/S, charged,
or other hypervalent structures.  It is suitable for diagnostic auditing only,
not as a training target or production rule.

The valence pass also lowered aggregate audited type accuracy from `786/798`
to `782/798`; per-profile type accuracy is not included in the report and must
be added before any GPU-shadow gate.  Profile-conditioned heads require a
declared inference-time rendering-profile input; supplying profile IDs only
from dataset rows would be a deployment mismatch even though it is not GT
element leakage.
