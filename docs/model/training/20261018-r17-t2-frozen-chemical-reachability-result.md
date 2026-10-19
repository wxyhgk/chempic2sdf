# r17 t2 frozen chemical-head reachability (2026-10-18)

This is a read-only CPU attribution over the existing t2 feature cache.  It
does not train, read sealed test data, alter the production checkpoint, or
change the frozen pair set.

Artifact:

`outputs/audits/chemvector-r17-targeted-source-pair-type-reachability-cpu-20261018-t2-v2/report.json`

On the disjoint holdout, true edges outside the frozen presence baseline were:

| profile | true non-baseline edges | type mix | frozen chemical top-1 |
|---|---:|---|---:|
| oa13 | 11 | aromatic 7, double 2, single 1, triple 1 | 3/11 (27.3%) |
| oa33 | 20 | aromatic 17, double 1, single 2 | 7/20 (35.0%) |

The calibrated presence route admitted none of the oa13 candidates and only
three oa33 candidates; just one of those three had the correct frozen type.
The candidate pool therefore contains real type errors, but the difficult
oa13 edges are blocked one stage earlier by presence routing.  A chemical
head-only update cannot make an absent pair appear.

For context, the already-completed readout-only type probe reports holdout
frozen type accuracy `.9174` / macro-F1 `.8754`, improving to `.9552` /
`.9298` with its diagnostic residual; aromatic-to-single errors fall from
207 to 71.  This confirms that the chemical head has learnable errors, but
the probe leaves pair routing unchanged and therefore is not a release path.

If an explicit candidate-union oracle were supplied, the holdout ceilings are
`+2.65 pp` for oa13 (11 nonbaseline true edges) and `+4.87 pp` for oa33 (20
nonbaseline true edges).  With the frozen type top-1 alone those ceilings are
only `+0.72 pp` and `+1.70 pp`, respectively.  These are capacity bounds, not
achieved metrics.

Decision: do not spend the next bounded run on a standalone type-head-only
training or promotion.  If a diagnostic is needed, it should be an oracle
capacity probe on an explicitly fixed candidate union, not a production route.
Any useful intervention must jointly improve candidate routing and type
classification; the existing joint residual shadow remains NO-GO and should
not be promoted.
