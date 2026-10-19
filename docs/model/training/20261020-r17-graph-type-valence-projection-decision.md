# r17 graph-level type/valence projection (CPU shadow)

Two target-free probes were run against frozen candidate edges and the frozen
production checkpoint `09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`.

* `tools/probe_r17_graph_type_valence_projection_cpu.py` on the m128 cache
  changed only non-aromatic bond types under a joint valence penalty.  On the
  fresh holdout, lambda 4 improved sanitize from 0.7534 to 0.7573 and reduced
  valence failures 20 to 17, but typed precision/recall moved from
  0.9816/0.9265 to 0.9812/0.9261.
* `tools/probe_r17_t3f_graph_type_correction_cpu.py` replayed the t3f frozen
  presence residual.  The edge set stayed fixed, but global projection
  collapsed bond types: oa13 typed precision/recall 0.882/0.921 to
  0.710/0.742; oa33 0.943/0.943 to 0.750/0.750.

Decision: NO-GO for global graph-level type projection.  Valence should be a
conditional repair invoked only after a sanitize conflict, with a trust region
that preserves high-margin chemical predictions.  No checkpoint or sealed test
was touched.

## Contract-corrected t3f replay

The first t3f projection script accidentally searched only
SINGLE/DOUBLE/TRIPLE (`range(3)`) and used reachable edges as its recall
denominator. That made its lambda-zero comparison invalid, especially for
aromatic edges. The corrected replay is
`outputs/audits/chemvector-r17-t3f-graph-type-correction-cpu-v2/report.json`:
lambda zero is exactly the frozen chemical argmax, aromatic is included in the
joint search, and recall uses all ground-truth bonds. At threshold 0.95, lambda
1 improves oa13 typed precision/recall by 1.09/1.10 percentage points but
reduces oa33 by 1.15/1.10 points; lambda 4 reduces oa13 by 3.26/3.30 points.
It therefore remains NO-GO, and the earlier projection report is not used as
evidence.
