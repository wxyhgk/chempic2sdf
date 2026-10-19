# Dual atomness/rescue preflight

The raw-objectness result was split into two independent heads:

1. atomness: `protected + strict_positive` versus `hard_negative`;
2. rescue: `strict_positive` versus `hard_negative`, with protected,
   near-unknown, and selected-unknown candidates masked.

The atomness head reproduced the strong result (core precision 0.978–0.979,
recall 0.992–0.994).  The rescue head failed decisively under the fixed
natural-prevalence gate:

- coordgen: precision 0.00038, recall 0.045;
- original: precision 0.00019, recall 0.019;
- rdkit: precision 0.00028, recall 0.029.

Report: `outputs/audits/chemvector-style-factorial-dual-objectness-cpu-preflight-20261010-r1/report.json`.

This confirms that the successful atomness sidecar must not be used as a
strict-selector rescue model.  Protected candidates should remain on the
frozen baseline path, while a future rescue head needs a different objective
(for example, constrained group ranking or explicit graph/bond assignment).
No rescue GPU run is authorized.
