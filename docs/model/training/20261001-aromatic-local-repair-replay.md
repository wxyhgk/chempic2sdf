# 2026-10-01 aromatic local-repair replay

The r3 chemical-head safety run was followed by a read-only replay of every
changed development witness. The replay reconstructed the fixed initial and
1200-step arms from the bound checkpoint and release manifest, then tried at
most one aromatic-bond demotion per candidate. It did not train, read sealed
data, or alter production inference.

Report: `outputs/audits/chemvector-aromatic-local-repair-replay-20261001-r2/report.json`.

Results:

- 10 fixed changed development images; 139 bounded final candidates.
- 79 candidates sanitized, but only 3 matched the released molecular identity,
  all on `fresh_dev_000054_rdkit`, which was not a new fallback case.
- The three r3 new fallback cases (`001401_coordgen`, `001833_original`,
  `001866_stress`) had no one-edge candidate matching the released identity.
- Some candidates matched the untouched initial arm, but those identities were
  still wrong; matching an earlier wrong prediction is not evidence of repair.

Decision: do **not** wire the local verifier into default inference. RDKit
sanitization can recover a compilable graph, but on the actual fallback
regressions it does not establish molecular identity. Keep the verifier as an
opt-in diagnostic and focus the next experiment on upstream bond/identity
errors rather than a compiler-side aromatic fallback.

The verifier unit suite remains green (`38 passed`).
