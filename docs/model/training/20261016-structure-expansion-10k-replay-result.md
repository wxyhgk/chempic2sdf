# Structure-expansion 10k frozen replay (2026-10-16)

The quarantine batch was reduced to a deterministic metadata plan with 1,100
train molecules and 160 holdout molecules. All selected molecules have the
same eight profiles, yielding 10,080 images. Molecule, connection, and
scaffold overlap between train and holdout is zero; the plan deliberately
does not claim a complete historical prior-use ledger.

The full batch was replayed through the frozen initial checkpoint with no
gradients or checkpoint writes:

| metric | result |
|---|---:|
| images | 10,080 |
| target atoms | 122,896 |
| raw matched | 120,457 |
| aggregate raw reachability | 98.015% |
| selected reachability | 95.530% |
| selector drop rate among raw | 2.536% |

The aggregate number is insufficient. The per-profile gate fails badly:

| profile | raw reachability |
|---|---:|
| oa00 | 99.824% |
| oa02 | 99.902% |
| oa11 | 99.635% |
| oa13 | 93.412% |
| oa20 | 99.948% |
| oa22 | 99.909% |
| oa31 | 99.570% |
| oa33 | 91.922% |

`oa13` and `oa33` are the large-serif/180-degree profiles. Because the
required gate is at least 98% for every core profile, this release is **not
training-admissible**. The result is useful: it identifies a rendering/model
coverage gap before any weights are exposed to these images. Do not train on
this pool or lower the gate. The next data action is to add targeted variants
for these two cells (and preserve clean controls), then repeat the same
identity and frozen-replay gates.

Artifacts:

- `outputs/audits/chemvector-structure-expansion-10k-release-plan-20261016-r1/`
- `outputs/audits/chemvector-structure-expansion-10k-frozen-replay-20261016-r3/report.json`
