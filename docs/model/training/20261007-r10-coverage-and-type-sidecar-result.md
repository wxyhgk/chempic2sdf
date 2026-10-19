# r10 direct AtomPair coverage and bond-type sidecar (2026-10-07)

## Decision

The direct AtomPair route remains the active research route, but neither r10
sidecar is production-ready.  No GPU shadow was started after the CPU results:
the edge scorer improved pair metrics while worsening the chemistry audit, and
the type residual recovered only part of that loss.

## r10 presence CPU trial

- 384 deterministic training images: 96 per rendering profile.
- Category coverage: 24 hetero-N/O, 80 aromatic, 120 double, 80 mixed, 80 triple.
- All positive pairs retained; same-image negatives sampled at a maximum 2:1
  ratio, with 2,727 nearby (distance <= 80 px) hard negatives.
- 300 CPU steps; 8,973-parameter presence residual; train-only threshold `.72`.

Fresh 95-image holdout, molecule-disjoint:

| metric | frozen baseline | r10 presence |
|---|---:|---:|
| edge precision | .994845 | **1.000000** |
| edge recall | .934625 | **.966102** |
| dangling | .003851 | .003745 |
| type accuracy | .963731 | .959900 |
| oracle-element sanitize | .800000 | **.768421** |

The edge gate passes, but chemistry does not.  Oracle element replacement is
post-hoc only; it is not used by the model.

## r10 bond-type CPU A/B

A 1,576-parameter type residual was trained on 3,168 positive rows while the
r10 presence state and selected edge set were frozen.  It improved oracle
sanitize from `.757895` to `.821053` (72 to 78 of 95) and aromatic failures from
20 to 14, but the serif30 profile remained `.608696` and the chemistry gate
still failed.  This is evidence that bond type/aromatic closure, not pair
presence, is now the bottleneck.

Artifacts:

- `outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r10-coverage/report.json`
- `outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-chemistry-r10-coverage/report.json`
- `outputs/audits/chemvector-identity-aligned-direct-graph-type-sidecar-cpu-r10/report.json`

Production checkpoint SHA remained
`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`;
sealed test was not read and no production merge occurred.

## Next bounded step

Do not run another GPU presence shadow.  Build a profile-specific aromatic/type
CPU trial, concentrating hard aromatic and serif30 examples, and require:
oracle sanitize >= `.95` overall, >= `.90` per profile, edge precision >= `.990`,
edge recall > `.940`, and dangling <= `.01`.  Only if all those gates pass is a
small type-only GPU shadow justified.
