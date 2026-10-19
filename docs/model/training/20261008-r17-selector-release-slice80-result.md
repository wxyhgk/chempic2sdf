# r17 selector retention on the release identity slice (2026-10-08)

The aromatic-aware selector replay was evaluated on an internally
identity-disjoint slice from the audited 10k release plan. It is **not** a
holdout independent of all earlier r17 work: the overlap audit found 28 train
and 46 holdout molecule-ID overlaps with the r17 plan (and 28/41 with
materialized r17 caches). The frozen production checkpoint,
raw proposals, pair head, chemistry heads, and sidecar weights were unchanged.
The slice contains 80 train molecules and 80 holdout molecules, eight fixed
profiles each (640 images per split). No sealed test content was read and no
checkpoint was written.

Artifacts:

- normal (historical, overlapping cohort): `outputs/audits/chemvector-r17-selector-graph-replay-10k-slice80-aromatic/report.json`
- target perturbation: `outputs/audits/chemvector-r17-selector-graph-replay-10k-slice80-aromatic-perturb/report.json`
- corrected m128 strict attribution: `outputs/audits/chemvector-r17-selector-graph-replay-m128-aromatic-v3/report.json`
- overlap audit: `outputs/audits/chemvector-r17-release-slice80-identity-overlap-v3/report.json`

| split | baseline recall | replay recall | baseline precision | replay precision | baseline sanitize | replay sanitize |
|---|---:|---:|---:|---:|---:|---:|
| train | .93346 | .94078 | .99878 | .99879 | .60313 | .60313 |
| holdout | .94081 | .94878 | .99846 | .99848 | .60469 | .60469 |

The historical slice holdout accepted 87 previously rejected raw atoms and
gained 77 true edges. Prefix retention remained 100%, precision did not
regress, and the sanitize rate was unchanged. The normal and target-perturbed
decision digests were identical:

`97b9378c86b6de5877d168afeb63e880e8a84e22a68c147d0cc9a332b9dd4fa2`

This confirms target-free candidate selection, pair enumeration, and safe
compiler acceptance on the internally disjoint slice. The result supports scaling the
same frozen replay to the full release only as an evaluation; it is not a
production merge or training authorization. The release's existing raw
reachability failure on oa13/oa33 still blocks training admission.
