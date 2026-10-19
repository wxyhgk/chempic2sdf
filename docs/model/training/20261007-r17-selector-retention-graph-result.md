# r17 selector-retention → graph replay

The raw-vs-selected audit showed that the remaining endpoint gap is mostly
selector retention, not raw pixel proposal absence. A bounded selector sidecar
was therefore trained for 200 steps while freezing vision, raw atom proposals,
pair, OCR, and chemistry. Existing selected atoms were copied unchanged; only
rejected raw candidates could be proposed.

The sidecar-only holdout gate passed at a train-calibrated threshold of `0.90`:

- selected-prefix retention: `1.0`
- fresh new-candidate precision: `.931`
- fresh potential bond reachability: `7351 → 7493`

Sending every proposed atom directly to the frozen pair head increased edge
recall, but caused a sanitize regression (`.9453 → .8516` fresh). This was
unsafe and rejected.

We then ran a target-free safe replay at threshold `0.95`: a rejected raw atom
was accepted only when adding it to the current candidate graph and rerunning
the compiler still sanitized. Existing selected atoms were never removed.

| split | baseline sanitize | safe replay sanitize | baseline edge recall | safe replay edge recall |
|---|---:|---:|---:|---:|
| train | .9141 | .9141 | .9421 | .9619 |
| dev | .9219 | .9219 | .9350 | .9566 |
| fresh | .9453 | .9453 | .9418 | .9581 |

Fresh precision remained `.99817 → .99820`; 123 newly mapped edges were
observed and all were true in post-hoc identity accounting. The safe policy
made no graph-validity regressions and preserved the old edge set.

Artifacts:

- selector sidecar: `outputs/audits/chemvector-r17-selector-retention-m128-v4/report.json`
- safe graph replay: `outputs/audits/chemvector-r17-selector-graph-replay-m128-v8/report.json`

This is still a CPU/GPU shadow, not a production authorization. The
aromatic-aware follow-up and separately bound identity slice are documented
below; neither authorizes a merge or unfreezes the production checkpoint.

The replay's target-perturbation sentinel cleared all target centers, labels,
and types before selection. The normal and perturbed decision digests were
identical (`a74636cde0a67395fa07b698dc0fc5eacd12ceb4201095f9c3273bb4779ab54f`),
covering candidate acceptance, pair counts, and compile outcomes. Post-hoc
identity metrics become zero in the perturbed report by design, but the
target-free routing invariant passes.

## Aromatic-aware replay (2026-10-08)

The replay was rerun with the trained aromatic context head and the real
aromatic bond contract. The audit was corrected so aromatic edges remain
`AROMATIC` at the sanitizer boundary and raw AtomPair slot ids are compacted
before graph compilation. This changes only audit attribution; the production
checkpoint and frozen model weights remain untouched.

Artifact:

`outputs/audits/chemvector-r17-selector-graph-replay-m128-aromatic/report.json`

| split | baseline recall | replay recall | baseline precision | replay precision | baseline sanitize | replay sanitize |
|---|---:|---:|---:|---:|---:|---:|
| train | .94213 | .95262 | .99764 | .99766 | .57617 | .57617 |
| dev | .93497 | .94749 | .99659 | .99664 | .57813 | .57813 |
| fresh | .94178 | .94907 | .99817 | .99819 | .63281 | .63281 |

On fresh, 106 rejected raw atoms were accepted, yielding 55 additional true
edges with no precision or sanitize regression. The selected prefix remained
fully retained; the production gate remains deliberately `false`.

The independent target-perturbation run is at
`outputs/audits/chemvector-r17-selector-graph-replay-m128-aromatic-perturb/report.json`.
Its decision digest is identical to the normal run:

`c5aef49f391c4b6125c88eaec4c4758cad3edd23d07fc4d02b39fb8cbbf81eea`

Therefore candidate selection, pair enumeration, and safe compiler acceptance
are target-free. The perturbed report's zero precision/recall is expected
because its audit target is intentionally empty, not a quality result.

Decision: keep this shadow-only and promote it to a larger identity-disjoint
replay panel. Do not merge it into production or start checkpoint training
based on compiler validity alone.

## Contract-corrected strict attribution (2026-10-08)

Before relying on the replay, an independent probe found that the element
context head had been trained on `presence >= 0.72` edges but the compiler
replay aggregated all unordered pairs. This produced out-of-distribution
element predictions. The replay now uses the training-consistent selected-edge
scope and an explicit zero-incident fail-closed row; three regression tests
cover sparse slots, aromatic boundary semantics, and unmapped/type errors.

Corrected artifact:

`outputs/audits/chemvector-r17-selector-graph-replay-m128-aromatic-v3/report.json`

On fresh, strict edge precision was `.99510 → .99434`, strict typed-edge
precision `.96357 → .96317`, and strict typed-edge recall `.91194 → .92613`.
Mapped element accuracy remained `.98966 → .98909`; the selected prefix and
sanitize rate were unchanged. The exact typed-topology count increased from
230 to 247 images. The strict gates in the receipt pass, but
`production_go` remains `false` because this cohort is not a complete
historical-identity-clean training release and the graph compiler is still a
shadow artifact.
