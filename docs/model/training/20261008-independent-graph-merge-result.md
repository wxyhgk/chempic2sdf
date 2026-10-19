# Isolated high-resolution graph merge exploration

## Motivation

The text-center sidecar failed both on candidate precision and on ownership:
appending sidecar rows to the frozen set-head changed the prefix context on
every check. We therefore evaluated a different contract: the production
selector runs unchanged, a high-resolution branch produces residual graph
candidates independently, and baseline-first locked assignment merges only
the graph candidates. The residual rows never enter the old set-head.

## Protocol

- checkpoint: frozen `initial.pt` (`09d16611...c88dbe1`);
- 120/1080 group-, molecule-, and canonical-SMILES-disjoint split;
- 1,313-parameter 384px branch, 200 in-memory steps;
- baseline candidates are immutable;
- residual candidates use a 6px geometric deduplication radius;
- residual assignment can claim only targets not claimed by baseline;
- sealed test was not read and no checkpoint was written.

## All-atom target branch

| split | baseline matches | locked merged matches | residual candidates | residual new matches | new matches near a bond endpoint |
|---|---:|---:|---:|---:|---:|
| train (120) | 3,897 | 3,911 | 600 | 14 | 14 |
| holdout (1080) | 31,902 | 32,013 | 6,946 | 111 | 109 |

Prefix ownership failures: `0` on both splits. Of the 111 holdout additions,
89 were implicit/non-text endpoints and 22 were visible-text endpoints.

The final column is only a proximity diagnostic: it counts new endpoints whose
nearest frozen bond endpoint is within 12px. It is not yet a valid bond
assignment.

This is the first experiment that separates candidate capacity from set-head
coupling: it adds 111 locked endpoints without changing the frozen baseline.
It is still not a deployable accuracy result; residual candidate precision is
only about 1.6%, and bonds/incidence have not yet been rebuilt for the merged
graph.

## Text-only target branch

A second 200-step branch trained positives only on residual visible-text
targets. It was worse:

| split | residual new matches | visible-text additions |
|---|---:|---:|
| train | 14 | 2 |
| holdout | 99 | 12 |

The all-atom objective therefore remains preferable. The result suggests that
text endpoint recovery currently benefits from the shared endpoint/geometry
signal; a sparse text-only objective does not provide enough stable positives.

## Decision and next step

Do not train a scorer or modify production yet. The isolated merge is a
promising candidate route, but the next gate is graph compatibility: for each
new endpoint, construct/score bond incidence without changing existing graph
edges. If the merged endpoints cannot be assigned to valid bonds, the atom
candidate gain is not useful. The merge implementation is
`protected_residual_merge.py`; the next graph stage can use
`residual_bond_attachment.py` to attach residual rows to unoccupied frozen bond
endpoints. Unit tests cover deterministic selection, immutable baseline
prefixes, and endpoint ownership.
