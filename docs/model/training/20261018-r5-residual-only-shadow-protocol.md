# r5 residual-only proposal shadow protocol

Status: protocol only. No shadow training has started and no production state
or checkpoint is modified by this document.

## Why this route

The fixed residual reachability panel shows that the reviewed vertex field is
not missing atom geometry. On the 160/80 train/holdout slice, all-candidate
coverage is high, while the production baseline protects most of the same
locations:

| slice | baseline-owned atoms | residual targets | residual candidate reachability | hard collision rate | combined bond coverage |
|---|---:|---:|---:|---:|---:|
| train | 78.63% | 362 | 88.12% | 2.37% | 95.49% |
| holdout | 72.22% | 175 | 86.86% | 4.36% | 95.45% |

The visible-heteroatom stratum is the main gap: on the same holdout, baseline
misses 78.81% of O and 63.10% of N, while implicit-C misses only 0.81%.
Therefore another endpoint-pair scorer is not the next experiment. The
bounded experiment should learn a residual-only proposal field with explicit
baseline ownership protection.

Evidence receipt:

```text
outputs/audits/chemvector-identity-aligned-r5-residual-only-reachability/report.json
```

## Frozen inputs and labels

Use the fixed r5 identity split, taking 160 train images and 80 holdout
images, with all six profiles retained per molecule and zero molecule/scaffold
overlap. Load `initial.pt` only into a fresh model. The production selector,
bond/OCR heads, graph assembler, and checkpoint remain frozen.

For each training image, run frozen baseline inference first. A ground-truth
atom is a residual positive only when no frozen baseline atom lies within 6 px.
Atoms within 6 px of baseline geometry are protected and are not residual
positives. Their cells, plus a 6--20 px ring, are hard collision negatives.
Labels are used in memory for the loss and holdout audit only; no target IDs,
elements, or labels may enter candidate JSONL.

Residual labels should be stratified/oversampled for visible O, visible N, and
the one-baseline/one-residual bond pattern. Implicit C remains in the replay
set but must not dominate the residual objective.

## Shadow model and loss

Train a fresh `ChemVertexDetector` and only the reviewed stride-4 vision block
in a sidecar model. All other production weights are frozen. The head emits
the normal dense vertex heatmap, offset, role, and aromatic channels; decoding
uses the existing fixed dense decoder. The residual heatmap target contains
only residual positives.

Use the existing focal heatmap/offset/aromatic loss on residual targets,
plus a small explicit baseline-collision penalty sampled at protected cells.
Use a per-image positive-presence term so an image with two residual atoms is
not diluted by the many background cells. Do not use target coordinates at
inference and do not attach a scalar post-hoc threshold sweep.

Fixed budget: at most 800 optimizer steps, batch size 8, one run, at most
300 seconds GPU wall time. A separate sidecar receipt/state and candidate
JSONL may be written; `initial.pt` and production state must remain byte
identical.

## Mandatory readouts and gates

Report train and holdout separately, and by O/N/implicit-C/profile:

* residual target recall and unique residual candidate precision;
* residual candidate hard-collision rate against protected baseline atoms;
* residual candidate count and residual targets per image;
* combined baseline+residual atom coverage;
* one-residual and both-residual target-bond coverage;
* distinct endpoint ownership, complete-edge precision/recall, dangling rate,
  and conditional bond-type accuracy in a separate frozen integration audit.

The shadow is diagnostic only. It is admissible for independent review only
if all fixed holdout conditions pass: residual recall >= 0.80, residual
precision >= 0.85, hard-collision <= 0.10, O/N recall >= 0.70, combined atom
coverage >= 0.95, one-residual bond coverage >= 0.50, complete-edge precision
>= 0.80, edge recall >= 0.50, dangling <= 0.05, and baseline prefix/input
hashes remain exactly unchanged. Any failed condition is `NO-GO`; do not tune
thresholds or promote the sidecar.

## Required artifacts

```text
outputs/audits/chemvector-identity-aligned-r5-residual-only-shadow/report.json
outputs/audits/chemvector-identity-aligned-r5-residual-only-shadow/sidecar-state.pt
outputs/audits/chemvector-identity-aligned-r5-residual-only-shadow/candidates.jsonl
```

The candidate artifact may contain only `sample_key`, `source: "shadow"`, and
finite `x`, `y`, `score` fields. It must contain all 1,170 accepted rows if
used in the full integration audit, and must fail closed on any target field.
