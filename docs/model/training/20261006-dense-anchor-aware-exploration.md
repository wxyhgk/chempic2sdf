# Dense anchor-aware endpoint exploration (2026-10-06)

This was a forward-only exploration on the fixed 1,200-image training cohort
with the production `initial.pt`. No optimizer step, checkpoint write, sealed
identity read, or deployment change occurred.

## What the current field already knows

The raw proposal gap is 232 targets. Scanning every 192x192 dense cell (not
only local heatmap maxima) found a predicted anchor within 6px for **232/232**
targets. This is only a capacity/parameterization upper bound: every target
has a nearby grid cell and the anchor-offset head is defined at every cell, so
the result may be largely a grid-coverage artifact. It must not be read as
proof that the visual encoder has identified the correct endpoint. The robust
finding is that the production decoder's target cells are suppressed by
heatmap ranking:

| diagnostic | count |
|---|---:|
| any dense cell within 6px | 232 / 232 |
| best cell in score top-128 | 31 / 232 |
| best cell in score top-512 | 81 / 232 |
| best cell in score top-2048 | 152 / 232 |
| target cell itself in top-128 | 45 / 232 |
| target cell is a local peak | 0 / 232 |

Artifact: `outputs/audits/chemvector-dense-anchor-reachability-20261006-r2/report.json`.

This distinguishes the next learning target: promote/shape dense endpoint
evidence and ranking, not another pair/chemical head or a larger candidate
capacity.

## Target-independent decode shadow

As a capacity probe, a decoder scanned the top 4,096 heatmap cells and greedily
suppressed cells using their predicted-anchor distance. It did not use target
coordinates. With a 6px suppression radius and 128 candidates it matched:

| view | matched targets | new vs selected | selected IDs lost |
|---|---:|---:|---:|
| immutable selected baseline | 35,799 | — | — |
| anchor-aware shadow | 36,137 | +339 | 1 |

Artifact: `outputs/audits/chemvector-anchor-aware-decode-20261006/report.json`.

把 shadow 候选送进现有 frozen set head 后，问题暴露得更清楚：6px 抑制虽然
raw shadow 可达 36,137，但 set head 只保留 35,610，并丢失 203 个原 selected
target IDs。8px 时保留 35,747（丢 71），12px 时保留 35,793（丢 48）。因此
不能直接替换 production decoder；候选生成和 set-head ownership 之间需要一个
“baseline immutable + residual sidecar”接口，而不是改变原候选集合的全局重排。

这组 downstream 数字在
`outputs/audits/chemvector-anchor-aware-decode-20261006-r2/report.json`。

This is not a deployment proposal: it replaces the production decoder and is
only evidence that ranking/suppression can expose useful existing geometry.

## Ownership-safe sidecar check

The same idea was constrained to an immutable selected baseline, a 12px guard,
and residual-only matching. Its candidate precision remained extremely low:

| sidecar K | train candidates | train positives | precision |
|---:|---:|---:|---:|
| 16 | 1,920 | 25 | 1.30% |
| 32 | 3,840 | 26 | 0.68% |
| 64 | 7,680 | 27 | 0.35% |

Artifact: `outputs/audits/chemvector-anchor-aware-sidecar-20261006/report.json`.

Therefore this cannot justify RescueScorer training or sidecar integration.
The useful result is upstream: current dense anchors are informative, while
heatmap confidence and target-independent candidate discrimination are not.

## Decision

1. Keep `initial.pt` frozen and keep the ownership contract fixed.
2. Do not train a scorer, append high-resolution candidates, increase top-K, or
   repeat pair/chemical-head sweeps.
3. The next bounded experiment, if GPU is authorized, should be a **dense
   heatmap ranking objective** on a train-only slice: promote GT endpoint cells
   into the candidate boundary while retaining all initially selected IDs.
   It must be evaluated with the existing decoder and selected-baseline
   retention before any broader trial.
4. A suitable preflight gate is: target-cell rank improvement on train and
   holdout, selected retention 100%, and no regression in existing exact/fallback
   metrics. Failure means no training continuation.

The immediate engineering prerequisite is therefore a separate residual path
that never feeds a changed 128-candidate set through the old set head. Until
that contract exists and its candidate precision is measured, no GPU trial is
admitted.
