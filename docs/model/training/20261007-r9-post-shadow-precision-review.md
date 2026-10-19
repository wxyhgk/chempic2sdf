# r9 direct AtomPair shadow：precision 回落审查

本记录审查 r9 direct AtomPair residual 的 CPU preflight、GPU shadow 与后验
oracle-element sanitize。全部实验只在 r6 fresh holdout 上做隔离 replay；没有
读取 sealed test，也没有写入 production checkpoint。

## 观测到的变化

GPU shadow step 800 相对 frozen proposal 的计数如下：

| | mapped edges | true edges | false positives | target edges |
|---|---:|---:|---:|---:|
| frozen | 776 | 772 | 4 | 826 |
| residual | 796 | 787 | 9 | 826 |

因此多恢复了 15 条真边，但同时增加 5 条假边：edge precision
`0.994845 -> 0.988693`（下降 0.615 个百分点），edge recall
`0.934625 -> 0.952785`（提高 1.816 个百分点）。dangling 仅由 3/779
变为 4/800（`0.003851 -> 0.005000`），不是 precision 回落的主因。
step 400 为 `P=.989924/R=.951574`，说明 step 800 继续优化 loss 并没有带来
更好的 precision。

CPU frozen-proposal 的精确后验 sanitize 审计为 `76/95=.800`；aromatic
失败 16 图，valence 失败 3 图。四个 profile 的 sanitize 分别为
`.8333/.8333/.7826/.7500`。这远低于 promotion 需要的 aggregate `.95` 与
per-profile `.90`，表明剩余问题主要是芳香环闭合/键型-拓扑一致性，而不是
单纯 threshold 或 pair precision。

报告：

- `outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-gpu-shadow-r9/report.json`
- `outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-oracle-sanitize-r9-cpu-replay/report.json`

## threshold 诊断与限制

后验诊断 sweep（仅用于观察，不能拿 holdout 选阈值）显示：

| threshold | edge precision | edge recall | dangling |
|---:|---:|---:|---:|
| .95 | .988693 | .952785 | .005000 |
| .98 | .991128 | .946731 | .005044 |
| .99 | .993622 | .943099 | .005076 |

`tools/diagnose_direct_graph_threshold_sweep_r9.py` 的 sweep 明确标记了
`diagnostic_holdout_labels_used=true`，不得把这些数值当作训练选择结果。
当前 train-only calibration 在 `.70-.99` 网格中都是 `807/807`，所以原有
F1 tie-break 到 `.95` 并不能提供有意义的 calibration。下一轮若只做校准，
应预先固定如下规则：

1. 仅在 train rows 计算 pair 分数；
2. 取最高的预注册阈值（建议网格上限 `.99`），条件是 train precision=1、
   train recall≥.995；
3. 阈值冻结后只对 fresh holdout 评估一次，禁止依据 holdout 回调。

## 是否值得第二轮训练

不值得再跑第二轮 GPU residual：它只能重排 edge presence，不能修复已经暴露
的 aromatic/topology sanitize 缺陷；800 步也已经出现 loss 继续下降而 precision
不再改善的迹象。

如果需要继续，最多先做一次 CPU-only 200--500 步小实验，但训练对象应改成
**chemistry-aware abstention/graph repair sidecar**，而不是再训练同一个
presence residual：

- 输入保持 target-free：pair hidden、presence、geometry、predicted type
  logits 及图上的 cycle/degree/valence 证据；
- 可以拒绝低 margin 的 aromatic/高价态边，或延迟到图闭合阶段；
- GT 只用于最后的 edge/type/sanitize 审计，不能进入 loss 或 router。

CPU 试验的必要门槛（任一失败即停止，不申请 GPU）：

```text
edge precision >= 0.990
edge recall    > 0.940 且严格高于 frozen baseline .934625
dangling       <= 0.01
每 profile recall >= .90
oracle sanitize >= .95
每 profile sanitize >= .90
```

只有在固定 fresh holdout 上同时满足这些门槛，才值得申请一次新的受限 GPU
shadow（≤800 steps/30 min，每 200 steps 检查，严格无改善早停）。production
checkpoint、OCR、selector 与现有 graph/head 均保持不变。
