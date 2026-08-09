# G3 P18-E0A Stress Bond 候选上限拆解

> 日期：2026-08-10
>
> 状态：已预注册，等待运行
>
> 类型：零训练能力审计，不创建模型 tag

## 目标

P18-E0 的 Atom exact 为 95%，但 stress Bond exact 只有 35.63%，使该风格联合 skeleton 只有
34.38%。本轮保持 v0.10.1 Bond head 和 fresh640 不变，只区分三种原因：

1. 正确 Bond 已在 dense 输出中，但被 top192 截断；
2. 正确对象进入 top512，但端点定位误差超过 12px；
3. 即使 top512/24px 仍不可达，dense Bond 对象本身没有生成。

## 固定协议

- 输入固定为 P18-E0 的 fresh640 manifest、GT 和 cache；
- 仍按完整 640 图原批次顺序前向，只保留 160 张 stress，保证 top192 可逐 tensor 重放；
- 同一 dense 输出同时解码 top192 与 top512；
- 固定报告 12px 下 top128/192/256/384/512，以及 top512 下 12/16/24px；
- 按 `long_bond`、`crowded_subgraph`、`crossed_bonds`、`near_nonbonded` 分桶；
- 24px 仅用于诊断定位误差，不能成为部署阈值；
- 不训练，不读取 sealed test、confirm、USPTO 或 MR-TADF。

## 决策规则

- top512 exact ≥60% 且相对 top192 提升 ≥20pp：`top192_ranking_truncation`；
- 否则 top512/24px exact ≥60% 且相对 top512/12px 提升 ≥20pp：
  `endpoint_localization_precision`；
- 否则：`dense_bond_generation`。

后续只针对判定的单一断点设计最小修复，不调整 Atom、incidence、presence 或 compiler。
