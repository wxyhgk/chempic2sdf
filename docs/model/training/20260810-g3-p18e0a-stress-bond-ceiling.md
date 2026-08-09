# G3 P18-E0A Stress Bond 候选上限拆解

> 日期：2026-08-10
>
> 状态：已完成，端点定位误差
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

## 实际结果

top-k 曲线直接排除了排名截断：

| 候选预算，12px | Bond recall | Bond exact |
|---:|---:|---:|
| 128 | 93.34% | 35.00% |
| 192 | 93.82% | 35.63% |
| 256 | 93.93% | 35.63% |
| 384 | 93.99% | 35.63% |
| 512 | 93.99% | 35.63% |

top192 扩到 top512 只多匹配 12/6,874 根 Bond，整图 exact 完全不变。正确对象不是被候选预算
挤掉，继续扩大 top-k 没有价值。

top512 半径曲线：

| 诊断半径 | Bond recall | Bond exact |
|---:|---:|---:|
| 12px | 93.99% | 35.63% |
| 16px | 95.80% | 46.25% |
| 24px | 97.92% | 64.38% |

因此按预注册规则判定为 `endpoint_localization_precision`。24px 只证明大多数候选的几何身份
存在，不能作为部署匹配半径。

分 operation 看，`crowded_subgraph` 和 `long_bond` 在 24px 分别达到 97.73%/97.56% exact；
`near_nonbonded` 为 41.18%，`crossed_bonds` 仍只有 14.63%。top192 缺失目标中 311/424 来自
`crossed_bonds`，是下一容量实验必须单独报告的主桶。

本轮运行耗时 11.77 秒，top192 的 ID、mask、endpoint 与 P18-E0 160 张 stress cache 逐图一致。
下一步冻结候选身份、score、type 与视觉源，只训练 proposal 级 endpoint residual capacity64；
禁止扩大 top-k、解冻 Bond head 或同时训练 incidence。
