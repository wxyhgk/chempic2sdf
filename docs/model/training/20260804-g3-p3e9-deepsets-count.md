# G3 P3-E9 DeepSets Count Train1760

> 日期：2026-08-04
>
> 状态：NO-GO；Count 方向强提升，molecule 泛化不足

## 目标与模型

冻结 P2-W base 与 P3-E1 expert，只训练独立 Count。模型不再读取 `memory.mean`，而是聚合：

- query existence、pointer峰值、熵、top1/top2 margin；
- candidate视觉特征、中心场分数、base/expert joint支撑和pointer投票；
- 排序后的前64个query/candidate全局证据。

query/candidate 分支分别使用 DeepSets mean/max 聚合，再预测0–128 Count。不会向 base/expert
反传梯度。

## 数据与协议

- train：440 molecule / 1,760图；dev、confirm各64 molecule / 256图。
- molecule跨split重叠0；每个molecule包含四种variant。
- 排除 P3-E1/P3-E7 已消费 molecule；route100 不参与训练或选择。
- 30 epoch、840 step、52,800 exposures；fixed-final，不选中间checkpoint。
- 门：dev/confirm accuracy至少50%，confirm MAE不高于1，route100 accuracy至少40%。

## 结果

| 集合 | 旧 Count exact / MAE | DeepSets exact / MAE |
|---|---:|---:|
| train1760 | 84/1760 / 8.27 | 1499/1760 / 0.43 |
| dev256 | 10/256 / 7.57 | 83/256 / 1.67 |
| confirm256 | 12/256 / 8.65 | **81/256 / 1.56** |
| historical route100 | 3/100 / 7.66 | **34/100 / 1.95** |

所有正式门均未过，结果为 `NO-GO`。但 confirm exact 提高6.75倍，route100提高11.3倍，证明
集合统计特征足以承载 Count。train 85.2% 对 confirm 31.6%的差距说明当前主要问题是分子级
过拟合，而不是继续增加特征。

## 决策与产物

下一轮保持输入特征不变，只扩大训练 molecule 数并缩小 Count 头，严格隔离数据量与容量。
不接入 router 或默认推理。

- checkpoint：`step-000840.pt`
- SHA256：`a974369ec7d240459d6459937ae05d64f59e345d81a260495bab13edb923e988`
- 输出：`outputs/smoke/chemvector-g3-p3e9-deepsets-count-train1760`
