# G3 P19-E6 Three-route Oracle

> 日期：2026-08-10
>
> 状态：已完成，Oracle GO
>
> 类型：零训练 Oracle 上限，不创建模型 tag

## 目标

P19-E5 的全局融合只差 crossed 一项 Pareto 门。训练 router 前，先用已有 eval 行级结果计算
旧模型、新模型、融合三路线的逐图 exact union，判断路由是否存在足够上限。

## 固定协议

- 输入固定为 P19-E4 新模型 eval rows、P19-E5 融合 rows 与旧模型 paired rows；
- 只按 sample key 对齐，不重新推理、不调参数；
- Oracle exact 表示三路线中至少一条整图 skeleton exact；
- 该结果读取 eval 标签，只是理论上限，不得称为部署结果；
- 不读取 fresh640、confirm、sealed test、USPTO 或 MR-TADF。

## GO/NO-GO

- overall Oracle 至少 553/640，并比融合至少多 10 张；
- 普通三布局至少 456、stress 至少 82；
- crossed 至少 7、near-nonbonded 至少 9；
- Oracle 只能选择现有三条路线。

通过才允许进入输出级 router capacity；失败则直接停止路由路线。

## 实际结果

| 分桶 | 新模型 | 旧模型 | 融合 | 三路线 Oracle |
|---|---:|---:|---:|---:|
| Overall | 538 | 531 | 538 | **553** |
| 普通三布局 | 456 | 453 | 456 | **462** |
| Stress | 82 | 78 | 82 | **91** |
| Crossed | 7 | 0 | 4 | **7** |
| Crowded | 30 | 29 | 30 | **32** |
| Long bond | 39 | 40 | 39 | **41** |
| Near-nonbonded | 6 | 9 | 9 | **11** |

640 张中 603 张的三路线 exact 状态一致，只有 37 张分歧。以融合为默认路线，Oracle 仍有
15 张可救 headroom。exact route pattern 为：三者都对 516、三者都错 87、new+fusion 15、
old+fusion 6、old-only 8、fusion-only 1、new-only 6、new+old 1。

## 决策

正式决策 `three_route_oracle_go`，全部门通过。router 在理论上可以同时达到 overall 553、
普通 462、stress 91、crossed 7、near 11，足以解决 P19-E5 的唯一 Pareto 缺口。

下一步进入输出级 router capacity，但必须遵守：默认 `KEEP_FUSION`，候选仅 `USE_NEW` 与
`USE_OLD`；输入只能是三路线自身的预测数量、分歧数量、归一化 margin 分布等可观测字段，不能
使用 stress operation 或 GT 错误字段。当前 eval640 已用于 Oracle，只能报告上限，后续不得
用它选特征、threshold 或 checkpoint；训练/校准必须来自 calibration 或新的 train-only cache。

## 产物

- `audit.json` SHA256：`e77828f9e64e28b2800a7ea62ba1bfc267c9ed6094cd8e5fa018e04c344a4069`；
- `rows.json` SHA256：`f5e48fa21c69db54506efd63b0589370d98efd4d6b5df9f39ee13fe2f67a2cfa`；
- 本轮无训练，不创建模型 tag。
