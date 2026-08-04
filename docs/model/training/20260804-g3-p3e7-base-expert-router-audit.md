# G3 P3-E7 Base / Expert Router 零训练审计

> 日期：2026-08-04
>
> 状态：known-count Atom selector 路由 GO
>
> 训练更新：0

## 目标

验证是否应停止强迫一个共享 checkpoint 同时保持 P2-W capacity 与学习 general，改用：

```text
P2-W safe base + P3-E1 general expert + conservative router
```

路由只使用模型自身分数、base/expert候选集合差异和微扰稳定性；GT只用于判分，不进入特征。

## 数据协议

- 从 train5k cache 选 512 个 molecule，每个 molecule 只取一张图。
- route-select / route-confirm 各 256，四种 variant 各 64。
- 与 capacity64、P3-E1 general192 在 molecule 层面重叠 0。
- 两个 split 之间 molecule 重叠 0。
- route-select 选择单特征阈值：在 0 break 条件下最大化 rescue，再最小化 route 数。
- route-confirm 只执行一次固定策略。
- 历史 route100 不参与特征或阈值选择，只做固定后诊断。

注意：512 张仍来自 train5k candidate cache，上游中心场训练曾消费 train5k；因此它是 Atom selector
层的 molecule-disjoint 审计，不等同于完整图像流水线的公开未见测试。route100 提供额外独立图像证据。

## 结果

| 集合 | Base exact | Expert exact | Oracle | Router exact | Rescue / Break |
|---|---:|---:|---:|---:|---:|
| route-select 256 | 15 | 157 | 158 | 119 | 104 / 0 |
| route-confirm 256 | 13 | 160 | 160 | **111** | **98 / 0** |
| historical route100 | 4 | 54 | 54 | **36** | **32 / 0** |

固定策略为：

```text
base_joint_score_mean <= -1.8862037658691406 -> expert
otherwise -> base
```

route-confirm 路由 177/256；route100 路由 68/100。confirm 与 route100 均为 0 break，且分别净增
98、32 张整图 exact。expert 在 confirm 中完全包含 base 的 13 张正确图，在 route100 中也完全
包含 base 的 4 张正确图；但策略仍按预注册零 break 原则保守保留 base。

## 决策

`base_expert_router_audit_go`。证明独立 expert 比共享参数受约束训练更有效。下一步实现可复用
shadow router，并同时报告 known-count 与冻结 P2-W predicted-count；本结果不得写成端到端准确率。

产物：`outputs/audits/chemvector-g3-p3e7-base-expert-router-audit/metrics.json`
