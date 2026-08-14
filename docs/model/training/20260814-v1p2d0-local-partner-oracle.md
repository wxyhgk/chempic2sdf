# v1 P2-D0 局部 Partner Oracle

> 日期：2026-08-14
>
> 状态：已完成，正式 GO
>
> 审计：`chemvector-v1p2d0-local-partner-oracle`

## 目标

P2-C 证明宽 endpoint 候选的 exhaustive pair recall 为 97.86%，但全图 `4x` line-score 只保留
77.47% 真键。P2-D0 检查这个损失是否来自全局排序被少数高分区域占满：不再全图取固定数量，
而是让每个 endpoint 保留自己的 top-K partner，再对无序 pair 取并集。

partner 在全图所有 endpoint 中按 line score 选择，不使用半径、最近邻或最大键长裁剪，因此
长键与短键拥有相同进入候选集的机会。这仍是零训练 Oracle，不输出 BondObject 或 SDF。

## 冻结协议

- P1-A 与 P2-B 正式 winner 保持冻结；
- endpoint 阈值固定 0.05，每图最多 256 个；
- 沿用 P2-C 的 48x9 ribbon、line score 和 gated blocker score；
- 比较每个 endpoint `top2/top4/top8/top16` partner 的无序并集；
- 端点对小于 4px 才作为近重复无效，不设置最大长度；
- 只读取已经暴露的 P2-B calibration 400 张；
- P2-B holdout、validation、internal test、MR-TADF、USPTO 均不读取。

## 指标与门槛

pair 命中仍要求两个无序端点分别在 GT BondObject visible port 的 6px 内。以 top8 的最佳固定
路线作为决策点，同时满足：

- overall pair recall 至少 95%；
- image exact 至少 30%；
- stress pair recall 至少 90%；
- crossed-bonds pair recall 至少 85%；
- 不使用最大键长或空间近邻裁剪；
- P2-B holdout、validation、internal test 读取数均为 0。

若 GO，top8 union 成为下一轮 pair scorer 的冻结输入候选协议；pair scorer 使用 P2-B fit 内
重新隔离的 fit/calibration 训练，最终才一次性读取 P2-B holdout。若 NO-GO，不直接训练，因为
训练头无法恢复预筛选阶段已经删除的真键。

## 结果

400 张 route-select 共 13,860 个 GT BondObject。`line_score` 与 gated blocker 的 recall
完全相同，前者以略少的候选数成为固定 winner：

| 每端 partner | 平均 pair/图 | Pair recall | Image exact |
| --- | ---: | ---: | ---: |
| top2 | 313.98 | 87.58% | 11.00% |
| top4 | 566.84 | 93.25% | 34.25% |
| top8 | 1,019.10 | 95.91% | 57.00% |
| top16 | 1,898.86 | 97.22% | 71.00% |

固定 top8 的 stress pair recall 为 94.03%、crossed-bonds 为 88.20%、long-bond 为 97.08%、
near-nonbonded 为 95.35%，全部预注册门通过。相比 P2-C 平均 18,340.09 个 exhaustive pair，
top8 将候选规模降低 94.44%，同时超过 95% overall recall。

最终判定 `local_partner_oracle_go`。这证明失败主要来自全图 pair 排名竞争，而不是必须依赖
最大键长或空间近邻规则。GO 只冻结 pair scorer 的输入候选协议，不代表这些约 1,019 个候选
已经完成真假分类。

## 下一步

P2-D1 使用 `line_score/top8 partner union` 作为固定候选生成器。训练轻量 pair scorer 时读取
完整 9x48 ribbon 序列、两端分数、长度、line summary 和 blocker，但 presence 目标必须使用
整图 listwise/hard-negative 约束。训练与 calibration 从 P2-B fit 的 2,014 个分子内部重新按
分子隔离；P2-B 的 100 分子 holdout 保持未读，等 checkpoint、pair 阈值和最终选择协议完全
冻结后再一次性确认。

## 产物

- audit：`outputs/audits/chemvector-v1p2d0-local-partner-oracle/audit.json`；
- audit SHA256：`194fce170a843becdd95890ce3e26dbafdda121e7dca3426b4c29d8d648184c3`；
- rows：`outputs/audits/chemvector-v1p2d0-local-partner-oracle/rows.jsonl`；
- rows SHA256：`227c9e8837be0fd95767a99a821ae827449040d59efd68c483c9fa46fe17e9eb`。
