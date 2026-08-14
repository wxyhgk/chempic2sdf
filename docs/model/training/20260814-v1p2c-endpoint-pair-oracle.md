# v1 P2-C 端点对候选 Oracle

> 日期：2026-08-14
>
> 状态：已完成，严格 NO-GO
>
> 审计：`chemvector-v1p2c-endpoint-pair-oracle`

## 目标

P2-B 的正式 winner 在阈值 0.35 下严格 NO-GO，但同一 calibration 上阈值 0.05 的宽候选覆盖
98.68% 唯一 port 和 98.15% 可配对 BondObject。P2-C 不再要求 endpoint 头独立完成排序和定
基数，而是验证一根键的两个端点和整段 bond ink 能否把真实 BondObject 排进稀疏候选对集合。

这是零训练 route-select Oracle，不输出键级、AtomObject 或 SDF。成功只允许进入新的配对模型
训练周期，不能作为端到端结果。

## 冻结输入

- P1-A separator SHA256：
  `490b5daae3a2059c3d335785a69e0c2a41eed32f0237ef77fc3556c36d134a74`；
- P2-B 正式 winner SHA256：
  `3ec7bf27e96f166e0f30e7c5c8b79f33860701acfeb3e16297015fc7a91511ca`；
- endpoint presence 阈值固定为 0.05；
- 每图最多保留 256 个 endpoint；
- 输入只使用 P1-A bond probability、endpoint 坐标和 endpoint 分数；
- 两个冻结模型均不更新参数。

阈值 0.05 来自已经暴露的 P2-B calibration 宽候选诊断，因此本轮仍只在相同的 100 个分子、
400 张 calibration 上做路线选择。P2-B holdout、validation、internal test、MR-TADF 和 USPTO
均不读取；本轮 GO 后必须另建新 scaffold confirm，不能在这 400 张上宣称泛化。

## 候选与证据

每张图枚举所有无序 endpoint pair，不设置最大距离或邻居数量裁剪。只删除端点距离小于 4px
的近重复 pair，不会删除长键。每个 pair 从 bond probability 读取：

- 48 个纵向位置；
- 垂直方向 `-8..8px` 的 9 条采样线；
- 每个纵向位置取横向最大 bond probability，使双键/三键仍作为一个 ribbon；
- 汇总整段均值、20% 下分位数和 `probability >= 0.5` 的覆盖率；
- 计算线段内部 `0.08..0.92` 范围内其他 endpoint 的 blocker 支持。

固定比较四条排序路线：

1. `endpoint_score`：只用两端较低的 endpoint 分数；
2. `line_score`：整段 ribbon 连续性加少量 endpoint 分数；
3. `line_blocker_score`：`line_score - 0.5 * blocker`；
4. `line_gated_blocker_score`：只有 blocker 超过 0.5 才线性惩罚。

开发期两张真实图冒烟中，`4x` pair recall 分别为 23.64%、76.36%、43.64%；这说明 line
evidence 有信号，同时直接 blocker 会被宽候选中的假端点污染，因此 gated 路线必须独立报告，
不能预设 blocker 必然改善。

## Oracle 预算

若图片有 `B` 个 GT BondObject，每条路线分别保留最高分的 `B/2B/4B/8B` 个 pair，并报告：

- BondObject pair recall：候选 pair 的两个端点在无序匹配下都距 GT port 不超过 6px；
- image exact：图片内所有 GT BondObject 都进入候选集合；
- selected pair 数量；
- normal/stress 及四种 stress operation 分桶结果。

同时报告 exhaustive pair Oracle。它只诊断 endpoint 候选是否具备几何可达性，不能代替稀疏预算。

## GO/NO-GO

在 400 张 route-select calibration 上同时满足：

- wide endpoint unique-port recall 至少 97%；
- exhaustive pair recall 至少 97%；
- 最佳固定路线的 `4x` pair recall 至少 95%；
- 最佳固定路线的 `4x` image exact 至少 30%；
- stress `4x` pair recall 至少 90%；
- crossed-bonds `4x` pair recall 至少 85%；
- 没有最大键长或近邻数量裁剪；
- P2-B holdout、validation 和 internal test 读取数均为 0。

若只有 exhaustive 通过而 `4x` 不通过，说明端点候选可达但当前确定性 line ranking 不够，下一步
应训练 pair scorer 或集合选择器；不能通过扩大到全部二次 pair 宣称 GO。若 wide endpoint 或
exhaustive 也不通过，则停止该候选源，不训练配对头。

## 实际执行

- route-select：100 个分子、400 张图；
- GT BondObject：13,860；
- 唯一 GT port：15,528；
- 平均 endpoint 候选 185.67，p95/最大值均为 256；
- 平均无序 pair 18,340.09，p95/最大值均为 32,640；
- 总耗时 9.02 秒；
- P2-B holdout、validation、internal test 读取数均为 0。

宽候选的 unique-port recall 为 98.68%，两个 port 同时可达的 BondObject recall 为 98.15%。
删除小于 4px 的近重复 pair 后，exhaustive pair recall 仍为 97.86%，image exact 77.50%。前两项
可达性门通过，说明 P2-B 宽候选可以作为下一阶段输入。

## 稀疏排序结果

| 路线 | 1x recall | 2x recall | 4x recall | 4x exact | 8x recall | 8x exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Endpoint score | 4.55% | 8.07% | 14.44% | 0% | 23.90% | 0% |
| Line score | 39.94% | 58.72% | 77.47% | 1.75% | 87.67% | 19.75% |
| Line + direct blocker | 32.07% | 44.06% | 56.30% | 0% | 71.35% | 1.50% |
| Line + gated blocker | 39.93% | 58.72% | 77.47% | 1.75% | 87.67% | 19.75% |

`line_score` 是预注册排序下的 winner，但 4x recall 比 95% 门低 17.53pp。stress 4x recall
为 70.88%，crossed-bonds 为 61.39%，分别低于 90%/85% 门；所有稀疏能力门失败。gated
blocker 与不加 blocker 几乎相同，direct blocker 则显著有害，说明宽候选中的假端点会污染
简单的“线段中间有点就删除”规则。

最终判定为 `pair_candidate_route_select_no_go`。这个结果不能通过把预算从 4x 增加到全量
18,340 pair/图来改写，因为那会恢复二次复杂度，也没有给后续分类留下可用 precision。

## 下一步

P2-D 不再手调 ribbon 权重或 blocker 阈值。先在已暴露 route-select 上审计“每个 endpoint
保留 top-K partner”的局部邻接候选，因为它不设最大键长，又能避免全局 top-K 被少数高分区域
占满。只有可控 pair 数下 Oracle recall 至少 95%，才训练一个读取完整 ribbon 序列、端点分数、
长度和 blocker 特征的轻量 pair scorer。

训练应使用 P2-B fit 内新的分子级 fit/calibration；P2-B 的 100 分子 holdout 继续保留，在
pair scorer 和阈值完全冻结后只读取一次。pair scorer 需要 listwise/hard-negative 目标，不能
继续用本轮手工线条总分冒充学习模型。

## 产物

- audit：`outputs/audits/chemvector-v1p2c-endpoint-pair-oracle/audit.json`；
- audit SHA256：`4188f36a408bd1f7595195571f6893add4d21d89772005dc2f23f60af1e4ee49`；
- rows：`outputs/audits/chemvector-v1p2c-endpoint-pair-oracle/rows.jsonl`；
- rows SHA256：`d7a2cee4bb7b7bb22847df9a8309b36f04a4b016a415dd53822fd9a92a2866f3`。
