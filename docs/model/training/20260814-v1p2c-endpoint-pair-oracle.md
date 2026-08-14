# v1 P2-C 端点对候选 Oracle

> 日期：2026-08-14
>
> 状态：已预注册，等待审计
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
