# v1 P2-D0 局部 Partner Oracle

> 日期：2026-08-14
>
> 状态：已预注册，等待审计
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
