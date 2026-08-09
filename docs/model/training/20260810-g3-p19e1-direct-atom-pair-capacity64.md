# G3 P19-E1 Direct AtomPair Capacity64

> 日期：2026-08-10
>
> 状态：已预注册，等待运行
>
> 模型：`chemvector-g3-p19e1-direct-atom-pair-capacity64`

## 目标

P19-E0 证明旧 v0.10.3 在 GT Atom 下仍有 98.60% Edge F1，但 skeleton exact 只有
73.28%，stress 只有 40%，其中 crossed_bonds 为 0/41。当前需要区分两种解释：旧权重训练
不足，还是 pair-local ribbon 架构本身连 64 张图都无法表达这些困难关系。

本轮只做 train-only capacity64。通过只表示架构与视觉输入有拟合容量，不表示已经泛化，
更不表示可以部署。

## 固定协议

- 数据只来自 P4 pilot train 的 stress 图，不读取 fresh 标签、sealed test、confirm、USPTO
  或 MR-TADF；
- 使用冻结视觉编码器与 GT Atom anchor；GT Atom 只用于隔离 relation 容量，不是部署路线；
- 32 easy：旧 v0.10.3 在固定 threshold 0.85 下 skeleton exact；
- 32 hard：旧模型 skeleton 不正确，其中 crossed_bonds 16 张、near_nonbonded 16 张；
- 每张图枚举并监督全部无序 AtomPair，不使用距离、最近邻、键长或空白中点裁剪；
- 不运行 Dense Bond、endpoint regression 或 incidence；
- Direct AtomPair 架构从确定性随机初始化开始，禁止加载 v0.10.3 参数；旧模型只用于构造
  easy/hard 分桶；
- 输入沿用 96D 冻结视觉 feature map、GT anchor 采样的 128D Atom feature、Atom score 与
  32x9 ribbon；只训练 presence，GT Bond 字段不进入 forward；
- 每张图的全部正/负 pair 分别求均值并各占 BCE 的一半，另加权重 0.5、margin 1.0 的整图
  正负排序损失；不做负采样，chemical type 本轮只报告、不作为训练或容量结论；
- batch 1、3,000 step、AdamW `2e-4`、weight decay 0、grad clip 1、固定 threshold 0.5；
- 每 250 step 评估；checkpoint 依次按 easy exact、hard exact、TP、少 FP、少 FN 选择。

## GO/NO-GO

- cache 必须恰好为 32 easy + 16 crossed + 16 near-nonbonded；
- 所有 GT Bond 必须存在于完整 AtomPair universe；
- 64 个 molecule 必须唯一，且每图 pair 数严格等于 `n*(n-1)/2`；
- 与 P18 fresh640 的 molecule、scaffold overlap 必须均为 0；
- 冻结视觉、Atom 与 selection-only 旧 relation state 必须不变；
- 固定 best checkpoint 连续两次全量重放的 prediction SHA256 必须完全相同；
- best easy 必须 32/32，零 break；
- best hard 至少 28/32；
- best overall 至少 60/64；
- crossed_bonds 与 near_nonbonded 必须分别至少 14/16；
- 总体 Edge precision 与 recall 必须分别至少 99.5%。

全部通过才判定 capacity GO，并用同一个固定 best checkpoint 在 P19-E2 重放 fresh640。
若 capacity NO-GO，说明当前 pair-local ribbon 表示连小集合都不足，下一步先改表示；若 capacity
GO 但 fresh640 仍低，则说明主要问题是跨图泛化，应增加整图 AtomPair context，而不是退回
Dense Bond endpoint 或盲目扩大同架构数据量。

本轮无论 GO/NO-GO 都保存 checkpoint、写回完整结果，并创建与模型名完全相同的 annotated
Git tag；tag 注释记录 checkpoint 路径、SHA256、64 张训练图和决策。
