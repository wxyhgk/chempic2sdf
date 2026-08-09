# G3 P19-E1 Direct AtomPair Capacity64

> 日期：2026-08-10
>
> 状态：已完成，容量 GO
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

## 实际结果

冻结前向扫描 120 张 train-only stress 图后，确定性选出 64 个唯一 molecule：32 easy、
16 crossed_bonds hard、16 near_nonbonded hard。与 fresh640 的 molecule/scaffold overlap
均为 0；全部 pair 数严格为 `nC2`，1,921 条 GT Bond 全部存在于 pair universe，冻结 state
未变化。

旧 v0.10.3 在同一集合的 skeleton exact 为 32/64；easy 为 32/32，hard 为 0/32。hard 中：

- crossed_bonds：Edge precision/recall 94.38%/85.78%，0/16 exact；
- near_nonbonded：Edge precision/recall 96.39%/97.39%，0/16 exact。

scratch 模型的学习曲线为：

| Step | Overall exact | Easy | Hard | Crossed | Near-nonbonded |
|---:|---:|---:|---:|---:|---:|
| 0 | 0/64 | 0/32 | 0/32 | 0/16 | 0/16 |
| 750 | 40/64 | 28/32 | 12/32 | 7/16 | 5/16 |
| 1,500 | 61/64 | 31/32 | 30/32 | 15/16 | 15/16 |
| 1,750 | 63/64 | 32/32 | 31/32 | 15/16 | 16/16 |
| **2,000** | **64/64** | **32/32** | **32/32** | **16/16** | **16/16** |

step 2,000 首次达到 64/64，按预注册字典序成为 best；后续 step 2,250--3,000 也保持
64/64，但不替换更早 checkpoint。best 对全部 1,921 条真键达到 100% precision、recall 和
F1，0 FP、0 FN；连续两次固定 checkpoint 重放的 prediction SHA256 完全相同。

正式决策为 `direct_atom_pair_capacity64_go`，所有预注册门通过。模型从随机初始化开始，
没有加载旧 Direct AtomPair 权重；实际可训练参数为 2,265,537，训练与 cache 构建总耗时
39.15 秒。chemical head 冻结，因此 typed exact 为 0，不属于本轮结论。

## 工程复核

首次运行在 cache 构建完成、训练 step 0 前停止：FP16 cache 与 BF16 autocast 在
`grid_sample` 中形成不支持的混合签名。修复为 cache 仍以 FP16 落盘，但搬到 GPU 时先恢复
FP32，再进入 BF16 autocast；单图 CUDA forward/backward smoke 与完整测试通过。失败运行未
更新任何模型参数，产物保存在同名 `-failed-fp16-autocast` 目录用于审计。

## 产物与下一步

- best：`outputs/checkpoints/chemvector-g3-p19e1-direct-atom-pair-capacity64/best.pt`；
- best step：2,000；
- best SHA256：`b22f23d1490e223a759a86dd230afd50d5e7f91e8036ecb45dbf22c6e11f9d47`；
- capacity cache SHA256：`f9987765b92d6b84db7048e0ccd871539e82c28ffb46014e5e7dbd69dfbd5d21`；
- sample key SHA256：`00abcb2575e8880ddacab5463ef2093e4cd1e80676f650d7e5d398f533ead32a`；
- result SHA256：`08c11a11c3521da93e98a347d9833bd39344631f29fc6a176f5146570ce72305`。

这仍然只是 64 张图上的容量结论。下一步 P19-E2 必须固定本轮 best、固定 threshold 0.5，
在未参与训练与选模的同一 fresh640 上零训练重放；不得根据 fresh640 重新选 step、调阈值或
改损失。若回退，结论是 pair-local 表示的跨 scaffold 泛化不足，不是否定 AtomPair identity。
