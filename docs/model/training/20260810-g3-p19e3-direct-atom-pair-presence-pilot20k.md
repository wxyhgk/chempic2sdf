# G3 P19-E3 Direct AtomPair Presence Pilot20k

> 日期：2026-08-10
>
> 状态：已预注册，等待运行
>
> 模型：`chemvector-g3-p19e3-direct-atom-pair-presence-pilot20k`

## 目标

P19-E1 的 64 张容量集达到 64/64，但 P19-E2 fresh640 只有 305/640；旧 v0.10.3 用 10k 图
训练后为 469/640。当前直接证据更符合训练覆盖不足，而不是 AtomPair identity 或模型容量失败。
本轮将相同 scratch presence 路径扩大到 20k 图，但不再使用 fresh640 选模型。

## 数据边界

- fit：P4 `pilot_train.jsonl` 全部 5,000 分子、20,000 图；
- internal dev：从 `full_train.jsonl` 中先删除 pilot 的全部 molecule 及所有同 scaffold 分子，
  再按固定 SHA256 排序选 160 个分子、640 图；
- fit/dev molecule 与 scaffold overlap 必须均为 0；
- internal dev 只用于本轮 checkpoint 选择，不属于外部或 sealed test；
- 不读取 fresh640、confirm_v2、sealed test、USPTO 或 MR-TADF。

## 固定训练协议

- 冻结 v0.10.1 vision/Atom，使用 GT Atom anchor 隔离 relation；
- Direct AtomPair 全路径从随机初始化开始，禁止加载旧 v0.10.3 或 P19-E1 参数；
- 每图枚举并训练全部 `nC2` 无序 AtomPair，不做距离、近邻、键长或中点裁剪；
- presence-only：每图正/负 BCE 各占一半，加权重 0.5、margin 1.0 的整图排序损失；
- chemical head 冻结，不训练类型；不运行 Dense Bond、endpoint 或 incidence；
- batch 2、单 epoch 20,000 images/10,000 steps、AdamW `2e-4`、weight decay 0、grad clip 1；
- 固定 threshold 0.5，每 5,000 张图评估一次；checkpoint 按 overall exact、stress exact、
  TP、少 FP、少 FN 的字典序选择；
- 旧 v0.10.3 只在同一 internal dev 用固定 threshold 0.85 运行一次，作为相对基线。

## GO/NO-GO

- internal dev 必须恰好 640 图且 fit/dev molecule/scaffold overlap 为 0；
- 冻结 state 不变，固定 best 连续两次重放 digest 相同；
- overall skeleton exact 不低于旧 v0.10.3；
- coordgen/original/rdkit 合计不低于旧基线；
- stress 至少比旧基线多 8/160；
- crossed_bonds 与 near_nonbonded 分别至少比旧基线多 2 张；
- paired rescue 必须多于 break。

全部通过才允许考虑扩大到 full80k 或另开一次固定 fresh 审计。若 20k 后普通分布仍回退，不能
再把问题归因于“只训练了 64 张”；下一步应增加整图 AtomPair context，而不是继续同架构堆数据。

无论 GO/NO-GO 都保存 checkpoint、写回结果，并创建与模型名完全相同的 annotated Git tag。
