# G3 P19-E3 Direct AtomPair Presence Pilot20k

> 日期：2026-08-10
>
> 状态：已完成，Pilot20k NO-GO
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

## 实际结果

从 full80k 中有 13,576 个 molecule 同时满足非 pilot molecule、非 pilot scaffold；固定 SHA
选出 160 个 molecule、640 图。fit/dev molecule 与 scaffold overlap 均为 0，四种 variant
各 160 张。训练完整消费 20,000 图、10,000 step、9,698,720 个 AtomPair 与 656,556 条
真键；全部真键均在完整 pair universe 中。

同一 internal dev640 的旧 v0.10.3 固定基线为：

- overall 531/640，Edge precision/recall 99.34%/98.94%；
- 普通三布局 458/480；
- stress 73/160；
- crossed 0/38、crowded 31/36、long 31/39、near-nonbonded 11/47。

scratch 模型曲线：

| Samples | Overall | 普通三布局 | Stress | Crossed | Near | FP / FN |
|---:|---:|---:|---:|---:|---:|---:|
| 5k | 343/640 | 306/480 | 37/160 | 4/38 | 0/47 | 3,393 / 0 |
| 10k | 393/640 | 348/480 | 45/160 | 5/38 | 2/47 | 1,712 / 4 |
| **15k** | **437/640** | **386/480** | **51/160** | **5/38** | **3/47** | **1,391 / 4** |
| 20k | 382/640 | 342/480 | 40/160 | 2/38 | 0/47 | 3,874 / 2 |

step 7,500 / 15k samples 为 best。其 Edge precision/recall 为 93.77%/99.98%，相对旧基线
逐图为 11 rescue / 105 break；crossed 单项为 5 rescue / 0 break，但 near-nonbonded 为
2 rescue / 10 break。固定 best 两次重放 digest 相同，冻结 state 未变化。

## 决策

正式决策为 `direct_atom_pair_presence_pilot20k_no_go`。split、冻结状态、重复重放与 crossed
增益门通过；overall、普通布局、stress、near-nonbonded 和 paired 净增益门失败。

本轮修正了“只是 64 张太少”的解释：20k 确实把 scratch presence 的跨 scaffold exact 提高，
但 balanced BCE + listwise margin 长期维持接近 100% recall，同时留下大量 FP；20k 终点还从
best 的 437 回退到 382。数据量有作用，但当前首要断点是 presence logit 的可分性/校准和假边
抑制，不应直接扩到 full80k。

下一步先冻结 best 做独立 calibration/eval 审计：从未使用的 full80k 余量再取两组 scaffold
隔离样本，只在 calibration 选一个全局 threshold，在 eval 固定重放。若阈值可恢复到或超过旧
基线，说明模型学到关系但 0.5 校准不对；若所有阈值仍有大量 FP，则停止当前 balanced/listwise
目标，改为保留旧 BCE prior 的 residual/hard-negative 训练或整图 pair context。

## 产物

- best：`outputs/checkpoints/chemvector-g3-p19e3-direct-atom-pair-presence-pilot20k/best.pt`；
- best step/samples：7,500 / 15,000；
- best SHA256：`ab25e9de7294e48baa22797a5800c9208de4af09d9fd5ef71410773cf9eaf1e0`；
- result SHA256：`e92bc27a602838fdfba543887a1e010e87abc73eac93f3e9f1bb5bd85c962a0c`；
- dev manifest SHA256：`4e2cefe593ad81c856ef254475309db5fbf33b17361fc24d9a538bc0380ab6f7`；
- best prediction SHA256：`1e8cd675f3b85eff67372cafdc71176e60c37a171127fd25714e8a39b30011e5`；
- 可训练参数：2,265,537；
- 总耗时：545.48 秒。
