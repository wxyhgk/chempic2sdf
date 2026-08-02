# 2026-08-01 ChemTraverse 可观察 Meta Pilot

> 状态：route-select NO-GO；协议修复保留，不运行 confirm 400
> 禁止读取：confirm 400、fresh dev、sealed test、MR-TADF、USPTO

## 假设

ChemTraverse v0.1 把数据库 `molecule_id` 放在自回归序列开头，但分子图片不包含该 ID。
P2 的 356 个完成样本中 ID exact 为 0，模型只复用了少数训练 ID；teacher forcing 随后使用真实
ID，推理则从错误 ID 前缀继续，形成不必要的 exposure shift。

`molecule_id` 不属于 SDF 化学结构。实验保留原 ID 于 ChemMol/meta sidecar，只把模型目标中的
ID 固定为 `sdf`。默认 loader 行为保持不变，只有显式传入 `--target-molecule-id sdf` 才启用。

## 固定对照

- 基线：ChemTraverse P2，5K train / dev500，3 epochs，3,750 steps。
- 候选：相同视觉 checkpoint、seed、数据、batch 4、学习率 `3e-4` 和等权 CE。
- 唯一变量：目标序列的 molecule_id 从源 ID 改为常量 `sdf`。
- 不使用 P1/P2 checkpoint 初始化；仍只加载固定视觉编码器的 116 个 tensor。
- 不使用统一 control weight，也不加入 count head、token corruption 或新解码约束。

## Route Select

只评估固定 `dev_route_select_100.jsonl`。基线为：完成 72/100、SDF valid 2/100、count-pair、
Skeleton、typed graph、chemical exact 全部 0。

候选必须同时达到：

- completed / grammar / scene >= 85/100；
- SDF valid >= 10/100；
- count-pair exact >= 5/100；
- Skeleton exact >= 3/100；
- typed graph exact >= 1/100。

任一不满足即 NO-GO，不运行 confirm 400。全部满足后才运行固定 confirm 400；不能按中间结果
改阈值、追加 epochs 或更换样本。

## 结果

- 3 个 epoch 的完整 dev teacher token accuracy：77.49%、79.99%、81.97%；原基线终点为
  81.43%。常量 meta 只带来约 0.54pp 提升。
- route-select100：completed/grammar/scene 67/100，SDF valid 1/100，count-pair 1/100，
  Skeleton、typed graph、chemical exact 全部 0。
- 首因：33 个 generation incomplete、66 个 count mismatch、1 个 topology mismatch。
- 未达到 completed 85、SDF 10、count-pair 5、Skeleton 3、typed graph 1 的任一核心门槛，
  因此 NO-GO，不运行 confirm 400。

结论：不可观察 molecule_id 是真实协议缺陷，固定 sentinel 应作为后续所有视觉模型的共同前置
修复；但它不是零 Skeleton 的主要充分原因。下一步进入 GT-count oracle 和 grammar-aware decision
loss，不得以本实验为理由直接追加数据或 epochs。
