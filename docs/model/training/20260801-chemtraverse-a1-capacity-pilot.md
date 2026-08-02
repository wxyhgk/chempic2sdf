# 2026-08-01 ChemTraverse A1 Capacity Pilot

> 状态：P0/P1 通过，P2 与 P2-B NO-GO；禁止启动正式 50K，不是正式模型版本
> 禁止读取：fresh_dev、sealed_test、MR-TADF、USPTO（capacity 阶段全部不使用）

## 目的

在正式 50K A/B 前，先证伪“50M ChemVector + ChemTraverse v0.1 是否能自由生成完整遍历序列”。
该阶段只验证表示、grammar、训练链路和基本未见分子泛化，不承担生产对照结论，不创建 Git tag。

## 初始化

- 模型：默认 ChemVector，50,165,232 参数，最大长度 3,072。
- 视觉编码器：`outputs/checkpoints/chemvector-v4-clean-50k/last.pt`。
- 视觉 checkpoint SHA256：
  `65a443d26bbffd41359604ac8147fa62a66ee18ca877693ebb3dea0b0a637095`。
- 只加载 `vision_encoder.*` 的 116 个 tensor。
- ChemTraverse token embedding、位置 embedding、decoder 和输出 projection 随机初始化。

## 数据

从冻结 50K manifest 按完整分子组确定性选择：

- `overfit_32`：8 个分子、32 张图。
- `train_5k`：1,250 个分子、5,000 张图，包含 `overfit_32`。
- `dev_500`：125 个新分子、500 张图，与 `train_5k` molecule ID 零重叠。

三份清单位于 `Data/sdf/chemtraverse_v0_1_capacity/`。所有 variant 保持四张完整，不按图片
随机拆分。

## 阶段门

### P0：运行时

- 默认 50M、batch 4 可完成最长目标的 forward/backward/update。
- periodic 和 `last.pt` 都能从中途 step 确定性恢复。
- P0/P1/P2 均使用 `--deterministic`；连续运行与中途恢复后的模型、优化器 tensor
  必须逐项一致。
- grammar generation 和 evaluator 能输出 ChemScene、SDF 与分阶段错误。

### P1：32 图过拟合

- teacher-forced token accuracy >= 99.5%。
- 32/32 grammar valid、ChemScene parseable。
- 至少 30/32 自由生成 sequence exact；未达到则停止，不运行 5K。

#### P1-A 结果

- 配置：`3e-4` 固定学习率，200 epochs，1,600 optimizer steps。
- teacher-forced token accuracy：99.9646%，通过。
- grammar valid / ChemScene parseable：32/32，通过。
- SDF valid：31/32；chemical match：27/32。
- 自由生成 sequence exact：25/32，未达到 30/32，P1-A 失败。
- 7 个非 exact 首分歧中，2 个是坐标 token 且化学结构一致；5 个是
  `role:TEXT -> role:IMPLICIT`，说明残差集中在可见文字节点角色。

训练曲线在约 step 1,000 后接近饱和，但固定 `3e-4` 仍有明显振荡。为区分“学习率未退火”与
“架构/目标不可学”，允许一次预先限定的 P1-B：从 P1-A `last.pt` 以 `1e-4` 继续 100 epochs
（800 steps）。P1-B 仍使用相同 32 张图和原门槛；若仍不足 30/32，则停止，不再追加步数，
也不运行 P2。

P1-B 最终结果：teacher token accuracy 99.9949%，grammar/scene 32/32，自由生成 sequence
exact、count-pair exact、Skeleton exact、Atom/Bond typed exact 均为 30/32，P1 通过。剩余
2 条都是 count mismatch，并产生高价态无效分子。

### P2：5K 未见分子

- 只在 `dev_500` 选固定终点，不读取正式 fresh_dev。
- P2 不继承 P1 的 decoder；从相同视觉 checkpoint 独立初始化，避免 32 图记忆污染泛化结论。
- 固定训练 3 epochs：15,000 张图曝光、3,750 optimizer steps、batch 4、学习率 `3e-4`。
- 每个 epoch 对完整 500 张 dev 做 teacher-forced 评估；自由生成只运行固定最终 checkpoint。
- grammar valid >= 95%。
- ChemScene parseable >= 90%。
- count-pair exact 和 Skeleton exact 必须显著高于随机/空输出；具体结果如实报告，不因结果
  追加数据或改变 dev。

#### P2 结果

- 训练：独立视觉初始化，5K train，3 epochs，3,750 steps。
- 完整 dev teacher token accuracy：76.83% -> 79.44% -> 81.43%。
- 自由生成：完成/grammar/scene 356/500（71.2%），SDF valid 13/500（2.6%）。
- count-pair exact 2/500（0.4%）；Skeleton、Atom/Bond typed、chemical exact 全部 0/500。
- 首因分桶：144 个 generation incomplete、354 个 count mismatch、2 个 topology mismatch。

P2 未通过，禁止进入正式 50K。失败集中在遍历的计数、停止和状态控制；下一步先审计控制 token
与属性/坐标 token 的条件准确率，再验证损失加权或抗暴露偏差训练，不能直接追加数据或 epochs。

控制 token 审计：`COMPONENT_END` 40.2%、`DOWN` 81.3%、`LINK` 74.3%、`UP` 94.5%，
而元素和 chemical bond type 分别为 96.3% 和 96.9%。因此预注册 P2-B：相同初始化、数据、
3 epochs 和学习率，只把 traversal control token 的交叉熵权重设为 4。先在固定前 25 个 dev
分子（100 张）route-select；基线为完成 72/100、SDF valid 2/100、Skeleton 0/100。P2-B
必须完成至少 85/100 且 Skeleton 至少 1/100，才运行其余 400 张 confirm。

P2-B route-select 实际为：完成/grammar/scene 56/100、SDF valid 5/100、count-pair、Skeleton、
typed graph、chemical exact 全部 0。相较基线完成 72/100 明显退化，未达到门槛，因此不运行
confirm 400。teacher 审计中 `COMPONENT_END` 从 40.2% 降到 6.6%，`LINK` 从 74.3% 降到
62.7%；统一乘 4 的控制损失被高频 `DOWN/UP` 主导，不能解决稀有结束决策。P2-B NO-GO。

下一实验不得继续统一加权或直接扩量。应在以下两条中先做小规模预注册对照：按控制符频率
分别加权（重点是 `COMPONENT_END`/`LINK`），或把 count/stop 作为独立集合级预测而非让长序列
交叉熵间接学习。两者都必须先过同一 route-select 100，不能读取 confirm、fresh dev、MR-TADF
或 sealed test 来选方案。

并行复核与后续协议见：

- `20260801-chemtraverse-p2-count-stop-forensics.md`：count/stop 取证与 GT-count oracle；
- `20260801-chemtraverse-p2-objective-review.md`：grammar-aware decision loss 与 context corruption；
- `docs/research/20260801-chemtraverse-decoder-route-review.md`：G0-G3 四臂解码路线矩阵；
- `20260801-chemtraverse-observable-meta-pilot.md`：不可观察 molecule_id 常量化实验，route-select
  NO-GO，但该协议修复保留。

只有 P1/P2 证明表示可学习后，才冻结正式 A1 50K 配置。A0 公平对照和 A2 真正的 Text/Bond
mask 辅助需要另行实现，不能把现有生产 pipeline 或 16 通道 visual auxiliary 冒充对应实验。
