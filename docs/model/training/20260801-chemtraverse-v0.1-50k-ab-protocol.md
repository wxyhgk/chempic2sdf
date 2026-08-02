# 2026-08-01 ChemTraverse v0.1 50K 等预算 A/B 协议

> 状态：协议、真实标签、50K 清单、tokenizer/loader 与单步训练 smoke GO
> 当前生产对照：v0.10.21
> MR-TADF：只作外部主应用 benchmark，不参与训练、阈值或 checkpoint 选择

## 目标

验证“原子/键交替遍历”是否比当前独立 Atom-set + Bond-set 更容易学到整图对象数量与拓扑。
本轮只允许改变输出表示和低权重视觉辅助，图片、训练样本、视觉编码器规模、优化步数、batch、
学习率计划和随机种子保持一致。

## 固定数据

- 训练：`style_train_100k` 中确定性预检通过的前 50,000 张。
- 验证：`fresh_eval_12k/fresh_dev`，只读 8,000 张。
- 禁止读取：`fresh_eval_12k/sealed_test`。
- 外部报告：训练和候选选择全部完成后，只运行一次 MR-TADF v1 1,100；USPTO clean 1k 作为
  风格 OOD 护栏，不用于调阈值。
- 所有 A/B 使用完全相同的 sample key 清单；源拒绝项从后续样本确定性补足。

冻结清单位于 `Data/sdf/chemtraverse_v0_1_ab_50k/`：扫描 50,273 张，拒绝 273 个坏源目标，
最终保留 50,000 张、12,569 个唯一分子。`samples.jsonl` SHA256：
`bd1108b2657de83bf615a4195b0257bab076c398ddbfab1921a947099bee4bac`。

## 三组实验

| 组 | 输出目标 | 坐标/Text-Bond 辅助 | 用途 |
|---|---|---|---|
| A0 | 当前 Atom-set + Bond-set | 当前配置 | 等预算基线 |
| A1 | ChemTraverse v0.1 | 无额外辅助 | 检验交替序列本身 |
| A2 | ChemTraverse v0.1 | 低权重坐标 + Text/Bond mask | 检验视觉接地辅助 |

A1/A2 都编译到同一个 ChemScene/SDF，不增加后处理型键长阈值、最近邻候选删除或 MR-TADF
特征规则。A2 的辅助损失不得参与推理解码，也不得改变主序列真值。

## 等预算约束

1. 三组各看 50,000 张图片一次，优化器 update 数一致。
2. 编码器参数量差异不超过 5%；decoder 新增参数和实际 FLOPs 单独报告。
3. 使用同一初始化视觉编码器；A1/A2 的新 decoder 随机初始化并记录 seed。
4. 只允许在 fresh_dev 选择 checkpoint；每组预先固定一个训练终点和最多一个最佳 checkpoint。
5. 不因某组收敛较慢而追加数据或 step；需要扩量时另立正式实验。

## 必报指标

- ChemTraverse grammar valid、ChemScene parseable、SDF valid。
- Atom count exact、Bond count exact、count-pair exact。
- Skeleton exact、Atom-typed exact、Bond-typed exact、Chemical exact。
- 数量正确条件下 Skeleton exact。
- 重复原子、漏原子、多余键、漏键、文字误作键按首因分桶。
- 单图 CPU/GPU 延迟、峰值显存、参数量和序列长度分位数。

## GO/NO-GO

新路线至少满足以下条件才进入更大训练：

1. fresh_dev 的 count-pair exact 相对 A0 提高至少 5 个百分点。
2. Skeleton exact 相对 A0 提高至少 5 个百分点，且数量正确条件 Skeleton exact 不下降超过
   1 个百分点。
3. Chemical exact 不低于 A0，SDF valid 不下降超过 2 个百分点。
4. MR-TADF 首因中的“原子集合多漏”和“交叉多余键”合计至少下降 15%，且不以 OCR 错误显著
   增加为代价。
5. 不允许读取 sealed_test，也不允许按 MR-TADF 结果回选 checkpoint。

若 A1 不优于 A0，则交替表示本身 NO-GO；若 A2 优于 A1，说明需要轻量视觉辅助；若 A1 已
明显优于 A0 而 A2 无增益，则保留更简单的 A1。只有产生正式 checkpoint 时才分配下一个模型
版本，并在提交代码、配置和训练记录后创建与模型名完全相同的 annotated Git tag。

## 实现 Smoke

ChemTraverse tokenizer 已支持结构记录、0-1023 坐标、118 种元素、dummy atom、开放 UTF-8
字节、bbox、闭环和立体引用。真实首样本 `train_000000_coordgen` 编码为 661 token，解码后与
规范序列逐字节一致。

50K 全量 token 审计为：最小 261、P50 781、P95 1,451、P99 2,144、最大 2,874，超过当前
3,072 上下文的样本为 0，因此 A/B 不需要按长度过滤或扩展位置 embedding。默认配置词表
2,525、模型 50,165,232 参数。

使用冻结 manifest 在 RTX 5090 上完成 1 个 tiny forward/backward/update：5,154,032 参数，
step-1 loss 8.053930，训练链路无异常。该输出位于
`outputs/smoke/chemtraverse-v01-train-step1/`，只验证接口，不是候选模型，不创建 tag。
