# 连键存在性专项：冻结特征，只接线现有输出层

**已实现独立的 presence-readout 损失原型，并完成 CPU 合成接线检查。没有加载项目 checkpoint、运行新真实图片推理或执行真实训练；没有新增识别准确率。** 上一轮定向 24 图的 10/24、固定完整开发集 170/800 均保持原记录。

用户确认“键连比较重要”后，本轮只推进上一轮建议的最小试验准备。没有将该确认解释为新 GPU 预算、数据训练准入或部署授权。

## 本轮实现

新增 [pair_presence_stage.py](../../../chemvector/training/curriculum/pair_presence_stage.py)，不修改生产模型、旧 `FrozenUpstreamPairStage`、默认 continuation 或 launcher。

`FrozenFeaturePresenceStage` 由调用者提供独立模型实例，保持原预测节点、筛选阈值和 pair 枚举；只允许现有 `pair.presence` 线性层的 **weight、bias 两个参数张量**产生梯度。不是新建一个专家或新分类头。

| 路径 | 本原型的处理 |
| --- | --- |
| vision / atom / bond / set | eval + 冻结参数 + no_grad；预测候选不变 |
| pair 的 ribbon / atom / score encoder 与 shared | eval + 冻结参数 + no_grad |
| pair.chemical | eval + 冻结参数；原始 chemical logits 不参与损失 |
| pair.presence | 用原 pair 输出的 detached hidden 重算同一线性层，只计算存在性 BCE |
| role / bbox | 保留冻结，本目标不调用 |
| OCR | 不加载、不调用、不更新 |

完整原 pair forward 在 `no_grad` 下运行，保留原行序；重新计算的 presence logits 必须与该路径逐位相等。所有预测 pair（含对应关系未知的 pair）都由预测特征计算，GT 只决定损失标签和资格，不改变预测输入。

沿用已有正类加权 BCE：`pos_weight = clamp(negative_count / positive_count, 1, 16)`，分母计数保持旧的最小 1 约定；去掉 chemical loss，不伪造一个 0 型分类损失。**没有修改 0.85 推理阈值**或引入新排序损失。

## 正负例与安全边界

- 复用原 6 px 匹配和 `pair_targets_with_unknown()`，**保留全部确认负例**，不抽掉跨中间原子的困难非边；真实正键进入同一存在性损失。
- 未匹配端点的 pair 不伪造为负例。未知 pair、未匹配节点、不可达目标键都继续计数，不通过改变分母隐藏。
- 无可用 known pair 时返回显式 `no_known_pairs`，不提供假的零损失或虚假 optimizer step。
- 冻结路径被外部切回 train、意外解冻、产生残留梯度，或 presence 参数被替换/与其他分支共享，均拒绝执行。
- loss 模块没有数据 loader、训练准入能力、optimizer、`train_step()` 或 CLI。它不是正式训练入口；旧 launcher 仍拒绝未登记的 `presence-readout-v1` 配方和未准入数据。
- 在同一模型与同一预测输入下，只更新 presence 可保持 **raw chemical logits** 不变。但下游芳香环提升和编译会依赖边存在性，**不保证最终键型、有效性或 SDF 身份不变**；正式验收仍要观察整图。

## 本轮验证是什么、又不是什么

[新增合成测试](../../../tests/chemvector/test_curriculum_pair_presence_stage.py)使用随机初始化的小尺寸**现有** `DirectAtomPairBond` 和合成上游张量，不读取项目 checkpoint、真实分子图片、身份索引或封存内容。

已验证：

- 正键的 logit 梯度向上、确认非边向下、未知端点 pair 梯度为零。
- backward 本身不改权重；无梯度传到输入、共享特征、chemical、视觉或其他头。
- 仅在临时随机小模型上执行一次 SGD 更新，改变的 state 恰为 `pair.presence.weight` 和 `pair.presence.bias`；hidden 和 raw chemical logits **逐位不变**。
- 换目标键型、不改连接关系时，presence loss 和特征不变；原/拆分 readout 一致。
- 重复 train/eval、全未知、零/单候选、异常数值、参数共享和外部模式修改等边界。

这次合成更新只证明参数隔离和损失接线正确，**不证明初始真实特征线性可分，也不证明这两个张量足以修复 N–N/N–S**。若后续定额试验不能同时降低漏连和误连，应记录 readout-only 路径的不足，不自动扩大到联合重训。

新增 **36 项合成测试**；专项及相邻测试 **128 passed / 1.46 秒**，更广相关回归 **725 passed / 57.12 秒**。两组计数有包含关系，不相加，也不是全仓库测试声明。原 initial SHA256、既有 release manifest、24 图报告及该报告绑定的全部 284 个输入保持不变。

专项复现命令（只执行合成单测，不启动真实试验）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-train/bin/python -m pytest -q \
  tests/chemvector/test_curriculum_pair_presence_stage.py \
  tests/chemvector/test_curriculum_pair_stage.py \
  tests/chemvector/test_curriculum_supervision_contract.py \
  tests/chemvector/test_curriculum_pair_gap_diagnostic.py
```

## 下一轮建议，尚未开跑

1. 完成本批固定 **17 个新身份 + 7 个旧训练控制**的来源用途、训练发布及清单绑定；复用已完成的排重回执，**不得重查私有身份或自动补候选**。原 quarantine 不改名放行。
2. 只申请一次 **CPU ≤300 秒、≤200 个 optimizer steps**的可学习性小试验，GPU=0、无自动重跑。此数值为提案，不是已有授权或已经接通的运行入口。优先在一次固定前向后缓存预测 pair 的冻结 hidden，避免每个梯度步都重算视觉网络。
3. 保留所有 24 张图及原错误分母，同时看真键漏连、确认非边误连、整图救回/改错；不能只报告训练 loss 下降。24 图若参与拟合，之后的结果必须标为训练侧，不得当成独立准确率。
4. 后续回归沿用原固定验证资料，不新建“好看”的测试子集。小试验不产生部署决定；任何有效性/整体提升声明仍需要固定完整开发集的严格身份验证。

三键 `C#C` 的端点筛选、`N=N` 的类型错误、`NH→NHH` 的文字解析错误仍是独立限制。presence-only 不更新这些路径；本轮没有宣称这些问题已修复，也没有开始扩大数据或更换模型架构。

前序：[24 图诊断结果](20260918-pair-gap-diagnostic24.md)、[固定候选身份排重](20260918-pair-gap-identity-review.md)。
