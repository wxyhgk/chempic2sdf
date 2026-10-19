# Pair-scale 回退后的 chemical-head-only 探针

## 状态

**方案草案，不是训练授权，不可运行。** pair-scale 万图训练已按完整覆盖完成，但整图开发验收为 NO-GO；默认仍保留 initial。本方案只定义一个更小的机制验证，不复用已消耗的万图训练授权，不读取封存测试，也不改默认模型。

## 为什么下一步缩到 chemical 输出层

已保存的 pair 模型包含 2,266,565 个参数，其中 `chemical` 末端为 1,028 个参数、`presence` 末端为 257 个参数，共享 trunk 为 173,056 个参数。当前 `FrozenUpstreamPairStage` 会训练整个 pair 网络，presence 与 chemical 的输入表征因此同时变化。已保存 trace 又显示，pair-scale 退步主要伴随已选边键型/芳香提升漂移。

因此下一次只验证一个问题：**initial 的冻结 pair 表征是否已经包含足够键型信息，只需校准 chemical 末端即可改善整图结果？**

必须冻结 ribbon encoder、atom encoder、score encoder、shared trunk 和 presence 末端，只允许 `pair.chemical.weight` 与 `pair.chemical.bias` 更新。不能只冻结 presence 末端而更新 shared trunk；那仍会改变 presence logits，不能称为冻结 presence。

## 草拟配方

- 初始化：原始 initial checkpoint；不得从 `pair-015443.pt` 或任何失败 delta 续训。
- 训练数据：只使用已发布 expanded-release 中原有 train split；固定、确定性的 1,200 张不同图片，按原子规模、键型桶及绘图样式交错。不得读取 validation 内容来筛训练图，不新增身份查询或数据。
- 可训练参数：严格限定为 chemical 线性末端的 weight/bias（1,028 个参数）；其余 pair 与上游/OCR 参数全部冻结并逐项校验哈希。
- 目标：沿用现有 matched pair 标签，只对已知真键行做 unweighted chemical cross-entropy。presence BCE 为关闭状态；未知 pair 不作为负例；空监督图不做 optimizer step。
- 资源：GPU 0、FP32、无 autocast/TF32；最多 1,200 个样本访问与 optimizer updates，整个调用墙钟最多 1,800 秒，只允许一次调用，无续训、无自动重试。
- 检查点：initial、400、800、1,200 次访问；达到空监督时如实记录实际 updates。每个检查点在相同固定 944 张开发图上评估完整分子及课程域，并保留逐图变化。
- 推理：阈值、候选、芳香提升和编译路径一律不变。验证 presence logits 和选中 pair 集合与 initial 完全一致；任一偏差立即终止。

## GO / NO-GO

这是拟议的研究探针门槛，不是部署规则：

- 任一非 chemical 参数变化、presence 输出不一致、开发分母/身份变化、非有限 loss 或预算超限：立即终止，候选作废。
- 最终 full-molecule exact 必须严格高于 initial；full valid 不低于 initial；fallback 不高于 initial；课程 exact 最多下降 2/144。
- 同时逐图列出 broken/rescued、RDKit 错误类别、芳香原子/键和环数变化。任何新增 fallback 必须为零；不得用 aggregate exact 掩盖芳香退化。
- 未全部完成固定检查点，或不满足任一门槛：保留 initial，结束本探针，不自动加步数、不改阈值、不接第二阶段。
- 即使通过，也只得到“冻结表征上的 chemical 末端有可用改进信号”；固定开发集已反复用于研究，不能据此声称独立泛化或授权部署。

## 开跑前仍需完成

1. 建立独立的新 launcher / execution contract，精确绑定 1,200 张清单、initial、train manifest、固定开发集、代码哈希和 1,800 秒预算；不能修改旧 pair-scale attempt 或伪装成已获授权。
2. 增加 CPU 合成测试，证明 optimizer 参数集合恰为 chemical weight/bias；presence 和共享 trunk 在训练前后逐位不变；presence logits 与 selected pair 集合保持一致；没有已知真键化学标签时不更新。
3. 用只读、无模型更新的 dry-run 估算此协议完整时间；超过 1,800 秒则缩小评估实现成本或不运行，不自动提高预算。
4. 为这一次新用途建立明确授权。当前记录只授权并记录了已经完成的万图 pair-only 实验，**没有授权该探针训练**。

## 证据

- [万图训练结果](20260925-pair-scale-training-result.md)
- [逐 pair trace](20260925-pair-scale-trace-result.md)
- [输出层退步归因](20260925-pair-scale-attribution.md)
- [芳香阈值审计](20260925-pair-scale-aromatic-threshold-audit.md)
- [pair-stage 参数范围](../../../chemvector/training/curriculum/pair_stage.py)
- [pair 网络末端定义及损失](../../../chemvector/components/graph/joint/direct_atom_pair_bond.py)
