# Text-center sidecar v2：严格协议结果

## 目的

验证 `text-center + center→chemical-anchor offset` 是否能在不改变 production
baseline 的前提下，补回文字原子端点。该试验只从冻结的
`initial.pt` 提取特征，未写入 checkpoint，也未读取 sealed test。

## 协议修正

- train/holdout 按 `group_key`、`molecule_id`、canonical SMILES 同时隔离；
- 正样本只包含 residual text，已有 baseline atom 不作为负样本；
- AP 按候选分数逐图计算；
- 追加候选不进入旧 set-head 的 selector context，单独检查 prefix ownership。

## 结果

| split | K | candidates | positives | precision | mean candidate AP |
|---|---:|---:|---:|---:|---:|
| train | 16 | 1,920 | 1 | 0.052% | 0.00278 |
| train | 32 | 3,840 | 1 | 0.026% | 0.00278 |
| holdout | 16 | 17,280 | 13 | 0.075% | 0.00219 |
| holdout | 32 | 34,560 | 18 | 0.052% | 0.00231 |

数据切分：`group_overlap=0`、`molecule_overlap=0`、
`canonical_smiles_overlap=0`。

将 residual 行直接拼入旧 set-head 的对照检查中，prefix context 在 train 和
holdout 均为 100% 改变，最大 logit 变化分别为约 6.47 和 7.30。因此即使
sidecar 质量改善，这种拼接方式也不能部署。

## 决策

该方向 **NO-GO**：不继续调 K、阈值、rank loss 或训练步数，也不从该试验
checkpoint 继续训练。下一步采用隔离式接口：baseline selector 独立执行，
residual selector 独立执行，最后只在 graph candidate 层做几何去重和合并。
对应实现为 `protected_residual_merge.py`，其测试验证 baseline prefix
逐元素不变。
