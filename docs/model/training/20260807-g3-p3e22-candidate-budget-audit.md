# G3 P3-E22 Candidate Budget 审计

> 日期：2026-08-07
>
> 状态：已完成，Candidate Budget GO
>
> 输出：`outputs/audits/chemvector-g3-p3e22-candidate-budget-audit`

## 目标

P3-E21因top128在train/dev未达到99% candidate recall而停止。本轮冻结同一center field、同一
train/dev/confirm manifest和同一regional-maxima排序，一次提取top256，对前128、192、256
分别计算6px最大一一匹配可达率，选择三个split都达到99%的最小预算。

## 协议

- 零训练，不运行任何selector、SDF或下游阈值；
- batch24、BF16、确定性算法；
- 预算只允许128、192、256，不根据分桶增加临时候选；
- 三个split必须同时达到99%，否则NO-GO；
- fresh_dev、sealed_test、USPTO和MR-TADF均不读取；
- 保存top256 cache，若审计GO则供下一周期复用。

## 结果

| split | top128 recall | top192 recall | top256 recall |
|---|---:|---:|---:|
| train | 98.8009% | 99.2409% | 99.3816% |
| dev | 98.9290% | 99.3549% | 99.4383% |
| confirm | 99.1095% | 99.4125% | 99.4798% |

三个 split 同时通过 99% 门的最小预算是 `192`，因此本轮判定
`candidate_budget_go`。top256 相对 top192 只增加 0.07--0.14pp recall，不足以抵消
后续 pointer memory、缓存和完整 SDF 周期的额外成本；P3-E23 固定使用 top192，不按
Count、variant 或 split 动态改变预算。

整图 candidate 可达率也随预算提高：train/dev/confirm 从 top128 的
79.13%/79.84%/83.75% 提高到 top192 的 82.69%/84.53%/87.50%。它不是后续模型的
准确率，只表示所有 GT Atom 都能在候选池中找到 6px 一一匹配的图片比例。

## 产物与复现

- 指标：`outputs/audits/chemvector-g3-p3e22-candidate-budget-audit/metrics.json`；
- top256 缓存：`outputs/audits/chemvector-g3-p3e22-candidate-budget-audit/top256-candidate-cache.pt`；
- 缓存 SHA256：`c796da5a33cfc954e3f4f25d9da9d70e70296a9b76d32e0736fd6d5cd2655720`；
- 固定视觉 checkpoint SHA256：
  `2869451245f50f98f5e3f5259179ca338e82ad81a8fd59de2d2037a10056404b`；
- 总耗时：155.76 秒；
- `fresh_dev` 与 `sealed_test` 均未读取。

复现命令：

```bash
python -m chemvector.evaluation.audits.atoms.audit_g3_p3e22_candidate_budgets
```

这是零训练容量审计，没有产生新 checkpoint，因此不创建模型 tag。下一步先验证现有
base/expert checkpoint 能否以 `queries=128, candidates=192` 严格加载，并审计 P3-E16
DeepSets 在 192 候选下的固定基线；通过后才恢复 P3-E21 的五路线训练与双下游完整 SDF。
