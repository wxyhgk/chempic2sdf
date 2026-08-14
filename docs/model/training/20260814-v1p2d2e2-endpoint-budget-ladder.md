# v1 P2-D2-E2 Endpoint Budget Ladder

> 日期：2026-08-14
>
> 状态：已完成，路由到 endpoint model retrain
>
> 审计：`chemvector-v1p2d2e2-endpoint-budget-ladder`

## 目标

P2-D2-E1 表明 fresh calibration 的 endpoint unique-port recall 只有 95.88%，且每图候选 p95
已达到 256。本轮冻结模型，只审计候选硬上限和阈值是否造成不可达，不训练下游 graph-set。

## 固定配置

同一次 endpoint field 前向后，固定比较九组解码：

- threshold：0.05、0.02、0.01；
- maximum points：256、384、512；
- 匹配半径保持 6px；
- 同一 P2-D2-E0 calibration 100 分子、400 图；
- fresh holdout、sealed test、MR-TADF、USPTO 均不读取。

前置 P2-D2-E1 audit SHA256：
`baab7dccc977500fccee4c9db7dfaf3afc0d5afe36cf3e663a7aeade3caeb0ad`。

## 路由规则

每组同时要求 unique-port recall 和 pairable-bond recall 至少 97%：

1. threshold 0.05 下 384/512 任一通过：`maximum_points_ceiling`；
2. 固定阈值未过，但 0.02/0.01 任一预算通过：`endpoint_threshold_ceiling`；
3. 九组均未过：`endpoint_model_retrain`。

本轮不以 precision 或 count exact 选择部署阈值；它只判断 GT 可达性。即使某个宽配置通过，仍需
重新运行 top-K pair Oracle，不能直接接推理。

## 执行命令

```bash
.venv-train/bin/python -m \
  chemvector.evaluation.audits.bonds.audit_v1p2d2e2_endpoint_budget_ladder
```

## 实际结果

| Threshold / 最大候选 | Unique-port recall | Pairable-Bond recall | 平均候选 | 饱和图 |
| --- | ---: | ---: | ---: | ---: |
| 0.05 / 256 | 95.88% | 95.16% | 213.53 | 36.25% |
| 0.05 / 384 | 96.91% | 96.51% | 235.33 | 6.25% |
| 0.05 / 512 | 96.93% | 96.55% | 239.11 | 0.75% |
| 0.02 / 512 | 96.94% | 96.56% | 335.90 | 6.50% |
| 0.01 / 512 | 96.94% | 96.56% | 512.00 | 100% |

384 到 512 仅增加 0.02pp port recall 和 0.04pp pairable recall；降低阈值则几乎没有额外
收益。九组配置均未同时达到 97%，按预注册规则判定 `endpoint_model_retrain`。crossed-bonds
在最宽 0.01/512 下仍只有 96.15% port 和 95.35% pairable；困难布局的正确峰并未出现在场中。

下一步保持 P1-A separator 与 endpoint 架构不变，先用 `style_train_100k` 扩量微调，验证问题
是否来自 P2-B 仅 8,056 张 fit 图的训练覆盖。训练前必须全量预检 ChemScene，并按 scaffold
拆分 fit/calibration/holdout；不能把异常布局带入训练。

## 产物

- audit：`outputs/audits/chemvector-v1p2d2e2-endpoint-budget-ladder/audit.json`；
- 逐图九配置明细：同目录 `rows.jsonl`。

SHA256：

- `audit.json`：`cfc3ba5551882c021b26d70c6ef2b5ecc25676a79ccb0f57edc0d22a221939f2`；
- `rows.jsonl`：`b3f67f78e7314423e4bd2d48fc74353defbe07915dfc5db0fc1898a7ba1e240b`。
