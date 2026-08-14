# v1 P2-D2-E2 Endpoint Budget Ladder

> 日期：2026-08-14
>
> 状态：已预注册，等待执行
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

## 产物

- audit：`outputs/audits/chemvector-v1p2d2e2-endpoint-budget-ladder/audit.json`；
- 逐图九配置明细：同目录 `rows.jsonl`。
