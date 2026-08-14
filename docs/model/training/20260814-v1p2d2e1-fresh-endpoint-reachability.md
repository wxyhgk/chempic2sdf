# v1 P2-D2-E1 Fresh Endpoint Reachability

> 日期：2026-08-14
>
> 状态：已预注册，等待执行
>
> 审计：`chemvector-v1p2d2e1-fresh-endpoint-reachability`

## 目标

P2-D2-E0 在 held-out 字体 fresh calibration 上得到 top8 pair recall 92.83%，低于 95% 门，
因此不能直接训练 graph-set selector。本轮保持同一 100 分子、400 图和所有冻结 checkpoint，
把损失拆成两层：

1. 宽 endpoint 候选是否覆盖 GT unique visible ports；
2. 穷举所有 endpoint pair 时是否仍能覆盖 GT BondObject。

本轮是只读诊断，不训练模型、不扫描阈值、不接键级、OCR 或 SDF。

## 固定输入

- P2-D2-E0 audit SHA256：
  `31c49f8fb9a52aeaabf0a1b9f3092ed28841e8c9637d180ba22753b94df45298`；
- P1-A separator、P2-B endpoint、threshold 0.05、maximum endpoints 256 均冻结；
- 复用 P2-D2-E0 calibration，不读取预留 fresh holdout；
- `sealed_test`、MR-TADF、USPTO、旧 validation/internal test 均不读取。

## 路由规则

- unique port recall 至少 97%，且 exhaustive pair recall 至少 97%：判定为
  `partner_selection_blocker`，下一步改候选稀疏化协议；
- 任一未过：判定为 `endpoint_reachability_blocker`，下一步重训 P2-B endpoint 场的风格与复杂度
  泛化，不训练下游 graph-set scorer。

97% 是 P2-C 原协议的预注册可达性门，不根据本轮结果修改。

## 执行命令

```bash
.venv-train/bin/python -m \
  chemvector.evaluation.audits.bonds.audit_v1p2d2e1_fresh_endpoint_reachability
```

## 产物

- audit：`outputs/audits/chemvector-v1p2d2e1-fresh-endpoint-reachability/audit.json`；
- 逐图明细：同目录 `rows.jsonl`。
