# v1 P2-D2-E5 Slots4 Local Partner Oracle

> 日期：2026-08-14
>
> 状态：已预注册，等待执行
>
> 审计：`chemvector-v1p2d2e5-slots4-fresh-local-partner-oracle`

## 目标

E4 将 fresh max512 endpoint pairable recall 提高到 99.69%，但下游 candidate graph 实际需要
稀疏 pair，而不是全量 endpoint 指标。本轮冻结 E4，回到更便宜的 max256 与每端 top8
partner union，复测 P2-D2-E0 同一 fresh calibration。

## 固定协议

- E4 checkpoint SHA256：
  `f9753b38c5484eea646e70b4b3e40bf0715f667804c239d510114939fdc9590d`；
- P1-A separator 不变；
- endpoint threshold 0.05、maximum endpoints 256；
- line score 每端 top2/top4/top8/top16 partner union；
- 不设置最大键长、空间半径或 GT 键数预算；
- 复用 E0 的 100 分子/400 图 calibration；
- fresh holdout、sealed test、MR-TADF、USPTO 均不读取。

旧 E0 top8 pair recall 为 92.8254%。本轮只替换冻结 endpoint checkpoint，其他候选代码不变。

## GO/NO-GO

沿用 P2-D0/E0 门：top8 overall pair recall 至少 95%、image exact 至少 30%、stress 至少 90%、
crossed-bonds 至少 85%。全部通过才允许训练 candidate graph/set scorer。

## 执行命令

```bash
.venv-train/bin/python -m \
  chemvector.evaluation.audits.bonds.audit_v1p2d2e5_slots4_local_partner_oracle
```

## 产物

- audit：`outputs/audits/chemvector-v1p2d2e5-slots4-fresh-local-partner-oracle/audit.json`；
- 逐图明细：同目录 `rows.jsonl`。
