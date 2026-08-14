# v1 P2-D2-E5 Slots4 Local Partner Oracle

> 日期：2026-08-14
>
> 状态：已完成，正式 GO
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

## 实际结果

400 张、16,656 个 GT BondObject：

| 候选预算 | Pair recall | Image exact | 平均 pair/图 |
| --- | ---: | ---: | ---: |
| 每端 top2 | 87.64% | 10.75% | 353.83 |
| 每端 top4 | 92.96% | 33.25% | 625.77 |
| 每端 top8 | **95.53%** | **59.50%** | 1,125.98 |
| 每端 top16 | 96.45% | 68.00% | 2,125.09 |

top8 coordgen/original/rdkit/stress 分别为 96.40%/97.02%/97.02%/91.67%；crossed-bonds、
crowded、long-bond、near-nonbonded 为 86.24%/94.58%/97.04%/92.18%。相对旧 E0 top8
提高 2.70pp，image exact 提高 23.75pp。

全部门通过，判定 `slots4_local_partner_oracle_go`。candidate graph/set scorer 的输入前提恢复；
下一步冻结 P1-A、E4 与 max256/top8 协议，训练共享 endpoint 的 node/edge/count 模型。该结果
仍是 Oracle，不是最终键预测准确率。

## 产物

- audit：`outputs/audits/chemvector-v1p2d2e5-slots4-fresh-local-partner-oracle/audit.json`；
- 逐图明细：同目录 `rows.jsonl`。

SHA256：

- `audit.json`：`dcbd21adb0c296f5c39c68578b5961625fe197b872ab759aeb5d4905ee51e0e7`；
- `rows.jsonl`：`af5d9de27782a40b2008c3c42eb7e37f9a26b498bbe0b83b767cf149f9b2c25b`。
