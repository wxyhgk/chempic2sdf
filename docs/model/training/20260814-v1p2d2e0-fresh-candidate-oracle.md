# v1 P2-D2-E0 Fresh Candidate Oracle

> 日期：2026-08-14
>
> 状态：已预注册，等待执行
>
> 审计：`chemvector-v1p2d2e0-fresh-candidate-oracle`

## 目标

P2-D1 证明独立 pair MLP 无法利用共享 endpoint、重复候选和整图基数，但 candidate graph
是否值得训练仍取决于固定候选协议在新风格上的可达性。本轮冻结 P1-A、P2-B、endpoint
threshold 0.05 和每端 top8 partner union，在 held-out 字体 `fresh_dev` 上先复核候选 Oracle。

若 fresh 风格下 Oracle 已明显下降，下一步应修 P1-A/P2-B 的风格泛化，不能用 graph-set
scorer 掩盖上游不可达；若仍达到 95% pair recall，才进入 P2-D2 联合 node/edge/count 训练。

## 数据隔离

数据固定为 `chemvector_joint_v1/fresh_eval_12k`，commitment SHA256：
`13a456afb2c475a73fc4dca82df5fe62d4a31277d6b584637d9cd92736024512`。

按 `SHA256(20260821:molecule_id)` 对 2,000 个 fresh_dev 分子排序：

- 前 100 个分子、400 张图只用于本轮 candidate Oracle；
- 后 100 个分子、400 张图预留给冻结 P2-D2 的一次性确认，本轮不读取其 tar 成员；
- `sealed_test` 保持 000 权限，读取数必须为 0；
- MR-TADF、USPTO、旧 validation 和 internal test 均不读取。

四个图片变体必须按分子留在同一侧，不能发生同分子不同渲染泄漏。

## 固定协议

- P1-A separator SHA256：
  `490b5daae3a2059c3d335785a69e0c2a41eed32f0237ef77fc3556c36d134a74`；
- P2-B endpoint SHA256：
  `3ec7bf27e96f166e0f30e7c5c8b79f33860701acfeb3e16297015fc7a91511ca`；
- endpoint threshold 0.05、每图最多 256 endpoint；
- line score 每端 top2/top4/top8/top16 partner union；
- 不设置最大键长或空间半径裁剪；
- 无 GT 键数参与候选选择。

## GO/NO-GO

400 张 calibration 同时满足：

- top8 overall pair recall 至少 95%；
- top8 image exact 至少 30%；
- top8 stress pair recall 至少 90%；
- top8 crossed-bonds pair recall 至少 85%；
- fresh holdout 和 sealed test 的图片读取数均为 0。

GO 只说明 P2-D2 输入候选在新风格上仍可达，不代表键识别模型有效。NO-GO 时停止 graph-set
训练，回到 endpoint 场和语义分离器做 held-out 字体诊断。

## 执行命令

```bash
.venv-train/bin/python -m \
  chemvector.evaluation.audits.bonds.audit_v1p2d2e0_fresh_candidate_oracle
```

## 产物

- audit：`outputs/audits/chemvector-v1p2d2e0-fresh-candidate-oracle/audit.json`；
- calibration 明细：同目录 `rows.jsonl`；
- 固定拆分摘要：同目录 `split.json`。
