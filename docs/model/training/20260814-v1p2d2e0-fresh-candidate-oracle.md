# v1 P2-D2-E0 Fresh Candidate Oracle

> 日期：2026-08-14
>
> 状态：已完成，严格 NO-GO
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

### 首次执行的数据修正

首次执行在第 200 多张图按 fail-closed 原则中止，没有写入部分指标。定位发现
`chembl:CHEMBL5856698` 的 `rdkit` 和 `stress` 布局把 atom 15 与 atom 19 放在完全相同的
可视坐标。该分子虽然通过了数据侧 chemical roundtrip 和图层验收，但不能无损编译为
ChemScene v0.2，属于视觉布局异常。

因此整分子四种布局固定排除，再按同一 SHA256 排序从后备池补入下一分子；不只跳过报错的两张
图片。拒绝 ID、原因和补位后的 calibration/holdout 哈希写入 `split.json`。这次修正只清理
不可表示标签，不使用模型输出，也不改变预注册能力门。

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

## 实际结果

修正源异常后完整运行 100 个分子、400 张图，共 16,656 个 GT BondObject：

| 候选预算 | Pair recall | Image exact | 平均 pair/图 |
| --- | ---: | ---: | ---: |
| 每端 top2 | 83.33% | 7.50% | 357.42 |
| 每端 top4 | 89.70% | 20.50% | 643.04 |
| 每端 top8 | 92.83% | 35.75% | 1,161.94 |
| 每端 top16 | 94.31% | 43.50% | 2,175.28 |

top8 stress/crossed/long-bond recall 分别为 90.11%/85.83%/93.14%，三个分桶门和 image exact
门通过，但 overall recall 低于 95%，最终判定 `fresh_candidate_oracle_no_go`。即使将候选翻倍
到 top16 仍未达到 95%，因此不能直接训练 graph-set scorer；下一步在同一 calibration 拆解
endpoint 与 pair 的可达性。

## 产物

- audit：`outputs/audits/chemvector-v1p2d2e0-fresh-candidate-oracle/audit.json`；
- calibration 明细：同目录 `rows.jsonl`；
- 固定拆分摘要：同目录 `split.json`。

SHA256：

- `audit.json`：`31c49f8fb9a52aeaabf0a1b9f3092ed28841e8c9637d180ba22753b94df45298`；
- `rows.jsonl`：`d86e3fc5a6d033cb6f18eaef8f745522507dc18172dea83efe58fb8c8b54f286`；
- `split.json`：`21d8548af326d81afb321afc829e35aa522402bbe70ec2c3a893e501e4e6e5af`。
