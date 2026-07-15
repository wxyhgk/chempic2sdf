# 评审与验收报告

本目录保存模型人员对 `pilot_10k` 的评审结果，不保存训练 checkpoint。

计划至少包含：

```text
oracle_bond_layer.json
multiscale_ocr.json
soft_mask_ablation.json
decision.json
```

当前 Oracle semantic layer 实验已经完成：

- [`ORACLE_SEMANTIC_LAYER.md`](ORACLE_SEMANTIC_LAYER.md)：中文结论和决策；
- [`oracle_bond_layer.json`](oracle_bond_layer.json)：机器可读指标；
- [`MULTISCALE_OCR.md`](MULTISCALE_OCR.md)：多尺度 OCR 训练与完整 MR-TADF 结果；
- 完整逐样本结果位于
  `outputs/evaluations/chemvision-pilot-oracle-bbox-v073/`。

`decision.json` 用于决定是否启动 `full/`，应记录 Go/No-Go、实验 checkpoint、
普通 scaffold validation、字号挑战集和隐藏 benchmark 的结果。
