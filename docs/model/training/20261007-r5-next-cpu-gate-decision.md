# r5 下一步 CPU gate 决策

## 已完成的排查

在 599 张 identity-disjoint fresh holdout 上，对 text seam 做了六种只读变体：

- raw：edge P/R `.99164/.78926`，dangling `.26349`；
- glyph mask：`.99164/.78910`，dangling `.21812`；
- inpaint：`.99164/.78894`，dangling `.21852`；
- seam route：`.99163/.78831`，dangling `.25850`；
- mask+seam：`.99163/.78799`，dangling `.21921`；
- inpaint+seam：`.99163/.78862`，dangling `.23139`。

六种变体的 oracle sanitize 约 `.708`，全部低于 recall `>.85`、dangling `<=.05` 门槛。因此 text mask/inpainting/seam route 判定 NO-GO，不启动 GPU。

报告：

`outputs/audits/chemvector-identity-aligned-line-first-text-seam-mask-endpoint-ceiling-r5/report.json`

## 唯一允许的下一步

回到 frozen vertex candidates，做一个局部 bond-patch residual classifier 的 CPU-only feasibility：

1. candidate identity、OCR、node ownership、prefix edges 全部冻结；
2. 只对 candidate pair 使用 target-free 的局部 patch、长度/角度、ink support、parallel-rail、glyph proximity、degree 特征；
3. 仅预测 keep/drop 与 SINGLE/DOUBLE/TRIPLE/AROMATIC，不发现新节点；
4. 参数不超过 3k，200--500 CPU steps，train-only 选阈值，fixed holdout 只评估一次。

已有的 751 参数 endpoint-pair CPU preflight（300 steps）只能达到 holdout pair
precision `.8688`、recall `.7846`；它不是 graph merge，也没有达到本节 edge gate，
所以不能直接升级为 GPU 或 production。新的 bond-patch 试验必须在完整 graph
roundtrip 上重新验收，而不是沿用 pair-level 数字。

## Gate

必须同时满足：edge P `>=.95`、R `>=.80`；非芳香 type accuracy `>=.90`；sanitize `>=.95` 且六 profile `>=.90`；dangling 不恶化；相对 frozen sidecar 的 type 提升至少 3pp 或 sanitize 提升至少 2pp。任一失败即停止，不做 GPU shadow。

## 隔离保证

production initial SHA 仍为
`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`；sealed test 未读取；candidate artifact 仅含 `sample_key/source/candidates[{x,y,score}]`，1170 行、19884 个候选，未发现 GT 字段。该试验不写 production checkpoint。
