# r6 diversity panel：contract 与 endpoint ceiling 结果

## 数据 contract

在全局排除固定 r5 source 与既有 6k targeted identities 后，公开 catalog
仍有 1325 个候选 connection identity。本轮选出 240 个新 identity，计划
960 张图，覆盖 DOUBLE-only、TRIPLE-only、AROMATIC、mixed multiple-bond、
hetero N/O multiple 五类，以及四种短键/斜角/线宽/字体组合。

渲染结果：

- planned `960`
- accepted `952`
- rejected `8`
- rejected 全部为 `Typography clearance would erase a bond segment`
- fresh holdout `95` 张
- source overlap `0`
- required files/hash/graph identity/SDF identity 全部通过

contract 报告：

`outputs/audits/chemvector-identity-aligned-line-first-targeted-diversity-r6-contract/report.json`

## endpoint ceiling

严格 new-LSD + glyph alias proposal 的 fresh holdout 结果：

| metric | result |
|---|---:|
| edge precision | `.86789` |
| edge recall | `.62833` |
| dangling | `.33850` |
| node precision / recall | `.71568 / .81576` |

报告：

`outputs/audits/chemvector-identity-aligned-line-first-targeted-diversity-r6-endpoint-ceiling/report.json`

该批数据虽然 contract 合格，但 proposal ceiling 远低于 `P>=.965917`、
`R>.850031`、`dangling<=.05`，因此不能训练 bond sidecar，也不能启动 GPU。
这说明继续单纯增加分子数量或样式不会解决当前问题；必须先改 endpoint
proposal/ownership contract，再生成下一批数据。

补充的 glyph-alias radius sweep（train/dev 选半径，fresh holdout 一次评估）
也没有改善：dev 选择 `20 px`，fresh holdout edge P/R 为 `.86789/.62833`，
dangling `.35630`。因此仅扩大 glyph alias 半径也判定 NO-GO。

随后做了 target-free geometry ports probe：从 glyph bbox 四边/法向和 LSD
endpoint 偏移生成候选。最佳配置（`box_margin=12px`、offset `±4px`）将
node recall 提到 `.9922`，但 edge P/R 只有约 `.885/.726`，dangling
约 `.220`；profile 之间差异很大，弱 profile 仍明显失败。候选数量增加并没有
转化为可用 bond proposal，因此同样 NO-GO。

报告：

`outputs/audits/chemvector-identity-aligned-line-first-geometry-ports-r6/report.json`
