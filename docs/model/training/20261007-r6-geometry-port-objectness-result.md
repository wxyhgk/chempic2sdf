# r6 geometry-port objectness CPU 试训

在 glyph boundary port + incident offset 候选场上做了 300 步 CPU sidecar
（205 参数，固定 train/dev/fresh_holdout，labels 仅在 proposal 后置生成）。
候选场本身覆盖很高，但正负候选不可分：train 正例率约 `57%`，训练阈值
没有找到 precision `>=.90` 的 operating point；保守阈值 `.99` 在 fresh
holdout 选择 0 个候选，candidate precision/recall 和 edge recall 均为 0。

报告：

`outputs/audits/chemvector-identity-aligned-line-first-geometry-ports-objectness-cpu-r6/report.json`

结论：增加几何 offset 候选会制造大量近邻/重复候选，单独训练 objectness
无法把它们安全过滤；不启动 GPU，也不把该候选场接入 production。
