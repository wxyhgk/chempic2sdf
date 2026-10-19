# r6 text-anchor ownership 状态机 probe

尝试仅在 glyph bbox seam 与 segment 入射方向一致时，把 endpoint 路由到
glyph center，避免旧版无条件 glyph alias。对 margin `6/10/14/18px` 和
cosine gate `0.55/0.70/0.85` 做了 12 个 target-free 变体，fresh holdout
95 图一次性评估。

最佳 edge recall 仅 `.7070`，edge precision `.8862`，dangling 最低仍
`.4419`；node recall `.9478`。因此状态机虽然提高了部分节点覆盖，反而引入
大量 unresolved edges，不能进入 CPU 训练或 GPU。

报告：

`outputs/audits/chemvector-identity-aligned-line-first-text-anchor-state-r6/report.json`
