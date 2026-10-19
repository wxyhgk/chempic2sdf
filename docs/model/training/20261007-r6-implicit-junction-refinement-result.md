# r6 implicit-C junction refinement

只在已有 LSD endpoint cluster 内，用多条 incident segment 的共点最小二乘
重定位；不新增 offset candidate，避免 geometry-port 的 candidate flood。
fresh holdout 95 图结果仍很差：最佳 beta `.1` 的 edge P/R 为
`.8491/.5448`，dangling `.5337`；beta `1.0` 的 edge P/R 为
`.8701/.5351`，dangling `.5548`。

报告：

`outputs/audits/chemvector-identity-aligned-line-first-implicit-junction-refinement-r6/report.json`

结论：implicit-C 共点重定位也不能恢复缺失 bond endpoint，不进入训练。
