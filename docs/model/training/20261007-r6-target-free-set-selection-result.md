# r6 target-free multi-candidate set selection

在 r6 fresh holdout 的 95 张图上完成了一个只读 deterministic probe：每个 LSD
endpoint 保留 cluster/glyph alternatives，使用图像几何重投影误差、共享候选的
duplicate collapse 和 degree penalty 做 target-free greedy set selection。

结果：edge precision `0.8830`、edge recall `0.6029`、dangling `0.6730`，node
recall `0.9478`。proposal gate 要求 `P>=.965917, R>.850031, dangling<=.05`，
三项均失败，因此该表示不能进入 CPU scorer 或 GPU shadow；没有进行训练。

独立脚本：[probe_line_first_target_free_set_selection_r6.py](../../../tools/probe_line_first_target_free_set_selection_r6.py)

报告：[report.json](../../../outputs/audits/chemvector-identity-aligned-line-first-target-free-set-selection-r6/report.json)

production initial checkpoint SHA、sealed test 和 production selector 均未触碰。
