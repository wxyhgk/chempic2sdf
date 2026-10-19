# r5 unreachable GT bond endpoint strata

脚本：`tools/audit_line_first_unreachable_bond_strata_r5.py`

报告：`outputs/audits/chemvector-identity-aligned-line-first-unreachable-bond-strata-r5/report.json`

这是只读 CPU 归因；未训练、未读 sealed test、未改 production。

653 条 unreachable GT bonds 的结果：

- 两端均有 frozen vertex candidate：`561/653 = .859`；
- 两端均有 LSD endpoint cluster：`395/653 = .605`；
- source pair 最大项为 `implicit_cluster|text_seam`：371（56.8%）；
- `text_seam|text_seam`：78；`implicit_cluster|implicit_cluster`：132；
- 任一端无 cluster 的 bond 仅 72 条。

按 profile，两端均有 cluster 的比例：cov13 base48 `.425`、cap52 `.853`、stroke16 `.421`；cov33 base `.714`、cap32 `.680`、stroke32 `.763`。按类型，DOUBLE 仅 `.043` 两端均有 cluster，但 `.609` 两端均为 text seam；TRIPLE 两端 cluster 全部存在但 vertex candidate 仅 `.25`。30–60° 方向和 30–50px 键长是明显弱 strata。

结论：候选原子大多已经存在，主要缺口是文字 seam 到线端点的连接以及短/斜线端点检测，不能通过 node identity 或 type router 训练解决。下一步应做 target-free text-seam bridge 与 double-bond bundle endpoint 的 deterministic ceiling；在其通过前不启动 GPU。
