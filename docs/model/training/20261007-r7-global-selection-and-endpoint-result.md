# r7 全局候选选择与端点身份 deterministic 结果

本轮只做 CPU/read-only deterministic probe，未读取 sealed test、未训练、未使用 GPU，production initial checkpoint 保持
`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`。

## 结果

1. **化学/拓扑 set selection**

   `tools/probe_line_first_global_set_constraints_r6.py` 在 frozen edge/type candidates 上测试 parallel-rail 去重和 degree cap。
   baseline holdout 为 P=`.965917`、R=`.850031`、dangling=`.00695`、sanitize=`.94515`。任何会删除候选的约束都会损失 recall；最大收益只有 sanitize `+.0021`，不能恢复 unreachable endpoint，也不能通过严格 gate。

2. **segment-port bipartite assignment**

   `tools/probe_line_first_segment_port_bipartite_r7.py` 在 50 张 fresh holdout 上测试 many-to-one、one-to-one、directional 三种 target-free assignment：

   - many-to-one P/R=`.8924/.7589`，dangling=`.1496`
   - one-to-one P/R=`.8839/.6964`，dangling=`.3132`
   - directional P/R=`.9288/.6406`，dangling=`.3988`

   均未达到 proposal gate；方向兼容性只能换取 precision，不能补回端点。

   额外的 target-free edge-support/degree 筛选也验证了同一结论：support≥3 时 P=`.988` 但 R 仅=`.181`；support≥2 时 P/R=`.891/.750`、dangling=`.131`。这是可 abstain 的高精度子集，不是可训练的完整图 proposal。

## 决策

停止“只在已有线段/候选上做全局删选、化学约束或 glyph 赋值”的路线，不启动 CPU scorer 或 GPU shadow。当前瓶颈仍是 proposal 阶段的端点身份/可达性，下一步必须改变 endpoint proposal 表示（例如保留显式的 segment endpoint hypothesis set，并让节点身份与 text-seam/implicit-C 分开建模），先在 50--100 张 fresh holdout 上验证 reachability ceiling，再决定是否训练。
