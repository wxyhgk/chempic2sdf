# r6 多路线 CPU 讨论与决策

本轮并行评估了六条路线，全部保持 production initial SHA
`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`，未读
sealed test。

| 路线 | 结果 | 决策 |
|---|---|---|
| local bond-patch residual | CPU 300 steps；P/R `.9578/.7888`，完整 graph R `.6226`，type `.00025` | NO-GO；冻结 proposal 是硬上限 |
| graph→render critic | aromatic critic 可把 sanitize 提到 `.9219`，但 precision `.9585`、弱 profile recall `.8052` | 后置 validator，不解决 endpoint |
| r6 diversity data | contract 952/960 通过；raw endpoint P/R `.8679/.6283`，dangling `.3385` | 数据量不是主因，NO-GO |
| glyph alias sweep | dev 选 `20px`；fresh holdout P/R `.8679/.6283`，dangling `.3563` | 半径扩展无效 |
| geometry ports | node recall `.9922`，edge P/R `.8850/.7264`，dangling `.2196` | 候选泛化造成大量假边，NO-GO |
| geometry-port objectness | CPU 300 steps、205 参数；无 train threshold 达到 precision `.90` | 不训练 GPU，不接 production |
| text-anchor state machine | 12 个 seam margin/cosine 变体；最佳 edge P/R `.886/.707`，dangling `.442` | NO-GO |
| implicit-C junction refinement | 只重定位已有 cluster；r6 最佳 edge P/R `.849/.545`，dangling `.534` | NO-GO |

## 当前结论

继续堆数据、扩大 glyph alias、训练 edge scorer、训练 render critic 或 graph
walk 都不能越过 raw endpoint/ownership contract。下一轮必须先设计新的
端点身份协议（例如单独的 text-anchor/implicit-C ownership 状态机或显式
abstention），并在 50--100 图上做 deterministic ceiling；没有达到完整
proposal gate 前不再启动 GPU。

本轮 text-anchor 与 implicit-C 协议的 ceiling 也已经失败，因此下一步不再
继续调 radius、offset 或 cluster 参数。新的实验必须改变 proposal 表示：
保留 segment/port 的不确定性集合，由全局图约束做一次 target-free set
selection；只有该集合 ceiling 达标后，才允许 tiny scorer CPU 试训。
