# PNG Ownership Realization v0.5

本目录复用冻结 Q16 整数 line renderer，穷举探索协议中的 90 个 phase/width/cap/AA 配置。每根 carrier 独立
`rasterize_support`，再验证 support union 与三根线共同 `rasterize_gray` 的非白像素逐像素相等。

fixture 使用两根 `opposite half verticals`：V1 在水平 carrier 终止，V2 从水平 carrier 起始。ownership universe 是
全部 cross-carrier overlap；任何没有合法 intersection option 的 pixel 都会令该配置得到 0 个合法 partition，不会
被局部裁掉。ambiguous triple-overlap pixel 的全部二元 assignment 被穷举；只有满足 seed、8-connectivity、互斥、
global exact cover 和逐像素 carrier incidence 的 partition 才计数。

运行：

```bash
.venv-train/bin/python -m \
  formal.chemgraph.audit.png_ownership_realization_v0_5.build_results

.venv-train/bin/python -m unittest \
  formal.chemgraph.audit.png_ownership_realization_v0_5.test_ownership_realization -v
```

`representative_edge_w5_round_box4.png` 的两份 witness 使用同一个 primitive program 和同一 PNG，只改变 locally
admissible ownership annotation。它不建立两个真实 semantic world，更不是两个非同构化学图的碰撞，也不证明
PNG extractor 能恢复 oracle per-carrier support。UniqueGraph、SDF 和训练仍不授权。

完整结果：90 个配置的 support union 全部逐像素 exact；36 个配置有至少两份 locally admissible partition，最大已
枚举数为 25,032。15 个配置因 exact seed 不可用得到 0；3 个配置因 `|A| > 20` 按探索容量门报告 `null`，数学
计数未知。全部配置均保留在 `results.json`。
