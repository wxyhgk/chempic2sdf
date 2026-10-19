# r8 endpoint-hypothesis deterministic ceiling

本轮在 r6 diversity fresh holdout 的 95 张图上运行只读 CPU probe，未读取
sealed test、未训练、未使用 GPU，也未写 production checkpoint。production
initial SHA 前后均为
`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`。

脚本：`tools/probe_line_first_endpoint_hypothesis_r8.py`；机器报告：
`outputs/audits/chemvector-identity-aligned-line-first-endpoint-hypothesis-r8b/report.json`。

## 表示

每个 LSD segment port 保留候选集合，而不是先把端点压成单一 cluster：

- text-seam：沿 incident ray 进入 glyph box interior（最多 64 px）；
- implicit-C：沿 incident ray 的反向显式生成 8/16/24 px hypothesis；
- 通过图像几何 support vote 选择共享节点；GT 仅在 proposal 后计算指标。

## 结果

| variant | precision | recall | dangling |
|---|---:|---:|---:|
| ray_anchor | 0.5717 | 0.3523 | 0.0872 |
| extended | 0.5899 | 0.3692 | 0.0931 |

相比 r7 没有达到 proposal gate（P≥.965917、R>.850031、dangling≤.05），且
假边显著增加。保留多候选集合本身不能解决 endpoint identity；继续扩大 ray
offset 或 alias 半径属于同一失败 ceiling，不值得训练。下一次 proposal 必须
引入局部图像端点 objectness/ownership 或显式 junction topology，而不是继续
做 deterministic 参数 sweep。
