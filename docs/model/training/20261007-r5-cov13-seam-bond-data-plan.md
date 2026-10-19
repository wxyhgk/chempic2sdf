# r5 cov13 / text-seam / multiple-bond 数据与训练计划

本计划只针对已公开的 train/holdout 诊断，不读取 sealed test，也不改变
production checkpoint、OCR 或默认 graph route。当前 production initial SHA
必须保持 `09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`。

## 证据

`unreachable-bond-strata-r5` 显示，653 条不可达 bond 中：

- DOUBLE 只有 4.35% 同时有两个 cluster，60.87% 两端都落在 text seam；
- 30–60° 方向的双端 cluster 率为 43.98%；
- `cov13_base48`/`cov13_stroke16_48` 双端 cluster 率约 42%，而 `cov13_cap52` 为 85%；
- 主要缺口为 `implicit_cluster|text_seam`（371）和 `text_seam|text_seam`（78）。

因此优先修复渲染/端点可见性与 identity，不再继续盲扫 detector 参数。

## 数据矩阵

按分子 identity 分层并保持 identity-disjoint split。每个分子只生成固定、可追踪的 style 组合：

- `bond_length_cap`: 32/40/48/52；
- `font_size`: 24/28/32/36；
- `bond_line_width`: 1.6/2.4/3.2；
- rotation: 0/15/30/45/60/75/90°，重点覆盖 30–60°；
- N/O incident DOUBLE/TRIPLE、aromatic bridge、degree≥3 junction 各有独立配额；
- multiple-bond rail offset/spacing 使用三档，并单独统计 text-seam 距离 `<2`、`2–4`、`>4` px。

继续使用 `stereo-aware-v1` 的 glyph bbox/mask contract。若要改变文字与键间距，新增明确的 renderer profile/version；不得覆盖旧 profile。每张新增图片先通过 text-seam contract、layer/graph round-trip 和 split identity 检查，才可进入训练候选。

## 低成本训练顺序

1. CPU proposal/identity gate：无训练，先确认各困难 strata 可达。
2. 只训练 node identity keep head（冻结 OCR、bond sidecar、production graph，最多 800 CPU steps）。
3. node gate 通过后，冻结 node，训练 edge/type sidecar（最多 800–1200 CPU steps）。
4. 两个 CPU gate 都通过后，才允许单卡 GPU shadow（最多 1200 steps/30 min）。

每阶段只用 train split 选择阈值；holdout 完整评估一次，禁止 sealed test。

## 停止条件

任一条件失败立即 NO-GO：full holdout node P<.90 或 R<.85、edge P<.96592、edge R≤.85034、任一 profile edge R<.75、sanitize<.90、DOUBLE/TRIPLE type recall<.80、text-seam unresolved>10%、dangling>.05，或 production SHA 改变。失败后不继续堆数据、换 detector 或启动 GPU；先回到失败 strata 的数据契约。
