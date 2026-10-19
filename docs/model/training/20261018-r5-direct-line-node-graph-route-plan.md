# r5 direct line-node graph route：CPU ceiling 与最小 shadow 计划

## 目的

验证 line segment endpoint graph 是否可以作为分子图的中间核心表示。该路线不复用 endpoint ownership scorer，也不把文字 OCR 混入第一阶段：LSD segment endpoint cluster 直接成为 provisional node，segment 本身直接成为 provisional edge；可见文字仅在后续作为 `TEXT_ANCHOR` 节点交给冻结 OCR。

production checkpoint、selector、OCR、edge/type head 和 sealed test 均保持冻结。

## CPU ceiling（训练前）

对固定 train/holdout 及六个 profile 做 target-free deterministic replay：

1. LSD segment → endpoint clustering → unordered direct edges；
2. 只使用端点几何、segment 长度/角度、局部 ink、端点 incident degree 和 glyph-evidence；
3. GT 只在 proposal 构造完成后用于审计；
4. 分别报告 node P/R、edge P/R、bond-type accuracy、raw dangling、self-loop、oracle-element sanitize 和每 profile 指标。

CPU ceiling 必须同时满足：

- holdout edge precision `>= .965917`，recall `> .850031`；
- edge-type accuracy `>= .90`；
- raw dangling `<= .05`，self-loop `= 0`；
- oracle-element sanitize `>= .95`；
- 六个 profile recall `>= .75`、sanitize `>= .90`；
- 与 frozen baseline 相比不能损失已正确 prefix 的保留率。

若 CPU ceiling 不通过，不训练 router，不启动 GPU。

早期 direct-graph ceiling replay（`tools/audit_line_first_lsd_direct_graph_ceiling_r5.py`）在 474-image holdout 上报告 edge P/R `0.9678/0.9471`、oracle sanitize `1.0`；但该数字是在 unresolved 节点被过滤、且 GT 只用于事后映射的 synthetic ceiling。修正严格审计中的 alias 重复计数后，direct route 的真实可评估指标为 node P/R `0.4805/0.6802`、edge P/R `0.9705/0.5690`、dangling `0.5783`（六 profile edge recall 约 `.542-.597`）。因此 direct line-node graph 不能作为当前中间图或 source-router 训练入口。

保留该 ceiling 仅作为“LSD 线段提议容量”的上界记录，不得把 synthetic edge/sanitize 数字当作可部署能力，也不得在 direct graph 上继续做 type router 或 GPU shadow。后续路线回到 frozen vertex-candidate ownership + frozen edge/type sidecar；所有改动先做 node identity、端点丢失和 dangling 的 target-free/事后归因审计。

## 最小 source/router sidecar

只有 CPU ceiling 通过后才允许拟合一个不超过约 3k 参数的小头。输入仅来自 direct line-node 证据；输出为 `REJECT`、`LINE_JUNCTION`、`TEXT_ANCHOR`、`IMPLICIT_C` 的 source/presence 及 bounded local offset。不得读取 atom id、bond、GT 坐标或 OCR token；标签在 proposal 生成后以内存后验方式建立。

训练顺序：

1. CPU frozen-feature fit，最多 500 steps；
2. 200-step full-holdout 检查无严格 graph 改善即早停；
3. 仅当 CPU fit 同时保持 precision/sanitize/profile floors，才允许一次独立 GPU shadow，最多 800 optimizer steps或30分钟；
4. GPU shadow 必须使用 train-only threshold，full holdout 只评估一次；任何 production SHA 变化、sealed-test 访问、非有限 loss 或 dangling/self-loop 增加立即终止。

GPU shadow 通过条件：node P/R `>= .90/.85`、edge P/R `>= .965917/.850031`、type `>= .90`、raw dangling `<= .05`、self-loop `=0`、sanitize `>= .95`，并且每 profile 达到 recall `.75`/sanitize `.90`。通过也只是诊断证据，不授权 production merge。

## 三个 CPU 前置 gate

在任何参数拟合前，按以下顺序执行；每一关失败都停止后续工作：

### A. edge/type geometry gate

冻结 direct segment graph，只统计长度、角度、平行/重叠、端点 incident degree 与局部像素支持。不得用 GT edge 参与筛选。要求 edge precision/recall `>= .965917/.850031`、bond-type accuracy `>= .90`、self-loop `=0`、oracle sanitize `>= .95`，且六 profile 各自通过 recall/sanitize floor。该关验证几何图本身，不训练 edge scorer。

对于 bond type，不使用“同一 pair 的线段数量”作为最终类型判定：现有 multiplicity heuristic 的 holdout type accuracy 仅约 `.021`、sanitize 约 `.032`。先做 deterministic bundle grouping，把同一 direct node pair 的原始 segments 聚合，再提取 bundle count、平行间距、角度一致性、投影长度、raster strip support、ink width、端点 degree、dash-gap 等 target-free 几何特征。只有 bundle 特征对 type 有足够 ceiling，才考虑一个不超过 3k 参数的 `SINGLE/DOUBLE/TRIPLE/AROMATIC/OTHER` type router；该 router 不改 edge presence/ownership。

type router 的 CPU admission：edge set retention `100%`；整体及非芳香 conditional type accuracy `>= .90`；SINGLE accuracy `>= .90`，DOUBLE/TRIPLE/AROMATIC recall 各 `>= .75`；按预测 type 重建的 oracle-element sanitize `>= .95`，每 profile `>= .90`；edge P/R 相对 direct ceiling 下降不超过 `0.5pp`。任一条件失败即 NO-GO，不训练。

### B. node identity / OCR routing gate

冻结 node 集，分别审计三类节点：`LINE_JUNCTION`（至少两条 incident line）、`TEXT_ANCHOR`（glyph evidence seam）、`IMPLICIT_C`（无 glyph 但几何 junction）和 `REJECT`。OCR 只接收 `TEXT_ANCHOR` crop，不参与 line edge 生成。要求 source-tag macro-F1 `>= .85`，TEXT_ANCHOR recall `>= .90`，误把 line junction 路由到 OCR 的比例 `<= .05`，并且 source tagging 后 edge recall 相对 A 关下降不超过 `0.5pp`。标签只能在 proposal 后用于离线评估。

### C. source router frozen-feature gate

仅使用 A/B 可见的 target-free 特征训练一个 ≤3k 参数 MLP；输出 source tag 与 abstain 分数，不输出 atom id、bond type 或坐标。CPU 最多 500 steps，阈值仅 train 选择。full holdout 200-step 检查必须同时满足 B 关的 source 指标以及 graph P/R/sanitize/dangling floors；否则早停并 NO-GO。

## 当前归因结论（2026-10-07）

严格 full-holdout attribution（`tools/audit_line_first_node_edge_failure_attribution_r5.py`）显示，frozen vertex-candidate + edge sidecar 的 653 个漏边中：

- `0` 条来自 node keep 丢弃；`0` 条来自 edge score 阈值；全部属于原始 candidate reachability 缺失；
- aggregate recall 为 `.7981`，selected precision 为 `.9598`，dangling 为 `.0496`；
- cov13 三个 profile recall 仅 `.716-.785`，cov33 三个 profile 为 `.851-.861`；
- keep-all 与 drop-isolated A/B 完全相同，说明删孤立节点不是瓶颈。

因此下一步只允许做 raw endpoint/candidate reachability 归因（端点是否被 LSD 检出、是否进入 frozen candidate、是否被文字 seam 遮挡，以及按 bond 长度/角度/样式分桶）。在该归因前，不再训练 node/type router，不启动 GPU，也不改 production。

归因结果进一步显示：653 条 unreachable bond 中，`371` 条是 `implicit_cluster|text_seam`，`78` 条是 `text_seam|text_seam`，`132` 条是 `implicit_cluster|implicit_cluster`；cov13 base/stroke 的两端 LSD cluster 覆盖率仅约 `.42`，DOUBLE bond 两端 cluster 覆盖率仅 `.043`。一次 target-free local raster bridge 试验虽保持 sanitize `.979`，但 holdout recall 只有 `.743`、dangling `.623`，因此也判定 NO-GO。

下一步只做两项确定性诊断：text-seam 到 line-end 的受限桥接上界，以及 DOUBLE/TRIPLE 平行端点的 bundle/cluster 归因；不得将 raster bridge 直接接入训练或 production。

因此当前状态是：direct graph 表示 **NO-GO（synthetic ceiling 与严格 node/edge 指标不一致）**；source-router 训练和新的 GPU shadow 均不授权。只有 frozen vertex-candidate 路线在完整 holdout/profile 上通过归因后的 identity/edge gates，才可提出下一次受控 CPU 试训；production 仍冻结。
