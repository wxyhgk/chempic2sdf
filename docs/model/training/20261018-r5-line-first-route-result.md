# r5 line-first route：CPU gate、shadow 与严格整合结果

## 为什么转向 line-first

residual-only vertex shadow 找回了 O/N，但 holdout residual precision 只有
`0.6111`，one-residual bond coverage 只有 `0.3727`。继续训练 residual
ownership 或 endpoint-pair MLP 不能解决图连接瓶颈。

固定 r5 图片上，完全绕过 frozen bond endpoints 的路线为：

```text
raster LSD segments -> endpoint clustering -> provisional unordered edges
```

proposal 阶段不读取 target atom、bond、vertex candidate 或 sealed test。

## 证据

train-only 选择 cluster radius `10px` 的 CPU gate：

- train edge precision `0.9889`，bond recall `0.9615`
- holdout edge precision `0.9638`，bond recall `0.9059`
- holdout 248/474 图恢复全部键
- AROMATIC/DOUBLE/TRIPLE recall `1.0`，SINGLE recall `0.8993`

随后完成一次 800-step、17 维几何/墨迹特征 edge + bond-type shadow。固定
train-only threshold，不接 production：

- 80-image shadow holdout：edge precision `0.9524`，edge recall `0.9091`，
  bond-type accuracy `0.982`
- full 696/474 replay（固定 shadow state/threshold）：edge precision
  `0.9247`，edge recall `0.8813`，bond-type accuracy `0.9611`

最后用 frozen scored vertex candidates 做严格 candidate-aligned 的独立整合，
不使用 GT 中心构造 proposal：

- holdout 474：edge precision `0.9598`
- edge recall `0.7981`
- conditional bond-type accuracy `0.9667`
- train：precision `0.9856`，recall `0.9031`

严格整合报告：

```text
outputs/audits/chemvector-identity-aligned-line-first-edge-bond-candidate-integration-r5/report.json
```

## 决策

line-first edge/type shadow 通过诊断门，但仍不能直接生产化：当前主要缺口是
provisional node 与化学 atom identity 的稳定对应，而不是 bond order。下一步
只做一次独立 node-identity/graph-assembly shadow，要求：

- 固定 line-first edge/type state 和 train-only threshold；
- node identity 只用图像/候选/局部 OCR 证据，GT 仅作后置评估；
- holdout distinct-node precision >= `0.90`、node recall >= `0.85`；
- complete-edge precision >= `0.90`、edge recall >= `0.75`、type accuracy
  >= `0.90`、self-loop/dangling = `0`；
- production graph、selector、bond head、checkpoint 仍完全冻结。

若 node identity shadow 失败，停止 GPU 训练，转为扩大 line/node identity 的
针对性数据；不回到 residual pair scorer。

当前 node identity CPU 预检结果：

- 简单 node-identity MLP holdout precision `0.8589`，未达到 `0.90`，因此没有
  启动 node GPU shadow；
- 固定 greedy candidate alignment（cluster radius 16px）在全 holdout 的
  node precision/recall 为 `0.9309/0.8949`，但独立 80-image graph replay
  仍有 `2.67%` dangling edge；
- 这说明下一步应改进 node→candidate ownership 和 dangling veto，而不是
  再训练一层残余 atom head。

补充的 node-identity CPU v3 将“每个 line node 全局一对一分配一个 scored
candidate”作为冻结 assembly contract。holdout 80 张上得到 node precision
`0.9363`、recall `1.0`；但 MLP 没有筛掉任何节点，收益为零，因此不启动
额外 node GPU shadow。独立 graph replay 的 dangling 行在序列化前全部拒绝，
不会进入图；该规则必须和 edge recall 一起审计，不能靠阈值掩盖漏边。

固定 shadow state 的 graph/chemistry round-trip（80-image holdout）结果：

- edge precision `0.9893`，edge recall `0.8400`
- conditional bond-type accuracy `0.9892`
- self-loop `0`
- raw dangling `14/511 = 0.0274`，均在序列化前拒绝
- candidate-only C topology RDKit sanitize `0.875`
- oracle-element overlay sanitize `0.950`

报告：

```text
outputs/audits/chemvector-identity-aligned-line-first-graph-chemistry-roundtrip-r5/report.json
```

因此下一步不再训练 node MLP；优先做固定 assembly contract 的 chemistry
round-trip/元素识别接入审计。`oracle-element` 结果只能说明拓扑和键型大体
可行，不能当作真实元素识别准确率，也不授权生产合并。
