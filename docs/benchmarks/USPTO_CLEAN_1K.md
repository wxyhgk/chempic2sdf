# USPTO Clean 1K 公开 Benchmark

该基准从固定 revision 的 `USPTO-30K clean` 子集随机抽取 1,000 个可 sanitize
分子，用于 ChemVector 的第一版公开 OCSR 基线。它不包含缩写 S-group，也不混入
`large` 子集的模型容量问题。

## 固定协议

- 数据 revision：`58b3a773f01816d62126f11dec4d5e033e8ebf9d`；
- 随机 seed：`20260715`；
- 抽样方法：固定随机排列后依次选取 1,000 个可 sanitize 记录；
- 官方 PNG 字节原样复制，不裁剪、不缩放、不重新编码；
- 官方 V2000 Molfile 只增加 `$$$$` 记录分隔符，不经 RDKit 重写；
- 异常候选写入 `rejected.csv`，再从固定排列中补足数量。

构建命令：

```bash
.venv-train/bin/python \
  Data/benchmarks/public_ocsr_v1/build_uspto_clean_1k.py
```

输出目录：

```text
Data/benchmarks/public_ocsr_v1/prepared/uspto_clean_1k/
├── images/             官方图片
├── ground_truth/       官方 Molfile 封装的 SDF
├── manifest.csv        评测入口
├── rejected.csv        抽样时跳过的异常候选
├── summary.json        来源、seed 和数量
└── validation.json     图片/SDF hash 与读回验收
```

## SDF 比较协议

正式准确率不使用 SMILES 字符串。评测器先读回真值和预测 SDF，再比较：

1. 原子数、键数和片段数；
2. 原子元素、同位素、形式电荷、自由基和芳香性；
3. 键端点、键级和芳香性；
4. 带立体化学的完整图同构。

同时报告去立体图同构、分子式、原子数、键数和 Morgan fingerprint 相似度，便于
判断只错一个局部的预测。`chemical_match_rate` 在该 benchmark 中表示“带立体
化学的 SDF 属性图完全一致率”，不是 SMILES 或文本 SDF 完全一致率。

运行当前 v0.7.6：

```bash
.venv-train/bin/python -m chemvector.benchmark_public_ocsr \
  --checkpoint outputs/checkpoints/chemvector-v076-vertex-pilot-head-lr1e4-9k/best.pt \
  --benchmark-manifest Data/benchmarks/public_ocsr_v1/prepared/uspto_clean_1k/manifest.csv \
  --output-dir outputs/evaluations/uspto-clean-1k-v076 \
  --device cuda
```

评测保留 `correct_*.sdf`、`wrong_*.sdf`、`invalid_*.sdf`、原始/修复 ChemSVG 和
统一适配后的模型输入。运行中断后，使用同一命令会从 `records.jsonl` 续跑；配置
或 checkpoint hash 不一致时会拒绝写入原目录。

## v0.7.6 首次结果

最初使用当前最佳 checkpoint 完整评测 1,000 张官方图片：

| 指标 | 结果 |
|---|---:|
| SDF 有效 | 79 / 1,000（7.9%） |
| 带立体 SDF 图完全一致 | 5 / 1,000（0.5%） |
| 去立体 SDF 图完全一致 | 5 / 1,000（0.5%） |
| 原子数一致 | 32 / 1,000（3.2%） |
| 键数一致 | 37 / 1,000（3.7%） |
| 平均推理时间 | 0.0387 秒/张 |

921 个无效预测中，价态错误 674 个、断图 224 个、Kekulé 错误 23 个。输出位于：

```text
outputs/evaluations/uspto-clean-1k-v076/
```

该结果比 MR-TADF 低很多，不应解释为 SDF 比较过严，因为 SDF 有效率和计数指标
也同步很低。直接检查官方图片可见，clean 子集仍经常带有 `(THZ1)`、编号、标点等
分子外文字，同时 ChemDraw 字体、画布比例和布局也不同于当前合成训练域。模型会
把部分外围文字误识别成原子或线段。这是公开 benchmark 揭示出的真实域外问题，
不能通过清洗官方测试图片来掩盖；后续应在训练数据中加入独立的非分子文字负样本
与公开图风格增强，并继续用未修改的本基准验收。

### 页面框修复复核

旧输入预处理会把同时贴边的大型分子连通域误认为页面边框。1,000 张中有 20 张
触发，平均删除 75.8% 前景。修复后 v0.7.6 的结果为：

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| SDF 有效 | 79 | 88 |
| SDF 图完全一致 | 5 | 5 |
| 原子数一致 | 32 | 36 |
| 键数一致 | 37 | 41 |

字体扩展 OCR 全量微调把完全一致提高到 8、有效 SDF 提高到 90，但在 MR-TADF
上从 77.55% 退化到 71.27%，因此该 checkpoint 已否决，不能作为新的公开基线。
该 v0.7.7 实验结束时正式版本仍为 v0.7.6。

## v0.7.8 OCR 残差适配器

v0.7.8 冻结 v0.7.6 的全部已有权重，只新增 29,044 个 OCR 残差适配与角色头
参数。角色头在本次正式推理中保持影子模式，不参与删点。

| 指标 | v0.7.6 修复预处理后 | v0.7.8 |
|---|---:|---:|
| SDF 有效 | 88 | 167 |
| SDF 图完全一致 | 5 | 20 |
| 原子数一致 | 36 | 84 |
| 键数一致 | 41 | 81 |
| 平均推理时间 | 约 0.039 秒 | 0.0387 秒 |

输出位于：

```text
outputs/evaluations/uspto-clean-1k-v078-adapter-shadow-role/
```

该结果证明字体残差适配可以在不整体更新 OCR 的情况下迁移到公开字体域。剩余
833 个无效预测仍以价态错误为主，双键和拓扑问题按既定计划后置处理。

## v0.7.9 全点对骨架

v0.7.9 冻结 OCR、vertex 和视觉主干，使用全部原子点对训练 relation/pair encoder，
并采用 `edge_threshold=0.4`、`repair_candidate_mode=model_connect`。

| 指标 | v0.7.8 | v0.7.9 |
|---|---:|---:|
| SDF 有效 | 167 | 177 |
| SDF 图完全一致 | 20 | 22 |
| 原子数一致 | 84 | 87 |
| 键数一致 | 81 | 86 |
| 价态失败 | 579 | 532 |
| 断图失败 | 204 | 242 |

输出位于：

```text
outputs/evaluations/uspto-clean-1k-v079-relation-model-connect-t040/
```

概率约束减少了多余边和价态失败，但较保守的跨域边召回增加了断图。下一阶段需要
改进补边排序，而不是恢复无条件全点对墨线连接。

## v0.7.10 概率桥接

v0.7.10 不训练新权重。推理保留所有点对的连续边概率，以模型概率优先、墨线 gap
次优的顺序选择桥边；墨线仍无法连通时，只允许概率不低于 0.1 且连接不同分量的
点对作为兜底。

| 指标 | v0.7.9 | v0.7.10 |
|---|---:|---:|
| SDF 有效 | 177 | 192 |
| SDF 图完全一致 | 22 | 22 |
| 原子数一致 | 87 | 87 |
| 键数一致 | 86 | 87 |
| 价态失败 | 532 | 567 |
| 断图失败 | 242 | 192 |

输出位于：

```text
outputs/evaluations/uspto-clean-1k-v0710-probability-bridge/
```

断图减少 50 个，有效 SDF 增加 15 个，完全一致没有回退。价态失败增加是断图被
连接后暴露出的下一层错误，不代表新增了 35 个原本正确的结构。详细实现和门控见
[v0.7.10 独立实验档案](../model/training/20260715-v0.7.10-probability-bridge.md)。

## v0.7.18 联合 Proposal Graph

v0.7.18 固定 v0.7.10 主权重和现有 relation/repair，只把顶点选择替换为联合
node-edge proposal graph。graph threshold `0.87` 只在 MolGrapher validation 选择，
USPTO clean 1k 未参与训练、checkpoint 或阈值选择。

| 指标 | v0.7.10 | v0.7.17 keep | v0.7.18 graph |
|---|---:|---:|---:|
| SDF 有效 | 192 | 226 | 241 |
| SDF 图完全一致 | 22 | 19 | 29 |
| 原子数一致 | 87 | 101 | 115 |
| 键数一致 | 87 | 67 | 84 |
| 平均指纹相似度 | 0.0980 | 0.0964 | 0.1140 |
| 价态失败 | 567 | 393 | 436 |
| 断图失败 | 192 | 320 | 254 |

相对 v0.7.17 的 paired 结果为：保留 17 个旧正确、救回 12 个、破坏 2 个，净增
10 个完全一致。结果目录：

```text
outputs/evaluations/uspto-clean-1k-v0718-proposal-graph-t087/
```

该结果接受为 shadow 候选，但断图仍高于稳定 v0.7.10，尚不直接替换默认推理。

## v0.7.19 Graph Edge Prior

固定 v0.7.18 的节点和全部阈值，只在 `model_connect` 补桥时对 graph score 不低于
`0.85` 的点对使用 `max(relation_score, graph_score)`。结果只有 2/1000 个样本改变：

| 指标 | v0.7.18 | v0.7.19 prior |
|---|---:|---:|
| SDF 有效 | 241 | 242 |
| SDF 图完全一致 | 29 | 29 |
| 断图失败 | 254 | 252 |
| 价态失败 | 436 | 437 |

一个样本由断图变成 fingerprint similarity 为 0 的有效错误，另一个由断图变成价态
失败。该方案判定 NO-GO，不在本公开集上继续降低阈值。

## v0.7.20 跨字体全局 Proposal Graph

v0.7.20 的 checkpoint 和统一 node threshold `0.90` 只由 held-out 字体 validation
与 MolGrapher retention validation 选择。USPTO clean 1k 仅在内部选择冻结后运行。

| 指标 | v0.7.18 | v0.7.20 |
|---|---:|---:|
| SDF 有效 | 241 | 258 |
| SDF 图完全一致 | 29 | 30 |
| 原子数一致 | 115 | 131 |
| 键数一致 | 84 | 92 |
| 平均指纹相似度 | 0.1140 | 0.1268 |
| 有效预测平均相似度 | 0.4730 | 0.4916 |
| 断图失败 | 254 | 230 |
| 价态失败 | 436 | 433 |

输出：

```text
outputs/evaluations/uspto-clean-1k-v0720-style-global-t090/
```

公开域有小幅一致提升和明显断图减少，但 MR-TADF 外部回归失败，因此该 graph 仍不
替换稳定默认。完整决策见
[v0.7.20 训练档案](../model/training/20260716-v0.7.20-style-global-proposal.md)。

## v0.8.16 语义 Bond Endpoint Junction

v0.8.16 不再由平行 stroke 或旧隐式顶点检测器生成结构点。每个 Bond Object 固定
两个语义端点，端点聚合为 Junction；OCR 只能附着标签。

| 指标 | v0.7.10 | v0.7.20 | v0.8.16 |
|---|---:|---:|---:|
| SDF 有效 | 192 | 258 | 730 |
| SDF 图完全一致 | 22 | 30 | 131 |
| 原子数一致 | 87 | 131 | 600 |
| 键数一致 | 87 | 92 | 584 |
| 平均指纹相似度 | 0.0980 | 0.1268 | 0.4051 |

v0.8.16 将完全一致提高到 13.10%，说明语义端点骨架路线有效；但 216 个价态失败、
大量芳香键转单键以及元素标签错误表明端到端仍未过门槛。结果目录：

```text
outputs/evaluations/uspto-clean-1k-v0816-bond-endpoints/
```

## v0.9.39 BondObject 风格 MoE

v0.9.39 保留原 BondObject 专家和 MolGrapher 风格专家。gate 只用内部 MolGrapher/
ChemVision 来源标签训练，阈值 `0.90` 也只由内部验证选择；USPTO 不参与训练或调参。

同代码、同 checkpoint、同 1,000 张图片的 paired 结果：

| 指标 | v0.9.36 基线重跑 | v0.9.39 MoE |
|---|---:|---:|
| SDF 有效 | 69.1% | 68.5% |
| 化学完全一致 | 15.7% | 15.4% |
| 纯骨架 exact | 70.5% | 69.3% |
| 原子类型+骨架 | 28.9% | 28.5% |
| 键级+骨架 | 57.3% | 56.7% |

gate 将 56 张图片送入风格专家。纯骨架只救回 1 张、破坏 13 张；未路由的 944 张
纯骨架和化学 exact 全部不变。说明运行时回退正确，但“图片像哪个数据来源”不能
替代“哪个专家更可能正确”。该 gate 判定 NO-GO，不使用 USPTO 标签反向训练。

结果：

```text
outputs/evaluations/uspto-clean-1k-v0936-predicted-junction-relations-rerun/
outputs/evaluations/uspto-clean-1k-v0939-bond-style-moe/
```

## v0.9.41 Bond 专家收益路由

v0.9.41 不使用来源风格作为路由目标，只读取两个 BondObject 专家的预测数量、置信度、
端点聚类图、自环/重复 pair 和专家分歧。router 训练与阈值选择只使用内部数据，
USPTO 标签未参与。

| 指标 | v0.9.36 同代码基线 | v0.9.41 |
|---|---:|---:|
| SDF 有效 | 69.1% | 70.4% |
| 化学完全一致 | 15.7% | 16.0% |
| 纯骨架 exact | 70.5% | 72.7% |
| 原子类型+骨架 | 28.9% | 29.3% |
| 键级+骨架 | 57.3% | 59.1% |

纯骨架 paired 为救回 29、破坏 7，净增 22；化学完全一致救回 4、破坏 1，净增 3。
该结果接受为 shadow GO。输出：

```text
outputs/evaluations/uspto-clean-1k-v0941-bond-utility-router/
```

## v0.10.16 Structural Twin 公开 100 配对回归

使用 USPTO clean 1k 官方 manifest 中前 100 条 ready 记录，按原顺序同时运行 v0.10.10
Direct+OCR 与 v0.10.16 structural-twin shadow。样本不按触发或结果筛选。

| 指标 | v0.10.10 | v0.10.16 | paired break |
|---|---:|---:|---:|
| SDF valid | 43 | 43 | 0 |
| 非立体 exact | 0 | 0 | 0 |
| chemical exact | 0 | 0 | 0 |

v0.10.16 自然触发 3 张；三张均未破坏 valid。其余 97 张的 OCR 标签与预测图全部一致。
该结果通过未见公开非回退门，但两路 exact 都是 0，说明 USPTO 风格上的主要瓶颈仍在基础
Atom/Bond/OCR 链路，不能将本结果解释为公开准确率提升。

输出：

```text
outputs/evaluations/uspto-clean-100-v01016-structural-twin-ab/
```

报告 SHA256：
`7d6794ab950c015c9b75db0ddf5084ac03326a007c899dd22568e5373c508571`。

## v0.10.16 Structural Twin 公开 1K 配对回归

固定使用官方 manifest 的前 1,000 条记录，拆为四个互斥的 250 样本分片运行后合并。
structural-twin 自然触发 15 张；这 15 张同时运行 v0.10.10 baseline 和 v0.10.16 candidate。
其余 985 张经确定性验证无动作，因此使用 identity 结果，避免重复执行相同推理链。

| 指标 | v0.10.10 | v0.10.16 | rescue | break |
|---|---:|---:|---:|---:|
| SDF valid | 474 | 474 | 0 | 0 |
| 非立体 exact | 0 | 0 | 0 | 0 |
| chemical exact | 0 | 0 | 0 | 0 |

15 张触发样本的 OCR 文本 multiset 全部保持不变。Atom 数误差在 10 张改善、5 张变差；
Bond 数误差在 5 张改善、10 张变差。985 张无动作样本的 OCR 和预测图均无变化。

该测试通过公开 1k 配对非回退门，但不能宣称准确率提升：两路完整 exact 都仍为 0。
structural-twin 是安全的窄范围重复节点修复，不是 USPTO 风格下基础 Atom/Bond/OCR 错误的
主要解法，后续不再沿该证书继续局部优化。

合并输出：

```text
outputs/evaluations/uspto-clean-1k-v01016-structural-twin-ab/
```

报告 SHA256：
`cb1a3f8e432b711e608378ef7849518d99b6458682ad63766dd615b488dd9897`。

配置 SHA256：
`8a63051649f8e7c808f3ea0066af156957b008ade6d7a62d00054e5bafe5baf8`。

## v0.10.17 语义 BondObject + v0.10.9 OCR

v0.10.17 回到 v0.9.41 的双 BondObject 专家、utility router、semantic Junction relation 和
全部冻结阈值，只把 v0.8.8 CTC 替换为协议兼容的 v0.10.9 正确标签开放 CTC。USPTO 不参与
checkpoint、路由或阈值选择。

| 指标 | v0.9.41 | v0.10.17 | rescue | break |
|---|---:|---:|---:|---:|
| SDF valid | 704 | 785 | 98 | 17 |
| 非立体 SDF exact | 203 | 272 | 97 | 28 |
| Chemical exact | 160 | 215 | 75 | 20 |
| Skeleton exact | 727 | 727 | 0 | 0 |
| Atom-typed skeleton | 293 | 320 | 64 | 37 |
| Bond-typed skeleton | 591 | 632 | 53 | 12 |
| Labeled graph exact | 237 | 294 | 84 | 27 |

OCR 输出在 630 张变化，Atom/Bond 数量逐图不变。更正确的原子状态减少了错误价态和 aromatic
fallback，因此 valid、Atom-typed 和 Bond-typed 同时净增；但完整 1k 仍存在 paired break，
该组合只作为非推荐 shadow，不宣称逐图安全替换。

输出：

```text
outputs/evaluations/uspto-clean-1k-v0941-v0109-ocr-swap/
```

- summary SHA256：`18fee99d680c791f9e792f7c28e257d494c6860c53f60648b70f36745025de82`
- decomposition SHA256：`61339a4327f7f81b8a4ab1b67a3232f64a3e2e6dedbc4055fa24b65919be1db1`
- config SHA256：`9b67995ecbc75ab89e810a9010eba84022801ee9b75475a9612adfdd13bdbd48`

## v0.10.18 保守双 OCR 路由

v0.10.18 保持 v0.9.41 全部结构组件和 v0.8.8 主 OCR，同时加载 v0.10.9 OCR 专家。只有旧新
文字不同且内部 utility router 概率不低于固定 `0.95` 时才采用专家结果。USPTO 不参与 router
训练、checkpoint 或阈值选择。

| 指标 | v0.9.41 | v0.10.18 | rescue | break |
|---|---:|---:|---:|---:|
| SDF valid | 704 | 704 | 0 | 0 |
| Skeleton exact | 727 | 727 | 0 | 0 |
| Atom-typed skeleton | 293 | 301 | 8 | 0 |
| Bond-typed skeleton | 591 | 591 | 0 | 0 |
| Labeled graph exact | 237 | 244 | 7 | 0 |
| 非立体 SDF exact | 203 | 209 | 6 | 0 |
| Chemical exact | 160 | 162 | 2 | 0 |

1,378 个 OCR 分歧中只 route 12 次，改变 9 张图片。该版本的收益明显小于 v0.10.17，但完整
1k 所有 paired 指标零回退，因此作为独立的保守 shadow，而不是用汇总净收益掩盖局部破坏。

输出：

```text
outputs/evaluations/uspto-clean-1k-v01018-ocr-router-t095/
```

- Router SHA256：`26cc1f3233e583925273cfe83b64c6a5016c9ecccd7b792228f8aeabf29c175b`
- summary SHA256：`b526fe821cb16bb89c4ac0500fdd8f72ff620ade840a401e404054269c27f1b0`
- decomposition SHA256：`11192db46c7f29db48a1d7ff631533df2243f43733c2ee7c4fac1171fe510c8d`
- config SHA256：`52c9c3fce831926c3aee1ea936d95b83c9356757015fdde2c5ab21e41eecdb21`
