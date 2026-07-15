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
