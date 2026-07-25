# MR-TADF v1 Benchmark

MR-TADF v1 用于检验大型 B/N 富集、多环、拥挤布局和超长键分子的图片到 SDF
能力。该基准与现有 18 万训练分子没有结构或 Murcko scaffold 重叠。

从 v0.9.64 开始，该数据集是当前结构路线的主要能力测试集。它仍然不进入训练、不用于
阈值拟合或 hard replay；locked internal100 作为回归护栏继续同步报告。

## v0.9.64 新结构主测试

v0.9.63 joint object + MILP 解码协议在全量 1,100 张上的结果：

- 有效 SDF：874，79.45%；
- 去立体 SDF 图完全一致：611，55.55%；
- 化学图完全一致：611，55.55%；
- 合法但结构错误：263；
- 无效预测：226，其中 Kekule 失败 186、价态失败 40；
- 标准/长/极端长键 exact：68.49% / 55.37% / 16.23%。

本轮使用 Oracle 原子化学状态，只测结构对象与图编译，不是完整 OCR 端到端成绩。结果位于
`outputs/evaluations/chemvector-v0964-mr-tadf-v1-main1100/`，每个样本均保存输入、真值
SDF/渲染和预测 SDF/渲染。

## v0.9.65 多尺度 Bond proposal fallback

逐层审计发现极端长键的 Bond proposal 完整覆盖只有 32.02%。v0.9.65 保留单尺度主路径，
只在其四级解码全部无效时调用已有多尺度 Bond proposal checkpoint，不按键长路由，也不
使用 MR-TADF 调阈值。

| 指标 | v0.9.64 | v0.9.65 |
|---|---:|---:|
| 有效 SDF | 79.45% | 88.55% |
| SDF 图完全一致 | 55.55% | 59.27% |
| 标准键长 exact | 68.49% | 71.80% |
| 长键 exact | 55.37% | 58.76% |
| 极端长键 exact | 16.23% | 21.49% |

结果位于 `outputs/evaluations/chemvector-v0965-mr-tadf-proposal-fallback-full1100/`。
本版还修复了 invalid 不保存 scene 的诊断缺口，1,100 个样本现在都有失败前或成功后的
ChemScene 可供前端检查。

## 数据构建

数据入口只能是审计后的
`Data/MR-TADF_SDF/audit/approved.csv`，禁止直接遍历原始 SDF 目录。

```bash
.venv-train/bin/python -m Data.sdf.build_mr_tadf_benchmark \
  --approved-manifest Data/MR-TADF_SDF/audit/approved.csv \
  --source-dir Data/MR-TADF_SDF \
  --output-dir Data/benchmarks/mr_tadf_v1
```

固定渲染协议：

- 768×768 PNG；
- 黑白配色；
- 保留源 SDF 二维坐标和相对键长；
- 旋转角度 0°、线宽 2.0、字体 16、padding 0.08；
- 每个分子只生成一张图，不使用随机增强；
- PNG 通过尺寸和非空像素验证后才写盘。

输出结构：

```text
Data/benchmarks/mr_tadf_v1/
├── images/            1,100 张输入 PNG
├── annotations/       原子、键和像素坐标
├── ground_truth/      1,100 份真值 SDF
├── manifest.csv       benchmark 加载入口
├── runtime_rejected.csv
└── summary.json
```

当前构建结果为 1,100/1,100 成功，运行时拒绝为 0，总占用约 102MB。按最大键长
与中位键长之比分层：

- 标准键长 `<1.75×`：695；
- 长键 `1.75×～2.5×`：177；
- 极端长键 `>=2.5×`：228。

v1 一旦用于模型对比，不再修改图片或真值。渲染协议变化必须建立 v2，不能覆盖
v1，否则不同模型的结果不可比较。

## 模型评测

```bash
.venv-train/bin/python -m chemvector.benchmark_mr_tadf \
  --checkpoint outputs/checkpoints/chemvector-v5-vertex-10k/last.pt \
  --benchmark-manifest Data/benchmarks/mr_tadf_v1/manifest.csv \
  --output-dir outputs/evaluations/mr-tadf-v1-v071 \
  --device cuda
```

评测器只加载一次模型，逐样本追加 `records.jsonl`，中断后使用完全相同的命令即可
续跑。`benchmark_config.json` 固化 checkpoint 与 manifest 的 SHA256，避免在同一
结果目录中混入不同模型或数据版本。

主要指标：合法 SDF、分子式一致、去立体结构一致、InChIKey 完全一致、原子数和
键数一致，以及三档键长难度的分层结果。

## v0.7.1 基线

稳定 checkpoint 在 RTX 5090 上的完整 1,100 张结果：

- 合法 SDF：71.91%；
- 分子式一致：34.09%；
- 去立体结构一致：33.82%；
- InChIKey 完全一致：33.82%；
- 原子数一致：61.27%；
- 键数一致：60.91%；
- 平均推理时间：0.099 秒/张；
- 失败分类：断图 23、价态 187、Kekulé 99。

InChIKey 完全一致率按难度分层：

- 标准键长：36.69%；
- 长键：34.46%；
- 极端长键：24.56%。

结果位于 `outputs/evaluations/mr-tadf-v1-v071/summary.json`。后续每个模型版本使用
独立输出目录，不覆盖该基线。

## 平衡字符替换 OCR 实验

checkpoint：

```text
outputs/checkpoints/chemvector-v072-ocr-balanced-3k/last.pt
```

评测结果：

```text
outputs/evaluations/mr-tadf-v1-v072-ocr-balanced-3k/
```

该实验只使用现有 ChEMBL/COCONUT 完整分子图做动态字符替换，没有把 MR-TADF
分子加入训练。与 v0.7.1 基线相比：

| 指标 | v0.7.1 | 平衡 OCR 3k |
|---|---:|---:|
| 有效 SDF | 71.91% | 83.64% |
| 分子式一致 | 34.09% | 45.00% |
| 化学完全一致 | 33.82% | 44.82% |
| 正确 SDF | 372 | 493 |
| 错误 SDF | 419 | 427 |
| 无效 SDF | 309 | 180 |

按键长难度的化学完全一致率：标准键长 47.05%，长键 40.68%，极端长键
41.23%。这说明收益来自 OCR 改进，而不是只对简单布局有效。

## Pixel OCR v0.7.3 候选

Pixel OCR 直接从原始像素裁取字符区域，避免 stride-8 特征丢失 B/P/N 的内部
笔画。没有使用 MR-TADF 分子训练。

| 指标 | 平衡旧 head | Pixel 最高准确率 | Pixel 原图修正 |
|---|---:|---:|---:|
| 有效 SDF | 83.64% | 75.55% | 81.36% |
| 分子式一致 | 45.00% | 52.91% | 50.36% |
| 化学完全一致 | 44.82% | 52.55% | 49.91% |
| 正确 SDF | 493 | 578 | 549 |
| 错误 SDF | 427 | 253 | 346 |
| 无效 SDF | 180 | 269 | 205 |

最高准确率输出：

```text
outputs/evaluations/mr-tadf-v1-v073-pixel-ocr/
```

原图修正版输出：

```text
outputs/evaluations/mr-tadf-v1-v073-pixel-original-refine/
```

最高准确率候选把 B 识别率从 68.50% 提升到 85.48%，但 F 从 96.41% 降到
68.30%。原图修正恢复了部分结构合法率，但 F 仍是下一阶段主要 OCR 问题。

## 多尺度 OCR v0.7.5 候选

ChemVision Layers pilot 从现有结构生成 10,000 张语义 SVG 图片，其中 9,000 张
用于训练，validation 和 internal_test 各 500 张并使用未见字体。MR-TADF 仍未
进入训练。

最终 checkpoint 为 500 step、batch 4，约 2,000 次图片采样；9,000 张是可用
训练数据池规模，不表示最优 checkpoint 已遍历全部图片。

v0.7.5 OCR 对同一原子中心裁取 `12/20/32px` 三种范围，使用共享 Pixel CNN
编码，再通过学习式权重融合。训练时只增加受控文字框锚点：

- B 标签全部增加一个 bbox 内锚点；
- N 标签只随机抽取 5% 增加 bbox 内锚点；
- 其他字符仍使用标准原子中心；
- 共享视觉编码器、vertex、stroke 和 relation 参数全部冻结；
- 最优 checkpoint 出现在 500 step，而不是总体 OCR 指标最高的后期 checkpoint。

checkpoint：

```text
outputs/checkpoints/chemvector-v075-pilot-multiscale-balanced-jitter-1500/checkpoint-0000500.pt
```

完整输出：

```text
outputs/evaluations/mr-tadf-v1-v075-multiscale-balanced-jitter-500/
```

| 指标 | v0.7.1 | v0.7.3 原图修正 | v0.7.5 候选 |
|---|---:|---:|---:|
| 有效 SDF | 71.91% | 81.36% | 89.73% |
| 化学完全一致 | 33.82% | 49.91% | 67.45% |
| 原子数一致 | 61.27% | 68.55% | 76.00% |
| 键数一致 | 60.91% | 68.64% | 76.18% |
| 平均推理时间 | 0.099s | 0.101s | 0.105s |

v0.7.5 的 1,100 张结果为：

- 完全正确：742；
- 有效但未完全一致：245；
- 无效：113；
- 断图：23；
- Kekule 失败：58；
- 价态失败：32。

按难度的化学完全一致率：

- 标准键长：72.37%；
- 长键：62.71%；
- 极端长键：56.14%。

实验同时证明 checkpoint 不能只按真值中心上的总体 OCR accuracy 选择。无控制的
bbox jitter、所有 N 全量复制、以及预测锚点续训都没有通过端到端门控；只有
B 全量、N 5% 的早期多尺度 checkpoint 在全部三个 MR-TADF 难度桶同时提升。

## v0.7.5 图错误诊断

诊断器使用 benchmark 的原子像素坐标将预测 ChemSVG 与真值图对齐，直接比较
原子属性和键集合，不使用 SMILES 作为错误分类依据。旧结果可以原地回填：

```bash
.venv-train/bin/python -m chemvector.benchmark_diagnostics \
  outputs/evaluations/mr-tadf-v1-v075-multiscale-balanced-jitter-500
```

358 个错误样本允许同时具有多个错误标签。按主要错误分类：

| 主要错误 | 样本数 |
|---|---:|
| 顶点检测或定位 | 255 |
| OCR 字符 | 95 |
| 键型 | 4 |
| 原子属性 | 1 |
| 芳香性标记 | 1 |
| 漏键 | 1 |
| 错连键 | 1 |

按并发错误标签统计：

- 142 个样本含 OCR 字符错误，其中 95 个没有同时发生顶点或拓扑错误；
- 230 个样本漏原子，48 个样本多原子；
- 244 个样本漏键，122 个样本有多余键；
- 64 个样本存在明确的交叉多余键；
- 78 个样本的多余键穿过第三个原子中心。

742 个化学完全一致样本全部通过图诊断一致性校验。芳香键与 Kekule 单双键的
等价表示已归一化，不会被误报为键型错误。

几个此前重点观察的样本在当前 checkpoint 上分别为：

- `GDG906`：单个 OCR 字符错误，顶点和键集合正确；
- `GDG907`、`GDG901`：SDF 完全正确；
- `GDG893`：漏 1 个原子，并连带漏 2 根键；
- `GDG751`：漏 1 个原子、漏 3 根键，并触发 Kekule 失败。

因此 v0.7.1 Oracle 得出的“OCR 是第一瓶颈”仍能解释多尺度 OCR 为什么带来大幅
收益，但不能直接作为 v0.7.5 的下一阶段结论。当前剩余错误首先要提高顶点召回和
定位稳定性；OCR 应继续作为并行分支维护，而不是继续单独扩大训练。

## v0.7.6 顶点召回 pilot

v0.7.6 使用同一 ChemVision pilot 的 9,000 张 train 图片，只训练 vertex head。
可训练参数为 92,548；与 v0.7.5 相比只有 10 个 `vertex_head.*` tensor 变化，
OCR、视觉编码器和 relation 权重逐位相同。

checkpoint：

```text
outputs/checkpoints/chemvector-v076-vertex-pilot-head-lr1e4-9k/best.pt
```

完整输出：

```text
outputs/evaluations/mr-tadf-v1-v076-vertex-head-pilot-9k/
```

| 指标 | v0.7.5 | v0.7.6 |
|---|---:|---:|
| 有效 SDF | 89.73% | 91.91% |
| 化学完全一致 | 67.45% | 77.55% |
| 原子数一致 | 76.00% | 82.45% |
| 键数一致 | 76.18% | 82.64% |
| 标准键长完全一致 | 72.37% | 81.01% |
| 长键完全一致 | 62.71% | 74.58% |
| 极端长键完全一致 | 56.14% | 69.30% |

v0.7.6 完全正确 853、有效错误 158、无效 89。它修复 156 个 v0.7.5 错误，
同时使 45 个旧正确样本回归，净增加 111 个正确。顶点主错误从 255 降到 191，
含 OCR 错误从 142 降到 81，说明错误顶点会级联污染 OCR。

`GDG906`、`GDG893`、`GDG751`、`GDG901` 和 `GDG907` 当前全部完全正确。
详细训练过程和第一次 NaN 失败记录见
[v0.7.6 独立训练档案](../model/training/20260715-v0.7.6-vertex-recall-pilot.md)。

## v0.7.8 OCR 残差适配器

v0.7.8 冻结 v0.7.6 的全部已有 tensor，只增加 29,044 个 OCR 残差适配器与
角色头参数。角色头直接参与删点的 200 张消融发生明显缺原子回归，因此正式评测
保持影子模式。

| 指标 | v0.7.6 | v0.7.8 |
|---|---:|---:|
| 有效 SDF | 91.91% | 91.36% |
| 化学完全一致 | 77.55% | 79.55% |
| 原子数一致 | 82.45% | 82.00% |
| 键数一致 | 82.64% | 82.00% |
| 标准键长完全一致 | 81.01% | 83.17% |
| 长键完全一致 | 74.58% | 76.27% |
| 极端长键完全一致 | 69.30% | 71.05% |

完整输出：

```text
outputs/evaluations/mr-tadf-v1-v078-adapter-shadow-role-full/
```

完全正确从 853 增至 875。有效率和计数指标小幅下降，但完全一致率提高 2 个
百分点，且 USPTO clean 1k 同时取得显著收益，因此接受为当前开发候选。详细过程
见 [v0.7.8 独立训练档案](../model/training/20260715-v0.7.8-ocr-residual-adapter.md)。

## v0.7.9 全点对骨架

v0.7.9 冻结 OCR 和 vertex，使用全部原子点对训练 relation/pair encoder，并让
repair 以模型边为主体，只补充连通分量之间的墨线桥边。

| 指标 | v0.7.8 | v0.7.9 |
|---|---:|---:|
| 有效 SDF | 91.36% | 92.00% |
| 化学完全一致 | 79.55% | 79.73% |
| 原子数一致 | 82.00% | 82.18% |
| 键数一致 | 82.00% | 82.18% |
| 标准键长完全一致 | 83.17% | 83.31% |
| 长键完全一致 | 76.27% | 76.27% |
| 极端长键完全一致 | 71.05% | 71.49% |

完整输出：

```text
outputs/evaluations/mr-tadf-v1-v079-relation-model-connect-t040-full/
```

该版本没有使用 MR-TADF 训练。完整过程见
[v0.7.9 独立训练档案](../model/training/20260715-v0.7.9-skeleton-all-pairs.md)。

## v0.7.10 概率桥接

v0.7.10 沿用 v0.7.9 checkpoint，只升级断图桥接协议。MR-TADF 仍未进入训练、
checkpoint 选择或参数调优。

| 指标 | v0.7.9 | v0.7.10 |
|---|---:|---:|
| 有效 SDF | 92.00% | 92.09% |
| 化学完全一致 | 79.73% | 79.73% |
| 原子数一致 | 82.18% | 82.18% |
| 键数一致 | 82.18% | 82.18% |
| 断图失败 | 14 | 12 |
| 价态失败 | 25 | 25 |

完整输出：

```text
outputs/evaluations/mr-tadf-v1-v0710-probability-bridge-full/
```

完全一致保持 877/1100，有效 SDF 从 1012 增至 1013，因此通过外部回归门控。
详细过程见
[v0.7.10 独立实验档案](../model/training/20260715-v0.7.10-probability-bridge.md)。

## Oracle 误差分解

Oracle 评测不训练模型，只在推理时将指定模块替换为真值，用来测量该模块修好后
的理论提升上限：

```bash
.venv-train/bin/python -m chemvector.oracle_mr_tadf \
  --checkpoint outputs/checkpoints/chemvector-v5-vertex-10k/last.pt \
  --benchmark-manifest Data/benchmarks/mr_tadf_v1/manifest.csv \
  --baseline-dir outputs/evaluations/mr-tadf-v1-v071-diagnostics-v2 \
  --device cuda
```

完整 1,100 张结果：

| 模式 | 有效 SDF | InChIKey 完全一致 |
|---|---:|---:|
| v0.7.1 基线 | 71.91% | 33.82% |
| 只替换 OCR 字段 | 91.82% | 75.82% |
| 只替换顶点坐标 | 72.91% | 36.00% |
| 同时替换顶点与 OCR | 97.73% | 94.91% |

OCR Oracle 只替换元素、电荷、同位素、显式氢和自由基字段，保留模型自己的坐标、
边和芳香预测。Vertex Oracle 只替换顶点数量与坐标，仍使用模型 OCR、芳香预测和
关系头。因此结果表明当前主要误差来自 OCR，少量顶点漏检会与 OCR 错误产生明显
级联；在顶点和 OCR 都正确时，现有关系头与几何修复已经可以达到 94.91% 完全
一致。

三个 Oracle 输出目录与普通 benchmark 使用相同协议，会自动显示在本地 benchmark
前端中。

## v0.7.20 跨字体全局 Proposal Graph 回归

v0.7.20 未使用 MR-TADF 训练、checkpoint 选择或阈值选择。两个内部域共同选择的
统一 node threshold 为 `0.90`，随后一次性运行完整 1,100 张外部回归。

| 指标 | 稳定 v0.7.10 | v0.7.20 |
|---|---:|---:|
| 有效 SDF | 92.09% | 84.82% |
| 化学完全一致 | 79.73% | 49.00% |
| 原子数一致 | 82.18% | 51.09% |
| 键数一致 | 82.18% | 51.00% |
| 标准键长完全一致 | 83.31% | 54.82% |
| 长键完全一致 | 76.27% | 46.89% |
| 极端长键完全一致 | 71.49% | 32.89% |

主要错误变为顶点过删：475 个错误样本含 missing atom，missing atom 事件共 1,235。
该结果判定固定 threshold 部署 NO-GO，不根据 MR-TADF 反向调整阈值。

输出：

```text
outputs/evaluations/mr-tadf-v1-v0720-style-global-t090/
```
