# ChemVector Joint v1 数据交付协议

## 用途

本目录用于下一阶段 Bond proposal、set presence、endpoint incidence 联合训练。该阶段停止
v0.9.95-v0.9.98 的 residual 补丁路线，直接扩大共享模型训练数据和整图监督。

交付分为两部分：

1. `style_train_100k/`：扩大字体、线宽、布局和多键间距覆盖；
2. `fresh_eval_12k/`：此前模型、阈值和人工分析均未见过的 fresh dev 与 sealed test。

不得使用 MR-TADF、USPTO-30K、当前 validation/internal 或其同结构重绘作为本目录来源。

## 一、style_train_100k

### 规模

- 25,000 个 molecule；
- 每个 molecule 四种渲染：`original / rdkit / coordgen / stress`；
- 共 100,000 张 PNG；
- 全部标记为 `train`；
- WebDataset 每 shard 约 5,000 张图。

### 分子来源

- 从 `webdataset_full_v0` 的 train molecule 中确定性抽取；
- ChEMBL/COCONUT 建议为 85%/15%；
- molecule_id、InChIKey 和 scaffold 必须保留；
- 四种渲染共享 molecule_id、ChemMol JSON 和规范化 SDF；
- 不得抽取 validation、internal_test、supplement test 或 benchmark 分子。

### 渲染

- 固定 768x768 PNG；
- 使用 `chemvision_layers_v0_2` 的语义 SVG/四层输出协议；
- 字体至少覆盖当前 v0.2 的 16 个 family；
- 按 family、font bucket、line width、rotation、multiple-bond offset 分层抽样；
- 每种 style 参数组合不得只绑定一种 layout；
- `stress` 四类操作数量尽量均衡；
- 不增加模糊、JPEG、透明背景等文档噪声，本阶段只解决结构风格；
- 保留黑白、有限元素色和灰度，但黑白不少于 60%。

### 样本成员

每张图至少包含：

```text
sample.png
sample.chemmol.json
sample.render.json
sample.layers.json
sample.sdf
sample.meta.json
```

## 二、fresh_eval_12k

### 规模与权限

- `fresh_dev`：2,000 个 molecule、8,000 张图；
- `sealed_test`：1,000 个 molecule、4,000 张图；
- fresh_dev 可在 commitment 固定后用于选模；
- sealed_test 在候选通过 fresh_dev 前不得读取图片、SDF 或逐样本 manifest；
- sealed_test 只允许读取 commitment、总样本数和最终 validation 状态。

### 新分子来源

优先从 ChEMBL 37 原始全集重新选择，不得从当前 180,000 molecule 直接抽取。必须满足：

1. molecule_id/InChIKey 不在以下任一数据集中：
   - `webdataset_full_v0`
   - `chemvision_layers_v0_1/v0_2`
   - `molgrapher_synthetic_v0_1`
   - MR-TADF benchmark
   - USPTO-30K benchmark
2. Bemis-Murcko scaffold 与所有已用训练、validation、internal molecule 的 scaffold 重叠为 0；
3. fresh_dev 与 sealed_test 之间 molecule、InChIKey、scaffold 重叠均为 0；
4. 分子必须通过 RDKit sanitize、SDF 往返和 InChIKey 复核；
5. 第一版继续限制为模型当前支持的单组分有机结构，不引入金属、聚合物和反应图。

### 分布

- 重原子数建议 8-80，按 8-20、21-40、41-60、61-80 分层；
- 环数、稠环、桥环、手性中心、B/N/S/Se 等元素按桶统计；
- 不针对 MR-TADF 特征过采样；
- 四种 layout/stress 比例与训练一致；
- fresh_dev 和 sealed_test 的复杂度分布应接近，但 molecule/scaffold 完全隔离。

### 字体隔离

- 将字体 family 先分成 train-family 与 heldout-family；
- `style_train_100k` 不得使用 heldout-family；
- fresh_dev 每个 molecule 至少一张 heldout-family；
- sealed_test 的 heldout-family 组合不得与 fresh_dev 完全相同；
- manifest 必须记录 `font_family_group=train_seen/heldout`。

## 三、验收文件

每个根目录必须提供：

```text
summary.json
validation.json
samples.csv
shards/
previews/index.html
commitment.json
```

`validation.json` 必须包含：

- tar/member 全量可读；
- PNG 尺寸与解码通过；
- ChemMol -> RDKit Mol -> sanitize -> InChIKey 往返通过；
- molecule/scaffold/font-family 跨 split 重叠为 0；
- 与所有旧数据集的 molecule/InChIKey/scaffold 重叠审计；
- 每个 molecule 四种 variant 完整；
- 语义 layers 与 PNG foreground IoU；
- 1,000 样本重复生成 hash 一致；
- rejected 与自动替换数量；
- 数据生成脚本、Git revision、RDKit 版本和随机 seed。

## 四、Commitment

在模型人员第一次读取 fresh_dev 图片前，必须固定：

- samples.csv SHA256；
- 每个 tar SHA256；
- molecule ID 集合 SHA256；
- InChIKey 集合 SHA256；
- scaffold 集合 SHA256；
- renderer/style schema SHA256；
- 字体 family 分组 SHA256；
- fresh_dev 与 sealed_test 样本数。

sealed_test 的逐样本文件权限保持关闭，直到模型人员明确记录：

```text
candidate checkpoint SHA256
训练数据 commitment SHA256
fresh_dev 指标
固定推理参数
```

## 五、完成标志

数据人员仅在以下条件全部满足后报告完成：

```json
{
  "status": "passed",
  "style_train_samples": 100000,
  "fresh_dev_samples": 8000,
  "sealed_test_samples": 4000,
  "old_dataset_scaffold_overlap": 0,
  "cross_split_scaffold_overlap": 0
}
```

在 `fresh_eval_12k/validation.json` 为 `passed` 之前，模型侧只运行现有数据上的工程 smoke，不得
把结果表述为新路线能力提升。
