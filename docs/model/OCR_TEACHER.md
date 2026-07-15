# OCR 路线与训练索引

本页只说明当前 OCR 设计和相关训练档案。详细训练过程不再堆放在同一个文件中。

## 当前原则

- 不使用 PaddleOCR 或其他通用 OCR 生成训练伪标签。
- OCR 只学习“图片区域到 UTF-8 字符串”，不负责化学理解。
- 标签不限制为元素表，支持 `R1`、`Ph`、`Me`、`A`、`X` 和任意后续字符串。
- 字符训练不能改变 SDF、顶点或键标签。
- 纯 OCR 课程默认冻结共享视觉、vertex、stroke 和 relation 分支。
- 不能只按内部 OCR accuracy 选择 checkpoint，必须通过端到端 SDF benchmark。

## 标签范围

- 周期表全部 118 个元素符号。
- `H/B/C/N/O/F/Si/P/S/Cl/Se/Br/I` 等常见字符提高采样率。
- `NH2`、`N+`、`O-`、`13C` 等组合字符串。
- `A`、`X`、`R1`、`R(n)`、`*` 等占位符。
- `Ph`、`Me`、`Et`、`tBu`、`Boc`、`TMS` 等简写。

这些标签全部作为原始字符串训练。后续是否用正则或化学规则解释字符串，属于
SDF 编译后的独立模块，不进入 OCR loss。

## 当前模型

v0.7.5 对每个锚点同时读取 `12/20/32px` 三个 ROI，三个尺度共享 Pixel CNN，
再由 scale scorer 融合。B 全部增加一个 bbox 内锚点，N 只抽 5%。

当前 checkpoint：

```text
outputs/checkpoints/chemvector-v075-pilot-multiscale-balanced-jitter-1500/checkpoint-0000500.pt
```

该 OCR 组件在 v0.7.6 中保持冻结且权重逐位不变。配合重新训练的 vertex head，
MR-TADF 达到 91.91% 有效 SDF、77.55% 化学完全一致；含 OCR 错误的样本从
142 降到 81，说明一部分 OCR 错误实际由错误锚点级联产生。

## OCR 训练档案

| 阶段 | 内容 | 结论 |
|---|---|---|
| [v0.5 解耦](training/004-v05-ocr-decoupling.md) | 独立 UTF-8 OCR head 和 synthetic replay | 局部预训练必须冻结视觉层 |
| [v0.7.2 字符替换](training/008-v072-replacement-ocr.md) | 完整分子图替换少量字符串 | 平衡 3k 有效，高比例 10k 失败 |
| [v0.7.3 Pixel OCR](training/009-v073-pixel-ocr.md) | 从原图 crop 读取细笔画 | B 提升，但固定 ROI 存在字符权衡 |
| [v0.7.5 多尺度](training/010-v075-chemvision-multiscale-ocr.md) | 12/20/32px 共享 CNN | 当前 OCR 组件，暂不生成 72 万全量语义层 |
| [历史 PaddleOCR](training/historical-paddleocr-teacher.md) | 通用 OCR 教师伪标签 | 已停止，仅保留复现说明 |

## 数据预览

生成完整分子字符替换预览：

```bash
.venv-train/bin/python -m chemvector.replacement_ocr \
  --shards Data/sdf/webdataset_full_v0/shards/train-000000.tar \
  --samples 16 \
  --output-dir outputs/diagnostics/replacement-ocr-preview
```

ChemVision pilot 数据报告：

- [`MULTISCALE_OCR.md`](../../Data/sdf/chemvision_layers_v0_1/reports/MULTISCALE_OCR.md)
- [`ORACLE_SEMANTIC_LAYER.md`](../../Data/sdf/chemvision_layers_v0_1/reports/ORACLE_SEMANTIC_LAYER.md)

## 新实验要求

新的 OCR 训练必须新建独立文档，使用
[`training/TEMPLATE.md`](training/TEMPLATE.md)，并在本页只增加一行摘要。
