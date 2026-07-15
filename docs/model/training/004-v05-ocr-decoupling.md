# 004：v0.5 几何与 OCR 解耦

> 状态：完成  
> 基础模型：ChemSVG v0.4

## 目标

在共享视觉编码器上增加独立 OCR head。ChemSVG 分支处理顶点、stroke 和键型；
OCR head 根据顶点锚点读取局部文字。OCR 使用 UTF-8 byte 输出，不把词表限制为
固定元素，因此 `R1`、`Ph`、`Me` 和其他未知字符串都能按普通文字表示。

## 数据范围

完整分子图片主要包含 `B/Br/C/Cl/F/I/N/O/P/S/Se/Si`。主数据没有的缩写和字符
最初使用 128×128 黑白局部图预训练。生成器加入单键、双键、三键和斜键干扰，
第一阶段不加入模糊、透明背景或扫描噪声。

```bash
.venv-train/bin/python -m chemvector.synthetic_ocr \
  --samples 64 \
  --output-dir outputs/diagnostics/synthetic-ocr-preview
```

## 局部 OCR 预训练

纯局部图训练必须冻结共享视觉编码器，只更新 OCR head：

```bash
.venv-train/bin/python -m chemvector.pretrain_ocr \
  --init-checkpoint outputs/diagnostics/chemvector-v4-replay-interp-010/model.pt \
  --output-dir outputs/checkpoints/chemvector-v5-ocr-frozen \
  --steps 10000 --batch-size 64 --workers 2 \
  --ocr-learning-rate 1e-4 --vision-learning-rate 0
```

即使给视觉层很小的学习率，局部 crop 分布也会拉低原 ChemSVG 几何准确率。
冻结后，原模型的 267 个基础 tensor 保持逐位相同。

## 完整分子联合训练

```bash
.venv-train/bin/python -m chemvector.train \
  --sequence-version 0.4 \
  --init-checkpoint outputs/checkpoints/chemvector-v5-ocr-frozen/last.pt \
  --train-shards Data/sdf/webdataset_full_v0/shards/train-000000.tar \
  --validation-shards Data/sdf/webdataset_full_v0/shards/validation-000000.tar \
  --no-shuffle --workers 0 \
  --batch-size 4 --gradient-accumulation 1 \
  --train-samples-per-epoch 1000 --epochs 1 \
  --learning-rate 1e-6 \
  --ocr-weight 1.0 --ocr-learning-rate-multiplier 10 \
  --synthetic-ocr-replay-weight 0.5 \
  --synthetic-ocr-replay-batch-size 64 \
  --visual-aux-weight 1.0 \
  --output-dir outputs/checkpoints/chemvector-v5-joint-1k
```

完整分子数据没有 `R1/Ph/Me` 等标签。不加入 synthetic replay 时，OCR head 会
快速遗忘开放字符。Replay 分支只更新 OCR head，视觉特征在 `no_grad` 下提取。

## 决策

几何与文字解耦保留。外部 PaddleOCR 教师停止使用，后续改为完整分子字符替换、
Pixel OCR 和 ChemVision 多尺度训练。
