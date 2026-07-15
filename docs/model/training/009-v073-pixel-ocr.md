# 009：v0.7.3 Pixel OCR

> 状态：候选，未打正式 tag  
> 最稳健 checkpoint：`outputs/checkpoints/chemvector-v073-pixel-ocr-original-refine-1k/last.pt`

## 假设

旧 OCR 从 stride-8 特征读取 16px 字符，通常只有约两个特征格宽，B 的两个孔、
P 的孔和 N 的斜线容易消失。即使使用真值锚点，B 准确率也只从 69.63% 提升到
72.62%，说明限制主要来自分辨率，而不是顶点坐标。

Pixel OCR 直接从原图按锚点裁取 `32x32` 区域，使用轻量 CNN 编码，再交给开放
UTF-8 byte decoder。辅助分类路由曾出现类别塌缩，因此推理默认使用 byte decoder。

## 训练

```bash
.venv-train/bin/python -m chemvector.pretrain_ocr \
  --dataset molecule-replacement --pixel-ocr \
  --ocr-roi-radius 20 \
  --init-checkpoint outputs/checkpoints/chemvector-v072-ocr-balanced-3k/last.pt \
  --train-shards Data/sdf/webdataset_full_v0/shards/train-*.tar \
  --validation-shards Data/sdf/webdataset_full_v0/shards/validation-000000.tar \
  --steps 3000 --batch-size 4 --workers 4 \
  --ocr-learning-rate 1e-4 --vision-learning-rate 0 \
  --output-dir outputs/checkpoints/chemvector-v073-pixel-ocr-3k
```

## 结果

| checkpoint | 有效 SDF | 化学完全一致 | B | F |
|---|---:|---:|---:|---:|
| 平衡旧 head 3k | 83.64% | 44.82% | 68.50% | 96.41% |
| Pixel OCR 最高准确率 | 75.55% | 52.55% | 85.48% | 68.30% |
| Pixel OCR 原图修正 | 81.36% | 49.91% | 80.24% | 68.95% |

最高准确率 checkpoint：

```text
outputs/checkpoints/chemvector-v073-pixel-ocr-balanced-2k/last.pt
```

## 决策

Pixel OCR 显著改善 B，但固定 ROI 在不同字号间存在冲突，并造成 F、C、Cl、I 的
权衡。该路线保留，下一阶段改为共享 CNN 的多尺度 ROI。
