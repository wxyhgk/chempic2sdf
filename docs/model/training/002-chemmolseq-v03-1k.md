# 002：ChemMolSeq v0.3 1k 训练

> 状态：历史基线  
> 协议：ChemMolSeq v0.3

## 目标

使用图片空间原子顺序，去除显式 record ID 和渲染专用键字段，并把二维坐标放到
化学图之后，降低自回归序列长度。

首个 shard 的 4,992 张图片全部通过化学往返，平均序列缩短 16.71%。同时发现
13 张增强图片的二维 E/Z 几何与化学标签冲突，正式训练需要通过 reject key 排除。

## 第一阶段：视觉接地

```bash
.venv-train/bin/python -m chemvector.train \
  --sequence-version 0.3 \
  --reject-keys-file Data/sdf/chemmolseq_v0_3_depiction_reject_keys.txt \
  --train-shards Data/sdf/webdataset_full_v0/shards/train-000000.tar \
  --no-shuffle --workers 0 \
  --batch-size 4 --gradient-accumulation 1 \
  --train-samples-per-epoch 1000 --epochs 10 \
  --token-corruption-probability 0.5 --visual-aux-weight 5.0 \
  --output-dir outputs/checkpoints/chemvector-v3-grounded-1k
```

## 第二阶段：前缀恢复

```bash
.venv-train/bin/python -m chemvector.train \
  --sequence-version 0.3 \
  --train-shards Data/sdf/webdataset_full_v0/shards/train-000000.tar \
  --reject-keys-file Data/sdf/chemmolseq_v0_3_depiction_reject_keys.txt \
  --no-shuffle --workers 0 \
  --batch-size 4 --gradient-accumulation 1 \
  --train-samples-per-epoch 1000 --epochs 20 \
  --token-corruption-probability 0.1 --visual-aux-weight 1.0 \
  --resume outputs/checkpoints/chemvector-v3-grounded-1k/last.pt \
  --resume-learning-rate 3e-5 \
  --output-dir outputs/checkpoints/chemvector-v3-finetune-1k
```

## 第三阶段：自回归精调

```bash
.venv-train/bin/python -m chemvector.train \
  --sequence-version 0.3 \
  --train-shards Data/sdf/webdataset_full_v0/shards/train-000000.tar \
  --reject-keys-file Data/sdf/chemmolseq_v0_3_depiction_reject_keys.txt \
  --no-shuffle --workers 0 \
  --batch-size 4 --gradient-accumulation 1 \
  --train-samples-per-epoch 1000 --epochs 25 \
  --token-corruption-probability 0 --visual-aux-weight 1.0 \
  --resume outputs/checkpoints/chemvector-v3-finetune-1k/last.pt \
  --resume-learning-rate 1e-5 \
  --output-dir outputs/checkpoints/chemvector-v3-final-1k
```

## 决策

v0.3 验证了短序列和空间排序的价值，但直接 graph 序列仍让模型同时承担视觉
解析与化学序列生成。后续主线转向 ChemSVG v0.4。
