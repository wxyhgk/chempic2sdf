# 001：100 样本视觉接地课程

> 状态：完成  
> 最终 checkpoint：`outputs/checkpoints/chemvector-grounded-final-100/last.pt`

## 目标

验证模型是否真正读取图片，而不是只在 teacher-forcing 下记忆结构序列。同时
打通训练、语法约束生成、RDKit sanitize 和 SDF 导出的完整链路。

## 环境与基础命令

训练环境使用支持 RTX 5090 的系统 PyTorch：

```bash
python -m venv --system-site-packages .venv-train
.venv-train/bin/pip install -r requirements-train.txt
```

冒烟训练：

```bash
.venv-train/bin/python -m chemvector.train_smoke \
  --image-size 768 --batch-size 4 --steps 3
```

正式训练入口：

```bash
.venv-train/bin/python -m chemvector.train \
  --train-shards 'Data/sdf/webdataset_full_v0/shards/train-*.tar' \
  --validation-shards 'Data/sdf/webdataset_full_v0/shards/validation-*.tar' \
  --batch-size 4 --gradient-accumulation 4 \
  --image-size 768 --epochs 1 \
  --output-dir outputs/checkpoints/chemvector-v0
```

## 问题定位

单纯 teacher-forcing 时，正确图片和打乱图片的 loss 几乎相同，自回归生成会固定
输出训练集中的另一个分子。这说明 decoder 可以依靠历史真值 token 完成训练，
但没有建立图片和结构之间的依赖。

`--visual-aux-weight` 在 24×24 patch 上增加 16 通道密集监督：12 种常见元素
位置和 `SINGLE/DOUBLE/TRIPLE/AROMATIC` 四种绘制键线。它只要求视觉 encoder
保留点线信息，不使用距离规则创建候选边。

## 三阶段课程

1. 视觉接地：50% 前缀遮蔽，`visual_aux_weight=5.0`，学习率 `3e-4`。
2. 前缀恢复：10% 前缀遮蔽，`visual_aux_weight=1.0`，学习率 `3e-5`。
3. 自回归精调：不遮蔽前缀，`visual_aux_weight=1.0`，学习率 `1e-5`。

高权重图片错配 hinge 损失在 100 样本实验中不稳定，未进入最终课程。

## 评估与 SDF 导出

```bash
.venv-train/bin/python -m chemvector.evaluate \
  --checkpoint outputs/checkpoints/chemvector-grounded-final-100/last.pt \
  --shards Data/sdf/webdataset_full_v0/shards/train-000000.tar \
  --samples 100 --generate-samples 20 \
  --output-dir outputs/evaluations/chemvector-grounded-final-100-eval
```

语法约束只保证记录格式、字段类型和端点引用合法，不会根据距离补边，也不会
替代模型判断图片中是否存在化学键。

## 结果

- 100 张图 teacher-forcing token accuracy：100%。
- 100 张图 exact sequence accuracy：100%。
- 前 5 个分子的 20 张四布局图片：20/20 完成自回归生成。
- 20/20 通过 RDKit sanitize，20/20 SDF 化学结构与真值一致。
- 错配其他分子图片后，序列 loss 增加到正确图片的 11.34 倍。
- 错配图片后，视觉 patch loss 增加到正确图片的 91.03 倍。

详细产物：

```text
outputs/evaluations/chemvector-grounded-final-100-eval/teacher_forcing_metrics.json
outputs/evaluations/chemvector-grounded-final-100-eval-20/generated/generation_metrics.json
```

## 决策

该实验只证明链路正确和训练集过拟合成功，不证明未见分子的泛化。后续课程扩展
到 1,000、10,000 和 scaffold 独立验证集。
