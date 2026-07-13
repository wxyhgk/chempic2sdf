# ChemVector

ChemVector 是面向化学结构图片识别的轻量视觉到结构序列模型。它借鉴
StarVector 的“视觉输入生成结构化代码”范式，但使用专用 ChemMolSeq 语法、
小型视觉编码器和自回归 Transformer decoder，不依赖通用 SVG 或 StarCoder。

## 当前基线

- 参数量：49,974,768
- 输入图片：768×768 RGB
- 视觉 token：24×24，共 576 个
- ChemMolSeq 词表：2,153
- decoder 上下文：3,072
- decoder：8 层、隐藏维度 512、8 个注意力头

全量数据中 ChemMolSeq 平均长度为 779 token，P99 为 2,036，最大为 2,613，
因此 3,072 的上下文可以覆盖全部第一版样本。

## 环境

训练环境使用系统中已经支持 RTX 5090 的 PyTorch，并单独安装数据依赖：

```bash
python -m venv --system-site-packages .venv-train
.venv-train/bin/pip install -r requirements-train.txt
```

## 冒烟训练

```bash
.venv-train/bin/python -m chemvector.train_smoke \
  --image-size 768 \
  --batch-size 4 \
  --steps 3
```

## 单卡训练

```bash
.venv-train/bin/python -m chemvector.train \
  --train-shards 'Data/sdf/webdataset_full_v0/shards/train-*.tar' \
  --validation-shards 'Data/sdf/webdataset_full_v0/shards/validation-*.tar' \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --image-size 768 \
  --epochs 1 \
  --output-dir outputs/chemvector-v0
```

第一轮正式训练建议从 100 个固定样本过拟合开始，再扩展到 1,000、10,000
和全量训练集。

`--visual-aux-weight` 启用 24×24 patch 密集监督。16 个通道分别表示 12 种
元素的原子位置和 `SINGLE/DOUBLE/TRIPLE/AROMATIC` 四种实际绘制键线。该任务
训练视觉 encoder 保留点线信息，但不使用距离规则创建候选边。

训练器也保留图片错配对比损失参数用于后续实验，但高权重 hinge 对比损失在
100 样本实验中不稳定，不属于下面已经验证的训练课程。

## 已验证的 100 样本课程

单纯 teacher-forcing 会忽略图片。实测正确图片和打乱图片的 loss 几乎完全
相同，模型自回归时会固定生成训练集中的另一个分子。当前通过以下三阶段解决：

1. 视觉接地：50% 前缀遮蔽，`visual_aux_weight=5.0`，学习率 `3e-4`。
2. 前缀恢复：10% 前缀遮蔽，`visual_aux_weight=1.0`，学习率 `3e-5`。
3. 自回归精调：不遮蔽前缀，`visual_aux_weight=1.0`，学习率 `1e-5`。

100 样本最终 checkpoint：

```text
outputs/chemvector-grounded-final-100/last.pt
```

最终实测结果：

- 100 张图 teacher-forcing token accuracy：100%。
- 100 张图 exact sequence accuracy：100%。
- 前 5 个分子共 20 张四布局图片：20/20 自回归完成。
- 20/20 通过 RDKit sanitize，20/20 SMILES/InChIKey 与真值一致。
- 错配其他分子图片后，序列 loss 增加到正确图片的 11.34 倍。
- 错配图片后，视觉 patch loss 增加到正确图片的 91.03 倍。

详细评估结果位于：

```text
outputs/chemvector-grounded-final-100-eval/teacher_forcing_metrics.json
outputs/chemvector-grounded-final-100-eval-20/generated/generation_metrics.json
```

这组结果只证明训练、视觉接地、自回归和 SDF 往返链路正确，不代表未见分子
的泛化准确率。下一步应将相同课程扩大到固定 1,000 样本，再使用独立 scaffold
validation 检查泛化。

## ChemMolSeq v0.3

开发中的 v0.3 使用图片空间原子顺序，去除显式 record ID 和渲染专用键字段，
并把坐标移动到化学图之后。现有数据无需重建：

```bash
.venv-train/bin/python -m chemvector.train \
  --sequence-version 0.3 \
  --reject-keys-file Data/sdf/chemmolseq_v0_3_depiction_reject_keys.txt \
  --train-shards Data/sdf/webdataset_full_v0/shards/train-000000.tar \
  --no-shuffle --workers 0 \
  --batch-size 4 --gradient-accumulation 1 \
  --train-samples-per-epoch 1000 \
  --epochs 10 \
  --token-corruption-probability 0.5 \
  --visual-aux-weight 5.0 \
  --output-dir outputs/chemvector-v3-grounded-1k
```

第二阶段从第 10 个 epoch 继续到第 20 个 epoch：

```bash
.venv-train/bin/python -m chemvector.train \
  --sequence-version 0.3 \
  --train-shards Data/sdf/webdataset_full_v0/shards/train-000000.tar \
  --reject-keys-file Data/sdf/chemmolseq_v0_3_depiction_reject_keys.txt \
  --no-shuffle --workers 0 \
  --batch-size 4 --gradient-accumulation 1 \
  --train-samples-per-epoch 1000 --epochs 20 \
  --token-corruption-probability 0.1 --visual-aux-weight 1.0 \
  --resume outputs/chemvector-v3-grounded-1k/last.pt \
  --resume-learning-rate 3e-5 \
  --output-dir outputs/chemvector-v3-finetune-1k
```

第三阶段从第 20 个 epoch 继续到第 25 个 epoch：

```bash
.venv-train/bin/python -m chemvector.train \
  --sequence-version 0.3 \
  --train-shards Data/sdf/webdataset_full_v0/shards/train-000000.tar \
  --reject-keys-file Data/sdf/chemmolseq_v0_3_depiction_reject_keys.txt \
  --no-shuffle --workers 0 \
  --batch-size 4 --gradient-accumulation 1 \
  --train-samples-per-epoch 1000 --epochs 25 \
  --token-corruption-probability 0 --visual-aux-weight 1.0 \
  --resume outputs/chemvector-v3-finetune-1k/last.pt \
  --resume-learning-rate 1e-5 \
  --output-dir outputs/chemvector-v3-final-1k
```

首 shard 的 4,992 张图片已全部通过 v0.3 化学往返，平均序列缩短 16.71%。
同时检测到 13 张增强图片的二维 E/Z 几何与化学标签冲突；正式全量训练前需要
扫描并替换这类样本。

## checkpoint 评估与 SDF 导出

评估命令先在指定样本上计算 teacher-forcing 指标，再对少量样本执行真正的
语法约束自回归生成。生成结果必须通过 ChemMolSeq 解析和 RDKit sanitize，
成功结果会写成 SDF：

```bash
.venv-train/bin/python -m chemvector.evaluate \
  --checkpoint outputs/chemvector-grounded-final-100/last.pt \
  --shards Data/sdf/webdataset_full_v0/shards/train-000000.tar \
  --samples 100 \
  --generate-samples 20 \
  --output-dir outputs/chemvector-grounded-final-100-eval
```

语法约束只负责保证记录顺序、字段类型、原子/键编号和端点引用合法，不会根据
距离补边，也不会替代模型判断图片中是否存在化学键。

从 checkpoint 继续训练时默认保留其中的 optimizer 学习率。如需进入低学习率
精调阶段，应显式传入例如 `--resume-learning-rate 3e-5`。
