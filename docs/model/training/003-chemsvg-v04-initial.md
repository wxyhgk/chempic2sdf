# 003：ChemSVG v0.4 初始训练

> 状态：当前协议基础  
> 协议文档：[`Data/sdf/CHEMSVG_V0_4.md`](../../../Data/sdf/CHEMSVG_V0_4.md)

## 目标

不再输出 `<b atom_id atom_id>`。每条 stroke 直接携带图片中的起点和终点坐标，
SDF 连接关系由端点确定性编译：

```text
<v 315 228 N 0 0 0 0 0 0 CHI_UNSPECIFIED>
<v 507 411 C 0 0 0 0 0 0 CHI_UNSPECIFIED>
<s 315 228 507 411 SINGLE SINGLE NONE STEREONONE>
```

这种表示把任务约束为点、线和文字图元，避免通过隐藏 atom ID 或距离启发式恢复
连接关系。完全重合的视觉顶点无法从图片唯一恢复，加载器直接跳过，而不是隐藏
歧义。

## 数据

现有 WebDataset 已包含 Render JSON，不需要重新生成图片。训练时继续排除已知
二维 E/Z 描绘冲突样本。

## 训练命令

```bash
.venv-train/bin/python -m chemvector.train \
  --sequence-version 0.4 \
  --reject-keys-file Data/sdf/chemmolseq_v0_3_depiction_reject_keys.txt \
  --train-shards Data/sdf/webdataset_full_v0/shards/train-000000.tar \
  --validation-shards Data/sdf/webdataset_full_v0/shards/validation-000000.tar \
  --batch-size 4 --gradient-accumulation 1 \
  --workers 0 --epochs 1 \
  --output-dir outputs/checkpoints/chemvector-v4
```

## 决策

ChemSVG v0.4 成为后续几何、OCR、relation 和并行推理实验的统一中间表示。
