# MolGrapher 固定 100 图过拟合集

本目录只用于判断模型和损失函数是否具备表达能力，不是训练集扩充，也不是公开
benchmark。任何在本目录上得到的准确率都不能作为泛化指标。

## 内容

- `train-000000.tar`：100 个固定样本，共 600 个 WebDataset 成员；
- `train-000000.json`：来源 shard、选择规则和固定 key；
- 每个样本保留原始 PNG、ChemMol JSON、Render JSON、ChemSVG 文本、SDF 和 meta。

样本来自 MolGrapher pilot train 前 500 张审计候选，按原子数、是否含可见文字和
是否含多键分桶轮询。自动筛查出的原子序号疑似污染样本未进入本目录。

## 相关审计

```text
outputs/audits/molgrapher-v0712-train-500/
outputs/audits/molgrapher-v0712-overfit-100/
```

审计规则用于构建诊断子集，不得直接据此删除全量 MolGrapher 数据。自动原子序号
检测存在保守误报，正式清洗前仍需人工复核 overlay。

## 重建命令

```bash
.venv-train/bin/python -m chemvector.audit_molgrapher \
  --shards 'Data/sdf/molgrapher_synthetic_v0_1/pilot_10k/shards/train-*.tar' \
  --output-dir outputs/audits/molgrapher-v0712-train-500 \
  --samples 500 \
  --subset-output Data/sdf/molgrapher_synthetic_v0_1/overfit_100/train-000000.tar \
  --subset-size 100
```

固定随机性来自确定性排序和分桶轮询，不依赖运行时随机抽样。
