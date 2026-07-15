# MolGrapher Synthetic v0.1

本目录用于把公开的 MolGrapher-Synthetic-300K 接入 ChemVector。原始数据只作为
外部训练来源，不改变 USPTO-30K 和 MR-TADF benchmark。

## 固定来源

- 数据集：`docling-project/MolGrapher-Synthetic-300K`；
- revision：`8a13a6a5d29ae40f98c6b92a5ca8b139d0e0a945`；
- 许可：Community Data License Agreement - Permissive 1.0；
- 原始结构来自 PubChem，图片由 MolGrapher/MolDepictor 使用 RDKit 生成；
- 官方字段：`id`、`image`、`mol`、`smiles`、`keypoints`。

完整来源、文件 SHA256 和许可要求见 `sources/source.json` 与
`sources/CDLA-Permissive-1.0.txt`。

## pilot_10k 协议

第一版只验证骨架迁移，不训练 OCR：

- 9,000 张来自跨越官方 train 排序轴的四个分片，不能只取相邻分片；
- 1,000 张来自官方 validation split；
- 输入模型的图片固定为 768x768，但原始 1024 图片仍保存在 Parquet 中；
- keypoints 与图片同步等比例缩放；
- 与 USPTO clean 1k、MR-TADF v1 按完整 InChIKey 去重；
- validation 与 pilot train 保持 InChIKey 和 Murcko scaffold 零重叠；
- query atom、superatom、坐标数量不符、异常价态和不能无损往返的记录全部拒绝；
- 不重新绘图，也不对异常样本自动修复。

每个 WebDataset 样本包含：

```text
sample.png
sample.chemmol.json
sample.render.json
sample.chemmolseq.txt
sample.sdf
sample.meta.json
```

构建命令：

```bash
.venv-train/bin/python -m Data.sdf.build_molgrapher_synthetic \
  --root Data/sdf/molgrapher_synthetic_v0_1 \
  --output Data/sdf/molgrapher_synthetic_v0_1/pilot_10k
```

交付必须满足 `pilot_10k/validation.json` 中 `status=passed`。模型训练只能读取
`pilot_10k/shards/`，不得直接遍历 raw Parquet。

## 许可说明

训练得到的模型属于计算结果。若发布原始数据或转换后的 WebDataset，必须保留数据
提供者署名、CDLA-Permissive-1.0 文本，并显著说明文件已经转换。这里只记录工程
合规要求，不构成法律意见。
