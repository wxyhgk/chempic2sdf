# 项目文档索引

## 模型

- [ChemVector 训练、推理与实验记录](model/CHEMVECTOR.md)
- [OCR 教师标注流程](model/OCR_TEACHER.md)

## 数据协议

- [数据构建说明](../Data/sdf/README.md)
- [ChemMolSeq v0.2](../Data/sdf/CHEMMOLSEQ_V0_2.md)
- [ChemMolSeq v0.3](../Data/sdf/CHEMMOLSEQ_V0_3.md)
- [ChemSVG v0.4](../Data/sdf/CHEMSVG_V0_4.md)
- [MR-TADF 外部测试集](../Data/MR-TADF_SDF/README.md)

## 参考代码

- [StarVector 上游版本说明](../third_party/README.md)

## 生成产物

- [outputs 目录规则](../outputs/README.md)
- `outputs/MANIFEST.csv`：历史输出目录的旧路径到新路径映射。

源码 API 文档暂时与模块实现放在一起。新增长文档统一放入 `docs/`，不要继续堆在
项目根目录或 `chemvector/` 包内。
