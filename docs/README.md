# 项目文档索引

## 模型

- [ChemVector 当前架构、结果与训练索引](model/CHEMVECTOR.md)
- [ChemVector 独立训练档案](model/training/README.md)
- [v0.9.13 Fresh Dev preflight 与 formal loader 门禁](model/training/20260722-v0.9.13-fresh-dev-preflight.md)
- [新训练记录模板](model/training/TEMPLATE.md)
- [OCR 当前路线与训练索引](model/OCR_TEACHER.md)
- [自动实验 Campaign、恢复与门控协议](model/EXPERIMENT_LOOP.md)

## 数据协议

- [数据构建说明](../Data/sdf/README.md)
- [ChemMolSeq v0.2](../Data/sdf/CHEMMOLSEQ_V0_2.md)
- [ChemMolSeq v0.3](../Data/sdf/CHEMMOLSEQ_V0_3.md)
- [ChemSVG v0.4](../Data/sdf/CHEMSVG_V0_4.md)
- [MR-TADF 外部测试集](../Data/MR-TADF_SDF/README.md)

## Benchmark

- [MR-TADF v1 构建、评测协议与 v0.7.1 基线](benchmarks/MR_TADF_V1.md)
- [USPTO Clean 1K 公开 Benchmark](benchmarks/USPTO_CLEAN_1K.md)
- [Public OCSR v1 数据交付规范](../Data/benchmarks/public_ocsr_v1/README.md)
- [Benchmark 真值/预测 SDF 对照前端](../frontend/README.md)

## AI 技术咨询

- [外部 AI 评审、项目决策与实验回填](ai_consultations/README.md)

## 参考代码

- [StarVector 上游版本说明](../third_party/README.md)

## 生成产物

- [outputs 目录规则](../outputs/README.md)
- `outputs/MANIFEST.csv`：历史输出目录的旧路径到新路径映射。

源码 API 文档暂时与模块实现放在一起。新增长文档统一放入 `docs/`，不要继续堆在
项目根目录或 `chemvector/` 包内。每次模型训练或消融实验必须在
`docs/model/training/` 新建独立记录，不再向模型主文档追加完整训练日志。
