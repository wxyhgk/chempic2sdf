# ChemPic2SDF

ChemPic2SDF 是面向化学结构图片识别的视觉到结构生成项目。模型直接读取分子
图片中的原子、键线和二维布局，生成 ChemMolSeq，并通过 RDKit 验证后导出 SDF。

当前版本：`v0.0.1`

## 当前状态

- 已完成 ChemMolSeq v0.2、专用 tokenizer 和严格化学往返验证。
- 已实现约 5,000 万参数的 ChemVector 单卡模型。
- 已实现二维视觉位置编码、原子/键线 patch 密集监督和三阶段训练课程。
- 已实现语法约束自回归生成、RDKit sanitize、SMILES/InChIKey 比对和 SDF 导出。
- 正在开发空间原子顺序的 ChemMolSeq v0.3，以降低不可见 canonical rank 对
  未见分子泛化的影响。
- 100 样本实验达到 100% exact sequence；前 5 个分子的 20 张四布局图片全部
  自回归生成正确并通过化学一致性验证。

训练方法、环境和实验结果见 [ChemVector 中文文档](chemvector/README.md)。

## 目录

- `chemvector/`：模型、训练器、推理与评估代码。
- `Data/sdf/`：ChemMolSeq 协议、数据构建工具、验证器和测试。
- `third_party/README.md`：StarVector 上游参考版本说明。

大型原始数据、WebDataset、虚拟环境、论文文件、训练 checkpoint 和生成输出均不
进入 Git 仓库。
