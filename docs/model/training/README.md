# ChemVector 训练档案

本目录保存每一次已执行训练或候选实验的独立记录。模型主文档只维护当前架构、
当前最佳结果和本目录索引，不再承载完整训练日志。

## 文档规则

1. 每次开始新的正式训练、消融实验或候选版本时，新建一个 Markdown 文件。
2. 新文件命名为 `YYYYMMDD-vX.Y.Z-topic.md`；历史实验使用已有实验编号命名，
   不补造未知日期。
3. 已结束实验的命令、指标和结论不得被后续结果覆盖。需要修正时，在原文增加
   “复核”小节，并说明修正原因。
4. 每篇必须记录数据范围、初始化 checkpoint、可训练参数、训练命令、门控指标、
   输出目录、失败方案和最终决策。
5. 新实验完成后，同时更新本页时间线和上一级 [ChemVector 主文档](../CHEMVECTOR.md)
   的一行摘要。
6. MR-TADF 只作为外部 benchmark，不得写成训练数据。

新建文档时复制 [训练记录模板](TEMPLATE.md)。

## 训练时间线

| 编号 | 阶段 | 主要结论 | 状态 |
|---|---|---|---|
| 001 | [100 样本视觉接地课程](001-grounding-100.md) | 证明图片到结构的训练和 SDF 往返链路成立 | 完成 |
| 002 | [ChemMolSeq v0.3 1k 课程](002-chemmolseq-v03-1k.md) | 缩短序列并改用图片空间原子顺序 | 历史基线 |
| 003 | [ChemSVG v0.4 初始训练](003-chemsvg-v04-initial.md) | 用视觉端点直接表达键，不依赖 atom ID | 当前协议基础 |
| 004 | [v0.5 几何与 OCR 解耦](004-v05-ocr-decoupling.md) | OCR 输出开放 UTF-8 字符串，局部预训练必须冻结视觉层 | 完成 |
| 005 | [v0.5 顶点、连线与墨线修复](005-v05-visual-relation-repair.md) | 全点对不设键长上限，repair 直接检查真实墨线 | 完成 |
| 006 | [v0.6 并行 CPU 推理](006-v06-parallel-cpu.md) | 删除部署 decoder，推理参数降至约 15.18M | 完成 |
| 007 | [v0.7.2 因子化 Edge/Bond head](007-v072-factorized-relation.md) | 局部指标提高但端到端无收益 | 否决 |
| 008 | [v0.7.2 平衡字符替换 OCR](008-v072-replacement-ocr.md) | 高比例扩展字符会破坏原分布，平衡 3k 有效 | 完成 |
| 009 | [v0.7.3 Pixel OCR](009-v073-pixel-ocr.md) | 原图 crop 解决 stride-8 字符细节损失 | 候选 |
| 010 | [v0.7.5 ChemVision 多尺度 OCR](010-v075-chemvision-multiscale-ocr.md) | 12/20/32px 融合将 MR-TADF 完全一致提高到 67.45% | 完成 |
| 011 | [v0.7.6 顶点召回与定位 pilot](20260715-v0.7.6-vertex-recall-pilot.md) | 只训练 92,548 个 vertex 参数，完全一致提高到 77.55% | 完成 |
| 012 | [v0.7.7 字体扩展 OCR pilot](20260715-v0.7.7-font-expanded-ocr.md) | 字体方向有效，但全头微调遗忘旧分布；保留数据，否决 checkpoint | NO-GO |
| 013 | [v0.7.8 OCR 残差适配器](20260715-v0.7.8-ocr-residual-adapter.md) | 冻结旧权重，只增 29,044 参数；USPTO 有效 88→167，MR 完全一致 77.55%→79.55% | 完成 |
| 014 | [v0.7.9 全点对骨架](20260715-v0.7.9-skeleton-all-pairs.md) | 全点对 relation + 概率约束 repair；USPTO 有效 167→177，MR 完全一致 79.55%→79.73% | 完成 |
| 015 | [v0.7.10 概率桥接修复](20260715-v0.7.10-probability-bridge.md) | 不训练新权重；保留全点对概率补桥，USPTO 有效 177→192，MR 完全一致不变 | 当前最佳 |
| 016 | [v0.7.11 MolGrapher 结构头插值](20260715-v0.7.11-molgrapher-interpolation.md) | 新域 Oracle 提升，但三档均未通过旧域联合门控 | NO-GO |
| 017 | [v0.7.11 结构残差适配器](20260715-v0.7.11-structure-residual-adapters.md) | Oracle 连线提升，detected 整图无收益，瓶颈仍是顶点集合 | NO-GO |
| 018 | [v0.7.11 Teacher Consistency](20260715-v0.7.11-teacher-consistency.md) | 新域局部指标恢复，但共享头仍严重遗忘未覆盖旧域 | NO-GO |
| 019 | [v0.7.11 保守结构 MoE](20260715-v0.7.11-conservative-structure-moe.md) | gate 只实现旧域回退，新域 detected exact 1.3% | NO-GO |
| 020 | [v0.7.12 集合式顶点](20260716-v0.7.12-atom-query-vertex.md) | Atom Query 计数 + dense 吸附在 100 图达到 95% vertex exact | 顶点 GO |
| H01 | [历史 PaddleOCR 教师方案](historical-paddleocr-teacher.md) | 通用 OCR 不再作为训练教师 | 停止 |

## 当前下一步

v0.7.11 MoE 已判定 NO-GO，稳定开发基线回到 v0.7.10。v0.7.12 的 shadow-mode
Atom Query 负责原子集合和数量，dense heatmap 仅提供一对一几何吸附，在固定 100 图
达到 95% vertex exact。接入现有 relation 后 skeleton exact 最高 23%，下一步固定融合
顶点训练检测后 relation hard negatives；未达到 85% 过拟合门槛前不扩大到 50k。
