# ChemPic2SDF

ChemPic2SDF 是面向化学结构图片识别的视觉到结构项目。系统并行检测原子顶点、
文字和可见键线，重建 ChemSVG，再通过 RDKit 校验并导出 SDF。

当前稳定 tag：`v0.7.1`。当前稳定开发基线为 `v0.7.10`；`v0.7.12` 正在以
shadow-mode 验证集合式顶点检测，尚未打 tag。

## 当前基线

- 完整模型约 5,149 万参数；无自回归 decoder 的部署路径约 1,518 万参数。
- GD906 可恢复为 `C80H44BFN2`，InChIKey 为
  `HJVIWKWTJWDQQZ-UHFFFAOYSA-N`。
- 500 张 scaffold 独立 CPU 盲测：完整拓扑 20.6%，化学完全一致 13.2%。
- v0.7.1 MR-TADF Oracle 表明当时第一瓶颈是 OCR：仅替换文字真值时，化学完全
  一致率由 33.82% 提升到 75.82%；顶点与文字同时使用真值时达到 94.91%。
- 完整分子字符替换 OCR 的平衡版 3000-step 实验，在未使用任何 MR-TADF 训练
  样本的前提下，将 1100 张外部测试的有效 SDF 从 71.91% 提升到 83.64%，
  化学完全一致率从 33.82% 提升到 44.82%。该 checkpoint 尚未打正式版本 tag。
- pixel OCR 直接从原图裁取字符区域，不再使用 stride-8 几何特征。候选模型在
  MR-TADF 上达到 52.55% 化学完全一致率和 75.55% 有效 SDF；原图修正版为
  49.91% / 81.36%。两者均未打正式版本 tag。
- v0.7.5 使用 `12/20/32px` 三尺度共享 Pixel OCR，并只对 B 全量、N 5% 采样
  文字 bbox 内锚点。训练数据池为 ChemVision pilot 的 9,000 张图片，最终
  checkpoint 位于 500 step、batch 4，实际约 2,000 次图片采样；在完整
  1,100 张 MR-TADF 上达到 89.73% 有效 SDF 和 67.45% 化学完全一致；正确、
  错误和无效 SDF 均已保存在 benchmark 输出中。
- v0.7.5 的 358 个错误样本中，255 个主要是顶点检测或定位问题，142 个含 OCR
  错误，122 个含多余键，其中 64 个是明确的交叉错连。下一阶段优先改进顶点
  召回和定位稳定性，OCR 作为并行分支持续维护。
- v0.7.6 使用同一 ChemVision pilot 的 9,000 张图片，只训练 92,548 个 vertex
  head 参数，其余模型权重逐位不变。MR-TADF 有效 SDF 达到 91.91%，化学完全
  一致达到 77.55%，三个键长难度桶全部提升；当前剩余 191 个顶点主错误。
- v0.7.9 使用 9,000 张图片训练全部原子点对的 relation/pair 参数；v0.7.10 沿用
  同一 checkpoint，以连续边概率恢复断图。USPTO clean 1k 有效 SDF 达到
  192/1000，MR-TADF 有效 SDF 达到 92.09%，化学完全一致保持 79.73%。
- v0.7.11 的保守结构 MoE 已复核为 NO-GO：USPTO/MR-TADF 的结果来自完全回退
  v0.7.10，MolGrapher routed detected skeleton exact 仅 1.3%。代码保留为实验
  基础设施，不作为模型能力提升。
- v0.7.12 在固定 100 张干净 MolGrapher 图片上验证集合式顶点：head-only 和解冻
  stride-4 block 的 vertex exact 分别为 87% 和 90%；Atom Query 负责计数、dense
  peak 负责一对一坐标吸附后，Count exact 达到 100%，4px vertex exact 达到 95%。
  接入现有 relation 后 skeleton exact 最高 23%，下一阶段转向检测后顶点上的关系
  hard-negative 训练。

当前架构与结果见 [ChemVector 中文主文档](docs/model/CHEMVECTOR.md)，每次训练的
完整命令、指标和决策见 [独立训练档案](docs/model/training/README.md)。

## 目录

```text
chemvector/       模型、训练、推理和评测 Python 包
Data/sdf/         ChemMol/ChemSVG 协议与数据构建工具
Data/images/      手工外部测试图片
Data/MR-TADF_SDF/ MR-TADF 外部测试集及渲染白名单
docs/             项目文档索引与模型说明
tests/            ChemVector 与数据协议测试
tools/            项目维护和目录整理工具
outputs/          checkpoint、评测和单图预测产物
paper/            本地论文资料，不进入 Git
third_party/      上游参考代码与版本说明
```

`outputs/` 已按用途分为：

- `outputs/checkpoints/`
- `outputs/evaluations/`
- `outputs/predictions/`
- `outputs/diagnostics/`

历史路径映射见 `outputs/MANIFEST.csv`。新实验不得再直接写入 `outputs/` 根目录。

## 常用命令

运行测试：

```bash
.venv-train/bin/python -m unittest discover -v
```

CPU 推理：

```bash
.venv-train/bin/python -m chemvector.infer_parallel_v6 \
  --checkpoint outputs/checkpoints/chemvector-v5-vertex-10k/last.pt \
  --image Data/images/GD906.png \
  --output-dir outputs/predictions/chemvector-v6-parallel-cpu-gd906 \
  --device cpu --cpu-threads 16
```

MR-TADF 渲染前审计：

```bash
.venv-train/bin/python -m Data.sdf.audit_mr_tadf
```

构建并评测 MR-TADF v1 benchmark：

```bash
.venv-train/bin/python -m Data.sdf.build_mr_tadf_benchmark \
  --output-dir Data/benchmarks/mr_tadf_v1

.venv-train/bin/python -m chemvector.benchmark_mr_tadf \
  --checkpoint outputs/checkpoints/chemvector-v5-vertex-10k/last.pt \
  --benchmark-manifest Data/benchmarks/mr_tadf_v1/manifest.csv \
  --output-dir outputs/evaluations/mr-tadf-v1-v071 \
  --device cuda
```

查看所有 benchmark 的真值/预测图片和 SDF：

```bash
.venv-train/bin/python frontend/server.py
```

浏览器打开 `http://127.0.0.1:8765`。页面会自动扫描普通评测和 Oracle 评测结果。
v0.7.6 结果已支持按 OCR、漏/多原子、漏/多键和交叉错连筛选。

v0.7.11 实验性 runtime 由 base checkpoint 和 packed bundle 组成：

```text
base:   outputs/checkpoints/chemvector-v079-relation-all-pairs-9k/last.pt
packed: outputs/checkpoints/chemvector-v0711-conservative-moe-packed/last.pt
```

packed bundle 为 983 KB，只保存 28 个变化的结构头 tensor 和 gate。该组合已判定
NO-GO，仅保留用于复现实验，不是当前正式推理候选。

生成完整分子 OCR 字符替换预览：

```bash
.venv-train/bin/python -m chemvector.replacement_ocr \
  --samples 16 \
  --output-dir outputs/diagnostics/replacement-ocr-preview
```

替换标签全部按普通字符串训练，覆盖 118 个元素符号、带氢/电荷/同位素写法、
`A/X/R1` 等占位符和 `Ph/Me/tBu/Boc` 等化学简写。该阶段不做化学理解。

完整文档入口见 [docs/README.md](docs/README.md)。
