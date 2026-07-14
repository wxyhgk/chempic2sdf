# ChemPic2SDF

ChemPic2SDF 是面向化学结构图片识别的视觉到结构项目。系统并行检测原子顶点、
文字和可见键线，重建 ChemSVG，再通过 RDKit 校验并导出 SDF。

当前稳定版本：`v0.7.1`。开发分支包含尚未发布的因子化 Edge/Bond head 实验。

## 当前基线

- 完整模型约 5,149 万参数；无自回归 decoder 的部署路径约 1,518 万参数。
- GD906 可恢复为 `C80H44BFN2`，InChIKey 为
  `HJVIWKWTJWDQQZ-UHFFFAOYSA-N`。
- 500 张 scaffold 独立 CPU 盲测：完整拓扑 20.6%，化学完全一致 13.2%。
- 当前主要瓶颈是全图价态、芳香环一致性和键级联合选择，不是顶点定位。

详细训练过程和实验结论见 [ChemVector 中文文档](docs/model/CHEMVECTOR.md)。

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

完整文档入口见 [docs/README.md](docs/README.md)。
