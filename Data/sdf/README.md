# SDF 原始数据

当前同时保留 ChemMolSeq v0.2 与空间顺序 ChemMolSeq v0.3。v0.3 协议见
[`CHEMMOLSEQ_V0_3.md`](CHEMMOLSEQ_V0_3.md)，训练加载器可直接从现有 JSON 动态
生成，不需要重建 75GB WebDataset。

请将原始分子结构文件放入 `raw/` 目录。

支持以下源文件格式：

- `.sdf`：可以包含一个或多个分子
- `.mol`：通常包含一个分子

`raw/` 中的原始文件应保持不变。后续的数据预处理程序会将规范化结构、
渲染图片、训练标注和数据集划分结果写入其他目录，避免污染原始数据。

请尽量使用保留二维原子坐标的 SDF 或 MOL 文件。这些坐标可以用于生成与
分子图片精确对齐的原子位置、化学键端点和线条几何标注。

## 第一版采样范围

第一版训练数据只保留单一连通分子，允许元素为
`B C N O F Si P S Cl Br I Se`，重原子数限制为 10 到 100，并要求结构能
通过 RDKit sanitize。暂时排除盐、多组分、聚合物、查询结构、R-group、
金属配位结构、同位素和显式氢原子。

默认采样规模：

- ChEMBL：150,000 个，按稀有元素类别分层采样
- COCONUT：30,000 个，侧重多环、手性、桥环、螺环和大环等复杂结构
- ChEBI：5,000 个独立专项测试结构，不进入主训练集

运行：

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python prepare_samples.py
```

结果写入 `prepared/`。必须先完成结构去重和数据集划分，再为每个分子生成
3 到 6 种图片，避免同一结构的不同绘图泄漏到训练集和测试集。

## ChemMolSeq v0.2

模型训练不再直接使用 `webdataset_full_v0` 分片中保存的旧版
`chemmolseq.txt`。`ChemMolWebDataset` 会读取 ChemMol JSON 和 Render JSON，
动态生成 ChemMolSeq v0.2 训练目标。

v0.2 协议和字段说明见 `CHEMMOLSEQ_V0_2.md`。它包含同位素、显式氢、
`no_implicit`、自由基电子、完整手性枚举、双键立体参考原子、显示键型和
量化二维坐标，并支持从模型输出重建 RDKit Mol 与 SDF。

运行单元测试：

```bash
.venv/bin/python -m unittest -v test_chemmol_sequence.py
```

验证单个训练分片的加载和化学往返：

```bash
.venv/bin/python chemmol_webdataset.py \
  webdataset_full_v0/shards/train-000000.tar \
  --limit 10 \
  --validate-target-roundtrip
```

全量 ChemMolSeq v0.2 验收：

```bash
.venv/bin/python validate_chemmolseq_v2.py \
  webdataset_full_v0 \
  --workers 8 \
  --output webdataset_full_v0/chemmolseq_v0_2_validation.json
```

## WebDataset 样例

`build_webdataset_sample.py` 从候选池构建 1000 个分子的 v0 样例：830 个
ChEMBL 和 170 个 COCONUT。分子按 Murcko scaffold 分组切分，每个分子生成
`original`、`rdkit`、`coordgen` 和 `stress` 四张 768 x 768 RGB PNG。

```bash
.venv/bin/python build_webdataset_sample.py
.venv/bin/python chemmol_webdataset.py \
  webdataset_sample_1k/shards/train-000000.tar --limit 100
```

样例输出位于 `webdataset_sample_1k/`：

- `shards/`：PyTorch WebDataset tar 分片
- `structures/`：每个分子一份规范化 SDF
- `schemas/`：ChemMol 和 Render JSON Schema
- `molecules.csv`：分子、来源、scaffold、split 和结构引用
- `samples.csv`：图片样本、布局、颜色和分片位置
- `rejected.csv`：被严格往返验证隔离的候选

每个图片样本包含 `.png`、`.chemmol.json`、`.render.json`、`.meta.json` 和
`.chemmolseq.txt`。`.venv/`、`__pycache__/` 和 notebook checkpoint 不属于
数据包内容。
