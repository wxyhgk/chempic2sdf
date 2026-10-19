# SDF 原始数据

当前保留 ChemMolSeq v0.2、空间顺序 ChemMolSeq v0.3、坐标图元 ChemSVG v0.4、
对象场景协议 ChemScene v0.1/v0.2、[ChemTraverse v0.1](CHEMTRAVERSE_V0_1.md)，
以及 v0.9 使用的可见物理图元协议 Visual SVG v0.1/v0.2。
训练加载器可直接从现有 JSON 动态生成旧序列标签；Visual SVG 容量集使用独立的
确定性构建器生成，不需要重建 75GB WebDataset。

## ChemTraverse v0.1

ChemTraverse 将原子与键交替输出，树边使用 `down/up`，闭环使用对已出现原子的 `link`。
双键和三键各自始终是一条 bond 事件。遍历由图片坐标唯一确定，四面体手性使用相对于
邻居坐标排序的 `TCW/TCCW`，不会因原子遍历重排而翻转。协议解析不要求 RDKit sanitize，
因此无效模型预测也可以完整保存和复现。

## ChemVector Safe Suppression v0.1

[Safe Suppression v0.1](chemvector_safe_suppression_v0_1/README.md) 为 `v0.10.13`
提供 12,500 个全新 scaffold 分子和 50,000 张未见图片。它将 8,000 张 route-select 与
42,000 张 `000` route-confirm 严格隔离，并用方向性删除动作替代存在标签偏差的
same-GT identity 目标。旧 sealed test 在该版本中继续保持未读。

## Visual SVG v0.1

[Visual SVG v0.1](visual_svg_v0_1/README.md) 分成两层：`visual-primitives-0.1`
保留完整图元、文字框、owner 和 bond group 审计信息；模型目标 `visual-svg-0.1`
只生成 q2047 坐标下的 `LINE`、`TRIANGLE`、`QUAD` 和 `TEXT_BOX` 序列。

物理图元不等于化学拓扑。双键的两条线和三键的三条线仍分别只对应两个化学端点，
交叉线内部不会自动成为 Junction，多边形顶点也不是额外原子。本阶段不输出文字内容、
元素、键级、价态、化学图或 SDF；这些由后续 OCR、primitive grouping 和 ChemScene
编译阶段处理。

`visual_svg_v0_1/capacity_100` 已通过数据验收：100 张图对应 100 个唯一分子，
`original/rdkit/coordgen/stress` 各 25 张，平均/最长目标序列为 375.22/1,191 token，
目标 bond mask IoU mean/min 为 95.292%/93.799%。MR-TADF、USPTO 和公开 holdout
均未进入该容量集。

v0.9.0 先验证旧 cross-attention，1,000 step 后错配损失比仍为 0.999985，判定
视觉未接地。v0.9.1 的 Visual Prefix 使用 raw-ink content-only value 支路后获得稳定
视觉梯度；最终 clean checkpoint 在完整 100 图上得到：

| 指标 | 结果 |
|---|---:|
| teacher token / sequence exact | 99.976% / 99% |
| mismatch loss ratio | 834.72 |
| free completed / grammar valid | 100% / 100% |
| primitive exact | 99% |
| LINE / 结构图元 / TRIANGLE / QUAD 数量 exact | 均为 100% |
| TEXT_BOX 数量 exact | 100% |
| line / text-box / polygon F1 | 99.867% / 100% / 100% |
| bond IoU mean / min | 94.825% / 49.138% |

训练决策见 [v0.9.0](../../docs/model/training/20260721-v0.9.0-raster-to-visual-primitives-capacity.md)
和 [v0.9.1](../../docs/model/training/20260721-v0.9.1-visual-prefix-grounding-and-capacity.md)。
这些数字只说明固定 100 图的视觉接地与模型容量，不是未见分子泛化结果，也不替换
当前稳定生产推理 `v0.7.10`。

## Visual SVG v0.2 Bond Group

[Visual SVG v0.2](visual_svg_v0_2/README.md) 在 v0.1 精确 physical primitive 之上保留
真实 group partition，把单线、双平行线、三平行线、虚楔键和实心楔键作为一等视觉
对象。每个 group 内嵌完整 q2047 members，因此可以无损展开回 v0.1 模型目标；target
不包含 bond ID、atom ID、键级或其他化学语义。

capacity100 已完成 100/100 结构往返：6,320 个 physical primitives 聚合为 3,493 个
bond groups 和 976 个 text boxes，单图最大 group-set 对象数为 132。v0.2 当前只用于
group set predictor 容量实验，尚未生成 pilot 全量 tar，也没有读取 `internal_test`。

## ChemScene v0.1

ChemScene 把图片表示为 `Junction`、`TextObject` 和 `BondObject`。双键和三键各自
仍是一个 BondObject，内部平行线端点不会成为额外顶点。对象端点直接引用坐标，
不使用最近邻、键长阈值或隐藏 atom ID。

```bash
.venv-train/bin/python -m Data.sdf.validate_chemscene_v0_1 \
  Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-000000.tar \
  --output Data/sdf/chemscene_v0_1_validation.json
```

## ChemScene v0.2

[ChemScene v0.2](CHEMSCENE_V0_2.md) 删除独立 Junction 与末端最近邻 OCR 附着，改为
联合 `AtomObject + BondObject`。AtomObject 同时拥有 chemical anchor、文字区域和全部
incident bond ports；BondObject 的两个端口必须逐项引用对应 AtomObject。现有 WebDataset
可动态构建 v0.2，无需重渲染图片。

```bash
.venv-train/bin/python -m Data.sdf.validate_chemscene_v0_2 \
  Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-000000.tar \
  --limit 500 \
  --output Data/sdf/chemscene_v0_2_validation.json
```

## ChemSVG v0.4

v0.4 不再用原子编号表达键连接。模型生成视觉顶点坐标和 stroke 起止坐标，
确定性编译器根据端点创建 RDKit Bond。该路线没有按原子距离枚举候选边，也
没有最大键长阈值。

```bash
.venv-train/bin/python -m Data.sdf.chemmol_webdataset \
  Data/sdf/webdataset_full_v0/shards/train-000000.tar \
  --sequence-version 0.4 \
  --limit 100 \
  --validate-target-roundtrip
```

首 shard 在排除 13 张已知 E/Z 描绘冲突图片后，有 4,956 张完成无损往返，
另有 23 张因为视觉顶点完全重合而由 v0.4 加载器跳过。

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
.venv-train/bin/python -m unittest -v tests.data_sdf.test_chemmol_sequence
```

验证单个训练分片的加载和化学往返：

```bash
.venv/bin/python chemmol_webdataset.py \
  webdataset_full_v0/shards/train-000000.tar \
  --limit 10 \
  --validate-target-roundtrip
```

MR-TADF 外部 benchmark 使用独立的审计和构建入口：

```bash
.venv-train/bin/python -m Data.sdf.audit_mr_tadf
.venv-train/bin/python -m Data.sdf.build_mr_tadf_benchmark \
  --output-dir Data/benchmarks/mr_tadf_v1
```

渲染器只读取 `Data/MR-TADF_SDF/audit/approved.csv`，不会为异常或重复 SDF
生成 PNG。

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
