# ChemPic2SDF

将化学分子结构图片识别为分子图，并导出 SDF。

ChemPic2SDF 是一个研究中的光学化学结构识别（OCSR）项目，自研模型代码位于
`chemvector/`。目标是识别正确的原子、键类型与连接关系；SVG 用于数据渲染、
标注和可视化，最终输出以分子图为核心。

> 当前公开的是研究源码、配置、测试和实验记录。仓库不包含模型权重、完整训练数据
> 和评测产物；仅克隆仓库还不能直接运行预训练模型推理。

## 当前状态

项目沿自研模型路线推进，不以 MoE 作为主线。文字与结构分别提供证据，
通过 ChemScene 表示组装分子图，再由确定性编译流程导出 SDF；
不要求先把文字全部识别正确，才开始预测连接关系。

当前 Direct AtomPair 路径的主要流程为：

```text
原图 → 共享视觉特征 → 原子候选与集合筛选
                           ├→ AtomPair 连接与键类型预测
                           └→ 文字角色、文字框与 OCR
                                      ↓
                                ChemScene → SDF
```

| 部分 | 状态 |
| --- | --- |
| 默认推理配置 | `structure-v01010-direct-aromatic-ocr` |
| 冻结研究参照 | `structure-v01016-structural-twin-ocr`，保留为非默认对照 |
| 后续探索 | 课程数据、候选覆盖、键连接与键型泛化；实验结果通过检查后再考虑替换默认模型 |
| 训练策略 | 原子／基本键 → 官能团 → 小分子 → 大分子，并回放已有能力；各阶段效果仍需验证 |

推理配置见 [模型注册表](frontend/inference_profiles.json)，职责边界见
[分子图路线说明](docs/model/20260910-graph-first-contract.md)。

历史记录涉及不同数据集、真值辅助（Oracle）、固定样本容量实验和开发集，
不能合并成一个“项目准确率”。完整分子正确率、局部原子／键指标和 SDF 有效率
需分别看待；各次实验的条件、结果和采用结论见
[训练与实验档案](docs/model/training/README.md)。
部分旧文档中的“当前”描述属于当时的实验状态。

## 环境准备

主要依赖为 Python、PyTorch 和 RDKit。请先根据机器与 CUDA 环境选择合适的
[PyTorch 安装方式](https://pytorch.org/get-started/locally/)；
`requirements-train.txt` 不包含 PyTorch。

```bash
git clone https://github.com/wxyhgk/chempic2sdf.git
cd chempic2sdf

python -m venv .venv-train
source .venv-train/bin/activate

# 已有适配机器的 PyTorch 时可跳过此行
python -m pip install torch
python -m pip install -r requirements-train.txt
```

下面的命令均在项目根目录、已激活虚拟环境的情况下执行。
个别研究工具还需要额外依赖，以对应实验文档为准。

## 使用

### 查看结果与上传图片

```bash
python frontend/server.py --host 127.0.0.1 --inference-device cpu
```

浏览器打开 <http://127.0.0.1:8765>。前端支持评测结果对照、错误筛选、
图片上传和 SDF 下载。已有评测结果需放在 `outputs/evaluations/`；
上传推理需先准备所选 profile 引用的全部 checkpoint。新克隆的仓库没有历史结果，
也不会自动下载权重。

已配置 CUDA 的机器可以使用 `--inference-device cuda`。
更多说明见 [前端文档](frontend/README.md)；实际可选模型以注册表为准。

### 检查模型基础组件

```bash
python -m pip install pytest
python -m pytest -q \
  tests/chemvector/test_direct_atom_pair_bond.py \
  tests/chemvector/test_direct_atom_role.py
```

这是两组组件测试，不需要本地训练权重。完整测试入口为
`python -m pytest tests/`，部分集成检查依赖本地数据或额外环境。

### 数据构建与训练

数据协议和构建工具见 [数据文档](Data/sdf/README.md)。
训练入口见 [课程训练模块](chemvector/training/curriculum/README.md) 及
[运行合同](chemvector/training/curriculum/RUNNING.md)。

训练前需固定数据清单、划分与检查结果，并设置步数、时间和开发集验收标准。
训练数据与评测数据按分子身份分组隔离；外部测试不用于训练或阈值拟合。
短试训、局部指标改善及生成 checkpoint 均不代表默认模型已经更新。

## 目录导航

| 目录 | 内容 |
| --- | --- |
| [`chemvector/`](chemvector/README.md) | 模型组件、训练、推理、评测与工作流 |
| [`Data/sdf/`](Data/sdf/README.md) | 数据协议、渲染与数据构建工具 |
| [`docs/`](docs/README.md) | 架构说明、实验记录与文献调研 |
| [`frontend/`](frontend/README.md) | 评测对照与图片上传界面 |
| [`tests/`](tests/) | 组件与数据协议测试 |
| [`tools/`](tools/) | 数据检查、实验诊断与维护脚本 |
| [`experiments/`](experiments/README.md) | 自动实验配置与门控流程 |
| [`formal/`](formal/chemgraph/README.md) | 几何歧义与可识别性的形式化研究 |
| [`outputs/`](outputs/README.md) | 本地生成产物的目录约定，产物不随仓库发布 |
| [`third_party/`](third_party/README.md) | 上游参考代码说明 |

`chemvector/` 按职责分层，但保留 `python -m chemvector.xxx` 等历史模块入口；
代码放置规则见 [源码维护手册](chemvector/README.md)。

## 进一步阅读

- [完整文档索引](docs/README.md)
- [模型架构与历史结果](docs/model/CHEMVECTOR.md)
- [训练与实验档案](docs/model/training/README.md)
- [OCR 路线与记录](docs/model/OCR_TEACHER.md)
- [自动实验与门控协议](docs/model/EXPERIMENT_LOOP.md)
- [文献调研](docs/research/README.md)
