# ChemVector 源码维护手册

`chemvector/` 保存图片到化学结构模型的 Python 源码。这里不保存训练数据、checkpoint、
benchmark 结果、生成图片或临时缓存。

本目录采用两套彼此独立的组织方式：

- **物理目录**表达代码归属，方便查找和维护。
- **逻辑模块名**保持历史兼容，公开导入仍使用扁平的 `chemvector.xxx`。

顶层只允许保留 `README.md`、`__init__.py` 和职责目录。新增 `.py` 文件不得直接放到顶层。

## 分层原则

```text
workflows
   |
   +--> training / evaluation / inference / tools
                          |
                          +--> components
                                  |
                                  +--> core

configs 由上述入口按需读取
Data/ 和 outputs/ 位于包外
```

各层含义：

1. `core`：跨路线共享、变化较慢的基础设施。
2. `components`：可复用的模型头、数据结构、算法和确定性编译组件。
3. `training/evaluation/inference/tools`：带明确副作用或执行目的的命令入口。
4. `workflows`：编排多个入口的长流程，不承载新的模型算法。
5. `configs`：与代码版本绑定的小型静态配置。

新代码应尽量遵守“执行入口依赖组件，组件依赖核心”的方向。历史模块存在少量交叉依赖，
可以在实际修改时逐步收敛，但不得再让 `core` 依赖某个具体训练实验。

## 完整目录

```text
chemvector/
├── README.md
├── __init__.py
├── core/
├── components/
│   ├── atoms/
│   │   ├── chemistry/
│   │   ├── objects/
│   │   └── selection/
│   ├── bonds/
│   │   ├── groups/
│   │   ├── objects/
│   │   └── selection/
│   ├── graph/
│   │   ├── joint/
│   │   ├── proposals/
│   │   └── correction/
│   ├── evidence/
│   │   ├── data/
│   │   ├── models/
│   │   ├── pairing/
│   │   └── refiners/
│   ├── ocr/
│   └── vision/
├── training/
│   ├── general/
│   ├── atoms/
│   ├── bonds/
│   │   ├── groups/
│   │   ├── objects/
│   │   ├── relations/
│   │   └── routing/
│   ├── graph/
│   │   ├── proposals/
│   │   ├── relations/
│   │   └── scenes/
│   ├── evidence/
│   ├── ocr/
│   └── vision/
├── evaluation/
│   ├── benchmarks/
│   ├── audits/{atoms,bonds,evidence,general,graph,ocr,vision}/
│   ├── evaluators/{atoms,bonds,evidence,general,graph,ocr,vision}/
│   ├── oracles/
│   └── shadows/
├── inference/
├── tools/
│   ├── builders/
│   ├── calibration/
│   ├── diagnostics/
│   └── packaging/
├── workflows/experiment_loop/
└── configs/
```

## 各目录职责

| 目录 | 应放内容 | 不应放内容 |
|---|---|---|
| `core/` | 主模型、数据加载、图像/SDF I/O、几何、公共指标、稳定路径 | 某个版本的临时实验逻辑 |
| `components/atoms/chemistry/` | 元素、价态、原子化学状态及动作 | 键集合模型 |
| `components/atoms/objects/` | AtomObject、atom query、count、角色 | 训练循环 |
| `components/atoms/selection/` | 原子保留、删除、去重和安全证书 | benchmark 入口 |
| `components/bonds/groups/` | 双/三键 group、carrier、membership、packing | 单图 CLI |
| `components/bonds/objects/` | BondObject、键端点、键语义 relation | 数据下载脚本 |
| `components/bonds/selection/` | 键集合选择、路由、等价组、junction assembler | checkpoint |
| `components/graph/joint/` | Atom-Bond 联合图、incidence、graph MILP | 仅用于报表的代码 |
| `components/graph/proposals/` | proposal 生成、保留和集合适配 | OCR 字符模型 |
| `components/graph/correction/` | ChemScene 修正、结构门和证书 | 无真值 benchmark |
| `components/evidence/data/` | Evidence Carrier 数据协议和冻结输入加载 | 训练结果 |
| `components/evidence/pairing/` | pair cache、matching、scoring 和 evaluation | CLI 参数解析 |
| `components/evidence/refiners/` | coverage、veto、local residual 和 neighborhood | 原始 WebDataset |
| `components/evidence/models/` | extent、component、endpoint 等可复用模型 | 一次性转换脚本 |
| `components/ocr/` | CTC、文字对象、OCR teacher、文字路由 | 化学字符串展开规则 |
| `components/vision/` | stroke、vertex、line、primitive、视觉证据 | SDF benchmark 结果 |
| `training/` | 会优化参数并产生 checkpoint 的入口 | 可复用模型定义 |
| `evaluation/benchmarks/` | 固定测试集端到端评测 | 修改 checkpoint 的代码 |
| `evaluation/audits/` | 只读因果分析、失败分桶、可达性审计 | 训练数据回写 |
| `evaluation/evaluators/` | 某个组件或中间指标的评测 | 正式数据构建 |
| `evaluation/oracles/` | 使用真值替换局部组件的能力上界 | 生产推理 |
| `evaluation/shadows/` | 不替换默认模型的 shadow 评测 | 默认部署选择 |
| `inference/` | 单图、批量推理、生成、repair | 训练循环和真值读取 |
| `tools/builders/` | cache、manifest、评测包和教师数据构建 | 模型 forward 定义 |
| `tools/calibration/` | 阈值和路由校准 | sealed test 读取 |
| `tools/diagnostics/` | compare、confirm、mine、replay | 可复用核心算法 |
| `tools/packaging/` | merge、pack、export、freeze、interpolate | 大型权重文件本身 |
| `workflows/` | 多阶段 campaign、恢复、门控和报告编排 | 新模型层 |
| `configs/` | 小型、静态、可审计配置 | checkpoint、图片、tar、运行时快照 |

## 文件放置判断

增加文件前按下面顺序判断：

1. 它是否定义一个可被多个入口复用的模型、数据结构或算法？放 `core/` 或 `components/`。
2. 它是否执行反向传播并保存权重？放 `training/`。
3. 它是否只读 checkpoint 和真值并报告指标？放 `evaluation/`。
4. 它是否只读取图片并生成预测？放 `inference/`。
5. 它是否构建/冻结/合并数据或校准阈值？放 `tools/`。
6. 它是否串联多个上述命令？放 `workflows/`。

如果仍然无法判断，不要新建 `misc/`、`common/`、`utils2.py` 或 `temp.py`。先明确调用者和副作用，
再选择归属；无法明确通常说明文件承担了过多职责，应先拆分。

## 导入协议

### 公开逻辑路径

虽然文件已经物理分层，公开 Python 模块名仍保持扁平：

```python
from chemvector.model import ChemVectorModel
from chemvector.bond_object import BondObjectDetector
from chemvector.proposal_graph import ProposalGraphModel
```

命令也保持旧入口：

```bash
python -m chemvector.train --help
python -m chemvector.evaluate --help
python -m chemvector.build_mr_tadf_causal_audit --help
```

`chemvector/__init__.py` 将受控源码目录加入包搜索路径，所以不需要在顶层保留几百个转发文件。

### 禁止物理路径导入

不要这样写：

```python
from chemvector.core.model import ChemVectorModel
from chemvector.components.bonds.objects.bond_object import BondObjectDetector
```

同一个文件若同时以 `chemvector.model` 和 `chemvector.core.model` 加载，Python 会创建两个模块
对象，进而破坏 dataclass 相等、异常类型捕获、注册表和 `isinstance`。子目录是源码所有权边界，
不是公开 Python 命名空间。

### 文件名必须唯一

兼容层通过文件名解析逻辑模块。所有参与搜索的目录中，除 `__init__.py` 外，不允许出现两个
同名 `.py` 文件。检查命令：

```bash
find chemvector -type f -name '*.py' ! -name '__init__.py' -printf '%f\n' \
  | sort | uniq -d
```

命令无输出才表示没有冲突。

## 新增代码示例

### 新增可复用 Bond 组件

假设新增 `bond_order_context.py`：

```text
chemvector/components/bonds/objects/bond_order_context.py
tests/chemvector/test_bond_order_context.py
```

其他模块使用：

```python
from chemvector.bond_order_context import BondOrderContext
```

### 新增训练入口

假设训练该组件：

```text
chemvector/training/bonds/relations/train_bond_order_context.py
docs/model/training/YYYYMMDD-vX.Y.Z-bond-order-context.md
```

运行：

```bash
python -m chemvector.train_bond_order_context --help
```

模型定义不得直接写进 `train_bond_order_context.py`；可复用部分放回 `components/`。

### 新增 benchmark

端到端固定数据集评测放：

```text
chemvector/evaluation/benchmarks/benchmark_example.py
```

局部组件评测放 `evaluation/evaluators/<domain>/`，错误归因放
`evaluation/audits/<domain>/`。不要用文件名 `test_*.py` 代替正式 benchmark；单元测试属于
`tests/`。

## 路径与文件系统

源码不得通过 `Path(__file__).parents[n]` 猜测项目根目录。统一使用：

```python
from chemvector.paths import PACKAGE_ROOT, PROJECT_ROOT
```

- `PACKAGE_ROOT`：`chemvector/`。
- `PROJECT_ROOT`：项目根目录。

文件归属：

| 内容 | 位置 |
|---|---|
| 原始/生成训练数据 | `Data/` |
| 正式和 smoke checkpoint | `outputs/checkpoints/` 或 `outputs/smoke/` |
| benchmark 结果 | `outputs/evaluations/` |
| 因果审计资产 | `outputs/audits/` |
| 每次训练记录 | `docs/model/training/` |
| 文献和路线调研 | `docs/research/` |
| 小型静态配置 | `chemvector/configs/` |

禁止向 `chemvector/` 写入 `.pt`、`.tar`、`.png`、运行日志或临时 JSON。需要缓存的工具必须接收
显式 `--output`/`--output-dir`，不得默认写到源码旁边。

## 命名规则

- 可复用组件：能力名，例如 `semantic_bond_relation.py`。
- 训练入口：`train_*.py` 或 `pretrain_*.py`。
- 组件评测：`evaluate_*.py`。
- 固定测试集：`benchmark_*.py`。
- 只读审计：`audit_*.py`。
- 数据构建：`build_*.py`。
- 推理入口：`infer_*.py`；共享生成逻辑可使用 `generation_*.py`。
- 校准、合并和导出分别使用 `calibrate_*`、`merge_*`、`export_*`。

源码文件名不包含正式模型版本号。版本号属于训练文档、checkpoint 输出目录和 Git tag；确有
协议代际并存时才允许 `generation_v4.py` 这类兼容文件，并应在模块文档中说明淘汰条件。

## 移动文件检查清单

移动现有模块时依次完成：

1. 确认目标职责目录，检查没有同名模块。
2. 物理移动文件，不创建顶层空壳 wrapper。
3. 搜索 `__file__`、`with_name()` 和硬编码的 `chemvector/...` 物理路径。
4. 项目路径改用 `chemvector.paths`。
5. 检查随代码冻结的 SHA256；路径维护导致内容变化时，明确更新并重新测试门禁。
6. 验证旧逻辑导入和旧 CLI。
7. 运行相关测试；大范围移动必须运行完整 `tests/`。
8. 更新本 README、文档索引和受影响的配置路径。
9. 清理 `__pycache__`、`.ipynb_checkpoints` 等生成缓存。

兼容 smoke：

```bash
python - <<'PY'
import chemvector.model
import chemvector.train
import chemvector.evaluate
print(chemvector.model.__file__)
print(chemvector.train.__file__)
print(chemvector.evaluate.__file__)
PY

python -m chemvector.train --help
python -m chemvector.evaluate --help
python -m pytest tests -q
```

## 正式训练与版本

源码整理不产生模型版本。只有正式训练生成候选 checkpoint 时才：

1. 在 `docs/model/training/` 新建独立中文记录。
2. 记录数据 commitment、样本数、初始化权重、命令、参数量和门控结果。
3. 提交对应代码、配置和训练记录。
4. 创建与模型名完全一致的 annotated Git tag。
5. 在 tag 注释中记录 checkpoint 路径、SHA256 和 GO/NO-GO。

Smoke、容量实验和中间 step 不单独占正式版本 tag。MR-TADF 可以作为主应用 benchmark，但不得
进入训练、阈值选择或 checkpoint 选择。

## 已知限制

当前扁平逻辑模块兼容层是为了保留大量历史实验，具有两个约束：

1. 所有逻辑模块文件名必须唯一。
2. 制作 wheel 或发布 pip 包时，必须验证深层源码目录都被打包并运行 import smoke。

等 ChemTraverse 新路线稳定、历史实验进入只读状态后，可以单独立项迁移到真正的分层公开模块
名。迁移必须统一更新代码、测试、文档和 checkpoint loader，不能在日常实验中零散混用两种
导入方式。
