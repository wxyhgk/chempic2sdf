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

### ChemScene 推理边界

`infer_chemscene_hybrid.py` 只负责编排 checkpoint、模型前向和输出流程。下列确定性算法已经从入口拆出，
训练、评测和其他推理路线应直接复用对应逻辑模块：

| 逻辑模块 | 物理位置 | 职责 |
|---|---|---|
| `chemvector.chemscene_assembly` | `components/graph/joint/` | Junction/BondObject 组装、原子标签转换、ChemScene 校验与 SDF 编译 |
| `chemvector.junction_text_attachment` | `components/graph/joint/` | 将 OCR 结果一对一挂接到已选 Junction，不允许文字创建结构点 |
| `chemvector.graph_prediction_correction` | `components/graph/correction/` | 将图修正头输出应用到 BondObject 集合 |
| `chemvector.text_interior_bond_veto` | `components/bonds/selection/` | 删除完全位于同一文字框内部的假 BondObject |

旧代码仍可从 `chemvector.infer_chemscene_hybrid` 导入这些函数，但该入口只做兼容重导出，不再保存实现。
组件不得反向依赖 `inference/`、`training/`、`evaluation/`、`tools/` 或 `workflows/`。

Hybrid 推理所需的 selector、BondObject、OCR、graph corrector 和 junction-aware checkpoint 统一由
`chemvector.hybrid_checkpoint_loader` 装配。`ChemSceneHybridPipeline.__init__` 只保存运行参数并绑定显式的
`HybridRuntimeComponents`，不得直接调用 `torch.load`；入口模块中也不得出现任何 `torch.load` 调用。
invalid-only OCR fallback（`load_compatible_fallback_ctc_ocr`）同样由 `chemvector.hybrid_checkpoint_loader`
提供，入口只重导出以保持历史调用兼容。可选专家必须以完整组合出现，组合约束在读取任何
checkpoint 前统一校验。

Direct AtomPair Bond 入口（`chemvector.infer_direct_atom_pair_bond`）的 checkpoint 装载与
OCR 回填、structural-twin 抑制、场景张量转换算法位于 `chemvector.direct_scene_inference`；
入口只做编排与兼容重导出，不得再保存实现，也不得出现 `torch.load` 调用。该入口的模块名
被预注册审计的子进程命令固定，不得改名或移动。

Joint 入口（`chemvector.infer_joint_object_structure`）的冻结组件与 AtomPair 专家级联
（group / graph-context / one-move / retype-verifier / line）由
`chemvector.joint_component_loader.load_joint_components` 统一装配并执行跨 checkpoint
协议校验；AtomPair 精炼与 decode attempt 构建位于 `chemvector.joint_atom_pair_refinement`。
两个推理入口共享的占位画布、图像准备与碳占位化学回退位于 `chemvector.placeholder_scene`，
入口只兼容重导出，入口之间不得互相导入。

遗留入口 `infer_parallel_v6`、`infer_detected_v5`、`infer_shadow_v0928` 本身就是零实现形态：
`main` 只编排已外部化的 `chemvector.*` 组件（生成、修复、SDF/图像 IO），不含 `torch.load`，
也不保存可复用算法；`infer_shadow_v0928` 额外持有自己的 profile 校验与命令构造
（profile 冻结 checkpoint 清单，由子进程方式调用 `infer_chemscene_hybrid`）。边界测试
`test_legacy_entries_keep_zero_implementation_form` 守卫这一形态，不得向这些入口内联实现。

单图命令行参数由 `chemvector.hybrid_cli.build_hybrid_parser` 唯一定义。入口模块只解析该协议，不得再次
内嵌第二份参数表；这样前端、benchmark 和命令行可以针对同一套默认值与 choices 做兼容测试。

ChemScene JSON、SDF、诊断 PNG 和 `metrics.json` 统一由 `chemvector.hybrid_output` 写入；模型流水线
不直接接触 RDKit 绘图或结果目录。无效结构的诊断 SDF 写入位于 `chemvector.diagnostic_sdf`，benchmark
仅兼容重导出，不再被推理代码反向依赖。

Junction 的视觉候选融合、集合选择、角色抑制和 batch padding 位于
`chemvector.hybrid_junction_selection`；该模块不依赖 OCR。结构集合确定后，
`chemvector.hybrid_ocr_inference` 独立完成 CTC crop、主/专家 OCR 路由和标签回填。OCR 组件只接收已经
冻结的 Junction 集合，只能修改文字角色的标签，不能新增、删除或移动结构点。入口中的
`ChemSceneHybridPipeline._junctions_batch` 现在只编排这两个阶段。

Junction 选择内部又分为候选准备和集合定稿。候选准备保留主文字阈值与过滤阈值的不同职责，按
text-first 容量规则合并候选并生成视觉、像素和端点证据；集合定稿执行按角色阈值、同角色抑制、
空集校验和变长 batch padding。角色及候选索引顺序是下游原子编号协议的一部分，不能按全局分数
重新排序。

主 OCR 与专家 OCR 的 utility 路由位于 `chemvector.hybrid_ocr_routing`。该模块只处理两组文字预测的
分歧、动态特征标准化和保守替换；不依赖 Junction、crop 或 checkpoint 装载。旧的
`chemvector.hybrid_ocr_inference.route_ocr_predictions` 仍重导出同一函数，保证历史调用兼容。

BondObject 的多尺度特征准备、基础/风格专家推理和 utility router 选择统一放在
`chemvector.hybrid_bond_routing`。该模块只返回过滤后的键对象和路由诊断，不接触 Junction、OCR 或
ChemScene 编译；pipeline 保留一个薄委托方法，以兼容现有内部调用。

无效 ChemScene 的原子状态恢复候选由 `chemvector.hybrid_atom_decoding` 生成。它可以重新读取同一文字
crop 并提出元素、电荷、显式氢等候选，但不能改变 Junction 或 BondObject；化学有效性重排仍由
`structured_atom_decoder` 负责。

图级后处理分为两个边界：`chemvector.hybrid_graph_refinement` 负责 graph corrector、语义键级和稀疏
键存在推理；`chemvector.hybrid_junction_bonds` 负责 BondObject 集合选择、端点聚类及随后一次语义键级
细化。二者都显式返回诊断字典，不允许通过 pipeline 全局状态传递结果。

更高层的推理编排也分为三个阶段：`chemvector.hybrid_component_routing` 统一处理单图/批量的 Junction
来源与 Bond 来源分派；`chemvector.hybrid_scene_compilation` 在固定组件上处理 assignment、ChemScene
组装及 invalid-only 恢复；`chemvector.hybrid_cli_execution` 负责模型装载、图片准备、可选 OCR fallback 和
结果写盘。入口 `main()` 只连接这四个步骤，不再保存业务流程实现。

其中场景编译内部继续按固定顺序分为五段：先执行可选的初始图修正，再确定显式或学习式 assignment，
随后完成首次 ChemScene 组装和结构化原子状态重排；只有首次场景无法编译时才进入 `invalid_only`
恢复；最后按 assembly、assignment、graph、atom、component、固定运行元数据的顺序合并 diagnostics。
恢复流程复用首次 assignment，结构化原子候选只计算一次，修正后场景可以使用同一候选再重排一次。
这些顺序属于兼容协议，修改阶段 helper 时必须同步通过事件序列和 diagnostics 覆盖顺序测试。

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

已知测试状态：G3 P1-B/P1-D/P1-E 的冻结 hash 链测试（`test_audit_g3_p1b_final`、
`test_audit_g3_p1d_final`、`test_audit_g3_p1e_final` 各一个方法，以及
`test_train_g3_p1e_aligned_stroke_feature_sufficiency` 的
`test_dependency_sources_are_preregistered`）自 G3 campaign commit `ec2b846` 起即失败：
预注册的 `EXPECTED_*_SHA256` pin 与同 commit 提交的源文件（`g3_presence_ranker.py`、
`g3_boundary_band_loss.py`）实际 hash 不符。2026-08-30 调查确认：pin 与 checkpoint
payload、预注册文档三方一致（训练时真实记录），训练版源文件在 `ec2b846` 被修订后无
处归档、不可恢复，且 P1-B/D/E 均为终态 NO-GO 实验。决策：不削弱审计（不 re-pin、不改
审计逻辑），4 个方法以 `@pytest.mark.xfail(strict=True, reason=...)` 隔离并显式记录；
strict 为金丝雀——若未来正式 re-pin 或恢复源文件使测试通过（xpass），会显式报错，
必须人工清除标注。审计实现与 pin 常量原样保留。

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
