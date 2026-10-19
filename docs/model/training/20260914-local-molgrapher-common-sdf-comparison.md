# 2026-09-14：本机 MolGrapher 的固定 200 图实测

## 结论

本机完整 MolGrapher 在同一批已暴露开发输入上，非立体完整化学图正确 **111/200（55.5%）**。
冻结自研研究参照 v0.10.16 为 **59/200（29.5%）**，部署默认 v0.10.10 为 55/200（27.5%）。
这是实际同图、共同 SDF 评分器的比较，不是拿论文成绩代替本地运行。

相对 v0.10.16，MolGrapher 救回 61 张、丢失 9 张，净增加 52 张、26 个百分点。
两者共同正确 50 张、共同错误 80 张。它不是逐样本完全替代自研模型的证明。

## 分版式结果

下表每组分母为 50，指标为非立体完整图正确数；括号中是百分比。

| 版式 | 自研部署默认 v0.10.10 | 自研研究参照 v0.10.16 | 本机 MolGrapher |
|---|---:|---:|---:|
| original | 17（34%） | 17（34%） | 37（74%） |
| rdkit | 20（40%） | 20（40%） | 34（68%） |
| coordgen | 13（26%） | 17（34%） | 34（68%） |
| stress | 5（10%） | 5（10%） | 6（12%） |
| 合计 / 200 | 55（27.5%） | 59（29.5%） | 111（55.5%） |

不含 stress 的 150 张，MolGrapher 为 105/150（70%），v0.10.16 为 54/150（36%）。
这只是固定子组的补充统计，不移除 stress，不把主分母改成 150。四种版本全部正确的分子，
MolGrapher 只有 1/50，说明跨布局稳定性仍然不足。

本机 MolGrapher 的输出详情：

- 200 次调用完成，调用异常 0；生产过程约 352.16 秒，含加载和逐图调用开销，不是吞吐 benchmark。
- 193 个 SDF 通过共同 RDKit 严格解析/完整 sanitize；其中 111 正确、82 有效但错误；7 无效。
- **33 个结果是本机原生失败回退产生的单个 `C`、置信度 0**，全部仍计入 200 分母，
  归为有效但错误，不能将“193 个有效 SDF”解释为 96.5% 的识别准确率。
- 33 个占位结果分布：original 1、rdkit 5、coordgen 5、stress 22。
- 使用 `gc_no_stereo_model`，不识别立体化学。共同评分器记录的立体 exact 为 68/200，
  仅供完整输出审计，不能解释为该模型具有立体识别能力；本轮主比较只用非立体 exact。

## 本机系统身份与执行边界

用户指出本机已有 MolGrapher 后，在项目相邻目录找到可运行的完整安装。此前只检查当前仓库
`third_party/` 和 `.venv-train` 的记录不能代表整机安装状态。

- 源码：`/home/zhenghe/GK/Project/MolGrapher/MolGrapher`。
- Git HEAD：`33c8c4d244dd8485098f30d6697f92a4cf93d4c8`，本地提交
  `feat: add aligned sdf export test workflow`，另有用户未提交改动，全部保留。
- 这是一份包含本地导出扩展的 MolGrapher，不声称是未经修改的官方仓库/论文复现。
- 环境：`/home/zhenghe/miniconda3/envs/MolGrapher/bin/python`，Python 3.11.15，
  torch 2.10.0+cu128，torchvision 0.25.0+cu128，Lightning 2.1.3，
  PaddleOCR 2.10.0，CPU Paddle 3.3.0，RDKit 2025.9.6，NumPy 2.4.6。
- RTX 5090 的 CUDA 张量计算和正式图网络推理成功；驱动 590.48.01，torch CUDA 12.8。
  现有 Paddle 未编译 GPU 支持，原生 OCR 调用使用 CPU；未换装/降级任何依赖。
- 模型参数约 39.53M：关键点检测器 12.8722M、图分类器 26.6585M。

使用已有的两个 checkpoint，运行前后 SHA256 均一致：

| 文件 | SHA256 |
|---|---|
| `data/models/keypoint_detector/kd_model.ckpt` | `54f67ea2e0bf2c91d472ca245f72cc3a0fa008f0a3ee2cfc54fde12a817f9b79` |
| `data/models/graph_classifier/gc_no_stereo_model.ckpt` | `30f4e1569c793e3d6f37ea780d524fa45d660660834bab002cdfdb09eebc15ff` |

输入来自冻结副本中的既有 200 PNG，逐张核对哈希后复制到新实验目录。MolGrapher 原生流程
会在输入旁写 `_preprocessed.png`，所以没有把冻结目录直接交给它。保留默认 caption removal、
OCR、图后处理和回退；不注入 ChemVector 的预测或修补规则，不传入真值，不训练、不调阈值。

运行适配仅为：按本机 `test.py` 补回 NumPy 已删除的 `np.int0 = np.int64` 别名；
关闭清理和可视化；启用原生 MOL/SDF 导出；每次输入一张，配置 `nb_workers=0`。
每个样本调用一次，无失败重试/置信度筛选。`align_rdkit_output=False` 保留默认坐标生成方式，
不声称这些导出坐标恢复了原图位置。

Lightning 在内存中升级旧 checkpoint 元数据；没有运行永久升级命令，磁盘权重未修改。
原生源码/配置与实验模块共 79 个文件、两个 checkpoint、9 个已有 OCR 缓存文件，以及全部输入
均有哈希绑定；Python 包版本另行记录。这不是打包整个可启动环境的容器快照。

## 验证

推理和评分分进程、分环境：先用 MolGrapher 环境完成生产并封存 receipt，之后才由当前
`.venv-train`（RDKit 2026.3.3）读取图真值，使用未修改的 `profile_comparison.scoring.score_sdf`。
主指标直接评分原生导出的 SDF，不在失败时改用 SMILES 或另一个编译器结果。

- 冒烟：按固定顺序的前 4 张成功，正式运行重新加载并重新预测，不沿用冒烟结果。
- 两次独立加载的前 4 张，12 个 MOL/SDF/预处理 PNG 文件逐字节一致；SMILES、molblock、
  OCR 文本、置信度和原子/键数量也一致。
- 正式生产模型 state_dict SHA 前后相同：
  `65b93efb20322529f80e1388ebde6fbaefab792cb73cd5c569feb5967fa0a376`。
- 原生 molblock 与 SDF 的三个共同指标全部一致；另一次直接 RDKit 复核确认 200 个
  MOL、molblock、SDF 的图签名一致（共同 sanitize 失败也按失败保留）。
- 200 个真值 SDF 控制通过。独立脚本重新解析 MolGrapher/v0.10.10/v0.10.16 共 600 个
  预测 SDF，重现所有有效性、非立体与立体结果，并复核配对统计。
- 外部对照、既有 profile 比较、基线冻结相关测试共 **54 passed**。
- 冻结的 v0.10.16 副本再次通过 5,215 文件校验；前端推荐配置未改。

## 工件与复现

工件目录：`outputs/experiments/molgrapher-200-20260914/`。
包含 `protocol.json`、`runtime.json`、`producer.log`、`images/`、`native/`、
`invocations.json`、`prediction_receipt.json`、`scores.json`、`report.json`。
`native/` 中保留每张原始调用结果、MOL、SDF，以及原生 `smiles.jsonl`。

| 工件 | SHA256 |
|---|---|
| `protocol.json` | `0d8e120b8a95c2cd857076a32013d08842511e836956e15f49cadf0ca86d9503` |
| `prediction_receipt.json` | `794b8f47b9b3fcd852d59ca05094a259c51fba6a8887b242b4357193b86cc652` |
| `scores.json` | `9c1f7802498b85f0264bcd9b6642e560b3f8fbd08592e90152120ad9061ffdda` |
| `report.json` | `54de515e098c1bd6f8b9d71d8fbd1a050812f0343e7fcb7bf7495a6caf7a27ca` |

从 chempic2sdf 根目录运行，生产目标必须使用新目录，以下使用新名称示意：

```bash
/home/zhenghe/miniconda3/envs/MolGrapher/bin/python -u -m chemvector.experiments.external_ocsr.molgrapher --output-dir outputs/experiments/molgrapher-200-rerun
.venv-train/bin/python -m chemvector.experiments.external_ocsr.score --output-dir outputs/experiments/molgrapher-200-rerun
```

## 对后续方向的含义

同一批图片并非让所有模型都只能达到约 30%；现有成熟图识别系统在三个非 stress 版式上
显著提高了这次实测的正确数。这个完整系统比较同时改变了权重、训练数据、结构识别、
预处理、OCR、后处理与数值环境，**不能把增益单独归因于某一个模块或架构**。

stress 上两者都很低，是需要单独审查输入扰动与失败机制的共同难点；不能把它当作已被
外部模型解决，也不能仅凭低分就认定这些图片或真值错误。优先复核配对表中的 80 个共同
错误与 9 个自研正确而 MolGrapher 错误的样本，再决定需要补什么。没有据此启动新训练或
自动切换前端模型。

这 200 图是 50 个分子的四种合成布局，属于反复使用的开发诊断子集，不是 MolGrapher
官方 benchmark，不代表独立泛化性能，也没有证明与第三方预训练数据零重叠。
