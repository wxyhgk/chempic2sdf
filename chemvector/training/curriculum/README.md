# 自研 ChemVector：课程数据接入与定额继续训练

**最新回归（2026-09-19）：presence delta 不可替代 initial。** CPU 无更新评估在 30 分钟上限停止，共完成 919/944 图；课程验证 63/144 → 87/144，但已测完整分子 167/775 → 141/775，救回 3、改错 29，有效 SDF 531→395。剩余 25 张即使全部正确，delta 最多 166，仍小于 initial 已得到的 167；这是决策上界，不是完整 800 图实测分数，因此不自动追加推理。3 张历史回放 SMILES 不同但计分字段相同；超时未到达末尾全模型状态检查，已独立复核保存结果与源文件。GPU=0、训练更新=0，初始模型继续保留；相关测试 780 项通过。下一步先准备训练侧旧能力保护，开发集不转训练。见 [泛化退步、预算边界及下一步](../../../docs/model/training/20260919-presence-full-regression.md)。

**前序小试验（2026-09-19）：一次 CPU 连键存在性试验完成，训练 10/24 → 19/24，原干净验证 8/16 → 11/16。** 只更新现有 presence 末层 257 个参数，冻结上游、共享 pair 特征、原始 chemical logits、role/bbox/OCR；66.001 秒、200 步、GPU=0。训练侧可达漏键 12→0、确认误连 1→0，原来严格正确的图改错 0；但 `CCNN` 补边后暴露原 DOUBLE 键型错误，变为无效，不能称全链路修复。24 图是拟合数据，16 图仅为旧干净回归，当时尚未重跑完整开发图，默认仍保留 initial（历史 170/800），不部署或自动追加训练。固定子集已独立发布，未重查私有身份；该阶段相关回归 752 项通过。见 [小试验结果、剩余错误与下一步](../../../docs/model/training/20260919-presence-readout-result.md)；更广回归以上方最新记录为准。

**前序准备（2026-09-18）：连键存在性输出层已独立接线，仅完成合成验证。** 新 `FrozenFeaturePresenceStage` 冻结所有共享特征与其他分支，只允许现有 `pair.presence` 的 weight/bias 接收梯度；全部确认非边保留，未知端点不当作负例。随机小模型的一次合成更新确认 hidden 与原始 chemical logits 逐位不变；不是项目权重训练，也没有新真实图推理或准确率结果。该准备阶段未接入默认 launcher、未授权 GPU/真实更新，最新训练状态见上方结果。见 [实现边界及当时 CPU 小试验提案](../../../docs/model/training/20260918-presence-readout-preparation.md)。

**前序诊断（2026-09-18）：固定 17+7 干净图完成 initial-only CPU 回放，严格身份 10/24。** 这是定向 quarantine/train 诊断，不是独立验证准确率。91 根目标键中 89 根可达，仍漏连 12 根，主要为 N–S（8/11）和 N–N 单键（3/8）；另发现 3 处 `NH→NHH→解析失败→保留碳占位`，三键控制则被 selector 阻断。优先准备冻结其他路径的存在性专项验证，不直接启动全 pair 或联合训练。CPU 24.346 秒，相关回归 689 项通过；权重/OCR/标签不变，无 GPU、训练或新身份查询。见 [24 图诊断与下一步](../../../docs/model/training/20260918-pair-gap-diagnostic24.md)。

**前序排重（2026-09-18）：固定 18 个补缺候选已完成一次获准身份核验，公开排除 1 个，剩余 17 个。** 私有侧保守比较无候选重合；公开侧命中甲磺酰胺 `CS(N)(=O)=O`，不补选新身份。17 张新隔离图与 7 张既有训练控制组成 24 图诊断准备清单，基础正/负例及四种键型标签覆盖仍齐全。一次私有核验授权已消耗，该排重阶段相关回归 655 项通过，没有模型推理、训练或 GPU 使用，数据仍 quarantine，initial 不变。见 [身份排重结果](../../../docs/model/training/20260918-pair-gap-identity-review.md)；其中待执行诊断描述为历史阶段，以本页最新结果为准。

**前序产物（2026-09-18）：N–N/N–S 补缺隔离图已生成，18/18 通过适配。** 8 个 N–N 单键身份、8 个 N–S 单键身份及 2 个 N/S 非边控制，各一张干净 reference；另引用 7 张既有训练图保护四种键型及真假 O–O。三个 subagent 参与候选、生成接口、监督/权限审查和合成测试。该生成阶段无模型推理或 GPU，未训练、未进行新外部身份查询，数据仍 quarantine。见 [生成结果、图册及准入边界](../../../docs/model/training/20260918-pair-gap-quarantine.md)，其中待授权描述是生成阶段的历史状态，以本页最新结果为准。

**最新准备（2026-09-18）：暂停联合续训，保留 initial（170/800）。** 干净 16 图回放仍为 8/16；基础错误见 [clean16 报告](../../../docs/model/training/20260918-clean16-basic-errors.md)。本轮训练侧核对发现：62 个符合小分子范围的课程训练身份缺 N–N、N–S 单键正例，跨节点负例已有覆盖。隔离 `pair_stage.py` 已完成合成梯度测试，但未接入默认 launcher、未更新真实权重，也没有新的 GPU 授权。覆盖清单、单组件边界和下一步见 [pair 专项准备](../../../docs/model/training/20260918-pair-stage-readiness.md)。下方保留历史接入与试训记录，不应将旧“下一步/尚未完成”段落视为当前状态。

**历史状态（2026-09-15）：扩量后的冻结 role 试训已完成，在 800 步早停；完整分子 170/800 → 117/800，保留 initial，未部署。** 耗时 323.450 秒，未追加运行；结果见 `docs/model/training/20260915-expanded-freeze-role-result.md`。独立扩量 release 训练 13,151 条 / 12,985 张不同图片，固定验证 944 条不变。本批私有核验凭据仅继承，不再查询；公开复核与用途审查已完成。11,124 张兼容候选中排除与旧训练像素相同的 298 张，新增 10,826 张；旧训练的 166 条内部像素重复原样保留。发布为 `outputs/datasets/chemvector-expanded-release-20260915/`，源 quarantine、旧 release、checkpoint 和前端不动。

新增 opt-in `freeze-role-v1` 协议冻结 role/OCR，保留六个可训练组。四张新图 CPU 检查及本次真实 GPU 运行均确认 role 状态保持不变，但整图仍退步；未据此改试其他冻结组合。任务输出为 `outputs/experiments/chemvector-expanded-freeze-role-20260915/`；一次授权已执行，不自动重跑。授权与准备见 `docs/model/training/20260915-expanded-training-readiness.md`。下述模型诊断与旧 release 为历史记录。

2026-09-14 用户明确要求继续使用自己的模型。主线是 **现有 ChemVector 权重与模型的继续训练**，不是 MolGrapher 微调。外部模型仅保留为已有对照，不成为本模块依赖。

当前已接通数据、准入、训练/验证循环及完整权重包。2026-09-15 的 mixed release 为训练 2,325 图、验证 944 图，源隔离数据不动。**第一轮实际 GPU 续训已在 1,800 步按三次不改善早停**：完整分子匹配 170/800 → 66/800，未得到更好模型，保留 initial，未部署。耗时 414.60 秒，单卡用量上界 0.11517 GPU-hours。下一步只做文字角色/裁框接口的无更新定位，不自动启动第二轮训练。最新结果见 `docs/model/training/20260915-mixed-continuation-result.md`；下方保留早期历史。

**已完成的接口诊断**：24 个固定分层案例、五组 CPU 无更新对照，仅换回 initial role 后正确数 4 → 15；仅换回 bbox 仍为 4，两者回退为 14。48 次 initial/last 回放均重现旧结果，结构与权重不变；58.15 秒，GPU=0。这是错误富集案例定位，不是总体准确率。原先提出的完整 800 图 role 回退推理对照尚未执行；后续改为上方待确认的冻结训练提案，不宣称该推理对照或 role 根因审计已经完成。详见 `docs/model/training/20260915-text-interface-diagnostic.md`。

## 已完成

- `data.py` 把新课程数据的 graph / same-render layers 转成现有 ChemVector 原子、文字区域、键和连接目标张量。
- 不通过旧 SVG 坐标→ID 字典构图；对象身份通过 ID 绑定，即使坐标重合也不合并。仍报告现有 spatial atom loss 无法表示的同格原子。
- `IMPLICIT/TEXT` 来自实际可见字符；隐含氢不补进文字。原子锚点与字符墨迹框中心分开。
- 芳香化学键与画面中的 Kekulé 单/双键分开。楔线端点顺序跟随 drawing 记录，完整化学图原样保留。
- 使用现有 q1023 量化及 text-bbox-ray 端口约定。端口是派生的模型监督，不冒称 SVG 笔画的实测端点。
- 不把新数据的 bitset mask 交给旧 categorical loss；不添加 OCR、元素分类或官能团分类损失。
- quarantine 样本只能显式用于诊断；默认训练读取及训练 collator 都拒绝未准入数据。

首批报告：`outputs/audits/chemvector-curriculum-cpu-20260914/report.json`。

检查 751 张已通过渲染合同的图，**751/751 可转换为现有目标张量**，共 98 个结构，没有空间格碰撞。原生成器另有 158 张被拒绝，仍计入覆盖缺口，未被本适配器修复。

新版生成器在独立 `ink-aware-v1` 模式修复了上述排版问题：101 个结构的 909 张图全部通过原有成图校验；目标适配 **909/909**，失败和空间格碰撞均为 0。最终报告：`outputs/audits/chemvector-curriculum-v02-cpu-final-20260914/report.json`。第一次 v0.2 报告也保留，其中旧模板固定列出的“仍需修复渲染拒绝”不适用于实际 0 拒绝数据；最终版改为按实际拒绝数列出，不改写历史报告。

`tests/chemvector/test_curriculum_data.py` 验证现有 `spatial_atom_object_loss`、`build_pair_targets` 和 `direct_atom_pair_loss` 可在 CPU 消费新标签，执行有限损失及反向检查。测试只初始化微型现有原子头/假 logits，不读取训练 checkpoint、不更新优化器，不能解释为整模型训练或识别效果验证。

## 运行

```bash
CUDA_VISIBLE_DEVICES='' .venv-train/bin/python -m chemvector.training.curriculum.preflight \
  --data-root outputs/datasets/curriculum-v0.2-ink-aware-20260914 \
  --output outputs/audits/chemvector-curriculum-cpu-new
```

输出目录必须不存在。输入逐样本核对内容哈希、同源文字绑定、SDF 身份、对象索引、类型与 target 容量。报告不会授予训练权限，也不会调用已有训练入口。原图、旧权重、旧实验及前端配置不修改。

## 身份排重：只读缓存，失败关闭

`admission.py` 不打开外部评估图片/结构，不重建旧缓存，只接受显式给定的现有索引。运行示例（输出必须是新目录）：

```bash
CUDA_VISIBLE_DEVICES='' .venv-train/bin/python -m chemvector.training.curriculum.admission \
  --data-root outputs/datasets/curriculum-v0.2-ink-aware-20260914 \
  --exclusion-index Data/sdf/chemvector_joint_v1/work/old_overlap_index.json \
  --output outputs/audits/chemvector-curriculum-admission-new
```

本轮结果在 `outputs/audits/chemvector-curriculum-admission-20260914/`：

| 提案 | 独立分子 | 图片 |
| --- | ---: | ---: |
| 已确认公开评估重叠，排除 | 12 | 108 |
| 候选训练 | 75 | 675 |
| 候选验证 | 14 | 126 |

同一分子的所有变体一起处理。已知完整 InChIKey 或 14 字符连接层前缀命中均排除；候选内部按忽略立体/同位素的 Murcko scaffold 分组，无环结构用完整 core 身份分组，固定种子做约 80/20 的组级提案，训练/验证组交集为 0。36 个分子命中外部 scaffold，报告风险而非当作精确身份重复；共同芳环不自动等价于同分子。

旧索引 `benchmark_inchikeys` 的 30,818 项中有一个空键，旧合并列表也包含一个空键，分别计数，不能推断为两个不同未解析分子。非空合法键继续用于正向排除，无效键必须进入覆盖缺口；没有任何有效评估键则直接报错。缓存还报告 11 条旧 USPTO 记录未解析，fresh-dev/sealed-test 排除身份清单未检查，旧混合列表无法区分训练和评估归属。

因此所有报告和提案均为 `training_authorized=false`、`partially_checked`，原数据继续 quarantine。新的 `launcher.py` 已接入完整清单绑定与预算门禁，会在打开 manifest、加载模型或启动 worker 前拒绝本报告；默认 loader 同样拒绝这些数据。CPU 适配通过与身份准入是两个独立检查，二者都不是模型准确率。

## 续训内核与实际权重梯度结果

新增模块互相分离：

- `continuation.py`：只组合现有组件、现有损失和参数分组，不加载数据、不创建新专家；`train_step()` 提供受限更新原语。
- `checkpoints.py`：按前端当前登记哈希核验 pair/role/bbox/OCR 文件，记录全部 9 个相关 checkpoint 哈希。沿用原加载器，包括 query checkpoint 内的视觉权重更新；不恢复旧 optimizer、不加载 OCR 模型或评估逐样本。
- `runtime.py`：不依赖 torch 的外层子进程时限，CPU worker 隐藏 CUDA；最多允许显式指定 1 张 GPU、时限不超过 12 小时，GPU/训练请求必须先检查准入。超时杀掉本次创建的进程组、不自动重试，输出目录不可覆盖。这是复用的运行保护，不是完整训练发布器。
- `gradient_preflight.py`：公开 CPU 诊断入口；读取实际旧权重，执行 train-mode forward/backward，但不创建 optimizer、不更新或保存模型权重。

当前参数分组为 vision（实际使用的 stride 4/8/16）、atom、bond、set_head、pair、role、bbox，共 **13,130,546 个拟训练参数**。未使用的视觉 stride-32、token norm/projection 保持冻结；OCR 转写完全不参与。七组是现有组件的参数分组，不是新增七个专家。参数组使用一套暂定学习率，未开展超参数搜索；未验证正式训练效果。

与旧 pair-only 训练不同，这里不将视觉和原子路径包在 `no_grad` 内，也把依赖上游特征的旧键候选和 set 筛选损失接回。最终键头仍使用真实预测、经过 set 筛选的原子对，GT 只做匹配和标签，不注入 GT 候选。role/bbox 的裁剪锚点 detach，避免局部文字分类损失把原子坐标拉到便于识字的位置。top-k、阈值与匹配依旧不可微，不把这一限制称作已解决。

最终报告：`outputs/audits/chemvector-curriculum-gradient-final-20260914/report.json`，运行记录在同目录 `runtime.json`。3 张显式选定诊断图逐张得到：

| 诊断样本 | 目标原子 | 筛选后匹配原子 | 目标键 | 最终键头可达正例 |
| --- | ---: | ---: | ---: | ---: |
| 乙醇 reference | 3 | 3 | 2 | 2 |
| 四氢呋喃 reference | 5 | 5 | 5 | 5 |
| 甲基硼酸 serif | 4 | 3 | 3 | 2 |

每张图的七组均得到有限非零梯度，285 个拟训练参数张量均有梯度；前后整个模型 state 哈希及旧 checkpoint 文件哈希不变。外层实测约 8.23 秒，180 秒上限，GPU 训练用量为 0。甲基硼酸的缺失正例说明筛选仍会阻断部分监督，不是诊断成功就已解决漏检。

```bash
CUDA_VISIBLE_DEVICES='' .venv-train/bin/python -m chemvector.training.curriculum.gradient_preflight \
  --data-root outputs/datasets/curriculum-v0.2-ink-aware-20260914 \
  --sample-keys curr_ab1de819ede91df4__reference curr_92623ccd36bc322b__reference curr_f0d91cc2b0b4b0bf__serif \
  --output outputs/audits/chemvector-curriculum-gradient-new --wall-seconds 180
```

诊断最多允许 4 个显式样本，目的为梯度接线检查，不是模型准确率评测、容量实验或课程效果实验。CLI 的内部 worker 参数不能当成正式训练入口。

`train_step()` 会在 forward 前拒绝未准入报告、诊断 batch、缺失/耗尽步数额度和错误 optimizer 参数集合；反向后检查梯度有限性、裁剪范数，再预占步数并更新。合成微型单测验证了实际 CPU optimizer step 与额度耗尽后权重不变；**这不表示对课程数据或真实旧权重执行了训练**。清单绑定由新增 `release.py` / `launcher.py` 执行；手工修改报告布尔值不构成数据准入。

## 完整任务接线更新

- `release.py`：批准报告必须绑定 manifest 原始字节 SHA256，并明确来源、公开评估、fresh-eval 身份、分组及 target 检查。跨来源的同分子/core/scaffold 不能进入不同 split，样本 metadata 哈希继续绑定 PNG/图/SDF 等 payload。不提供 quarantine 晋升脚本。
- `launcher.py`：训练只消费明确发布的 same-render 数据根和样本清单，验证还支持显式固定的原图/SDF 来源；禁止自动搜数据。训练必须同时保留 curriculum 与 full_molecule 两个域，混合循环和种子写在受审清单中。每次正式运行显式给步数和时间上限，CPU/CUDA 都执行准入检查，无自动重试/断点续跑。
- `evaluation.py`：固定原有阈值和芳香环处理，冻结 OCR，预测接口不接收 GT 图/数量/框。按预测 MolBlock 的 canonical isomeric SMILES 比较完整化学身份，无效输出保留在分母，记录救回/改错/fallback。未新增立体识别能力，现有路径的缺陷会计为错误。
- `bundle.py`：将七组件全部 state 与架构配置一起保存，CPU 严格重新加载，不回退到旧视觉/原子/键权重；OCR 作为外部冻结依赖固定哈希。先写本次临时文件，完成后独占发布，已有文件不可覆盖；中途写坏不产生可用 checkpoint。

CPU 合成端到端测试实际执行了 3 个微型更新步骤、固定比例采样、完整身份评分接口、最佳权重选择和 patience 早停，并核对重新加载状态。这些测试的预测结果是合成控制，不是准确率实验。

真实旧权重另做了一次独立 CPU roundtrip：完整结构 state 一致、OCR 权重未变，乙醇样本保存前后均预测 `CCO`。产物是标记 `optimizer_steps=0` 的未训练副本，不是新训练模型；正式报告见 `outputs/audits/chemvector-curriculum-bundle-final-20260914/report.json`。

接口和准入合同见 [RUNNING.md](RUNNING.md)。前端默认配置、旧 checkpoint、旧数据和历史报告均不替换。

## 完整分子训练来源接入

`full_molecule_data.py` 从显式指定的 `style_train_100k` 读取训练来源，不导入旧联合数据生成器，不打开 fresh-dev/sealed-test 逐样本数据，不调用模型或 optimizer。

已有来源实际包含 25,000 个分子、100,000 张图；这不等于当前权重训练过全部图片。旧 tar 有图、ChemMol、render、layers、SDF，但没有新入口要求的 same-SVG 字符记录，不能仅改 schema 就当作同源文字监督。

此入口复用的是**旧训练分子的化学身份，重新生成完整分子图像**：

- 验证 train-only commitment、三种身份集合摘要、完整 samples.csv 哈希和四变体归属；非训练 commitment 在读取分子清单前拒绝。
- 从公开评测缓存中按完整 InChIKey 或连接层正向排除，再按来源 × 重原子区间分层、固定哈希顺序抽样；试点每个 core/scaffold 取一个结构。未解析评测记录继续记为缺口。
- 对选中结构所在训练 tar 验证完整 SHA256，并验证四变体中 ChemMol/SDF 的字节哈希和化学身份。清单、原结构 ID、原图片哈希和来源 commitment 保留在每个新样本的 provenance。
- 新图仍只有既有两种字体、九种样式和一个布局引擎。**不保留旧像素、16 字体覆盖、四布局和旧原子序号**；新图与新 SDF 按新序号一致，不能混称原图 replay。
- 这些是旧 train 身份，不是独立未见验证集。新数据继续 quarantine，不能被 launcher 用作正式训练或冒充训练准入报告。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-train/bin/python -m chemvector.training.curriculum.full_molecule_data \
  --source-root Data/sdf/chemvector_joint_v1/style_train_100k \
  --exclusion-index Data/sdf/chemvector_joint_v1/work/old_overlap_index.json \
  --output outputs/datasets/full-molecule-new-pilot \
  --molecules 200 --workers 8 --task-timeout 60
```

200 个结构是接入和复杂成图质量试点，不是最终训练规模，也不是已冻结混合比例。来源分层近似均衡，用于暴露问题，不代表原数据 ChEMBL/COCONUT 的 85/15 分布。

完整输出 `outputs/datasets/full-molecule-v0.2-rerender-20260914/`：1,800 张计划中通过 1,482、拒收 318；190 个结构至少一张通过，130 个结构九样式全通过，10 个结构无图。通过图 **1,482/1,482** 可转换为现有目标且无空间格冲突。数据继续隔离，复杂度/来源分层失败率与后续重点见 `docs/model/training/20260914-full-molecule-data.md`；本轮相关 CPU 回归 345 项通过。

初次试点遇到 `Degenerate auxiliary junction` 后中止，部分产物保留于 `full-molecule-v0.1-rerender-20260914`，不可当作完整数据集。对应几何是 RDKit 的辅助连接腿不足 0.5 px；新版 builder 拒收整张图片并记录该明确质量原因。文字/键归属不唯一、化学身份错误仍中止，不自动归为普通拒绝；归属器和原严格像素检查保持不变。

真实完整分子 CPU 反向检查见 `outputs/audits/chemvector-full-molecule-gradient-20260914/report.json`：19/24/59 个原子的三张参考样式均覆盖七组有限非零梯度；筛选后匹配原子分别为 19/23/59，可达正键分别为 20/23/63（目标 20/25/63）。权重 state 前后哈希一致，optimizer 步数 0、GPU 使用 0。这只验证旧结构组件能消费这些新标签，三张图按可用性选择，不能代表数据集准确率或泛化能力。

### 公开身份缓存缺口定位

独立只读探查重新核对 30,000 个 USPTO 公开源结构的 manifest SHA256，并重做解析/身份转换，没有读图片、修改源结构或重建缓存。记录见 `outputs/audits/chemvector-public-identity-gaps-20260914/report.json`（探查转录，不是准入报告）。

共 29,987 条得到合法身份键，**13 条未解决**：8 条解析/价态失败、3 条在 InChIKey 转换时发生 Kekulé 异常、2 条返回空字符串。旧 `indexed=29989` 把两条空键算入成功，旧 `unresolved=11` 因而不是所有缺口；空键集合只有一个元素，也不表示只有一条空键记录。未尝试强行改变价态、展开缩写或用未经复核的替代结构放行。

在旧 25,000 个 style-train 分子中，按连接层筛查发现 18 个公开缓存正向命中，本次完整分子选择已排除。当前负向筛查仍不能关闭这 13 条未解析记录及 fresh-eval 身份排除缺口。

### 立体笔画保护的成图修复

显式 `--typography stereo-aware-v1` 生成 `full-molecule-v0.3-stereo-aware-20260914`：同样 200 个结构 × 9 样式，合格图从 1,482 增至 1,712，救回 230、零退步；完全缺图的结构从 10 降到 2。全部新合格图通过目标适配，相关 CPU 回归 374 项通过。

修复虚楔短横线被当作普通键裁剪的问题，并有界平移碰撞文字；不改字号、化学图或锚点，全部受保护笔画与排版前向量证据逐图核对。文字位置/基线允许改变且有记录；旧模式和旧产物保留用于复现。完整结果及剩余 88 个失败变体见 `docs/model/training/20260914-stereo-aware-typography.md`。

这是数据成图质量提升，**不是识别准确率提升或训练准入**。没有启动 GPU、更新权重或关闭外部身份排重缺口。

### 独立数据侧核验请求

`review_request.py` 仅整理候选侧身份，不读取隐藏评估样本、不批准训练。当前本地请求位于 `outputs/audits/chemvector-identity-review-request-20260914/`：287 个候选分子 / 2,513 张图，12 个已知公开重合分子 / 108 张图列为排除项；三个跨来源共有骨架组需要统一分组。旧 train 完整分子不能作为未见验证集。

13 条公开身份异常中，9 份备用 SDF 的 MolBlock 与源文件逐字节相同并复现失败，另 4 条无备用 SDF；此路径未关闭任何缺口。当前需要获准的独立数据侧排重流程，而非改写 quarantine 标志。详见 `docs/model/training/20260914-data-admission-handoff.md`，含交接内容、访问边界与真实验证集依赖。新增请求和相关准入/发布回归 41 项通过，未启动 GPU。

用户后续已批准执行隔离核验：`private_review.py` 配合数据侧 `identity_review_worker.py` 完成固定 287 个候选对全部 3,000 个 fresh/sealed 分子的身份检查。最终凭据 `outputs/audits/chemvector-private-identity-review-20260914-conservative/receipt.json` 为 `checked_conservative`，候选完整键/连接层/保守骨架并集命中均为 0，未解析为 0；两条源骨架冲突保留两种分组共同排除，没有改动原标签。评估两部分的新连接层/骨架组交集为 1 / 3，因此还不能声称新口径下验证测试完全独立。完整流程及边界见 `Data/sdf/chemvector_joint_v1/IDENTITY_REVIEW.md`。这补充身份凭据，不自动完成公开身份准入或训练发布。

### 当前候选公开排除与验证输入解耦

`public_review.py` 逐条核对 30,000 份官方源结构，13 条身份异常保留；`ctab_inventory.py` 屏蔽全部不透明 SUP/alias 原子，仅用组外确定的非氢原子下界排除不可能同结构的候选。不能排除的 5 个完整分子 / 44 张图整体保守排除，余下 **282 个分子 / 2,469 张图**，凭据见 `outputs/audits/chemvector-public-candidate-review-20260914/`。这是当前候选的证据，不是修复完整公开身份索引或批准训练。剩余候选严格继承已核验子集，没有新查询隐藏数据。

评估器现在直接接受图像 + 单条 SDF（也保留旧 graph 真值），不再调用训练 collator；`pixels.py` 与训练共用逐位一致的原图归一化。训练目标门禁未改变，真实 legacy 验证 loader 与混合 release 尚待接入。相关完整回归 211 项通过，未启动 GPU。具体方法、限制和下一步见 `docs/model/training/20260914-public-candidate-screen.md`。

## 本轮验证

2026-09-14：以下 CPU 回归 **251 passed**。覆盖旧版行为、新版排版同源性、化学图/SDF 不变、子进程超时清理、现有 loss 消费和反向、身份筛查缺口及拒绝覆盖输出。人工抽看参考联系页及小字号同位素/Serif 硼酸原图，不声称全量人审通过。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-train/bin/python -m pytest -q \
  tests/data_sdf/test_curriculum_atoms_bonds.py \
  tests/data_sdf/test_curriculum_functional_groups.py \
  tests/data_sdf/test_curriculum_gallery.py \
  tests/data_sdf/test_curriculum_pipeline.py \
  tests/data_sdf/test_curriculum_typography.py \
  tests/chemvector/test_curriculum_data.py \
  tests/chemvector/test_curriculum_admission.py
```

续训增量的 `test_curriculum_continuation.py`、`test_curriculum_runtime.py` 共 25 项通过；与上述课程测试及被复用的 pair/role/bbox/spatial-atom/dense-bond/proposal-set 组件测试合并运行，最终 **301 passed（17.74 秒，CPU）**。实际未准入报告也验证了在 worker 启动和输出目录创建前被拒绝。超时测试使用短暂挂起的合成子进程，没有启动 GPU 任务。

启动器更新后再加入 `test_curriculum_release.py`、`test_curriculum_bundle.py`、`test_curriculum_launcher.py`，合并回归最终 **327 passed（28.66 秒，CPU）**。真实旧权重 roundtrip 的最终外层运行耗时约 3.67 秒，独立产物为约 74 MB 的未训练完整结构副本；原文件未覆盖、真实 optimizer 更新数为 0。

## 为什么不能直接运行旧键头脚本

当前默认模型由 v0.10.3 Direct AtomPair、v0.10.5 role、v0.10.8 bbox、v0.10.9 OCR 等已有组件组成。
本轮按前端登记哈希核验了这四份 selected checkpoint，并以 CPU map_location 读取元数据；它们对应阶段的 `samples_seen` 均为 10,000。它们继承更早训练，不能说整模型从来只见过 10,000 张图。

`train_direct_atom_pair_bond.py` 将视觉编码器、原子、旧键及 set 模块冻结，只训练 direct pair head。单纯给它换上课程数据不能使原子检测与视觉骨干共同学习。已有 joint proposal 训练器也冻结视觉骨干，且不是当前 Direct AtomPair 完整训练通路；不要把文件名中的 joint 当成已经满足目标。

上述续训内核已经沿用现有组件完成参数组和结构损失的 CPU 梯度覆盖检查。OCR 本轮不重训，数据中的字符记录用于结构区域监督和审计。CUDA 反向数值、显存和吞吐尚未验证，不能从 CPU 检查推出正式 GPU 训练无风险。

## 下一步与预算边界

1. 排版拒绝修复和组级提案已完成；下一步完成合法来源和评估重叠准入，扩充真正不同的结构/连接环境。新课程只是补充，完整分子样本持续参与，不靠重复这 101 个结构凑训练规模。
2. 启动器、训练/验证循环和整套权重保存加载已接通。下一步是提供真实获准的 mixed-data release，而不是再增加架构。当前支持统一 same-render target 格式；旧 WebDataset shards 不能直接塞进清单，必须先经过独立且获准的数据适配/发布。冻结数据配方、训练范围、指标和停止条件后才训练，不进行架构或超参数搜索。
3. 首轮预算建议保持 12 GPU 小时（准备/吞吐 1、主训练 9、验证 2）；这是上限，不是吞吐或效果承诺。本次实际 GPU 训练用量为 0。
4. 完整 SDF/图正确率为终点，同时记录救回、改错、无效输出和耗时。反复使用的 200 图只用于诊断，不能当新独立验证集。

**尚未完成**：完整评估排除准入、真实训练/验证混合清单发布、完整分子来源适配与化学环境扩量、GPU 数值/资源检查和正式训练。外层时限与内部步数保护已接入任务入口并有 CPU 验证，但不是已完成的 GPU 训练试验。因此本批数据仍未准入训练，不能把工程链路通过解释成准确率提高。
