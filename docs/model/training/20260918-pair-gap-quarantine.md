# N–N / N–S 基础教材补缺：18 张隔离图已生成

**18 个固定新身份各生成一张 reference 图，18/18 生成成功、18/18 通过目标适配、零拒收。** 另引用 7 张既有训练控制，25 图检查清单满足本次所需的正/负例和四种键型覆盖。整批生成及适配 CPU 用时 15.582 秒，模型前向/优化器/GPU 均为零。

新数据仍为 `quarantine / training_authorized=false / external_overlap_status=unchecked`。**这不是新发布训练集，不是准确率结果，也还不能开始训练。** 原 initial 和已有训练/验证清单未改变；完整开发 170/800、clean16 的 8/16 均保持原记录。

## 本轮多人协作与范围

三个 subagent 分别负责候选化学与本地排重审查、旧 renderer/适配流程复用及合成测试、单组件监督与生成权限边界复核。主 agent 整合固定候选、实现生成入口、执行生成、查看图像和运行回归。

采用审查提出的三项具体改进：

1. 除各 8 个 N–N/N–S 单键正例，再固定 2 个 N/S 共存但不直接相连的控制，避免把元素对当作连键规则。
2. 复用已有 DOUBLE/TRIPLE/AROMATIC 控制，因为 pair loss 同时训练 presence 与四分类 chemical 头，不应把新面板变成全单键训练。
3. 追加旧输入哈希时，禁止覆盖生成前绑定的原摘要；若旧控制 metadata 中途变化立即失败。这个完整性缺口在实际生成前已修正并加测试，未发现实际旧数据变化。

不改变现有模型、OCR、默认推理或训练接线，不扩展 MoE/循环架构，不进行模型结果筛选、自动重试或样式扫描。

## 固定候选与控制

所有新分子都是手工指定的中性、单连通、无同位素/自由基/原子映射/指定立体信息的 3–7 节点分子；总节点数与重原子数相同，不借显式 H 绕过范围。

| 用途 | 固定 canonical SMILES |
| --- | --- |
| 8 个 N–N 单键正例 | `CNN`、`CCNN`、`CNNC`、`CN(C)N`、`CC(=O)NN`、`NNC(N)=O`、`NNC=O`、`COC(=O)NN` |
| 8 个 N–S 单键正例 | `CS(N)(=O)=O`、`CCS(N)(=O)=O`、`CCCS(N)(=O)=O`、`CCS(=O)(=O)NC`、`NS(N)(=O)=O`、`CNS(N)(=O)=O`、`CNS(=O)(=O)NC`、`CN(C)S(N)(=O)=O` |
| 2 个 N/S 非边控制 | `NCCS`、`NC(N)=S` |
| 7 个既有训练身份，只读引用 | `OO`、`CS(C)(=O)=O`、`CCS(C)(=O)=O`、`N=N`、`C#C`、`c1ccncc1`、`CCN` |

N–S 正例只覆盖本批磺酰胺/氨磺酰胺环境，不代表所有硫氧化态。`NCCS` 的 N/S 图距离为 3；硫脲中 N/S 图距离为 2，都没有直接 N–S。氨基脲和氨磺酰胺还包含 N/N 非边，不能看到两个 N 就连单键。

本地只读排重基于当前发布清单的 14,095 条记录 / 2,389 个 canonical 身份，覆盖已发布 train 和已暴露 validation。18 个候选与该清单的 full InChIKey、连接层前缀及项目 group_key 均无重合。**没有读取外部身份索引或封存内容，也没有调用身份服务。** 零本地重合不等于外部准入；无环 group 使用完整 core，不代表官能团或近邻化学环境完全不重合。

## 成图与验收

复用 `catalog.merged_records()`、原 `build()`、`Profile()` 和 `ink-aware-v1`，没有另造 renderer。固定配置：768×768、白底黑线、DejaVu Sans 24 px、2 px 线宽、无旋转；每身份只生成一个参考样式。全部任务在读取生成结果前写入 plan，失败不替换候选或换样式。

每图保留 PNG、SVG、graph、same-render layers、字符记录、SDF、metadata 和分层 mask。原 preflight 核对同源哈希、像素/标注绑定、SDF 身份往返、原子/键索引、目标容量及空间格碰撞。**18/18 通过，空间格碰撞为零。** 实际调用默认训练 loader 18 次，全部因未准入而拒绝。

已查看含全部 18 个分子的联系页，并抽看 `CNN` 的原始 768px 图。联系页为前景裁剪缩略预览，仅供人看；模型输入文件未裁剪、未放大或重写。自动成图检查通过和预览检查不等于真实域风格覆盖，也不是所有原图的逐像素人工审查。

### 25 图清单的标签覆盖

下面是新隔离图与既有控制合并后的标签统计，类别相互重叠，不能相加当作图片总数。

| 目标/控制 | 分子数 | 边/非边实例 |
| --- | ---: | ---: |
| N–N 单键 | 8 | 8 |
| N–S 单键 | 8 | 12 |
| C–N 单键 | 15 | 21 |
| 真实 O–O 单键 | 1 | 1 |
| N/N 确认非边 | 6 | 7 |
| N/S 确认非边 | 2 | 3 |
| O–S–O 两端跨中间 S 的 O/O 非边 | 10 | 10 |
| SINGLE | 22 | 62 |
| DOUBLE | 16 | 26 |
| TRIPLE | 1 | 1 |
| AROMATIC | 1 | 6 |

O–S–O 几何见证来自实际生成后的像素锚点：共享中间邻居位于端点线段 10%–90%，垂距 ≤2 px；不是仅凭 SMILES 或两跳路径推断共线。未把该规则接入预测，也未修改旧图的标签。

当前两个标签覆盖检查成立：`all_planned_targets_compatible=true`、`minimal_target_coverage_passed=true`。预测候选能否覆盖这些目标尚未检查，`predicted_reachability_checked=false`；`training_ready=false`。

## 产物与下一步

- 图像及标签：`outputs/datasets/chemvector-pair-gap-quarantine-20260918/`。
- [图片总览](../../../outputs/datasets/chemvector-pair-gap-quarantine-20260918/reference-contact-01.png)；[离线图册与原图/标签入口](../../../outputs/datasets/chemvector-pair-gap-quarantine-20260918/index.html)。
- 审计：`outputs/audits/chemvector-pair-gap-generation-20260918/`，含 plan、report、runtime、18 图适配记录、25 图 target coverage、mixed-origin inspection panel。
- `candidate-identities.json` 固定了本批 18 个候选的身份交接内容，状态为 `not_submitted`；不是排重凭据。旧批次一次性授权不适用于本批。

下一步需要本批新的隔离身份核验授权：数据侧只读评估排除身份，仅返回重合候选和统计，模型侧不开放封存内容，也不授予训练许可。完成本批公开/私有准入和分组审查后，再固定训练侧清单、检查 initial 的预测候选可达性，最后另行确认单组件预算和验收条件。不能把本轮 18 张图或 25 图检查清单直接作为已获准的训练 manifest。

这批数据是小规模基础能力验收教材，不是用 18 张替代完整训练集；先确认缺项补对，再考虑每种连接环境的身份和样式扩量。

## 核验记录

专项合成与相邻测试 **91 passed / 1.07 秒**。其中 subagent 编写的 45 项覆盖固定候选、纯本地身份检查、控制选择和正负键统计；另有入口预算/禁止覆盖/哈希承诺测试。测试数量有包含关系，不相加。

较广相关回归 **634 passed / 56.50 秒**，覆盖 curriculum、pair/role/bbox/OCR、推理/编译及原课程 renderer、排版、图册；包含本轮新增的 57 项测试，不是全仓库测试声明。独立 subagent 复核了固定候选、7 个旧训练控制、19 个运行前绑定输入、61 个最终绑定输入、198 个新样本文件哈希，以及从 graph/真实锚点重算的全部 11 项覆盖，均一致。主 agent 另核对原 initial checkpoint、旧 clean16 及 pair-readiness 报告哈希未变。

- plan SHA256：`ef40075f2db8346c24de1da6f84c1d0e168ea004680a5c1a7b562ef61542ae20`。
- report SHA256：`1b97e1df97abaebc5a4a3c904c925236f27330f68c8431f38e6a9c002140cea8`。
- runtime SHA256：`e9313ae12db5171546932619e28253c0f59f5b1f803a8530ce5a57ff89c6c97b`。

前序：[基础错误](20260918-clean16-basic-errors.md)、[连键专项准备与数据缺口](20260918-pair-stage-readiness.md)。
