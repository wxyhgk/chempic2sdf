# N–N / N–S 补缺：固定候选身份排重完成

**本批 18 个固定候选完成一次获准的隔离身份核验；私有侧未命中，公开侧排除 1 个，剩余 17 个。** 与 7 张既有训练控制组成的 24 图诊断准备清单仍满足基础标签覆盖。未训练、未做模型推理、未使用 GPU；不是准确率提升或训练发布。

## 一次性核验与公开排除

用户针对本批 18 个候选的身份核验询问回复“继续推进”。[授权记录](releases/20260918-pair-gap-identity-authorization.json)绑定固定请求、候选和实现哈希，只允许一次隔离数据侧身份比较，不开放封存样本、不授予训练许可。本次已执行并消耗该授权。

`pair_gap_review_request.py` 组装原始 18 个候选，未按公开结果预先删减私有请求；`pair_gap_identity_review.py` 在调用未改动的 runner/worker 前独占创建并 fsync `private-attempt.json`。即使中断且未生成回执，也不自动重试或回收额度。

- 私有侧：覆盖 3,000 个评估身份，完整 InChIKey、连接层、保守 group 比较的候选重合均为 **0**，未解析数为 **0**；状态 `checked_conservative`，CPU 2.917 秒。
- 原有 2 处 group 绑定不一致通过既有保守并集规则处理，未修复源数据。回执仍报告评估内部跨 split 的连接层/group 重合；候选无重合不等于原评估隔离问题已解决。
- 公开侧：核对 30,000 条 USPTO 源记录及现有公开缓存，继承上述回执，无第二次私有查询；CPU 16.233 秒。
- 29,987 条公开记录正常解析。原有 13 条全局未解析记录继续保留，针对固定候选的 **234 = 13 × 18** 对均由元素数量下界证据排除可能重合，剩余歧义对为 **0**；不宣称修复了全局身份缺口。

唯一排除项为甲磺酰胺 `CS(N)(=O)=O`，候选 ID `curr_8addb0acfb426fe9`。其公开命中为 `uspto_10k_US20200361876A1-20201119-C00005`，旧公开缓存亦为正向命中。这是实际身份重合，不是仅因未解析而保守排除。

剩余 **8 个 N–N 单键正例、7 个 N–S 单键正例、2 个 N/S 非边控制**。没有补选新分子；原 18 图 quarantine、原 25 图检查清单和历史 `not_submitted` 交接文件保留不变，排除只体现在新清单中，不删除图片。

## 排除后的 24 图覆盖

[筛选后清单](../../../outputs/audits/chemvector-pair-gap-public-identity-20260918/screened-panel.json)绑定请求、回执、公开结果和原生成证据。它包含 17 张新隔离图及 7 张原训练控制，**不是训练 manifest**。下列类别互有重叠，不能相加当作图片总数。

| 目标/控制 | 分子数 | 边/非边实例 |
| --- | ---: | ---: |
| N–N SINGLE | 8 | 8 |
| N–S SINGLE | 7 | 11 |
| C–N SINGLE | 15 | 21 |
| O–O SINGLE | 1 | 1 |
| N–N NONEDGE | 6 | 7 |
| N–S NONEDGE | 2 | 3 |
| O–S–O 两端 O/O 跨中间 S 的非边 | 9 | 9 |
| SINGLE | 21 | 60 |
| DOUBLE | 15 | 24 |
| TRIPLE | 1 | 1 |
| AROMATIC | 1 | 6 |

`minimal_target_coverage_passed=true`，但 `predicted_reachability_checked=false`、`training_authorized=false`、`training_ready=false`。标签中包含正确键，不代表模型实际预测的节点足以支持这些键。

## 验证与不变项

- 请求、授权、单次预占和回执哈希一致；剩余 17 与排除 1 严格构成原 18 个候选，没有替换。
- 既有 `release_evidence.verify_reviews()` 接受该回执链及精确剩余子集；这只验证身份材料，不授予完整训练准入。
- 请求/生成绑定输入及全部 **198 个生成 payload 文件**哈希不变。封存目录权限保持 **000**，未读取私有图片/SDF/tar，未改 split。
- 原 `initial.pt` SHA256 仍为 `09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`；既有 expanded release manifest 仍为 `67ccb1307f7af3ca19f16ee6fd900d1a2e8c3c62dd0b9c0ec28704ddd93aa05e`。
- 新增 **21 项合成测试**覆盖固定候选、输入绑定、一次性预占、中断后禁止重跑和 CPU 入口。测试中的私有调用均为 mock。
- 专项相关测试 **65 passed / 0.99 秒**；较广相关回归 **655 passed / 56.72 秒**，包括 curriculum、pair/role/bbox/OCR、推理接口和课程生成/排版/图册。两组数量有包含关系，不相加，也不是全仓库测试声明。

模型前向 **0**、optimizer steps **0**、GPU **0**、自动重试 **0**。完整开发集 **170/800** 和 clean16 **8/16** 仍为此前 initial 的记录，本轮没有新的识别准确率。

## 产物与下一步

- [固定请求](../../../outputs/audits/chemvector-pair-gap-review-request-20260918/request.json)，请求目录内保存已消耗的 `private-attempt.json`。
- [私有侧允许输出的回执](../../../outputs/audits/chemvector-pair-gap-private-identity-20260918/receipt/receipt.json)。
- [公开侧排重报告](../../../outputs/audits/chemvector-pair-gap-public-identity-20260918/review/report.json)，同目录保留剩余/排除候选及未解析记录的逐对证据。
- [24 图准备清单](../../../outputs/audits/chemvector-pair-gap-public-identity-20260918/screened-panel.json)。

下一步准备固定 **17 + 7** 图、只用 initial 的 CPU 候选可达性诊断：明确 quarantine 诊断范围，保持原标签，不注入 GT 节点；区分“原子没找到”“筛选丢端点”和“端点齐全但键判错”。本轮未启动该诊断。现有仅允许 admitted-train 的诊断入口不能通过伪造准入标志复用。

先根据可达性判断连键专项是否有足够有效监督，再完成来源用途、分组与正式数据发布审查，最后单独确定训练预算和验收条件。一次身份核验不授权训练，也不自动续用旧 GPU 额度。

前序：[18 图生成与历史 25 图覆盖](20260918-pair-gap-quarantine.md)、[连键专项准备](20260918-pair-stage-readiness.md)、[clean16 基础错误](20260918-clean16-basic-errors.md)。
