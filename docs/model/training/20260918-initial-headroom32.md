# 32 图 initial 收益空间验收：先对齐监督与输出，再申请训练

已完成第 2 步的**训练前检查**：32 个已准入完整分子训练样本，CPU 用时 **42.775 秒**，上限 300 秒；GPU、优化器更新和自动重试均为零。没有进入单组件试训，保留原 initial 和固定开发集 **170/800** 基线。

本次发现两项具体障碍：当前 direct 预测链路没有给出立体信息；局部监督评分与最终化学身份存在不一致。它们支持先修清接口契约，**不支持立即重训 pair 或 bbox，也不支持把历史退步全部归因于立体化学**。

## 范围与原定结果

仅从已发布清单的 `train/full_molecule` 中按固定哈希顺序选取 32 张图，`molecule_id` 与 `canonical_smiles` 均不重复。选择种子为 `20260918-initial-headroom-v1`，在加载模型和图片前保存 plan；没有看预测后换图，也没有新身份核验或封存数据访问。

原 initial、冻结 OCR、原阈值和默认编译流程保持不变。真值只用于预测后的评分，不提供 GT 候选、GT 裁框或图修复。32 次预测另加第 0、15、31 张的默认路径一致性回放，共 35 次上游前向；3 次回放全部一致。

| 原定观测项 | 结果 |
| --- | ---: |
| 严格 SDF 身份正确 | 8/32 |
| 有效 SDF | 27/32 |
| 使用 fallback | 3/32 |
| selected 与目标在原 6 px 匹配规则下一一对应 | 20/32 |
| 固定候选上的局部 pair 任务完全匹配 | 6/32 |
| 角色入口与 OCR 字面标签完全匹配 | 12/32 |
| 满足严格 pair 单组件筛选条件 | 0/32 |
| 需复核的文字接口候选 | 2/32 |
| 足以支持 bbox 训练的案例 | 0/32 |

**8/32 是训练样本诊断，不是新开发集准确率，不能与 170/800 直接比较。** 此处的“局部 pair 完全匹配”也不是图同构身份评分。

合计 991 个目标原子、1,069 根目标键；raw 匹配 981 个原子，selected 匹配 970 个，selected 总数 987 个，17 个 selected 未匹配；可达正键 1,034 根。不可达的 35 根仍留在目标分母。几何匹配失败不等于图像中的真实原子必然没有被预测。

样式分布为 serif 5、font_small 8、font_large 7、line_thick 7、reference 2、bond_short 2、line_thin 1；没有为均衡样式重新抽样。

## 保存结果的事后分析：立体信息确实缺失，但不是全部原因

以下分析只读取已保存的 32 份结果；不是事先登记的新评分，也没有重新推理。原始 `report.json`、严格 SDF 指标和分母均不修改。

| 目标是否含指定立体信息 | 图数 | 严格身份正确 | 仅在分析时移除双方立体信息后的匹配数 |
| --- | ---: | ---: | ---: |
| 否 | 9 | 8 | 8 |
| 是 | 23 | 0 | 3 |
| 合计 | 32 | 8 | 11 |

事后比较用 RDKit 解析保存的 SMILES（`removeHs=False`），在副本上 `RemoveStereochemistry`，再以 `canonical=True, isomericSmiles=True` 输出；保留同位素、电荷等身份字段。目标立体分组与保存的对齐图标记逐图一致。无效预测继续计错。

保存的编译前 scene 中，987 个原子全部为 `CHI_UNSPECIFIED`；1,026 根键全部为 `STEREONONE`、`direction=NONE`。保存的最终有效 SDF 身份中也没有指定原子/键立体信息。代码对应：

- `chemvector/direct_scene_inference.py:256` 的 `direct_scene_tensors()` 只将 pair presence 和 chemical logits 转成编译张量，display/chemical 使用同一分布，没有传递独立的立体预测。
- `chemvector/training/curriculum/evaluation.py:95` 提供的是空占位 scene，不是标签；`shadow_chemscene_v02_oracle.py:90` 的默认原子手性未指定，`:450` 的键方向和立体字段固定为 NONE/STEREONONE。
- `chemvector/direct_scene_inference.py:81` 的 OCR 属性写回也使用 `CHI_UNSPECIFIED`。

**这是当前预测链路的能力缺口，不是 ChemScene/SDF 数据格式不支持立体化学。** `Data/sdf/chemscene_v0_2.py` 已有立体字段、引用检查和 roundtrip 通路；也不能假定只改一个导出常量就能从图片恢复正确手性。

三张严格失败、去立体后相同的图为：

- `curr_3b3fc1cbb2cc2063__bond_short`
- `curr_9ca66e07763eb2e3__font_small`
- `curr_26004e1f8c9f19a8__font_small`

因此 24 张失败中只有 3 张在本次比较下仅剩立体差异；另外 21 张仍有非立体身份错误或无效输出。23 张含立体目标中的 20 张还存在其他错误。不能说“补立体就都能对”，也不能用这项静态缺口解释 initial→last 的全部退步。

### 为什么严格 pair 候选是零

原定筛选逐层留下：错误图 24 → 候选一一对应 14 → 角色/文字字面路径正确 7 → 基础原子字段正确 7 → 无目标立体信息 **0** → 孤立 pair 错误 0。

七张其余条件较干净的错误图都带立体目标，而当前 pair 任务不能单独补出它。**零候选不是“pair 没有错误或永远不值得训”的证明**，只是这批图不满足本轮“只改 pair 有望恢复严格身份”的保守验收条件。基础字段只含元素、电荷、同位素、自由基，不包含氢计数、芳香性或完整立体语义，筛选通过本来也不保证修复成功。

## 监督代理与最终身份不一致的见证

8 张严格正确的图中，**5 张仍被局部 pair 指标标为错误**：

- `curr_f0fecfaa0d52598b__font_small`、`curr_02a159288a942d09__serif`、`curr_b2cf85f40af040bb__line_thick`：局部 SINGLE/DOUBLE/AROMATIC 标签不一致，但经过现有芳香一致性处理及 SDF 编译后身份相同。这不代表所有芳香键误判都可忽略。
- `curr_6de40b2274fd34b2__serif`、`curr_084b60e449eb413a__font_large`：selected 原子总数与 GT 相同，但原 6 px 匹配下少匹配一个文字原子；相连的预测边因此被标成 extra，最终身份却正确。这不证明任意未匹配候选都应强行匹配。

另有严格正确图 `curr_e1d661d8e18e9de9__line_thick` 的 `HN→NH` 被字面评分记为差异。字符串不相等与化学属性错误必须分开。

当前 `build_pair_targets()` 将含未匹配端点的 pair 留为负标签，值得在后续监督契约中明确“确认负例”和“对应关系未知”的区别。但**本轮没有负样本采样、反向传播或优化器更新**，未证明上述边一定进入了历史训练损失，更未证明这些代理差异就是全部退步根因。

## 两个文字接口候选不等于两个 bbox 故障

- `curr_120c681aa0931c96__reference`：候选和局部 pair 任务正确，`H2N→PH2`，编译元素 N→P；预测框与文字框 IoU 约 0.804。去立体后仍错，存在真实属性问题，但没有受控裁框对照，不能区分 bbox 与 OCR 责任。
- `curr_26004e1f8c9f19a8__font_small`：`HN→NH`，IoU 约 0.700，基础原子字段相同，去立体后身份正确。不能据此认定化学 OCR 错误或批准 bbox 训练。

## 下一步：有限的 CPU 契约修补，不先开 GPU

本轮验收到此结束，不追加另一批图，也不轮流试训各头。继续使用自研模型，不改成 MoE，不换 MolGrapher，不重训 OCR。

建议顺序：

1. **先统一监督语义。** 用上述保存见证和合成小图建立契约测试，区分显示键型与规范化学键型、字面 OCR 与解析属性、未匹配端点与确认负键。先明确目标标签和匹配规则，再提出最小实现修改；不能简单放宽半径、抹掉错误或改低正式评分标准。此步不需要新模型推理。
2. **补齐必要输出的设计与 roundtrip 测试。** 明确 wedge/dash 的端点方向、原子手性顺序以及 E/Z 引用如何从预测进入 scene/SDF；用合成图验证重排、坐标变化和往返不丢语义。这是表示/编译验收，不是模型已能识别立体化学的证明，也不是本轮已经实现的新立体头。
3. 上述契约可用后，才定义一个可表达、可达的单组件学习目标，重新申请有上限的小训练。原固定 800 图严格身份仍是后续验证主指标；非立体/立体分项只能解释结果，不能替换主分数或制造改善。

不再把“继续喂更多图、局部 loss 下降”当作整图必然进步的依据；本轮也没有否定用户提出的阶段式课程路线。

## 实现、验证与产物

新增 `chemvector/training/curriculum/headroom_metrics.py`、`headroom_audit.py` 及 `tests/chemvector/test_curriculum_headroom.py`。本轮不修改生产推理、原 checkpoint、发布清单或前端，不新增训练入口。

- 新增 21 个参数化测试用例；专项与相邻测试 **58 passed / 1.00 秒**；curriculum + direct pair/role/bbox 回归 **370 passed / 46.94 秒**。两组有重叠，不相加，也不是全仓库测试声明。
- 独立保存产物复核通过：固定选择、32 份结果重新汇总、严格 molblock 身份重算、全部 43 个绑定输入 SHA256、initial 文件 SHA256、3 次回放标记、状态前后一致、CPU 限额和零训练记录。
- initial state：`9d7f66f79329479e05c38f8b91d95118adb9f12c32a44e5bd9d83a458a8b452e`；OCR state：`2882fc3cd8b24e6fb1124202f3dcf9fd7ff802f5db9cdc1020e26d0e4edb8d9f`。
- 目录 `outputs/audits/chemvector-initial-headroom32-20260918/` 包含原 plan、32 份 case、report、worker.log、runtime；独立的 `postcheck.json` 保存本节事后复核，不覆盖原 report。
- plan SHA256：`2b5aedf18a4ab18613c19ca33a658a3471aaec2dff4cda73fa913ff5ce4d2aac`。
- report SHA256：`80787a19a7fbfb526a568de228e722f492068fd946cb77e5ef0942a80febaa68`。
- runtime SHA256：`0bbfeb84f2bf76d3f262c96e81a79b8086fd5d738b830dd37c8d60491804bf63`。

前序依据：[低算力路线讨论](../../research/20260918-low-budget-training-review.md)、[第 1 步接口定位](20260918-upstream-interface-diagnosis.md)。
