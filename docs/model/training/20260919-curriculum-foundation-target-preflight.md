# 基础扩量：筛选后目标适配与固定身份核验申请

本轮承接 [基础数据扩量](20260919-curriculum-foundation-expansion.md)，只做训练前数据准备。
不改变自研 ChemVector 架构，不启动训练、OCR 更新、模型前向或公开/私有身份查询。
上批已经消耗的单次身份核验授权不适用于本批新增结构。

## 固定分母与标签检查

全量目标检查仅处理既有内部质量交接中的 **20,289 张不同 PNG/RGB、3,468 个结构**。
原始 20,904 图中，96 张冲突身份图和 519 张同身份重复图保持原文件不变、也不进入 collator。
检查前复算原始 PNG/RGB 冲突与既定筛选政策，不重新挑选候选。

逐图使用未修改的 `load_sample(purpose="diagnostic")` 和 `CurriculumCollator`：

- 核验源 manifest、语义图、SVG/PNG、文字记录和样本元数据的哈希及同源合同。
- 保留原始 768×768 RGB，核验实际 tensor 形状、dtype、CPU 设备、有限值与范围。
- 默认 query 容量为原子 128、键 128；核验目标数量与 mask、192×192 空间格碰撞。
- 原子 ID 与完整化学图保持不变；键端点按绘图方向引用正确原子槽。
- 化学键类型、绘图键类型和 wedge/dash 等方向分开核验，不把芳香性当成单双线画法。
- 每张图分别验证训练 loader 和训练 collator 仍拒绝隔离数据，包括只篡改内存 purpose 的情况。

新增检查器绑定启动时的源代码和输入哈希，结束再次核验。逐图日志保留失败项，
损坏产物立即失败，不把损坏悄悄转换成新的“排除候选”；不完整执行不能产生通过结论。
内部 CPU 上限 900 秒，由临时 user service 提供 960 秒外层硬上限及 10 秒终止宽限。
没有自动重试，也不把“进程消失”当作成功。

## 结果与产物

**全量完成：20,289 / 20,289 图通过，3,468 个结构，0 失败，CPU 耗时 297.545 秒。**
最终状态为 `passed_target_compatibility_only`，不是训练准入或准确率提升。
GPU、模型前向、优化器更新、外部身份比较均为 0；临时 service 已结束。

| 检查项 | 结果 |
| --- | ---: |
| 原子目标实例 | 168,216 |
| 有可见文字的原子目标 | 44,622 |
| 键目标实例 | 158,466 |
| 空间同格原子碰撞 | 0 |
| 单图最大原子 / 键数 | 16 / 17 |
| 训练 loader 拒绝隔离样本 | 20,289 / 20,289 |
| 训练 collator 拒绝隔离样本 | 20,289 / 20,289 |

化学键实例：SINGLE **109,506**、DOUBLE **10,485**、TRIPLE **3,561**、AROMATIC **34,914**。
绘图类型另计：单线 127,650、双线 27,255、三线 3,561；其中实楔、虚楔各 1,632 条。
这几组计数不是可互加的独立样本；楔形方向与化学单键分别占各自标签字段。
元素仍覆盖 C/N/O/S/P/B/Si/F/Cl/Br/I。全部是基础/局部环境，不能声称已覆盖大分子训练。

独立重算逐图日志的样本顺序、原子/键总数、最大 query 用量及完整分母，
并复核 24 个预检输入绑定，全部一致。预检报告 SHA-256：
`3b599c7d2165c60295d928ea7fdbbe72a40a3d76c37eaf289c35c3696136003c`。

- [全量预检计划](../../../outputs/audits/curriculum-v02-target-preflight-20260919/plan.json)
- [逐图日志](../../../outputs/audits/curriculum-v02-target-preflight-20260919/samples.jsonl)
- [最终报告](../../../outputs/audits/curriculum-v02-target-preflight-20260919/report.json)
- [运行输出](../../../outputs/audits/curriculum-v02-target-preflight-20260919.log)

## 身份核验申请的边界

只有完整预检全部通过，才从完全相同的候选名单生成 `candidate-identities.json` 和 `request.json`。
申请绑定质量交接、预检报告/逐图日志、原始 manifest、身份算法、RDKit 版本和评估集 aggregate commitment。
只读取这一份 aggregate commitment，不访问其列出的评估样本路径、封存结构或身份清单。

请求的比较规则沿用 full InChIKey、连接前缀和去立体/去同位素 core/group。
申请固定后不根据私有比较结果补换分子。返回范围仍限定为本批候选的排除 ID 和计数，
不允许返回私有身份、结构、图片或工作日志。**生成申请不等于获得授权或已做排重。**

申请已生成并复核，固定 **3,468 个候选结构 / 20,289 张图**，没有再筛掉或增加候选。
29 个输入绑定及候选字节哈希均已复核；状态 `not_granted_for_this_batch`。

- [本批固定申请](../../../outputs/audits/curriculum-v02-identity-review-request-20260919/request.json)
- [仅本批候选身份](../../../outputs/audits/curriculum-v02-identity-review-request-20260919/candidate-identities.json)
- request SHA-256：`0a74762ff63af549beb52978c783c547e2a31df67ac8656ed7c7dd9e2c5eee58`
- candidate SHA-256：`cedcb3700a0bd9082227799a746e74c232f254029c224fd071ba5600cedd389d`

没有创建授权记录、调用身份核验 worker、读取公开排重缓存或打开 fresh-dev/sealed-test 样本。
仍无 train/val/test 划分、发布训练清单或训练预算变更。

## 后续训练不能原样复用 presence-only 配方

这批数据包含单双三键、芳香键及立体画法，但本轮没有测量模型的预测候选端点可达性，
也没有训练准确率结果。GT 中存在某条三键不意味着当前 selector 保留了它的两个端点。

后续顺序建议：

1. 对本批固定候选取得一次性隔离身份核验授权，完成独立身份准入和分组划分，再发布数据。
2. 使用固定、覆盖不同键型的已准入切片测量真实预测候选可达性；未知端点对应继续排除出负例标签。
   不机械限定 `reference` 样式：去重后有 173 个无文字身份的 reference 已由同像素的其他样式代表，
   reference-only 抽取会丢掉这些基础碳骨架环境。
3. 根据可达性准备定额试训：把“是否连键”和“连什么键”分开统计、按任务选择可训练参数，
   同时回放既有完整分子；不能把上一轮只更新 257 个 presence 参数的配方复制过来声称学会键类型。
4. 试训需要另外确定预算与授权；仍从 `initial` 出发，不自动晋升之前完整分子回归退步的 checkpoint。

现有适配器保留了元素、电荷、同位素及完整化学图，但并未因此新增 OCR 转写、
元素/电荷/同位素分类或官能团分类损失。更多原子种类也不能被解释为这些分类能力已经学到。

## 实现与测试

- [筛选候选目标预检](../../../tools/preflight_curriculum_v02_targets.py)
- [固定身份申请](../../../tools/prepare_curriculum_v02_identity_request.py)
- [合成契约及集成测试](../../../tests/data_sdf/test_curriculum_v2_target_preflight.py)

新预检/申请测试与既有质量交接、目标适配测试合并执行：**69 passed / 28.62 秒**。
包括名单增删/重复、标签/引用错配、坐标碰撞、query 溢出、非有限 tensor、隔离拒绝、
中断/不完整状态、候选分母保留、源文件不变以及只有 aggregate 输入的申请打包。
这不是全仓测试，也不是模型性能评估。
