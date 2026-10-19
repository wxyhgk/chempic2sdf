# CPU 接口定位：上游输出变化与部分裁框退步，不继续六组联合更新

已完成约定的第 1 步：一次 CPU 诊断，**64.357 秒**，外部上限 300 秒；GPU=0、optimizer steps=0、自动重试=0。没有训练、换阈值、扩样或部署。

结论：在固定诊断样本中，**仅改变 set 筛选头的权重不能解释主要整图退步；上游特征/候选输出的改变足以令原有下游失效，bbox 变化另有局部因果证据。** 这不等于全部 800 张开发图都已归因，也未区分 vision、atom、bond 三者各自的责任。

## 固定范围与对照

沿用旧 probe 已固定的 12 张开发图（包括控制）与 12 张训练诊断图（8 张完整分子、4 张课程图），不根据新结果选图。原清单来自 `outputs/audits/chemvector-freeze-regression-probe-20260915/worker.log` 首条 JSON。

本次权重是 selected 标签修复后试训的 initial/last。两者 role state 完全相同，统一使用 initial role 与原冻结 OCR。开发图只读已发布原 PNG/SDF；不使用 SDF 坐标冒充像素标注。训练诊断图以 diagnostic 用途加载已有同源图像标注，只做推理后覆盖评分，不向模型注入真值。

生成部分 G = vision + atom + bond；筛选部分 S = set_head。每张图只计算两份 G 输出，然后在完全相同的缓存输入上比较 initial/last S。下游直接复用现有 pair/文字/编译流程。

| 对照 | G | S | pair | bbox | 严格 SDF 正确 /12 |
| --- | --- | --- | --- | --- | ---: |
| initial 原样 | initial | initial | initial | initial | 10 |
| 只换筛选头 | initial | last | initial | initial | 10 |
| 只换生成部分 | last | initial | initial | initial | 5 |
| 换整个上游 | last | last | initial | initial | 5 |
| last 仅回退 bbox | last | last | last | initial | 6 |
| last 原样 | last | last | last | last | 4 |

这是历史错误富集的诊断集，**不是新的准确率估计**。当前这 12 张的类别是 7 张新增错误、3 张保持正确、1 张救回、1 张始终错误；没有沿用旧权重的分类冒充本轮分类。全部 24 次 initial/last 原样回放与已有保存预测的身份、valid、exact、fallback 一致。

## 1. 筛选头权重不是这组案例的主要解释

固定 initial G，换 S 后 12/12 图的 selected mask 完全相同，整图仍为 10/12。固定 last G，换 S 只有 `fresh_dev_000953_original` 改变了两个候选选择，selected 数从 24 到 26；该图两组都错误，总体均为 5/12。

因此不能继续用“set 头被训坏了”概括本轮退步；但也不能说筛选决策无关。它依赖上游特征、分数和几何，上游改变时，即使 S 的权重完全不动，决策仍会改变。

### 具体见证：位置仍在，但同一个旧筛选头拒绝了它

`fresh_dev_000263_rdkit` 的 selected 数从 19 到 18。比较两份模型预测的位置（不是 GT 位置）：

| 输入 | 候选索引 | proposal 分数 | 初始 S 给出的 logit | 是否保留 |
| --- | ---: | ---: | ---: | --- |
| initial G | 27 | 0.243132 | −0.028051 | 是 |
| last G | 24 | 0.182480 | −0.689143 | 否 |

两个预测位置仅相距 **0.369 px**；阈值仍为 −0.25。last S 在后一位置给出 −0.662392，也拒绝。故该例不是“那个位置在 raw 候选中完全消失”，而是上游输出改变后，原有筛选器也不再保留附近候选。此反事实没有进一步分开特征、分数、位置和上下文的贡献，也不能把预测位置对应当作独立像素真值认证。

## 2. 文字框的局部回归再次成立，但不能全局回退

固定 last 的同一结构包、候选锚点、role 和 OCR，仅把 bbox 换成 initial：

- 救回 3 张新增错误：`fresh_dev_001209_coordgen`、`fresh_dev_001099_original`、`fresh_dev_000175_stress`。
- 改错 1 张本轮救回图：`fresh_dev_001143_coordgen`。
- 3 张保持正确的控制图不变；总数从 4/12 到 6/12，但不能据此选混合权重或部署。

同锚点下观察到的文字变化（last bbox → initial bbox）：

- `001209_coordgen`：`CH → Cl`，恢复整图。
- `001099_original`：包括 `N → S`、`NH → N`，恢复整图。
- `000175_stress`：`NH → NH2`，恢复整图。
- `001143_coordgen`：包括 `NH → NHH`，回退反而破坏原本正确结果。

共有 6 张图、11 处 OCR 字符串变化；这些是预测字符串，不是每一处都已独立标注正确。bbox 对照两组的角色概率、开关决策及完整结构摘要完全相同，故这里的变化不是换了 OCR 权重或候选集合。

## 3. 小训练样本的覆盖没有随整图退步一起下降

12 张训练诊断图共 160 个目标原子、166 根目标键；使用原有 Hungarian 匹配及 6 px 半径。

| G / S | raw 匹配原子 | selected 匹配原子 | selected 未匹配候选 | 可达正键 |
| --- | ---: | ---: | ---: | ---: |
| initial / initial | 160 | 159 | 0 | 165 |
| initial / last | 160 | 159 | 1 | 165 |
| last / initial | 160 | 160 | 2 | 166 |
| last / last | 160 | 160 | 2 | 166 |

训练样本覆盖略升，同时多出未匹配候选。不能把覆盖率提升等同完整分子泛化提升，也不能用这 12 张外推开发图的像素级召回。两个数据子集的职责保持分开。

## 实现与验证

- 新增 `chemvector/training/curriculum/upstream_diagnostic.py`：固定样本/6 组对照、缓存输入不变检查、原样回放硬门槛、原权重/来源哈希检查和外部 CPU 时限；没有训练入口或运行时超参搜索。
- `evaluation.py` 提取 `from_predicted_atoms()` 复用原 pair 推理代码，默认 `predict_structure()` 调用同一逻辑；未改变模型、阈值、化学规则或默认 fallback 行为。
- 可选 trace 新增失败前完整 scene（含原子和键）、第一次 sanitize 错误和最终编译错误。72 次开发图预测中的 9 个无效输出均保存了失败前 scene；未把无效输出删除或改成有效。
- 专项及相邻测试 **67 passed / 7.14 秒**；更广的 curriculum 与 direct pair/role/bbox 回归 **349 passed / 47.65 秒**。两个计数包含重叠，不相加；不是全仓库测试声明。新增测试共 16 个参数化用例。
- initial/last/OCR 的参数及缓冲区摘要前后相同；来源 checkpoint、原 initial/last、前端 registry、发布清单和绑定输入核验通过。独立只读产物复核重新计算六组正确数、干预救回/改错和输入文件哈希。

## 产物

目录：`outputs/audits/chemvector-upstream-interfaces-20260918/`，含运行前 plan、12 份训练覆盖记录、12 份开发图完整对照、report、worker.log、runtime。

- plan SHA256：`43fb2edf59febc4fd604f9a2699bbbcb6651a4bd1a406bebe895382d7cd31360`。
- report SHA256：`0ea0b333a0e187f31f2a5a016a20092f10a4500f0ea322834f94f98451f1feb9`。
- runtime SHA256：`d3bafdd9ae50b1e863e971d5249233eb3d7cb748a1007179057ad533684c9cec`。
- 执行状态：`completed / exit_code=0`；耗时 64.356664 秒，无自动重试，未追加另一轮推理。

## 第 2 步的门槛，而非立刻开训

保留完整 initial 和原 170/800 开发基线；不采用诊断中的任意混合权重，也不再使用六组共同更新的旧配方。当前证据支持先保护上游输出与文字裁框接口，**不支持直接认定应该重训 set，也不支持从本次结果就断言 pair 值得再训**。

下一步只能先做约定的 32 张已准入完整分子训练图的 initial 收益空间验收：在优化前，确认哪里确有错误、该组件是否有能力在冻结其他接口的条件下修复、如何同时守住正确控制样本。若上游不可达或目标组件本来就正确，就不启动优化器。若收益空间成立，再为一个组件制定一次小训练预算并取得明确确认。

本次没有执行该 32 图验收或新的 GPU 训练。诊断已按预定范围结束，不再为了给每个模块分配责任比例而扩样扫描；长期架构和阶段课程方向仍未被这次局部实验否定。
