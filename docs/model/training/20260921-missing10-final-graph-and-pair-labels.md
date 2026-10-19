# 10个定位漏点的最终图追溯与键监督重放

## 结论

这10个点不能统一解释成“原子没识别出来”。仅使用固定58训练回放集中的原7张图、
原始initial保存预测和同源标注，结果是：

- **3个点：最终元素、相邻节点和键型均正确**，只是不满足原6px定位口径。
- **5个点：最终N节点存在、元素正确，但合计缺7条连接**；另有1条最终键型不符。
- **2个点：有明确的已选空间候选，但编译失败，没有保存最终MolBlock**；
  因而无法确认最终元素和邻接，不能断言原子丢失。
- 按当前代码、原6px匹配重放，10个点涉及的**19条不同的真实连接全部进入可采样负例池**，
  没有成为键正例。这是监督边界问题的证据，不是历史梯度或训练退步的因果证明。

没有GPU、模型前向、checkpoint加载、优化器更新、预测修改、阈值修改或新训练标签。
CPU工作进程1.416秒，相关测试84项通过。

## 逐点结果

样本列省略共同前缀`curr_`。节点和键编号均为原始标注的0起始ID。
这里的距离是**最近已选且可明确对应的候选**，不一定是此前图册里的最近raw候选。

| 样本 | 节点 / 文字 | 已选候选距离(px) | 最终图中的局部结果 |
| --- | --- | ---: | --- |
| 89a886092f8bcd05__font_large | 1 / NH | 6.3588 | N存在；应连0、2，实际只连2；缺键0 |
| cb5b7da426023d14__font_large | 9 / NH | 6.0239 | N存在但孤立；缺与8、10的连接，即键8、9 |
| cb5b7da426023d14__font_large | 19 / N | 7.1887 | N存在；缺与18、20的连接，即键18、19；与25的最终键为单键而非芳香键 |
| 00f41c00f7bf4670__font_large | 13 / O | 6.1695 | O及其与12的双键正确 |
| 00f41c00f7bf4670__font_large | 16 / N | 6.0102 | N存在；与15、26相连，缺与17的连接，即键16 |
| 00f41c00f7bf4670__font_large | 34 / HN | 6.1219 | N存在；与35相连，缺与33的连接，即键33 |
| 6de40b2274fd34b2__serif | 10 / HO | 6.5644 | O及其与9的单键正确；整图规范SMILES也正确 |
| 4ef506af64e8f0b8__bond_short | 38 / C | 8.5850 | 已选候选slot41；无最终MolBlock，局部化学结果未知 |
| 08d02946e879406f__font_large | 12 / N | 6.0913 | N及其与11、13的芳香键正确；整图错误在其他位置 |
| 5c99d711703c1c0c__font_small | 14 / Br | 8.1041 | 已选候选slot20；无最终MolBlock，局部化学结果未知 |

显式C那张的保存错误为F价态3，Br那张为O价态4；不能把这些整图编译错误直接归因于C或Br漏点。
cb5样本的芳香键差异是**最终图**结果，可能包含芳香一致性后处理的影响，未单独归因于键分类头。
局部“正确”只覆盖元素、相邻端点和标准化键型，不等于所有氢数、自由基、立体属性均已独立核对。

## 怎样对应，避免用答案凑图

1. 按原保存logit及固定`-0.25`阈值恢复已选候选。
2. 使用坐标建立只读诊断对应：双方唯一最近、没有多个候选争抢同一目标，
   且距离严格小于该目标最近其他目标间距的一半。**不使用元素或键作为配对依据**。
   所有7张图的已选候选均通过这些歧义检查；08d样本的目标5没有对应已选候选。
3. 从保存MolBlock的坐标精确恢复q1023锚点，逐点连接到候选slot，验证完整一对一来源。
   不使用SMILES遍历次序，不进行坐标拟合，不凭元素相同猜测节点身份。
4. 用RDKit标准化芳香性后比较键型，避免把正常Kekulé单/双交替序列误报成芳香键错误。
5. 对没有MolBlock的两张图保留未知；没有重建或臆测编译前的图。

这套对应仅用于回答“保存的点代表什么”，**不是替换官方6px匹配，也不是训练标签发布**。
NH节点1的最近raw候选是slot115，而最终图来自slot5；显式C、Br也存在raw最近点与已选点不同，
因此不能直接拿上一轮raw最近候选当作最终图节点。

## 19条连接为什么进入负例池

离线重放直接调用当前生产实现，没有复制一个放宽半径的匹配器：

1. `match_atom_targets`仍使用原768尺度、6px门限。
2. 重放匹配的目标ID集合与各图保存的`matched_target_ids`完全一致。
3. 10个诊断对应候选的正式目标映射均为`-1`。
4. `build_pair_targets`先把所有pair的`presence`初始化为False；遇到任一端未匹配时，
   直接继续，**没有给这个pair设置忽略掩码**。
5. `select_training_pairs`把所有`~presence`行纳入负例池，之后才按距离等信息采样。

因此，19条真实连接对应的候选pair都有`presence=False`，具备被当作无键负例采样的资格。
其中也包括3个局部正确点的4条连接。所有19条均去重核对，不是双端重复计数。

本轮没有运行负例采样或反向传播。**不能说19条一定在历史某一步被采样、获得梯度，
更不能据此证明近期试训退步的唯一原因**。近期只调整selector/heatmap读出层的试训也不能
被描述成更新了pair head。本轮证据限定为当前监督构造在这些保存候选上的行为。

代码依据：

- `chemvector/training/graph/relations/train_direct_atom_pair_bond.py`：匹配、pair目标和负例池。
- `chemvector/training/curriculum/continuation.py`：已选候选重匹配后调用上述函数。
- `chemvector/training/curriculum/evaluation.py`：最终图编译及OCR路径。

## 下一步顺序

1. **先审查“不确定归属的pair是否应参加无键监督”**：做独立的监督掩码原型，
   分开有键、可靠无键、归属不确定三类；先统计固定58中损失了多少有效负例和保留多少正例。
   不把这次空间诊断对应直接提升为正例，不一刀切删除所有未匹配候选的监督。
2. 如需补齐C/Br的最终局部证据，仅对原两图做有上限的编译前trace，先验证保存基线一致性；
   不扩样本，不自动重训，不把CPU/GPU候选差异混入解释。
3. 监督边界和回归测试通过后，再决定是否值得申请小额试训；不扩大6px验收半径来制造改善。

## 文件与验证

- 实现：`tools/selector_missing_graph_trace.py`，`tools/audit_selector_missing_graph_trace.py`。
- 测试：`tests/chemvector/test_selector_missing_graph_trace.py`。
- [逐点JSON报告](../../../outputs/audits/chemvector-selector-missing10-graph-trace-20260921/report.json)
- [资源回执](../../../outputs/audits/chemvector-selector-missing10-graph-trace-20260921/runtime.json)
- [此前10点局部图册](../../../outputs/audits/chemvector-selector-coordinate-contract-20260920-r1/index.html)

验证命令：

```bash
CUDA_VISIBLE_DEVICES='' .venv-train/bin/python -m pytest -q \
  tests/chemvector/test_selector_missing_graph_trace.py \
  tests/chemvector/test_selector_coordinate_contract.py \
  tests/chemvector/test_curriculum_selected_matching.py \
  tests/chemvector/test_selector_radius_reference.py
```

结果：**84 passed / 1.41秒**。独立复核42项输入SHA、7个逐图报告SHA、
5张MolBlock的规范SMILES/节点数/键数、19条监督连接和7条缺键去重，全部通过。
未读取开发/封存测试逐图内容，未新增身份查询；旧预测、配置及SHA绑定代码均未更改。

报告SHA256：`4254d5c8568234f2346951aa0c4b9bf829d424cebd36efdc3634d581a46dbd54`。
