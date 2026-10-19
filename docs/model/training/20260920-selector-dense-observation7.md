# 7图定位观察：10个缺口分为7个无邻近峰、2个排名排除、1个低分兼排名排除

## 本轮结论

完成原10个raw缺失目标所属7张旧训练/回放图的initial定位观察。
**7张图的候选槽/网格顺序、6px有效关系、原指派和缺失目标全部复现；参数未改变。**
CPU完整工作进程4.421秒，硬上限120秒，每图一次前向，共7次，无重试、无GPU、无训练。

不能把10个缺口全部归为同一种错误：

| 缺口类别 | 目标数 | 当前保存输出说明什么 |
| --- | ---: | --- |
| 6px内没有局部峰 | 7 | 增大top-k也不能让这些非峰进入原解码器；优先核对峰位置和锚点监督 |
| 6px内有超过0.01的局部峰，但不在top-128 | 2 | 存在被排名筛掉的空间合格候选；不是“附近完全没有响应” |
| 6px内有局部峰，但低于0.01且不在top-128 | 1 | 排名和低分两项限制同时存在，不能只标一个原因 |

这里的“合格”仅指空间距离，不证明元素或连接语义正确。
这些是固定7图的**解码机制观察**，不是已经测明的训练因果，也不是新准确率。

下一步优先做**峰监督与化学锚点定义的一致性核对**：旧原子头训练目标、当前标签，
是否把化学节点位置与文字墨迹/框中心混在一起；若一致，再检查局部峰监督的损失/梯度。
先做这项静态与CPU检查，不直接解冻上游，不增加候选容量或启动新试训。
另21个“raw有候选但selector未保留”的initial缺口尚未由本轮解决。

## 固定范围和隔离

沿用上一轮SHA绑定的58图报告，仅取其中10个无6px内raw候选的目标，涉及7张旧训练图。
不读取64开发图逐图产物、封存测试或新增分子身份信息。

初始checkpoint：`outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt`。
SHA：`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`。
完整模型状态前后均为：`9d7f66f79329479e05c38f8b91d95118adb9f12c32a44e5bd9d83a458a8b452e`。

使用既有bundle恢复器，没有自建替代头或外部OCR加载。
仅执行 `vision.forward_stroke_resolution(pixels)` 和 `atom(features)`。
下游bond、selector、pair、role、bbox模块装有拒绝前向的hook，实际均未调用。
原子头内部自身的role/text几何输出保留；这不涉及OCR识别或下游文字角色头。

模型eval且所有参数冻结；CPU float32、2线程、无autocast。
像素使用原RGB768归一化，不resize、不换字体、不增强。
原128容量、0.01原子峰门限、6px匹配半径不变，不运行selector分数筛选。

每图流程：记录一次前向预约 → image-only前向 → 持久化完整dense快照 → 解读GT → 写逐图结果。
输入hash校验可以读取标注文件字节，但目标坐标只在快照持久化之后解析，绝不传入前向。
快照保存heatmap logits、offset logits、anchors，以及原decoder的indices/scores/mask/geometry。
每个NPZ均做无pickle往返核对。运行日志顺序和每图仅一次已独立复核。

## CPU与历史GPU的复现边界

原GPU和本轮CPU的数值**不是逐位相同**，没有把它们冒充为同一次前向。

- 7/7图的活动槽和有序grid ID完全相同，总计872个候选（6×128+104）。
- 全部6px成员关系、原raw指派、缺失目标ID完全相同。
- 共同grid的坐标分量最大差异为0.0000457764px。
- 候选分数最大差异约0.00000154972。
- 10/10关注缺口在CPU依然存在。

旧GPU没有保存dense热图，因此不能证明每个未入选格子的历史数值相同。
本轮是复现同一离散候选/缺口的CPU解释，不是恢复丢失的GPU张量。

## 为什么只检查GT格会漏掉另一种错误

10个GT所在格全都不是局部最大值，首个排除环节均为3×3局部峰筛选。
其中7个GT格分数已≥0.01，3个仍低于门限。
这些格上已经输出的坐标距离目标为0.680～4.617px，均在原6px内，
但未成为入选候选；不能把这个GT辅助观察当作实际救回预测。

如果只看到这里，就容易得出“10个都是峰位置问题”的过度结论。
因此继续检查了**全部解码坐标处于目标6px内的dense格**，不仅是GT格：

- 共96个目标-格关系，其中93个不是局部峰，3个是局部峰。
- 共59个格分数≥0.01，但局部峰和top-k条件仍需独立满足。
- 3个局部峰分别属于3个目标，均未进入原top-128；其中2个已过阈值、1个没过。

三个邻近峰的具体证据：

| 样本/目标 | 标签 | 峰分数 | 前面严格更高的峰数 | 到目标距离(px) | 该图第128峰分数 |
| --- | --- | ---: | ---: | ---: | ---: |
| `curr_cb5b7da426023d14__font_large` / 9 | NH | 0.0110832 | 315 | 4.7193 | 0.0256811 |
| 同图 / 19 | N | 0.0148713 | 212 | 4.4335 | 0.0256811 |
| `curr_00f41c00f7bf4670__font_large` / 16 | N | 0.00717673 | 728 | 4.8469 | 0.0334219 |

没有为了这些目标扫描K、降低阈值、扩大半径或注入GT坐标。
单独降低门限不会改变top-k和非峰屏蔽；单独增加K也不能处理7个无邻近峰的目标，
而第三个低分峰仍受门限限制。两个排名缺口说明容量是局部因素，
但尚未测量扩大容量带来的额外坏点、selector兼容性、计算代价或完整分子收益。

另外7个目标为：
`curr_89a886092f8bcd05__font_large:1`、
`curr_00f41c00f7bf4670__font_large:13,34`、
`curr_6de40b2274fd34b2__serif:10`、
`curr_4ef506af64e8f0b8__bond_short:38`、
`curr_08d02946e879406f__font_large:12`、
`curr_5c99d711703c1c0c__font_small:14`。

## 独立核验与工程交付

新增：

- `tools/selector_dense_observation.py`：无GT前向API、快照契约、历史复现和后置目标解释。
- `tools/audit_selector_dense_observation.py`：固定7图、120秒硬上限、每图一次的入口。
- `tools/review_selector_dense_observation.py`：仅读快照，独立NumPy邻域/距离核验与7/2/1分类。
- 两个新测试文件，共39项合成测试；覆盖禁止下游前向、输入/快照契约、范围隔离、
  数值与离散复现的区别，以及“GT被抑制但附近另有峰”的反例。

独立核验45个输入hash、7个快照hash、7个逐图hash；无新增模型前向。
NumPy双精度sigmoid和距离复核与实际float32观察在这批邻域的峰、门限、半径成员判定一致；
这不是更换生产数值算法，遇到不一致将明确失败，不添加平局规则。

相关11个测试文件合并回归：**221 passed / 2.64秒**，不是全仓或模型质量验收。

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv-train/bin/python -m pytest -q \
  tests/chemvector/test_selector_dense_review.py \
  tests/chemvector/test_selector_dense_observation.py \
  tests/chemvector/test_selector_localization_contract.py \
  tests/chemvector/test_selector_coordinate_contract.py \
  tests/chemvector/test_selector_radius_reference.py \
  tests/chemvector/test_selector_stable_subset.py \
  tests/chemvector/test_selector_node_semantics.py \
  tests/chemvector/test_selector_node_groups.py \
  tests/chemvector/test_curriculum_bundle.py \
  tests/chemvector/test_curriculum_runtime.py \
  tests/data_sdf/test_curriculum_v2_selector_matching.py
```

本轮入口为 `python -m tools.audit_selector_dense_observation`，
后置复核为 `python -m tools.review_selector_dense_observation`；前者要求全新输出目录，
后者要求新的复核文件。当前结果已存在，两者都会拒绝覆盖，不应为了重跑删除记录。

输出目录：`outputs/audits/chemvector-selector-dense-observation7-20260920/`。
主报告、快照、逐图记录、事件日志、资源回执均保留；邻域复核单独写新文件，不改主报告。

主报告SHA：`7ba69b5f75241fced837611a9dbdef236a1855fdbbd290f01ff50154f78ec1ff`。
邻域复核SHA：`d6829dc8d2b7f2f5a8395a91189cc151d34b7f59607a999dfdb7cbf6d9db601f`。

生产代码、checkpoint、历史指标、原始数据、默认initial均未修改。
未训练、未运行OCR、未生成新的端到端准确率，也未发布新训练标签。
