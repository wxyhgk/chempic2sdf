# 固定58图半径内匹配参照：判定假回退存在，但不是主要缺口

## 结论和下一步

复用已有 `radius_assignment`，完成26张拟合见证图、32张旧训练/回放图的
raw候选及4个保存阶段重放。**所有历史原指标精确复现，没有修改原matcher或模型。**

当前默认initial在这58图、1090个原子目标上的32个未匹配目标，可以拆成：

- **10个**：raw候选中没有任何点在原6px半径内，涉及7张旧图。
- **21个**：raw中有合格候选，但selector没有保留下来。
- **1个**：selected中有合格候选，被原“先全局指派再截断”判定漏掉。

这里是固定训练/回放图的原子端点覆盖，不是完整分子准确率或独立测试结果。
换匹配参照只澄清第三项，不能生成前两项缺失的端点，也没有增加任何一条可达键。
因此不把匹配规则差异当作主要训练问题的修复，不启动新一轮试训。

下一项优先工作仍是原10个缺口所属7张旧图的initial dense定位快照：
查看GT格局部峰、原top-k、原阈值与offset，先检查能否复现历史候选。
建议仍按CPU上限120秒、每图至多一次、GPU/训练/OCR为零的原范围进行；
**本轮没有运行这项真实前向，也没有扩大训练授权**。
定位证据之后，再处理有候选却被筛掉的监督问题；此前exact-one目标的反例仍成立。

## 方法：参照独立，旧报告不动

新增计算模块 `tools/selector_radius_reference.py` 和固定范围入口
`tools/audit_selector_radius_reference.py`。
算法直接复用 `tools/audit_curriculum_v02_selector_matching.py` 中已有、通过穷举小矩阵测试的
半径内匹配函数；不是重新发明算法，也未改动这个旧文件。
此前参照仅用于另一批32张干净图，本轮扩展到当前固定58图、4阶段及稳定子集删除反事实。

固定条件：原float32坐标、768尺度、6px半径、logit≥-0.25、原保存候选和分数。

1. 原方法：Hungarian最小化所有分配距离，再拒绝6px外分配。
2. 独立参照：只允许6px内关系，允许未匹配；先最大化有效匹配数，再最小化距离。
3. 删除反事实：仍只移除502个稳定提议中实际被选中的片段，23个坐标敏感提议不动。
4. 原方法的删除后距离按原调用重新计算；参照用删除前成本矩阵切片，
   另外独立重算一次距离，区分匹配政策与矩阵形状/数值效应。
5. 除总数，还逐项记录目标ID、slot→target关系、可达键，以及删除前后的成本矩阵反例。
   不增加人工平局规则；最大匹配数相同不代表目标或键相同。

本批232项检查中，删除前后冻结的有效边和距离完全相同；参照下目标、关系、可达键均未丢失。
实际重算距离差异也全部为零。额外独立核对全部232项raw→selected矩阵的
6px成员关系，翻转为零，支持下面的缺口分类。
这些是当前固定产物的观测，不是任意未来输入的单调性保证。

## 结果

下面“前→后”均指诊断性的片段删除，不是新的模型预测。
每阶段共58图，目标1090个，源图键1150条。

| 保存阶段 | 原匹配目标数，前→后 | 半径内参照目标数，前→后 | 可达键，两种方法及删除前后均相同 |
| --- | ---: | ---: | ---: |
| initial（当前默认） | 1058→1058 | 1059→1059 | 1103 |
| retention-200 | 1061→1061 | 1062→1062 | 1108 |
| control-200 | 1055→1055 | 1055→1055 | 1100 |
| retention-400 | 1068→1067 | 1068→1068 | 1115 |

“可达键”只表示两端原子均有匹配候选，不表示模型预测了正确键类型或连接。
这张表不能用于宣布checkpoint通过验收或切换默认模型。

所有政策差异只发生在同一张 `curr_ddc8766e5cc2a898__reference`（`C#[13CH]`）。
共有5项前/后比较发生差异：initial前后、retention-200前后、retention-400删除后。
**不是5张图，也不是5个独立新增正确原子。**

raw候选的58张图中，两种算法的全部slot→target关系完全一致：
见证图99/99目标、81/81可达键；旧图981/991目标、1050/1069可达键。
因此在本批raw监督上，单独替换匹配政策也不会改变已有原子正负标签。

各阶段删除前的未匹配目标拆分：

| 阶段 | raw就没有候选 | raw有、selected没有 | 原指派额外漏判 |
| --- | ---: | ---: | ---: |
| initial | 10 | 21 | 1 |
| retention-200 | 10 | 18 | 1 |
| control-200 | 10 | 25 | 0 |
| retention-400 | 10 | 12 | 0 |

参照下没有“selected有邻近候选、但因竞争仍无法匹配”的剩余目标。
该拆分由独立核验脚本按逐图ID集合完成，不是只把汇总数相减。
这仍不能解释训练的全部因果原因：候选缺失可能涉及峰、排名、门限或偏移，
selector拒绝也不等价于某个单独loss错误，需后续对应证据。

## 三键反例确证

retention-400原selected槽为1、2、3、5。删除1、3后保留正确槽2和敏感槽5。

| 保留槽 | 到目标0的距离(px) | 到目标1的距离(px) |
| --- | ---: | ---: |
| 2 | 68.8663177490 | 3.8118102551 |
| 5 | 94.1752471924 | 23.5993747711 |

原Hungarian仍选择2→0、5→1，两个都在半径外，记0个匹配。
半径内参照保留唯一有效关系2→1，删除前后均记1个匹配。
另一个端点目标0仍没有被选中，三键依然不可达。

旧报告中的232项保护检查依然是231通过、1失败，原失败完整保留；
新参照的232项目标/关系/键非回退只说明有效空间关系未受删除影响，
不把这502个GT辅助提议自动升级成训练标签或部署过滤器。

## 核验、产物和边界

输出目录：`outputs/audits/chemvector-selector-radius-reference-20260920/`。
包含计划、资源回执、报告及58个逐图记录。
CPU工作进程用时 **1.666秒**，预算120秒，无自动重试。
主流程之外独立核对475个输入SHA、58个逐图SHA、原三键成本矩阵和全部232项关系保护。

报告SHA：`e01a31a88c0850d213b0cf747a056a86c0fd95e56eb896c26022390d9ec2e835`。
源稳定子集报告SHA：`462fc1555110ea42ed9379aeaf5d6746ac848141f9d035deb284cd283758c17f`。

新增测试位于 `tests/chemvector/test_selector_radius_reference.py`，
包括原matcher一致性、真实数值反例的合成复现、空选择、非法删除、固定范围隔离、
矩阵形状引入数值变化的模拟、平局不指定winner、相同匹配数但不同目标/键的检查。
测试只用合成数据；实际诊断只读取既有保存坐标、logit和元数据。

新增 **46项** 合成测试；相关16个测试文件合并回归 **310 passed / 9.84秒**，
不是全仓测试，也不是模型质量验收。命令如下：

```text
CUDA_VISIBLE_DEVICES='' .venv-train/bin/python -m pytest -q \
  tests/chemvector/test_selector_node_semantics.py \
  tests/chemvector/test_selector_node_groups.py \
  tests/chemvector/test_selector_coordinate_contract.py \
  tests/chemvector/test_selector_stable_subset.py \
  tests/chemvector/test_selector_group_objective.py \
  tests/chemvector/test_selector_localization_contract.py \
  tests/chemvector/test_selector_retention_release.py \
  tests/chemvector/test_selector_retention_trial.py \
  tests/chemvector/test_selector_full_loss_control.py \
  tests/data_sdf/test_curriculum_v2_retention_prototype.py \
  tests/data_sdf/test_curriculum_v2_selector_matching.py \
  tests/data_sdf/test_curriculum_v2_selector_gradients.py \
  tests/data_sdf/test_curriculum_v2_selector_adamw.py \
  tests/data_sdf/test_curriculum_v2_selector_loss_terms.py \
  tests/data_sdf/test_curriculum_v2_selector_loss_postcheck.py \
  tests/chemvector/test_selector_radius_reference.py
```

运行入口：

```text
CUDA_VISIBLE_DEVICES='' .venv-train/bin/python -m tools.audit_selector_radius_reference
```

入口要求全新输出目录，当前产物已存在时会拒绝重写；不得为了重跑删除历史记录。

真实模型/checkpoint加载、真实图片前向、优化器更新、GPU使用均为零。
未读取开发64逐图产物或封存测试，未查询身份，未新增数据准入。
原半径、阈值、生产matcher、历史报告、模型参数、默认initial均未改变。
