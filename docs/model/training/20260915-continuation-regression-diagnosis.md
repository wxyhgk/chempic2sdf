# 续训回归排查：确认连键监督存在伪负例，上游结构与文字框也有回归

用户要求排查后，仅运行有界 CPU 诊断、读取已保存训练日志和旧 checkpoint 元数据。**没有修代码、修改阈值、更新权重或启动新 GPU 训练。** 以下区分已复现的监督缺陷、局部组件对照，以及尚未完成的整体因果归因。

## 第一优先级：未选候选占用匹配，真实键被当成负例

当前 `continuation_losses()` 在 `continuation.py:106` 对全部原子候选进行真值匹配；到第 118 行只是遮掉未选候选的映射，没有对最终选中的候选重新匹配。

`build_pair_targets()` 在 `train_direct_atom_pair_bond.py:180` 将所有原子对初始化为无键，第 211 行遇到未匹配端点便跳过。因此，未匹配不等于被忽略：这样的原子对仍可能被采样为负例，参与 BCE。

完整问题链条：

```text
全部候选先匹配真值
  → 距离略近、但被筛掉的候选占用真值原子
  → 实际留下的正确候选没有映射
  → 它们之间真实存在的键被标成“无键”
  → 进入负例训练，梯度压低正确的有键判断
```

旧的最终连键头与文字框训练入口是在已经选中的候选上调用 `match_atom_targets()`。当前重用 raw 匹配的接法改变了这个语义。raw 匹配本身仍可用于 raw/set 监督；缺陷是把它直接沿用为最终 selected-pair 监督。

### 真实 C=N 样本的坐标证据

样本：`curriculum:curr_8419bd7a8bf844e9__bond_short`，化学结构 `C=N`，2 个原子、1 根双键。使用保存的 initial，在原图和原阈值下复现：

| 真值原子 | 被 raw 匹配占用的候选 | 是否被选择 | 到真值距离 | 实际已选候选 | 到真值距离 |
| --- | ---: | --- | ---: | ---: | ---: |
| 0 | 2 | 否 | 3.273 px | 1 | 4.246 px |
| 1 | 3 | 否 | 3.096 px | 0 | 4.204 px |

已选候选均在原 6 px 匹配半径内，且分别对应两个相距约 48 px 的真值原子。对 selected 候选重新匹配可得到唯一真实键；当前接线给出零匹配、零正键，唯一原子对被实际负例采样选中。

initial 对这根键的预测概率为 **0.990990**。该单对 BCE 对 presence logit 的导数为：当前负标签 **+0.990990**，正确正标签 **−0.009010**。梯度下降时方向相反；这不是只有覆盖统计异常，而是实际监督会惩罚正确的有键判断。此检查不创建 optimizer，不执行参数更新。

实际训练记录亦可核对：

- 本轮第 **12 步**，同一张图：selected=2，raw matched=2，selected matched=0，pair rows=1，reachable positive pairs=0，pair loss=4.717592。
- 本轮第 **324 步**，同一分子的 font_large 图：selected=2，selected matched=1，pair rows=1，positive pairs=0。
- 上一轮七组联合续训第 **486 步**，同一 bond_short 图：selected=2，selected matched=1，pair rows=1，positive pairs=0，pair loss=4.556982。

中间步骤没有保存全部候选坐标，不能宣称逐张重放了那些中间状态。上述初始权重坐标复现与两轮日志中的监督计数相互支持。

### 固定训练样本探查范围

在加载模型或图片前，固定本轮实际采样序列中的前 8 张完整分子图与前 4 张课程图。对 initial/last 各检查一次，共 24 次前向，没有梯度或参数更新。

initial 的 12 张里有 3 张发生 selected 映射差异，按旧 selected 匹配本应为正例、却被当前采样送入负例的原子对合计 **11 个**：

| 样本 | 目标原子 | 当前匹配 / 旧 selected 匹配 | 错送入负例的正键对 |
| --- | ---: | ---: | ---: |
| `curr_eacef754193a270a__font_large` | 16 | 13 / 16 | 4 |
| `curr_2d374a1e00db024e__font_large` | 16 | 13 / 16 | 6 |
| `curr_8419bd7a8bf844e9__bond_short` | 2 | 0 / 2 | 1 |

这三个样本对应真实本轮第 4、8、12 步；日志中的 selected 匹配/可达正键计数与该缺陷相符。last 在同一小组的映射对照未出现差异，不能因此否认前面已经发生的错误训练；也不能把 3/12 外推为整个数据集的错误率。

全 800 步日志中，26 步出现“selected 数量等于目标数量、raw 全匹配、selected 却丢失匹配”的模式。这里只把它当待进一步核对的筛查信号，不把 26 步全部断言为已确认标签错误。

## 第二项：固定组件对照定位结构和文字框回归

按固定哈希顺序、分子不重复，选择四种已知退步类型各 2 张，共 8 张；另选 2 张保持正确、2 张被本轮救回的控制图。先记录计划，再加载像素/模型。这里有意富集错误，不是新验证集。

对 initial/last 的上游结构（vision/atom/bond/set）、最终 pair 头和 bbox 做预先固定的 2×2×2 组合。角色头两份状态相同，统一使用 initial role；OCR 和阈值不变。真值不进入预测、裁框或组件选择，只做最终评分。

| 上游结构 | 最终连键头 | 文字框 | 8 张退步图中正确数 |
| --- | --- | --- | ---: |
| initial | initial | initial | 8 |
| last | last | last | 0 |
| last | last | initial | 3 |
| initial | last | last | 6 |
| initial | last | initial | 7 |

全部 8 个组合记录在 worker.log，不把最佳组合当作可部署结果。两个保持正确的控制图在全部组合中都保持正确；两个救回控制中，一个需要本轮结构和 pair 的配合，另一个由本轮 pair 带来改善。不能据此建议把所有组件简单回退。

在这组案例里，上游结构改变和 bbox 改变都有直接反事实证据；仅回退 pair 并不能消除大部分退步。比如候选数量出现 19→18、23→25，换 pair/bbox 无法补回或移除这些候选。

三个 bbox-only 救回案例在**完全相同的 last 结构、role 和 OCR**下，局部识别变化为：

- `fresh_dev_001209_coordgen`：initial bbox 为 `Cl`，last bbox 为 `ClH`。
- `fresh_dev_001099_original`：出现 `N`→`NH`、`S`→`Se`。
- `fresh_dev_000175_stress`：`NH2`→`NH`。

这证明“冻结 OCR/role”并不保证文字接口输出不变。上游结构仍未拆分到单个子模块，尚不能给整套 800 图分配各组件的因果责任。

## 配方差异：已确认，但不能直接当作唯一根因

旧 source checkpoint 的 `joint_multidomain_args` 确认：bond 冻结、bond loss 权重 0；presence 学习率 2e-6、incidence 学习率 2e-7；graph exact surrogate 权重 0.1。当前续训启用六组更新，把整个 set_head 设为 5e-6，且没有传 graph exact surrogate 权重，实际默认 0。

因此关联子模块的**名义每步学习率**相对最近一次对应训练提高 25 倍，这不等于实际参数更新或模型损害放大 25 倍。优化器、批量、裁剪和数据也不同；不能仅凭这个数确定主因。该图级 surrogate 是候选决策的近似约束，不是完整 SDF 正确性保证。

initial→last 的相对 state L2 变化：atom 4.305%、bbox 2.765%、pair 1.861%、bond 1.487%、set presence 0.325%、set incidence 0.203%、vision 0.067%、role 0。不同组件的比例不能直接当作效果影响排名；vision 统计还包含未训练部分。组件反事实比单看参数变化更直接。

## 建议处理顺序（本次未实施）

1. **先修监督接线**：raw/set 保留自己的匹配；最终 pair 和文字监督对实际 selected 候选建立正确映射。补充“加入一个未选候选不能让已有合法正键变为负例”的回归用例，以及 C=N 真实样本的标签/梯度方向检查。
2. 在固定训练样本上检查正键召回、负例有效性与角色/文字框可达监督，不通过降低匹配/推理阈值或注入真值候选掩盖问题。
3. 明确与旧权重兼容的参数分组、学习率和图级约束，再决定一次受限训练。先不要同时试多种架构、扩大图量或简单将所有子模块一起解冻。

标签缺陷已确认并应优先修复；“它解释了全部退步”尚未证明。原 initial、last、数据、训练协议和前端保持不变，诊断组合未保存成模型或部署。

## 诊断产物

- `outputs/audits/chemvector-freeze-regression-probe-20260915/`：固定计划、24 次训练匹配对照、12 张图全部 8 组组件对照、24 次 CPU/GPU 保存基线回放一致性，以及前后模型/OCR/输入摘要检查。77.520 秒，300 秒上限，completed。
- `outputs/audits/chemvector-false-negative-witness-20260915/`：固定 C=N 的坐标、候选筛选、目标和梯度方向证据。2.367 秒，120 秒上限，completed。
- 两目录均有 `worker.log` 和 `runtime.json`；runtime 保存完整诊断命令，无自动重试。两项 GPU=0、optimizer steps=0。
- 第一项 worker.log SHA256：`6fb9c41c9cf42f139d4a1706f5d76c0ee248dd720014f905e350b024c82a0b51`。
- 第二项 worker.log SHA256：`084a600babaa99aa3e2cef17bda5f3f7fa3822ea2b157845609f9afe3d7cfa9a`。

无新的私有身份查询、封存样本访问或训练数据发布。本报告仅记录排查结果，不是修复完成或新训练授权。
