# 2026-09-09 当前文字入口失败归因

## 范围

这是对已完成 text-entry-comparison 的事后诊断，不是新的盲测或训练 A/B。
固定处理当前入口 A 的 211 个失败可见文字对象：114 个 IoU 0.5 框未匹配，
97 个框已匹配但 raw string 不正确。不得挑选或丢弃其中的对象。

只读取已暴露的 fresh-dev 与冻结审计产物；不读取 sealed test、MR-TADF、USPTO。
不训练，不改生产入口，不重新运行结构检测器，不调 set、role 或局部峰阈值。

## 证据规则

- 按 `gt_key` 关联比较报告、proposal-ladder 和 peak-precision，检查文字、分子和
  source atom 身份。按 `(sample_key, proposal_index)` 关联冻结 deployed AtomObject。
- set 过滤：decoded 阶段有合法 anchor 匹配，该候选 presence logit < -0.25，且其
  精确身份不存在于 deployed 输出。若候选保留但失去匹配，另记 assignment competition。
- role 过滤：deployed 匹配的同一个 AtomObject 存在，`selected_as_text=false`。
  原记录没有负例 role 分数，因此不能据此声称只需微调 role 阈值。
- bbox：同一个 deployed AtomObject 有 TEXT crop，但该框与 GT IoU <0.5；如果
  框本身已满足 IoU，则另记 bbox assignment competition。
- 其余按 raw128、all-local-peaks、dense-anchor 阶段的几何可达性区分。
  独立阶段的一一匹配可能重分配，raw/decode 类别保留 assignment 不确定性。
- 6px anchor 匹配和 IoU 框匹配是不同契约，阶段归因不代表恢复该阶段即可恢复 OCR/SDF。

## 97 个 OCR 错误的裁框干预

在推理前保存本次协议和代码哈希，以及全部干预 crop 的 manifest。
使用同一个冻结 v0.10.9 CTC，CPU FP32 greedy，4px context，192×48 letterbox。
重放原预测框 PNG，并与 GT bbox +4px 的裁图配对比较。原框识别必须逐对象重现之前结果。

GT bbox 仅用于这次诊断裁图，不回写 text-entry-comparison，也不是部署候选。
换框同时改变位置、缩放、上下文与可能的截断：错误消失只能说明当前识别结果对裁框
干预有响应，不能直接认定“切掉了某个字符”。GT crop 上仍错误也不证明仅是识别器问题。

记录原预测框与加入 context 后的 GT bbox 面积覆盖，`<0.99` 作为几何检查标记；
不能将它当作 glyph ink 覆盖或邻接键线污染的测量。

## 执行与产物

```bash
.venv-train/bin/python -m chemvector.audit_text_failure_attribution --output-dir outputs/evaluations/text-failure-attribution-20260909
```

- `protocol.json`：冻结报告、输入产物和源文件 SHA，以及诊断方法。
- `failures.jsonl`：全部 211 个失败对象及分阶段证据。
- `crop_interventions.jsonl` / `ocr_counterfactual.jsonl`：OCR 执行前的 crop 清单及其后预测。
- `review.html`：全部失败样例，绿框 GT、红框对应 A 框，附原 OCR 输入及 GT crop。
- `receipt.json` / `report.json`：权重不变校验、汇总与全部产物 SHA。

实现分成纯归因函数、裁框/OCR/样例生成、输入校验与 CLI 三个模块。

## 实测归因（保留原冻结标签契约）

| 失败位置 | 数量 | stress 数量 | 核实依据 |
|---|---:|---:|---|
| set 过滤 | 60 | 41 | decoded 对应候选分数低于 -0.25，精确身份未部署 |
| 文字框偏移/尺度不合格 | 19 | 2 | 同一个已部署 TEXT 对象的框 IoU <0.5 |
| 局部峰不可达 | 18 | 11 | dense-anchor 可达，但 all-local-peaks 不可达 |
| role 判为非文字 | 16 | 6 | 同一个 deployed 对象的 selected_as_text=false |
| raw128 容量/匹配损失 | 1 | 0 | raw128 不可达，raw192 可达，候选 rank=177（从0开始） |
| 已匹配框内的 raw string 错误 | 97 | 24 | 原比较报告中的匹配框与字符串完整重放 |
| 合计 | 211 | 84 | 114 个框未匹配 + 97 个框内文字错误 |

18 个局部峰不可达对象均在历史 dense384 几何审计中可达，但不能把这算作真实文字
恢复或 SDF 恢复。60 个 set 损失的 presence logit 范围约 -2.203 至 -0.254，
不能简单认为它们全都靠轻微移动阈值就能解决；本轮未评估降低阈值增加的负例。

OCR 裁框干预：

- 原预测框 **97 / 97** 重现旧字符串，模型状态 SHA 前后不变。
- GT bbox +4px 后 **44 / 97** 变为原标签契约下正确；**53 / 97** 仍不正确。
- 97 个中，17 个原 crop 加上 context 后仍未覆盖 99% GT 框面积。
  44 个改善对象中只有 12 个带此标记，另外 32 个原 crop 已覆盖至少 99% GT 框面积。
  因此不能将全部改善解释为“裁掉了字”；缩放、位置和邻接内容也是可能因素。

## 样例检查发现：可见 C 被标成 CH2

这条发现改变了下一步的优先级。上述 53 个在 GT crop 上仍错的对象，不能全部归为
OCR 能力不足：其中 **21 个冻结 truth 为 CH2，而渲染文字实际为 C**。

检查链：

1. 原 crop 抽查：例如 `fresh_dev_001023_rdkit:t012` 显示单个 C，未画 H2；
   同一对象的 `layers.atoms.text` 却是 CH2。
2. 对 211 个失败中全部 truth=CH2 的 **23 个对象（3 张图、2 个分子）**做后验专项探针。
   用同一化学图与布局重新调用文字可解析的 SVG 后端，23 个均得到 `['C']`。
   其中 21 个属于 OCR 错误，1 个属于 bbox 失败，1 个属于 set 失败。
3. 代码路径：`Data/sdf/build_chemvision_layers.py` 的 `render_and_layers_with_svg`
   先用化学属性生成 `atom_display_text`；当中性碳实际存在文字框而默认文字为空时，
   回退到 `explicit_atom_text`。后者把 implicit H 也拼进字符串，因此写入 CH2。
4. `chemvector/core/data.py:rendered_atom_labels` 直接将这个 `layers.text` 当成实际
   绘制文字。最小回归用例为三个共线碳的 CCC：中间原子有两个隐式氢，绘图显示 C，
   当前化学属性回退却生成 CH2。新增测试稳定复现了这两个结果的差异。

重要边界：专项探针使用 noFreetype 文本后端，**不是原 PNG 的逐像素重放**；结论来自
原图抽查、23 个渲染文本节点和可复现生成规则的交叉证据。仅检查失败中的 CH2，
不是整个 fresh-dev 或训练集的标签污染率普查。也不能把23当作23个独立分子证据。

21 个该类 OCR 对象中，A 已有 6 个输出 C，但仍被原 CH2 truth 判错；GT crop 后有
4 个输出 C，其他多数仍误读为 O/F 等。**标签有问题不等于模型在这些对象上都读对了。**
本轮没有改写任何原标签或历史分数。44/53、90.17% 等原结果仍保留原契约，不能在
发现标签问题后继续将它们无保留地解释成纯粹的可见文字识别正确率。

专项产物：`outputs/evaluations/text-failure-label-probe-20260909`。

```bash
.venv-train/bin/python -m chemvector.audit_text_failure_label_probe --output-dir outputs/evaluations/text-failure-label-probe-20260909
```

## 建议顺序

1. **先修清文字监督契约。** 可见字符与元素、隐式氢、形式电荷是不同字段；
   OCR 目标必须来自实际绘制内容。先在生成器层设计修复和回归，再用受控 train-only
   检查确认同一路径是否污染训练样本。本轮尚未确认训练集污染范围，不能称其为已证实
   的训练失败主因；也不应未经版本化就重写冻结 fresh-dev 标签。
2. **再针对 set 的 stress 正例损失设计实验。** 它是本轮最大的已定位上游损失，
   60 个中 41 个在 stress；验证必须同时统计新增假原子/假文字，不能只追召回。
3. **随后是预测裁框的稳定性。** 19 个框定位失败，以及 44 个对裁框干预有响应的
   文字对象，为 bbox/裁剪协议提供了直接实验依据；不能仅优化 bbox IoU。
4. 保留其余 role、局部峰、识别器错误队列。暂不换 MoE、不复活旧独立 detector，
   不将这轮诊断转成继续大规模训练的默认授权。

## 验证

新增归因、候选身份边界、裁框重放、markup 转义、渲染标签回归测试，加上原相关
测试合计 **80 passed**。独立重算 211/114/97/44/53 计数，逐一验证全部 report
产物 SHA、194 份配对 crop 的 SHA 和冻结输入/源代码 SHA，全部通过。

归因 protocol SHA256：`f826220bd8064661a11c776dd163e0a8c37a81b1813bc035c934815030bd0b5a`。

归因 report SHA256：`a0e8299e808e2efdb0d8bba5968e19fd8007bd0b88a88d982936c0b79e15236c`。

标签探针 protocol SHA256：`2ae3763886c4a31f98885b9afd93cf916a5a6fac30ae8d69b80ea364a159b1fd`。

标签探针 records SHA256：`54d27f4fe384fed196fed8c5e205116cc2544e4344a027eeb11143ea52786544`。
