# 2026-09-09 独立文字入口小范围对照

## 问题与范围

验证现有独立文字框 checkpoint 是否能绕开 AtomObject 前置筛选，并改善有定位依据的
文字转写。不是新架构等预算训练 A/B，不训练新模型，不读取 sealed test、MR-TADF 或
USPTO，不改变默认推理。v0.7.33/v0.8.9 已有独立文字分支，不能把复用称作新发明。

## 运行前固定

- 从 R0 已暴露 fresh-dev 的 500 个分子中，按 `sha256(text-entry-v1:molecule_id)`
  取前 50 个，每个保留四个布局，共 200 张。选择不依赖错误标签或图片难度。
- A：冻结 R0 producer 中 v0.10.5 role、v0.10.8 bbox 的全部 predicted TEXT crops。
- B：独立 v0.8.9 bbox detector；历史阈值 0.3，最多 128 个对象，不搜索阈值。
- 两路使用完全相同的 v0.10.9 CTC、CPU FP32 greedy，192×48 crop、4px context。
  不引入 Paddle，不用识别结果筛框，不使用 GT 框裁图，不运行结构图分支。
- `freeze` 在推理前保存选择、全部模型依赖和代码/数据 SHA。`predict` 保存并冻结
  全部框及 OCR 文本；`summarize` 才利用 GT bbox 做计分。数据加载器本身仍会解码
  metadata，因此只声称真值不用于生成候选，不声称 GT 字节从未被读取。

## 指标和门

主匹配：IoU >= 0.5，最大合法一一匹配数，然后最大 IoU，不按字符串匹配。
同时描述 IoU 0.25/0.75 敏感性，不用结果选择阈值。

报告 bbox precision/recall、框匹配且 raw text 完全一致的 grounded exact、条件 OCR
exact、GT bbox 覆盖 >=90% 的数量、重复框/未匹配框/负框非空输出、整图全部文字
完全正确率，以及四布局和 A/B gain/loss。bbox 覆盖不等同于 glyph ink 覆盖。

诊断门：B grounded exact 至少高 2pp；框 precision 不得低于 A 超过 1pp；每图未匹配框
增加不超过 1；stress grounded exact 不下降。通过仅说明该独立 checkpoint 值得进行
未见数据确认；未通过不否定独立文字架构。训练史不同是不可消除的混杂因素。
无论结果如何，`training_authorized=false`、`promotion_authorized=false`。

## 执行

```bash
.venv-train/bin/python -m chemvector.audit_text_entry_comparison freeze --output-dir outputs/evaluations/text-entry-comparison-20260909
.venv-train/bin/python -m chemvector.audit_text_entry_comparison predict --output-dir outputs/evaluations/text-entry-comparison-20260909
.venv-train/bin/python -m chemvector.audit_text_entry_comparison summarize --output-dir outputs/evaluations/text-entry-comparison-20260909
```

推理与汇总分离。本协议绑定源文件哈希；如需修改汇总代码，必须另记代码/协议版本并
明确复用哪份已冻结预测，不能悄悄修改源文件后继续原协议。

## 2026-09-09 实测结果

50 个分子、200 张图、2,147 个可见文字对象。GPU 为 RTX 5090；两次 detector forward
逐 tensor 完全一致。两路合计生成 4,018 个 crop，模型状态摘要前后一致，没有训练。

| 主指标（IoU >=0.5） | A：当前原子条件入口 | B：历史独立文字入口 |
|---|---:|---:|
| 预测框数 | 2,101 | 1,917 |
| 匹配到的 GT 文字框 | 2,033 / 2,147 | 1,012 / 2,147 |
| 框召回率 | 94.69% | 47.14% |
| 框精确率 | 96.76% | 52.79% |
| 框匹配且字符串完全正确 | 1,936 / 2,147（90.17%） | 860 / 2,147（40.06%） |
| 已匹配框上的 OCR exact | 95.23% | 84.98% |
| 每图未匹配预测框 | 0.340 | 4.525 |
| 整图全部文字正确且无额外框 | 96 / 200（48%） | 4 / 200（2%） |
| stress 框匹配且字符串正确 | 84.53% | 37.94% |

B grounded exact 相对 A **-50.12pp**，所有诊断保护条件均未通过。
这里的“未匹配框”按 IoU 门槛定义，不等于其中每一个都是把键线当成文字：
B 在 IoU 0.25 下有 1,689 个匹配，而在 IoU 0.5 下只有 1,012 个，存在明显定位质量问题。
降低门槛仅是预先固定的敏感性分析，不改变正式判定。

| IoU 门槛 | A grounded exact | B grounded exact |
|---|---:|---:|
| 0.25 | 90.50% | 58.55% |
| 0.50 | 90.17% | 40.06% |
| 0.75 | 71.77% | 10.95% |

逐对象对照：843 个两路都正确，17 个仅 B 正确，1,093 个仅 A 正确，194 个两路均不正确。
这不是可部署的 Oracle 融合收益，未设计如何在推理时识别这 17 个对象。

当前 A 的 211 个不正确文字对象可分为：

- 114 个没有满足 IoU 0.5 的预测框，其中 stress 占 60 个；B 为其中 19 个提供合格框，
  但只有 12 个框和字符串同时正确。
- 97 个已有合格框但字符串错误；B 仅额外纠正其中 5 个。

114 个框未匹配不应全部归咎于 Atom selector：其中仍可能包含 role、bbox 或检测阶段
的问题。下一步应按既有 proposal-ladder 身份和本次框匹配结果拆解，再决定训练目标。

## 判读和下一步

1. **不直接替换为历史独立入口，不扩训练。** 这份旧 checkpoint 在当前域上不满足要求。
2. **不据此否定独立文字架构。** 两路训练史不同；v0.8.9 从更早的文字 detector 初始化，
   当时只训练 130 个 bbox offset 参数，不是与当前模型等预算的新训练对照。
   本次也没有复现其历史双 crop / OCR 分数回退策略，而是固定单框，控制识别器变量。
3. **优先做当前入口的错误归因。** 将 114 个框未匹配与原始候选、set、role、bbox
   分阶段对应；另审查 97 个框内转写错误是否源于截断、邻接键线或字符串识别。
   先找出可恢复的损失来自哪里，再选择监督目标，不以换 MoE 或加训练轮数为默认动作。
4. **识别器选择仍是另一个问题。** 两路复用相同现有 CTC 只为隔离检测入口，
   不构成“必须自己训练 OCR”的结论，也没有评估新的现成 OCR 模型。

## 验证与定位

- 新增协议/指标/预测收据测试与已有外部 OCR 回归测试合计 **62 passed**。
- 独立重算样本计数、逐对象正确数、各 arm 汇总、分子/variant 数量，全部一致。
- 逐一核对 4,018 个 crop SHA 和 report 引用的 artifact SHA，全部一致。
- A 的 2,101 个 crop 和历史 R0 同一份 crop SHA 一致；本次 CPU FP32 greedy 字符串
  与历史 R0 greedy **2,101 / 2,101 完全相同**。
- 无 sealed test、MR-TADF、USPTO 数据读取；无生产入口修改。

产物目录：`outputs/evaluations/text-entry-comparison-20260909`。

| 产物 | SHA256 |
|---|---|
| protocol.json | `c5c1c31dd43aee66838cdec8506d050a0cdf7788d2dbcd53c4a89f135b9d8a81` |
| predictions.jsonl | `0e13abf4623d155d961dc107964de13a94ee5ecb672957c21531279fca345d8c` |
| report.json | `a3fed25e12a96d8da7e0e466650bf2f7339af5ad380aab1c3221afb49b5f4f4a` |

实现按职责拆为 `text_entry_protocol.py`（输入冻结）、`text_entry_predictions.py`
（推理）、`text_entry_metrics.py`（纯指标）、`audit_text_entry_comparison.py`（CLI/汇总），
没有向原 `audit_external_ocr_candidates.py` 增加新的模型执行分支。
