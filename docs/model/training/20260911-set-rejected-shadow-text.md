# 2026-09-11：set 拒绝候选的影子文字实验

## 结论

**现有冻结模型能从 set 拒绝候选中找回文字，但全量放行到现有 role/bbox/OCR 的规则未通过预算，不能接入生产图。**

固定的 50 个分子、200 张已暴露开发图上，新增覆盖 50 个文字对象，其中 48 个来自历史
60 个 set 丢失对象；代价是 4,389 个额外裁剪。新增覆盖仅占额外裁剪的 1.14%。
这支持“set 删除前仍有有用的文字证据”，不支持“绕过 set 就解决了文字入口”。
本轮没有训练、替换 OCR、改阈值、运行 AtomPair 或重建 SDF。

## 冻结范围与运行

产物：`outputs/evaluations/set-rejected-shadow-text-20260911/`。

- 协议 SHA256：`1c1d83ab3fdfc455d2292870291533da12373152470d2aaf8d86506097b38a25`。
- A：2026-09-09 text-entry comparison 的 atom-conditioned 基线，2,101 条文字预测。
- 图像：原有固定 200 个 sample key；读取器只解码其 PNG，不读取 shard 元数据或额外图像。
- B：全部 `decoded_mask & ~selected_mask` 候选；不按真值、历史失败类别或 OCR 输出选择候选。
- 检测/role/bbox：CUDA BF16，batch 6；set 阈值 -0.25、容量 128，role 阈值取冻结 checkpoint 的 0.85。
- OCR：保留现有冻结 CTC，CPU FP32、greedy、batch 32，192×48 裁剪、4px context。
  使用它是为隔离 set 入口变量，不是重新决定自训 OCR 优于现成 OCR。
- 实际 GPU 为 RTX 5090，PyTorch `2.10.0+cu128`；本次 CUDA 推理正常完成。
- 候选产生不使用真值；真值仅在独立 score 阶段用于固定 IoU≥0.5 的文字盲匹配。

A 的部署候选 ID/坐标、文字选择、bbox、role 分数、裁剪 SHA 和所有原始 OCR 字符串
均精确复现；B 前后输入张量摘要与模型状态摘要一致。基线文字生产器未启用 structural
twin suppression，本实验亦未启用。这里验证的是 R0 文字生产器，不是端到端图输出。

## 结果

| 指标 | 数量 |
|---|---:|
| 历史文字目标 | 2,147 |
| A 已覆盖 / 未覆盖 | 2,033 / 114 |
| 全部 decoded / set 保留 / set 拒绝候选 | 25,396 / 7,444 / 17,952 |
| B 达到 role 阈值并执行 OCR 的裁剪 | 4,389 |
| B 对 A 未覆盖目标的新增合格匹配 | 50 |
| 其中历史原始字符串 exact | 35 |
| 其中历史 set 丢失对象 | 48 / 60 |
| B 未分配的重复框 | 186 |
| B 无 IoU≥0.5 合格目标的框 | 4,153 |
| 每图未带来新增匹配的额外框（含重复框） | 21.695 |

`50 + 186 + 4153 = 4389`，类别完整分割额外裁剪。A 的既有匹配不被 B 抢占或重新分配。
“无合格匹配”不等于全部为不存在的文字，也包含定位不准、只截到部分字形等情形。
全部 B OCR 字符串都非空，因此本次数据上仅检查“识别结果非空”不能筛掉任何额外框。

| 变体（各 50 图） | B 裁剪 | 新增框 | 历史 raw exact | 无合格匹配框 |
|---|---:|---:|---:|---:|
| original | 1,045 | 8 | 7 | 997 |
| rdkit | 1,157 | 6 | 3 | 1,100 |
| coordgen | 1,226 | 4 | 3 | 1,177 |
| stress | 961 | 32 | 22 | 879 |

文字框覆盖增加 2.329 个百分点，其中 stress 增加 5.893 个百分点。
50 个新增目标均非历史 CH2；35 个 exact 仍只是与历史标签一致，不是人工复核后的
可见文字准确率。所有历史 CH2 标签继续保守标记为有争议，未重标任何样本。

本次额外 role/bbox 累计约 0.616 秒，额外 CPU OCR 约 12.963 秒。
这些是单次诊断计时，不是经过独立基准测试的部署延迟承诺。

## 预定门槛与停止决定

门槛在推理前冻结，未按结果修改：新增框≥5、stress 新增框≥1、重复框比例≤50%，
以及每图未带来新增匹配的额外框≤1。
前三项通过，最后一项以 **21.695 > 1** 失败。

决定：`do_not_expand_shadow_branch_under_this_rule`。
`training_authorized=false`，`promotion_authorized=false`，未测量 graph/SDF 增益。
即便覆盖门槛通过，也不能用它抵消额外框预算失败。

## 定性核看（不参与门槛或调参）

从各类结果按 prediction key 排序取样，并查看保存的裁剪：

- `fresh_dev_000000_stress:p026`：可见 O，OCR 为 O，属于新增覆盖。
- `fresh_dev_000000_coordgen:p038` 和 `p047`：裁剪主要为两条平行斜线，
  role 分数分别为 0.98828125、0.99609375，OCR 均输出 S，均无合格目标匹配。
- 同图 `p023`：裁剪只有部分字形及相邻笔画，OCR 为 BH；不能把该类别全部解释成纯背景。
- `fresh_dev_000093_stress:p010`：可见字形周围含结构笔画，OCR 为 N+、历史目标为 N；
  框覆盖和原始字符串一致性是不同问题。

这些少量例子说明需要检查线段/残缺字形造成的误触发，但不能用来估计整个错误类别的比例。

## 下一步重点

先解决**文字区域的有效性判别**：被 set 拒绝的候选不应直接沿用现有 role 高分作为
可靠文字存在性证据。本次结果支持把线段、残缺字形和重复区域作为必须单独评估的负例。
这仍是待验证方向，不是已经确定应训练新模型或更换某个现成检测器。

不要马上扩大候选、只做去重、扫 role 阈值或写回图。重复框只有 186 个，去掉它们也无法
处理剩余 4,153 个无合格匹配框。下一项入口判别方案应独立定规则、再做预算检验；
通过之后才值得做文字与原子/键端点的关联实验。不能把本批反复使用的开发图当作未见确认集。

## 代码边界与验证

实现位于 `chemvector/evaluation/audits/ocr/set_rejected_text/`：
`protocol.py` 管输入与冻结，`inputs.py` 管选定 PNG，`checks.py` 管精确基线和候选完整性，
`runtime.py` 管模型执行，`metrics.py` 管固定 A 的增量匹配。没有修改旧 producer 或生产推理。

执行入口（已有产物目录不可重用）：

```bash
.venv-train/bin/python -m chemvector.evaluation.audits.ocr.set_rejected_text freeze --output-dir NEW_DIRECTORY
.venv-train/bin/python -m chemvector.evaluation.audits.ocr.set_rejected_text predict --output-dir NEW_DIRECTORY
.venv-train/bin/python -m chemvector.evaluation.audits.ocr.set_rejected_text score --output-dir NEW_DIRECTORY
```

新测试 24 项通过；本轮相关 OCR、归因、数据文字来源与 renderer 回归合计 **328 项通过**。
测试覆盖固定 A 匹配、文字盲分配、重复/无匹配分母、CH2 边界、精确基线、输入白名单，
以及所有达到 role 阈值的拒绝候选必须有且只有一个 B 预测。
