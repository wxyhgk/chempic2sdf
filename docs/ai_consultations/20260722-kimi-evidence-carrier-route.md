# 2026-07-22 Kimi：Evidence Carrier 新路线复核

## 提供的实验事实

- v0.9.7 typed affinity + exact-cover 在正确 proposal pool 上可完成 bond group partition；
- v0.9.8 free-query primitive-set 在 validation100 的 line F1 仅 `0.146%`、Bond IoU
  `11.39%`；
- capacity 图片平移 16px 后 line F1 从 `81.4%` 降至 `0.16%`；
- OpenCV LSD 在完整 raster 上可提供约 95 个线段候选/图；
- 普通 single/parallel group 有高 carrier direction support，hashed/polygon 不在普通线范围。

问题要求 Kimi 从第一性原理比较 LSD/evidence carrier、Dense VisualBondGroup 和 DETR
bond query，并给出可证伪的 GO/NO-GO 实验，而不是泛泛建议扩充数据。

## Kimi 的核心判断

Kimi 认为 evidence-carrier 路线在当前数据规模和已观察失败模式下更合理。原因不是经典
CV 天然优于神经网络，而是当前失败项恰好是 learned query 到绝对坐标的关联；LSD 把
“在哪里”变成平移等变的确定性 proposal，模型只承担 candidate-relative keep、type、
extent 和 affinity，样本复杂度显著降低。

它同时强调这条路线不是无条件 GO：proposal recall 是不可恢复的硬上限，必须先在目标域
做零训练审计。

## 对正式审计的复核

Kimi 读取了 evidence carrier audit，并指出几个不能被局部平均值掩盖的事实：

1. `parallel_3` 在 validation100 中只有 2 组，`100%` 不能作为可靠泛化结论；
2. raw long-bond endpoint coverage 在 q8/q32 仍很低，说明 fragment/extent 是普通线主问题；
3. q32 group coverage 虽在 95% 以上，但 structural sample exact 只有 `55%`，排除 hashed
   也只有 `76%`，整分子指标才是真正上限；
4. hashed、solid triangle 和 solid quad 平均超过 1 个 symbol/分子，特殊符号分支不是可
   忽略的边角功能；
5. synthetic full raster 与 oracle bond layer 接近，只能说明当前合成图文字干扰不大，
   不能外推到专利括号、箭头、caption、芳香圈和扫描噪声。

## 建议的三阶段门控

### A. 零训练目标域审计

- 多尺度、模糊、JPEG、分辨率、线宽和芳香圈腐化；
- 真实公开图的 candidate count、precision、fragment multiplicity 和每分子 full coverage；
- 长键、parallel、near-text、Markush bracket 分桶。

若中等腐化使单项 coverage 下降超过 20pp，先修 proposal，不训练 downstream head。

### B. Candidate-relative Keep/Extent

- 以 carrier 中心和方向规范化 oriented patch；
- 删除绝对 position embedding；
- 16/32/64px 平移后 F1 下降必须小于 2pp；
- keep AUC 高但端到端低时，判定 proposal geometry 是瓶颈，不继续扩大 head。

### C. 楔键与 SDF 语义

- hashed 先做 dash-cloud envelope，solid wedge 做 apex/base symbol detector；
- 最终使用 SDF sanitize、graph exact、stereo exact，而不是只看 line IoU；
- 端点必须分配给文字节点或 Junction，无法分配应成为显式失败态。

## 对当前实现的影响

本次意见强化了以下决定：

- v0.9.9 使用 evidence anchor + bounded/discrete extent，不回到 DETR/free query；
- target 必须 equivalence-aware，多个支持同一 GT line 的 anchor 不能互相当负样本；
- 训练和评估必须报告 per-sample exact、long-bond 和 translation-equivalence；
- hashed/polygon 独立建模；
- Visual SVG 指标用于诊断，最终仍以结构图与 SDF 为准。

完整 Kimi 会话可用以下 ID 恢复：

```text
session_1d6fc7cc-a4cf-4179-8778-d782a3f9e3d2
```
