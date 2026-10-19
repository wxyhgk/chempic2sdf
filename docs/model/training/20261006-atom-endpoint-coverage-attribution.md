# Initial 原子端点覆盖归因（2026-10-06）

## 范围

本记录只读重放 `chemvector-edge-label-reachability-20261005` 所绑定的 1,200 张
训练图和 `initial.pt`。使用生产相同的 raw atom decode、atom presence threshold
和 `match_atom_targets`（6 px），不写 checkpoint、不读 sealed test、不使用 GT
候选或 OCR 预测作为替代输入。

## 按样式汇总

`target / selected / matched / unmatched target / unreachable bond`：

| style | atoms | bonds |
|---|---:|---:|
| font_large | 5,606 / 5,598 / 5,430 / 176 / 304 |
| serif | 6,098 / 6,096 / 5,969 / 129 / 172 |
| line_thick | 6,383 / 6,354 / 6,325 / 58 / 93 |
| font_small | 6,540 / 6,520 / 6,503 / 37 / 57 |
| reference | 5,931 / 5,907 / 5,885 / 46 / 70 |
| bond_short | 4,060 / 4,030 / 4,017 / 43 / 65 |
| bond_long | 572 / 568 / 562 / 10 / 19 |
| line_thin | 638 / 633 / 632 / 6 / 10 |
| rotated | 476 / 476 / 474 / 2 / 3 |

未匹配 target atom 共 467 个，元素分布为 `C=223, O=149, N=97, Br=11,
Se=9, Cl=7, I=4, Si=3`，`B/P/S/F` 各 1。font_large 和 serif 的主要缺口是
可见文字端点（font_large 的 O/N 为 81/59，serif 为 54/30）；剩余大多是隐式
碳端点。不可达键以 SINGLE 为主，font_large 为 222/304，serif 为 142/172。

最大单图为 `curr_0c1e83af2dca5083__line_thick`（12 个 atom、21 条 bond 不可达），
其次为 `curr_811052177abf0bf6__bond_short`（7、12）和
`curr_f92f2304ee73e156__reference`（8、12）。

## 结论

这不是 chemical/presence 联合末层的同一问题：末层试训的唯一回归是已存在原子
之间的临界假阳性边；本审计显示还有约 467 个 target atom 根本没有可用 selected
端点，故 pair loss 无法使这些键可达。应先解决 endpoint coverage，不能把 unknown
转成负样本，也不能用 GT 辅助候选填洞。

## 下一轮准入方案（先诊断，后定额试训）

1. 固定 initial，输出每个 target atom 的 `text/implicit、element、style、raw
   candidate count、selected score、nearest distance`，把缺口分成 raw 缺失和
   selector 丢弃两类；raw 缺失不得路由到 selector 训练。
2. 只有在 selector 丢弃类有足够样本时，才建立小切片：从 train 中按
   `font_large/serif`、文字 O/N 和隐式 C 分层抽样；冻结 vision、OCR、pair 和
   chemistry，仅更新现有 atom presence/selector 参数，保留所有 initial 已匹配
   端点的 retention 约束，并使用候选内 hard negatives。最多 200 steps，先做
   shadow，再做 bounded trial。
3. 验收必须同时满足：缺口类 matched recall 提升；initial 已匹配端点 retention
   100%；固定 dev 的 valid/exact、fallback 不下降；否则 NO-GO，不延长步数、不解冻
   shared trunk。
4. 若 raw 缺失占主导，停止 selector 试训，改审计/训练上游 dense endpoint field
   （slot/capacity 与样式增广），保留当前生产 initial。

这与 2026-09-20 retention 试验不同：历史试验在 58 张图上混合了 raw 缺失、selector
拒绝和 Hungarian 竞争（10/21/1），并做了候选删除反事实；本方案先逐点归因，训练
切片只覆盖“raw 已有但 selector 丢弃”，不把 GT 提议注入模型，也不靠删除反事实
制造标签。

