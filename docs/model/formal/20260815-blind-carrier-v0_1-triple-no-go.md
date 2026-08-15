# Blind Carrier v0.1 三键表达能力 NO-GO

日期：2026-08-15

## 结论

Blind Carrier v0.1 的结构验证器本身已经冻结且实现正确，但它的两个几何合同组合后无法表达标准的三条
平行 lane。因此 v0.1 不得进入 20 图盲测，也不得用于决定训练。

```text
blind_sample_selection_authorized=false
proposer_execution_authorized=false
training_authorized=false
```

## 最小反例

令：

```text
h = ceil(width_q4 / 2) > 0
A = AA_MARGIN_Q4
R = h + A
S = 2h + A
```

`R` 是 terminal 到一条 lane endpoint 的最大距离，`S` 是两条平行 lane 的最小中心线距离。三条平行
lane 的外侧 endpoint 距离至少为 `2S`。若存在一个共享 terminal 点，则三角不等式又要求外侧距离不超过
`2R`：

```text
2S <= 2R
4h + 2A <= 2h + 2A
h <= 0
```

这与 `h > 0` 矛盾。该结论与模型、训练数据和图片风格无关。

Lean 定理：

```text
ChemGraph.BlindCarrierTripleExpressivityV01.
  no_shared_terminal_for_three_separated_parallel_lanes
```

源码：`formal/chemgraph/ChemGraph/BlindCarrierTripleExpressivityV01.lean`。

## v0.2 修复方向

v0.2 不再要求一个 terminal 点同时落入全部 lane endpoint 的 carrier 圆盘。每个 terminal 必须显式记录
引用的 `(segment_id, endpoint_index)`；validator 根据这些 endpoint 重算 terminal region，并要求 endpoint
圆盘的相交图连通。`center_q4` 只作为 endpoint 集合的确定性语义中心，不再充当所有圆盘的公共交点。

该表示保留以下门禁：

- 每条 lane endpoint 仍绑定自己的可见 segment endpoint；
- 同一 BondBundle 的 lane support 仍两两不交；
- 近重合 centerline 仍不能把一条粗线伪造成多键；
- 不同 Bond 的 X crossing 仍可共享交点像素；
- terminal region 必须由 validator 重算，proposer 不能任意放大。

v0.2 必须先有平行单键、双键、三键、chain、fan 和 crossing 的正例，再做反例与红队审计。只有 v0.2
validator 和 proposer 都冻结后，才允许选择 blind 20。
