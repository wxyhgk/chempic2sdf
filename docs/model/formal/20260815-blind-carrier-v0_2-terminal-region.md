# Blind Carrier v0.2 Terminal Region

日期：2026-08-15

## 结论

v0.2 修复了 v0.1 无法表达标准平行三键的问题。结构 validator、协议和只接受已提取结构证据的纯
适配器已经冻结，但完整图片 proposer 尚未实现，真实 20 图盲测尚未开始，训练仍暂停。

```text
blind_sample_selection_authorized=false
proposer_execution_authorized=false
training_authorized=false
```

## 表示变化

v0.1 把 terminal 当作一个必须同时靠近全部 lane endpoint 的点。v0.2 把它改成显式 endpoint region：

```json
{
  "center_q4": [4800, 6144],
  "endpoint_alignment": [
    {"segment_id": 0, "endpoint_index": 0},
    {"segment_id": 1, "endpoint_index": 0},
    {"segment_id": 2, "endpoint_index": 0}
  ]
}
```

validator 从 segment polyline 和 width 重算 endpoint 坐标及 carrier disk。`center_q4` 必须等于全部引用
endpoint 的整数 half-up 均值；圆盘相交图必须连通，但不再要求全部圆盘有公共交点。

## 已验证合同

- 标准三条等宽平行 lane 的相邻 disk 构成连通路径；
- disk 闭边界相切通过，外移 `1 Q4` 拒绝；
- one-segment lane 的 begin/end 必须绑定同一 segment 的两个不同 endpoint；
- Bond 每一侧必须精确覆盖全部 lane endpoint；
- 未选中的 segment 不能充当 terminal bridge；
- 同一 interpretation 中一个 endpoint 不能属于两个 terminal；
- 纯适配器进一步要求不同 terminal 的 endpoint disk regions 两两不相交，避免将同一可见 junction
  拆成两个节点；
- v0.1 的 canonical PNG、真实前景 support、geometry 双向绑定、bundle support 互斥、near-overlay 和
  ownership 门全部保留；
- chain、fan、不同 Bond 的 X crossing、single、double 和 triple 正例均可表达。

当前版本明确只支持 one-segment lane。multi-segment lane 的连续性、方向和端到端顺序仍不在已证明范围。

## Lean 对应

`ChemGraph.BlindCarrierTerminalRegionV02` 证明：

```text
S = 2c + A <= 2(c + A) = 2R
```

所以相邻 endpoint disk 相交，并可构造三节点相交图 `0-1-2` 的连通见证。该证明只说明 v0.2 合同
非空可满足，不证明真实图片的候选覆盖率或化学正确性。

## 验证结果

```text
v0.2 targeted validator tests  33/33
full formal audit             112/112
Lean build                     24/24
fixture double generation       exact
red team decision               GO_FOR_V0_2_STRUCTURAL_VALIDATOR_FREEZE
adapter unit tests               27/27
adapter red team                 GO
```

纯适配器冻结记录见 `20260815-blind-carrier-v0_2-adapter-freeze.md`。下一步只能实现并冻结完整图片
proposer，然后在已暴露图片上做公开 coverage/precision 审计；这些公开门通过前不得选择 blind 20。
