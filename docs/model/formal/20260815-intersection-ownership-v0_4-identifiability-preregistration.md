# Intersection Ownership v0.4 可辨识性预注册

日期：2026-08-15

## 问题

per-parent support partition 已经 exact，但同一 pixel 仍可被多个 carrier 引用。目标是判断现有 root 是否足以唯一
构造 named intersection support。

## 候选规则

以 cross-carrier overlap pixel 的 8-connected component 为候选 region，并要求：

1. component 恰含一个 exact geometry intersection anchor；
2. 每个 geometry intersection anchor 恰属于一个 component；
3. component 的 carrier union 等于 intersection 的 base-stroke fiber；
4. 全部 overlap component 都被解释。

## 预注册反例

- clean X：结构规则通过；
- disconnected far pixel：无 anchor component，拒绝；
- connected far snake：与 anchor 连成一个 component，结构规则会通过，但语义不可信；
- double-X bridge：一个 component 含两个 anchor，拒绝；
- three-carrier point：region carrier fiber 必须精确为三个成员。

如果 connected far snake 通过，则 component 规则只能得到结构完整性，不能得到 semantic soundness。下一版 root
必须增加可独立验证的 parent tube/width 或 primitive ownership 证据，不能临时拍一个像素半径。
