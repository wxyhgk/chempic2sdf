# Intersection Ownership v0.4 可辨识性 NO-GO

日期：2026-08-15

## 裁决

```text
8-connected component structural probe: GO
same-observation counterexample:          GO
unique named intersection support:        NO-GO
global unique pixel ownership:             NO-GO
production / UniqueGraph / SDF:            NO-GO
training:                                  NO-GO
```

问题不在 4-connectivity 还是 8-connectivity，也不在还差一个合适半径。现有
`geometry root + parent supports + overlap universe` 无法唯一决定 intersection support 的真实边界。

## 决定性反例

冻结 artifact 从完整 research `geometry_root + parent supports` 输入实际重跑 geometry/support constructor，确定性
投影出同一 observation。该 observation 包含两个 exact intersection seed 和五个连续 overlap pixel：

```text
pixel 18 -> carriers [0,1], seed I0
pixel 19 -> carriers [0,1]
pixel 20 -> carriers [0,1,2], bridge
pixel 21 -> carriers [0,2]
pixel 22 -> carriers [0,2], seed I1
```

两份 world：

```text
World A: I0={18,19,20}, I1={21,22}
World B: I0={18,19},    I1={20,21,22}
```

二者都满足：

- 每个 region 非空、8-connected、包含自己的 exact seed；
- region 两两不交；
- region 并集逐像素覆盖全部 overlap；
- region 中每个 pixel 的 carrier fiber 包含该 intersection 的 incident carriers；
- 同一个 source-bound observation SHA256。

但 bridge pixel 的 owner 不同。任何只读该 observation 的确定性 constructor 至少会对一个 world 错误。

冻结 artifact：

```text
support input SHA256:
a51d941903447556a4b1168947a5935f45b0c3c76e71f427640bb38d1f3d23d2

observation SHA256:
4a1d9a42f6be3f78b537fdfbc777e5cb6f1b7a63ff153bcfece9796a628e7a25

world A SHA256:
dfb5ea46278e3b0792a5af51e066a2ce22847d28727661da1b29da1ef9e8b083

world B SHA256:
2c77b77aeb1d8f04a7eef055a4e296f20d1db5853d2ef0fe63f35e0394436018

manifest SHA256:
cc4b53d9f35275b3cfb09b10e6bcefbee2aa2777510b5c5e9dd7bd5137821147
```

上述五个摘要作为代码内 trust anchor。只重算调用者提供的整条链只能证明内部一致性，无法拒绝同步替换；冻结
入口还必须逐 artifact 命中这些摘要。该区别由 `validate_self_consistent_bundle` 与 `validate_bundle` 明确分离。
trust anchor 保护的是本次 research witness 的身份，不是外部 PNG、production schema 或未来数据集的真实性证明。

## Component 规则反例

- disconnected far overlap：能拒绝；
- connected far snake：从 seed 延伸到 `(0,0)` 的任意长共享路径会被整个吞入 region；
- double-anchor bridge：一个 component 同时含两个 seed，无法唯一拆分；
- three-carrier partial fiber：region union 为 `[0,1,2]`，但逐 pixel fiber 可以只有 `(0,1)/(1,2)/(0,2)`；
- diagonal corner：8-connectivity 合并角接对象，4-connectivity 又会拆开合法斜线；
- glyph overlap：相同 component 可能同时来自文字笔画与 carrier。

不能用固定最大直径普适修复。两条半径 `r`、夹角 `theta` 的 stroke tube，其合法公共区沿 carrier 的尺度约为：

```text
r / sin(theta)
```

当 `theta -> 0` 时无界。固定阈值会误拒浅角合法重叠；不设界则 snake 可任意长。

## Lean 结论

`GlobalOwnershipNoGoV04.lean` 把 Observation、World、total OwnerPartition 与 OwnershipInterpretation 分离，证明：

1. 同 Observation 的两个 legal world 要求不同 owner partition 时，不存在同时 exact 的 root-only unique-owner
   constructor；
2. intersection support 不同时，不存在同时 exact 的完整 OwnershipInterpretation constructor；
3. total `Pixel -> Owner` 确保每个 world 内部唯一归属，但不能让 observation 自己决定选哪个 world。

有限 Boolean witness 只证明逻辑前提可满足；完整 research support-input schema witness 由上述 Python artifact
提供。它仍不是 PNG 或 production schema witness。Lean 文件 SHA256：

```text
c765ea855439995402069d8b2df2a9a2201dc0ee016203d0ef8fcfc1002d5832
```

## 下一版必须做

不能输出单一 owner，而应构造有限 owner candidate domain：

- stable intersection candidate ID；
- exact point，以及每个 incident carrier 的 exact `t` 和 start/end/interior role；
- 每个 overlap pixel 的 owner domain：intersection ID、fragment ID 或 residual/reject；
- multi-seed component 的 tie/equivalence group；
- width、AA margin、tube/raster kernel 等可审计 envelope 证书；
- glyph/text/charge collision 进入显式 ambiguity，不得硬分；
- candidate support RLE/count/hash/capacity；
- outside-any-envelope 的 closed reject 证书；
- parent provenance 与最终 PNG 绑定。

人为 Voronoi、最近 seed 或 ID precedence 可以作为规范化约定，但必须明确标记为 convention，不能声称是图片真值。

## 验证

```text
专项反例：15/15
全量 formal audit：245/245
Lean build：31/31 jobs
source -> observation projection：byte exact
5 个 artifact：内存重建逐字节一致
`validate_self_consistent_bundle`：内部完整 source/projection/world/manifest 门
`validate_bundle`：再校验五个冻结 SHA；同步全链替换被拒绝
```
