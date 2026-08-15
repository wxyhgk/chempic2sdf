# Intersection Ownership v0.4 可辨识性探针

本目录测试一个看似自然的规则：把 cross-carrier overlap 的 8-connected component 与其包含的 exact geometry
intersection anchor 对齐。

它能拒绝：

- 不含任何 anchor 的断开远端 overlap；
- 同一 component 合并两个 geometry intersection；
- component 的 carrier union 与 intersection fiber 不一致。

但它不能证明 semantic soundness。若从真实交点向远处添加一条由多个 carrier 共享的 8-connected 蛇形路径，整个
路径仍只有一个 component、包含一个 anchor，结构规则会全部接收为 intersection region。没有独立 width/tube、
primitive ownership 或 parent-support 来源证书时，无法判断这是合法粗线交点还是错误吸入的文字/噪声。

因此成功状态故意命名为：

```text
STRUCTURALLY_EXACT_NOT_SEMANTICALLY_IDENTIFIED
```

它不授权 production、UniqueGraph、SDF 或训练。

`ownership_worlds.py` 另给出 `v0.4-research JSON` 内的决定性桥像素反例；它不是 PNG 或 production schema
witness。同一个 observation 下，桥 pixel 归左交点或右交点的两份 world
都满足 region 非空、8-connected、包含 seed、两两不交、逐像素 carrier incidence 和 exact cover。两份 world 的
observation SHA256 相同而 owner partition 不同。

Lean 与 Python 是边界一致的独立证据层；当前没有 Lean 到 Python JSON parser 的形式化语义桥。

`build_counterexamples.py` 生成完整 `support_input.json`，实际重跑 geometry/support constructor 后投影出
`observation.json`，再生成 `world_a.json`、`world_b.json` 和 `manifest.json`。测试会从内存重建整链并逐字节重放。
`validate_self_consistent_bundle` 只重算 source projection、两份 world 和 manifest，证明调用者给出的 bundle 内部
一致；它不证明 bundle 就是预注册对象。`validate_bundle` 才是冻结 witness 的权威研究入口：除完整重算外，还逐一
校验五份 canonical artifact 的代码内 trust-anchor SHA256。`validate_world` 仅用于 world-relative 局部检查，不单独
证明 source binding 或冻结身份。
