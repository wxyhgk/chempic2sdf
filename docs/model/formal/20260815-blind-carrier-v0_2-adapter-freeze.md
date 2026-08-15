# Blind Carrier v0.2 纯适配器冻结

日期：2026-08-15

## 结论

`VisibleSegment + TerminalHint + SemanticBondGroup -> Blind Carrier v0.2` 的确定性纯适配器已经通过
正例、反例、容量边界、冻结 validator 和独立红队复核，可以冻结。

这不是完整图片 proposer。它不读 PNG、不预测线段、不预测 Bond 分组，也不证明真实分子识别准确。

```text
blind_sample_selection_authorized=false
proposer_execution_authorized=false
training_authorized=false
```

## 数学含义

适配器使用 endpoint carrier disk 表示端点的不确定区域。一个 terminal 内部的 disk 相交图必须连通；
不同 terminal 之间的 disk 必须完全不相交。因此单解释输出中的 terminal 恰好对应 disk 相交图的连通
分量，不能靠把 hint 平移 `1 Q4` 将同一个可见 junction 拆成两个图节点。

Lean 模块 `ChemGraph.BlindCarrierTerminalComponentsV02` 证明：

- `different_terminals_not_reachable`：不同 terminal 之间不可达；
- `same_terminal_reachable`：同一 terminal 内可达；
- `reachable_iff_same_terminal`：可达关系当且仅当 terminal 身份相同；
- `terminal_partition_unique_up_to_class_equivalence`：任意精确连通分量标号都诱导相同分区。

新增 Lean 源码 SHA256：

```text
843133e3acb30c74a3850b527bef9ba3c5b6d2dff76c7736a78829d97a76bf7d
```

上述新增定理不依赖任何 axioms。

标准平行三键的三个 endpoint disk 不必有共同交点，只需形成 `0-1-2` 连通路径。这保留了 v0.2 对
三键的表达能力，同时补上了 terminal 分区唯一性。

## 已封闭的反例

- `bool` 不能冒充任何整数输入；
- 一条物理线不能跨不同 Bond 拆成偶数/奇数像素的两个假单键；
- 跨 Bond 的 `1 Q4` near-overlay 被拒绝；
- 不同 hint ID 不能使用同一坐标；
- hint 坐标不同但派生 endpoint region 相交时也被拒绝；
- terminal 超过 64 个 endpoint ref 时 closed reject；
- one-segment-per-lane 的 512 lane 正边界通过，513 closed reject；
- 无向 Bond 交换 begin/end、整体重命名 opaque hint ID 时输出字节不变；
- 非预期 iterable 运行异常传播给 runner，不伪装成协议 reject。

## 验证结果

```text
adapter unit tests                 27/27
frozen v0.2 validator tests        33/33
full formal audit                 112/112
Lean default build                 25/25
red team                           GO
schema-invalid ok outputs           0
```

适配器相对路径 tree：

```text
1d883d7687e17204be9bfc65cc298c8c3f214686c134bb2efbdd0f7134284fe1
```

framing：排除 `__pycache__`，按相对路径 NUL 排序，对每个完整文件执行 `sha256sum`，再对清单完整
字节执行一次 SHA256。

## 剩余边界

- 当前只支持一条 lane 对应一条直 segment；折线、多段线和曲线不在合同内；
- 相交 terminal region 会保守拒绝，未来若要保留歧义，必须显式输出多个 interpretation；
- 输入的 segment、support、语义 Bond 分组仍由上游提供，其覆盖率和精度尚未验证；
- OCR、原子身份、化学价态、SDF 生成不属于本层；
- 不允许据此选择 blind 样本或开始训练。

下一步是实现公开图片 proposer：`PNG -> 墨迹分离 -> 物理 segment -> AtomPair 拓扑 -> support 分配
-> 冻结 v0.2 adapter`。只能先在合成图和已暴露 MR-TADF 上审计；公开门通过后才讨论 blind 20。
