# Owner Capacity v0.5 Axis Plan 预注册

日期：2026-08-15

## 冻结范围

本阶段为 12 个容量轴冻结 measurement point、计数公式、gate 顺序、category、证书和八份 L/H source witness。
生成过程没有调用 R0 candidate constructor。

```text
axis_plan SHA256:       06366e83731f1ca9dee116f995da14dd5a7e64c68f45e98f2a847c0e423bcd89
source_manifest SHA256: a06489212869743c414d8f87729c4ff6d3836f7bec3b7c5ef8af908ad7410118
witness count:          8
```

四个可构造轴为：

```text
max_carriers:        L=8, H=9, end-to-end candidate
max_intersections:   L=8, H=9, rejection-only
max_total_rle_runs:  L=65536, H=65537, end-to-end candidate
max_pixel_variables: L=4096, H=4097, end-to-end candidate
```

其余八轴由 Lean 条件算术与冻结 Python checker 标为 conditional unreachable、not-isolated、shadowed 或 unproven。
Checker 尚未从任意 Python source 重算全部 Lean 前提，手工 3,353,654-byte schema envelope 也尚未升级为 universal
byte-bound proof，因此这些轴一律保持逐轴 NO-GO。即使四个可构造轴全部确认，聚合状态仍只能是
`PARTIAL/NO_GO_GENERIC`。

## Source 闭包

每份 observation 都是 `synthetic_capacity` source：

- parent RLE、compiled child RLE、carrier/intersection/pixel counts 从 source 重算；
- parent-support union 被确定性渲染为真实 768×768 grayscale PNG 并绑定 SHA；
- renderer 绑定 `capacity_png.py` bytes；
- `geometry_renderer_binding=false`，不声称这些 PNG 来自冻结 straight-segment renderer，也不授权任何化学语义。

每个 H 的 count vector 已机械验证目标轴严格超界；其余 11 轴使用 exact/pre-gate count，加上尚待实现绑定的手工
output-byte envelope。故隔离状态仍是 conditional，正式确认器必须补完 schema dominance proof 后才能升级。Plan 同时
冻结 raw、canonical distinct、deduplicated 和 compiled measurement layers。

## 污染隔离

预注册期间一个 subagent 违反只读约束，使用另一套未冻结内存 fixture 提前调用 constructor。其 fixture 计数与当前
witness 不同，且没有 source/builder/dependency SHA 或 sandbox 证据，已写入 `EXPLORATORY_EXCLUSION.md` 并永久排除
在正式 verdict 之外。

## 下一门

提交本 plan 后才能实现并运行新的隔离确认器。确认器必须绑定 committed plan/source/builder/constructor SHA，独立
重算 count vector，验证 L 的非空 feasible world、H 的精确 target error 和全部副作用。当前仍为
`training_authorized=false`。
