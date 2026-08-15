# PNG Ownership Realization v0.5 探索协议

日期：2026-08-15

审计状态：本协议、实现和结果在同一未提交工作树内迭代，且根据结果后的互审修正过 full-overlap 口径、术语和
over-capacity 表达。因此它不是 commit-level preregistration，只能支持 exploratory computational result。独立确认
实验必须在实现前另行提交冻结协议。

## 目标

`Intersection Ownership v0.4` 的决定性 witness 来自完整 research support schema，但不是 renderer 直接生成的
per-carrier raster support。本实验只回答：现有冻结整数 renderer 是否能实际生成一张 raw PNG，其 per-carrier
support 对同一个 intersection universe 允许至少两份 locally admissible ownership annotation。

这不是“两个不同化学图生成相同 PNG”的实验。两份 partition 只是同一个 primitive program 的不同 support 注释。

## 冻结 renderer

复用：

```text
formal/chemgraph/challenge_v0_4/rasterizer_v0_4.py
```

必须记录源文件 SHA256、raw grayscale SHA256、canonical PNG SHA256，以及：

```text
union(per-carrier rasterize_support)
== {pixel | rasterize_gray(all carriers)[pixel] != 255}
```

不使用 PIL/Cairo，不读取模型、GT 或 semantic layer。

## R0 场景

三条独立 carrier：

```text
H : (10,20) -> (30,20)
V1: (18,10) -> (18,20)
V2: (22,20) -> (22,30)
```

坐标使用 Q16。phase：

```text
edge   = integer pixel edge
center = all coordinates + 1/2 pixel
```

探索矩阵：

```text
phase = edge, center
width = 2, 3, 4, 5, 6 pixels
cap   = butt, round, square
AA    = binary_1x, box_2x2, box_4x4
```

共 `2 * 5 * 3 * 3 = 90` 个配置；结果报告不得从该矩阵挑选后删除失败项。

两根竖线是 `opposite half verticals`：一根在 H 终止，另一根从 H 起始。本 fixture 专门验证 endpoint-touch 型的
两个相邻 intersection，不代表两条全长平行 carrier。

## Ownership universe

对每个 pixel 重建完整 carrier fiber。ownership universe 冻结为全部 cross-carrier overlap：

```text
U = {p | |carriers(p)| >= 2}
```

其中：

```text
L = {p in U | carriers(p) contains H,V1 but not V2}
R = {p in U | carriers(p) contains H,V2 but not V1}
A = {p in U | carriers(p) contains H,V1,V2}
```

若存在既不含 H/V1、也不含 H/V2 的 overlap，其 owner domain 为空，该配置的 admissible partition 数必须为 0；不得从
universe 删除后再计数。

intersection seed 使用 exact geometry crossing 所在 pixel；seed 必须实际位于对应 eligible support。Partition
annotation 由把 A 分给左/右两 region 的 total assignment 构成，并要求：

- 左 region 包含 L 和左 seed，右 region 包含 R 和右 seed；
- 两 region 非空、8-connected；
- 两 region 不交，union exact cover U；
- 左 region 每个 pixel carrier fiber 包含 H/V1；右侧同理包含 H/V2。

当 `|A| <= 20` 时穷举全部 `2^|A|` assignment；超过时该配置固定报告
`ambiguity_within_capacity=false`、`enumeration_status=OVER_CAPACITY` 和 `count=null`，不采样估计，也不从矩阵
删除。`null` 表示数学计数未知，不能当作 0。
若某个配置的 exact seed 不在对应 eligible support，该配置固定报告 `seed_available=false` 和 0 个合法 partition，
不能中止或从 90 格矩阵中删除。

## 指标

每个配置固定报告：

```text
foreground pixel count
ownership universe / L / R / A count
unowned overlap count
seed available
locally admissible partition count，或 over-capacity `null`
support-union exact
raw gray SHA256
canonical PNG SHA256
```

还要输出一个确定性 replay manifest；同一配置重复生成必须 byte exact。

## 探索裁决门

```text
R0 exploratory computational GO:
  至少一个探索矩阵配置 locally_admissible_partition_count >= 2
  且 support-union exact
  且每份 partition 通过全部局部 admissibility 条件

R0 exploratory computational NO-GO:
  90 个配置均未枚举出两份 locally admissible partition
  或决定性结果不能重放
```

即使 GO，也只证明：现有 renderer 的真实 per-carrier raster support 允许多份局部结构 annotation。它不证明：

- 两个非同构化学图具有相同 raw PNG；
- parent support extractor 能从 PNG 恢复 renderer oracle support；
- 文字与键不可区分；
- UniqueGraph/SDF 可用；
- 可以训练。

下一独立 R1 才测试 crossing-vs-junction 与 double-bond-vs-parallel-singles 的异图 raw gray / canonical PNG /
production tensor equality；未在有限矩阵发现碰撞也不能证明一般可辨识。
