# PNG Ownership Realization v0.5 结果

日期：2026-08-15

## 裁决

```text
actual renderer per-carrier support: GO
global overlap admits >=2 locally admissible labels: GO
same PNG/support, different local annotations:    GO
R0 exploratory computational result:              GO
preregistered independent confirmation:           NO-GO
conditional Lean antecedent bridge:               NOT FORMALIZED
distinct chemical/semantic worlds:                NOT ESTABLISHED
non-isomorphic chemical graph collision: NOT TESTED
PNG -> parent support completeness:       NO-GO
UniqueGraph / SDF / training:              NO-GO
```

现有 Q16 整数 renderer 可以直接生成同一张 raw PNG 和同一组 per-carrier support，而全部 cross-carrier overlap 对
两个相邻 intersection 存在多份满足 seed、8-connectivity、互斥、global exact cover 与 carrier incidence 的
ownership annotation。这里的 admissible 只表示通过探索协议的局部结构条件，不是化学语义 legal world。该结果不再
依赖手写 snake 或人工 support pixel。

这里的“探索协议条件”不代表已有先于结果的 Git commitment。协议、实现、
结果和审计修订发生在同一未提交周期，因此本包是 exploratory evidence，不是确认性证据。

## 冻结场景

```text
H :  (10,20) -> (30,20)
V1:  (18,10) -> (18,20)
V2:  (22,20) -> (22,30)
```

V1 在 H 终止，V2 从 H 起始。它是 endpoint-touch 型的 `opposite half verticals` fixture，不代表两条全长平行
carrier。所有 ownership 计算使用全部 cross-carrier overlap；没有 H/V1 或 H/V2 合法 option 的 pixel 会令对应
配置得到 0 个合法 partition，不会从 universe 裁掉。

## 完整矩阵

探索协议中的 90 个配置全部运行：

```text
phase: edge / center
width: 2 / 3 / 4 / 5 / 6 px
cap: butt / round / square
AA: binary_1x / box_2x2 / box_4x4
```

结果：

```text
configuration count:             90
support union exact:              90/90
unowned overlap nonzero:           0/90
locally admissible count >= 2:     36/90
max enumerated count:              25,032
seed unavailable -> zero:         15/90
ambiguity over capacity -> null:    3/90
```

没有从矩阵中删除失败配置。代表性结果：

```text
edge, width=5, round, box_4x4:
  overlap = 24 left-only + 10 triple + 24 right-only
  locally admissible partitions = 873

edge, width=5, square, box_4x4:
  overlap = 24 left-only + 12 triple + 24 right-only
  locally admissible partitions = 2,415

center, width=4, butt, box_4x4:
  overlap = 14 left-only + 1 triple + 14 right-only
  locally admissible partitions = 2

center, width=5, round, box_4x4:
  overlap = 20 left-only + 5 triple + 20 right-only
  locally admissible partitions = 32
```

## Artifact

```text
rasterizer_v0_4.py:
5d673ae1c33648b5498b4ebacfc1b91136fee5abdbae4fa1503a144a59816943

results.json:
2b396f1dc8c0e4d43bb34f852a0785800285e361bc7fb2cea889ab034bf8477a

representative_witness.json:
e00421918b62adc1b76c42ca1a9cb1b5c9981d3ee33d7e6f1f96dcebf63c268d

representative_edge_w5_round_box4.png:
0152efc7791f03515ef554739b545088b36d654aa081da9718d0c31eef3ecb4e

ownership_realization.py:
0ae56b499f62efa37ece7ba086214ad4483efabd5f910a61fca12c281617f5b9

test_ownership_realization.py:
482552be4535810de24d6a262900d0a4ef0802918600a1c9e5c3604c9bd71a5c
```

代表 PNG 是 `768 x 768` canonical grayscale PNG。witness 保存三条 carrier 的完整 RLE、全部 ownership universe、
unowned set 和两份不同 locally admissible partition annotation；二者引用同一 PNG。

## 关键纠错

探索阶段最初使用两条全长竖线，并只枚举 H 参与的局部 overlap，得到了 `2415/32`。但粗竖线会在 H 之外彼此
重叠，这些 V1/V2-only pixel 没有合法 intersection owner；忽略它们不满足 global exact cover。该口径在生成冻结
artifact 前撤回，最终 fixture 改为 opposite half verticals，并把 universe 改为全部 cross-carrier overlap。

## 结论边界

两份 annotation 共享同一个 primitive program、同一组 carrier supports 和同一 raw PNG，只改变 overlap pixel 的
region label。因此该实验给出了把 v0.4 抽象 NO-GO 连接到实际 renderer support 的可执行前提候选，但桥尚未进入
Lean，也仍未给出两个真实 semantic world 或两个非同构化学图的碰撞。

当前测试会重放完整矩阵和 witness，但仍 import 构造器的局部 admissibility helper；独立审计已在会话中复算 873，
却尚未作为不依赖构造器 helper 的 verifier 落盘。因此 commit-level preregistered confirmation 继续 NO-GO。

下一步应分别做：

1. 把这些 renderer world 接到 Lean `GlobalOwnershipNoGoV04` 的 Observation/World 接口，证明桥接条件；
2. 实现 factorized owner candidate domain，确保 873/2415 等解族不被单 winner 压扁；
3. 独立 R1 测试 crossing-vs-junction、double-bond-vs-parallel-singles 的 raw gray、canonical PNG 和 production
   tensor equality；
4. 只有 candidate graph truth coverage 与真实图片稳定性门通过后，才重新评审训练。

## 验证

```text
PNG ownership 专项：6/6
全量 formal audit：251/251
90 格 matrix：byte-exact replay
representative witness / PNG：byte-exact replay
git diff --check：通过
```
