# Fragment Support Partition v0.4 结果

日期：2026-08-15

## 裁决

```text
per-parent support partition research: GO
cross-carrier pixel ownership:          NO-GO
parent support extraction completeness: NO-GO
v0.4 production schema freeze:          NO-GO
public proposer / UniqueGraph / SDF:     NO-GO
training:                               NO-GO
```

本周期只证明：给定一个 base stroke 的 parent pixel set 和 exact ordered cuts，精确投影加右连续半开区间可以把
parent support 确定性分成 child supports。

## 冻结规则

```text
[t_i, t_{i+1})
final child includes t = 1
projection = cut -> right child
projection < 0 -> first child
projection > 1 -> last child
```

实现使用整数 dot product、`Fraction` 和 `bisect_right`。不使用浮点阈值。

## 已验证性质

- 每个 parent pixel 恰好进入一个 child；
- 同 parent 的不同 child support 两两不交；
- child support 并集逐像素等于 parent support；
- 任一 child 为空时 closed reject；
- RLE 必须严格排序、无 overlap、同行相邻 run 必须合并；
- parent/child RLE 均绑定 SHA256；
- RLE hash framing 与 v0.3 一致：`u32be(run_count)||u32be(row,start,length)...`；
- parent/child membership 总数、unique parent pixel 数和 output bytes 受容量门约束；
- parent 与 child 的总 run 数分别受独立、增量早停的容量门约束；
- 所有 cross-carrier overlap 逐像素列出完整 carrier fiber；
- declared 输出按 strict native JSON type 和 canonical bytes 比较，`bool/float/string` 不能冒充整数；
- 水平、垂直、两种对角线的 exact cut tie 均归右 child；
- 平移、90 度旋转和反射保持分区计数；
- 正好命中 pixel capacity 成功，capacity+1 closed reject。

## 红队边界

X 场景的中心像素会同时存在于两个 parent carrier。进一步给两者都注入远离交点的 `(0,0)`，per-parent 分区仍
数学自洽。因此当前输出会明确列出两个 overlap：真实中心与远端异常点，但不会擅自把它们声明为合法
intersection support。

这证明 per-parent cover/disjoint 不是 global ownership。下一层必须要求：

```text
cross-carrier overlap pixel
<-> pixel belongs to one named, geometry-valid intersection region
```

非 intersection overlap 必须 closed reject；intersection pixel 也不能继续同时归属普通 child owner。

## Lean 边界

Lean 证明 total bucket 诱导的 filter fibers 具有 unique ownership、pairwise disjoint 和 exact cover；进一步以 exact
cross multiplication定义右连续 bucket，并证明 tie、首尾和 cover/unique 性质。

Lean 不独立证明：cut 已约分、`0 < cut < 1`、cut 有序无重复、`normSquared > 0`、PNG 或 parent support extractor
正确。这些仍由 Python/root 前置门承担，未来 production schema 必须显式携带对应证书。

## 验证规模

```text
专项测试：25/25
全量 formal audit：230/230
随机 support partition：1,000/1,000
独立 integer oracle：100,000/100,000，与 Fraction+bisect mismatch=0
Lean build：30/30 jobs
```

最终实现哈希：

```text
FragmentSupportPartitionV04.lean:
bed2ba3cb736032ad997c9673c4af81b6ba6e627539b0fbe28123c3ed73d5d1b

support_partition.py:
c36eef270ebee85a842e445e0048b214dec7382544766b7b29a30062d0240e36

test_support_partition.py:
5679a0587433667de5f17a43782b21a06993bd7098faaa97f706434f3e3419d2
```

## 下一门槛

1. 从 v0.3 boundary support + canonical matching 独立重建 parent support，不能继续由 fixture 自报；
2. 构造 named intersection support universe；
3. 证明 intersection support 与普通 child support 的 exact-cover ownership；
4. 证明非交点区域跨 carrier support 不相交；
5. 公开 truth coverage 通过后，才评审 v0.4 schema freeze。
