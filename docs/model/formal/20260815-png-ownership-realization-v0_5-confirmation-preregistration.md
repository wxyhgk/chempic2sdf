# PNG Ownership Realization v0.5 独立确认预注册

日期：2026-08-15

## 身份与顺序

本协议在独立 verifier 实现和确认运行之前提交。它不把此前探索包装成盲测，而是确认已经公开的精确假设。

```text
exploratory evidence commit: 41dc1f4
candidate-domain preregistration commit: 030376f
```

冻结输入：

```text
rasterizer_v0_4.py SHA256:
5d673ae1c33648b5498b4ebacfc1b91136fee5abdbae4fa1503a144a59816943

representative PNG SHA256:
0152efc7791f03515ef554739b545088b36d654aa081da9718d0c31eef3ecb4e

representative witness SHA256:
e00421918b62adc1b76c42ca1a9cb1b5c9981d3ee33d7e6f1f96dcebf63c268d

exploratory matrix SHA256:
2b396f1dc8c0e4d43bb34f852a0785800285e361bc7fb2cea889ab034bf8477a
```

## 冻结确认假设

只确认一个事先公开的 representative，不重新挑参数：

```text
phase = edge
width = 5 pixels
cap = round
AA = box_4x4

H  = (10,20) -> (30,20)
V1 = (18,10) -> (18,20)
V2 = (22,20) -> (22,30)
```

确认值：

```text
all cross-carrier overlap U = 58 pixels
left-fixed                     = 24
triple / ambiguous             = 10
right-fixed                    = 24
unowned                        = 0
locally admissible partitions = 873
support union == PNG nonwhite = true
```

## 独立性门

确认 verifier 必须是新文件，并满足：

1. 不 import `png_ownership_realization_v0_5.ownership_realization`、`build_results` 或测试 helper；
2. 只允许 Python 标准库和冻结 `challenge_v0_4.rasterizer_v0_4`；
3. 自行实现 canonical JSON、PNG grayscale 解码、RLE 解码、carrier fiber、8-connectivity 和 subset enumeration；
4. 从冻结 Q16 segment/style 独立重跑三份 `rasterize_support` 和共同 `rasterize_gray`；
5. 独立计算全部 cross-carrier overlap，不信任 witness 自报 universe/count；
6. 穷举全部 `2^10=1024` assignment，不读取探索结果中的 `873` 作为循环边界；
7. witness 中两份 annotation 必须分别重验 seed、8-connectivity、disjoint、global exact cover 与 incidence；
8. PNG 必须由标准库解码为 768×768 8-bit grayscale，非白集合与 renderer support union 相等；
9. verifier 源码 SHA 和确认结果 SHA 必须写入结果文档；
10. 正常运行至少两次 byte exact，且 mutation tests 必须拒绝漏像素、双归属、断连、改 seed、改 RLE、改 PNG。

仓库测试可以调用独立 verifier 的公开入口，但不得反向让 verifier import constructor。

## 裁决

```text
confirmation GO:
  独立重算逐项命中全部冻结确认值
  且两次输出 byte exact
  且全部 mutation tests 被拒绝

confirmation NO-GO:
  任一确认值 mismatch
  或 verifier 依赖 constructor helper
  或只比较 SHA/自报 count 而未重建像素与 partition
```

即使确认 GO，授权范围仍只有：

> 对这个冻结 renderer fixture，完整 cross-carrier overlap 至少允许两份 locally admissible annotation。

它仍不建立真实 chemical semantic worlds、非同构图碰撞、PNG extractor completeness、UniqueGraph、SDF 或训练授权。
