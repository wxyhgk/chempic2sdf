# PNG Ownership Realization v0.5 独立确认结果

日期：2026-08-15

## 裁决

```text
preregistered independent confirmation:           GO
frozen renderer/support replay:                    GO
global overlap admits >=2 local annotations:       GO
conditional Lean antecedent bridge:                NOT FORMALIZED
distinct chemical/semantic worlds:                 NOT ESTABLISHED
non-isomorphic chemical graph collision:           NOT TESTED
PNG -> parent support completeness:                 NOT ESTABLISHED
UniqueGraph / SDF / training authorization:         NO-GO
```

确认协议先以 commit `61b7779` 独立提交，之后才实现 verifier。该 verifier 不 import R0 探索构造器、build script 或
测试 helper；它只依赖 Python 标准库和协议允许的冻结 Q16 rasterizer。

## 独立重建

`verify_confirmation.py` 重新执行：

1. 用标准库解析 PNG signature/chunk/CRC/IHDR/zlib/filter，得到完整 768×768 grayscale bytes；
2. 从冻结 Q16 segment/style 重跑三条 carrier 的独立 support 和共同 raw gray；
3. 自行严格解码 witness RLE，拒绝乱序、相邻未合并、overlap、越界和非原生整数；
4. 从三份 support 重建每个 pixel 的 carrier fiber 和全部 cross-carrier overlap；
5. 自行实现 8-connectivity 和 admissibility predicate；
6. 用 `itertools.product` 穷举全部 `2^10=1024` assignment，不调用探索枚举器；
7. 重验 witness 中两份 annotation 的 seed、连通、互斥、global exact cover、incidence 与差异；
8. 重验 closed configuration、所有自报标量、carrier support 和 ownership set。

确认值逐项命中：

```text
ownership universe:               58
left fixed:                       24
triple / ambiguous:               10
right fixed:                      24
unowned:                           0
locally admissible partitions:   873
support union == PNG nonwhite:   true
```

## Mutation

测试在 `enforce_frozen_identity=false` 时执行结构 mutation，因此不是仅凭 artifact SHA 拒绝：

- ownership universe 漏 pixel；
- carrier RLE 改位；
- annotation 删除远端 run 造成断连/漏覆盖；
- annotation 扩张造成双归属；
- 注入伪造 seed 字段；
- PNG byte/CRC 改写。

以上全部 closed reject。正常确认连续两次 canonical bytes 完全相同。

## Artifact

```text
confirmation protocol commit:
61b7779

verify_confirmation.py SHA256:
ec4c07e104cb3ccb0732cb3880d77cbf2774fa40ff0b0ce8e0bd448cff35893a

test_confirmation.py SHA256:
e23a6d1702710cc3723121c1a45930bbdb89bea9b2958a3dd6d7823bbccf2533

confirmation.json SHA256:
1be90a13348f815e2069188ac5289ed2a76ce8387aea31202e5cee7c23b32c02
```

## 数学边界

本确认严格建立的是：对冻结 renderer fixture 和预注册局部 admissibility predicate，完整 overlap universe 有 873 份
可接受 annotation；所以任何直接选一个 annotation 的规则都加入了 observation 之外的 convention/先验。

它尚未把 PNG bytes、RLE 与 `legalR0` 桥接进 Lean，也不证明这些 annotation 是真实化学语义 world。下一步可以把
这个确认 witness 实例化到 `GlobalOwnershipNoGoV04`，但结论仍只相对于 `legalR0`。异图碰撞和 PNG-only extractor
completeness 必须独立验证。
