# G3 P2-S Fixed Identity Listwise Existence Capacity64

> 日期：2026-08-04
>
> 状态：平滑整组排序NO-GO

## 实验

P2-S严格复用P2-R Action Cache与P2-J final初始化，冻结除65参数existence head外的全部组件。
目标为固定标签weighted BCE，加上覆盖每张图全部正负query对的平滑`logsumexp` listwise排序损失；
它强调越界query，但不使用动态`min-positive/max-negative`硬极值。

| 域 | P2-R BCE | P2-S BCE + listwise | identity exact |
|---|---:|---:|---:|
| easy32 | 964/967，29/32 | **963/967，28/32** | 21/32，954/967 overlap |
| hard32 | 799/807，27/32 | **798/807，26/32** | 24/32，791/807 overlap |

Count仍为双域32/32，但listwise没有改善固定identity排序，结果回到P2-J final水平。

## 决策

- `fixed_identity_listwise_existence_capacity_no_go`。
- 不再调线性head的BCE/listwise权重、margin或epoch。
- P2-R强制cache identity已证明pointer解码双域32/32；P2-R和P2-S两种目标均不能让共享线性head
  完成排序。P2-T在相同cache上只把65参数线性head替换为8,577参数MLP，用BCE-only严格隔离容量。

```text
model = chemvector-g3-p2s-fixed-identity-listwise-existence-capacity64
checkpoint = outputs/smoke/chemvector-g3-p2s-fixed-identity-listwise-existence-capacity64/step-001600.pt
SHA256 = 4da80d092b2da351c0709ade84dcda00babb77a8eecc69a13b7b6025b540e58b
Action Cache SHA256 = d2134087c5c1d79cb698a3aefa90fcfeb6e3defeba3e602e1e1317a5362e8aca
training images = 64
calibration samples seen = 6,400
trainable parameters = 65
decision = NO-GO
```
