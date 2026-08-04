# G3 P2-L Existence Boundary Query Set Capacity64

> 日期：2026-08-04
>
> 状态：协议通过 / 极值边界明确NO-GO

## 受控改动

P2-L从与P2-J相同的随机初始化重开，架构、64张图、4,800 steps、pointer-mass Hungarian、
唯一pointer和Count均不变。唯一新增项是权重1.0、margin 1.0的整图极值损失：最低正query
existence必须高于最高负query。

## 结果

| 域 | P2-J | P2-L | Count exact |
|---|---:|---:|---:|
| easy32 | 963/967，28/32 | **603/967，0/32** | 32/32 |
| hard32 | 798/807，26/32 | **456/807，0/32** | 32/32 |

训练全程每个4图batch的`boundary_exact_images`均为0。pointer和coordinate一度快速收敛，但
boundary loss长期停在约1.34，object BCE停在约0.57；极值query随Hungarian身份变化，梯度持续
追逐新的最弱正/最强负，最终使existence排序塌缩。

## 决策

- 正式结论：`existence_boundary_query_set_capacity_no_go`。
- 禁止调margin、加权重或追加epoch；P2-L目标停止。
- P2-M回到P2-J final，冻结encoder、decoder、pointer和Count，仅训练65参数existence head。
  query标签只由冻结pointer mass确定，Hungarian不再读取existence分数；在固定标签上再测试边界。

```text
model = chemvector-g3-p2l-existence-boundary-query-set-capacity64
checkpoint = outputs/smoke/chemvector-g3-p2l-existence-boundary-query-set-capacity64/step-004800.pt
SHA256 = aa9c94f1a65da766744968d2f7f1fbae93a7238ba1a7f75163cc8ef5ba06286f
training images = 64
samples seen = 19,200
decision = NO-GO
```
