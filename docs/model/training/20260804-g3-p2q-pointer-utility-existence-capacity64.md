# G3 P2-Q Pointer Utility Existence Capacity64

> 日期：2026-08-04
>
> 状态：utility head NO-GO / 固定标签协议错误已定位

## 实验

P2-Q冻结P2-J主干，为每个query增加10个target-independent特征：pointer峰值、top1-top2差、
归一化熵、概率加权与argmax候选的raw score、normalized score、rank，以及空间方差。
`64D hidden + 10D utility`进入9,877参数MLP，只用固定标签weighted BCE训练1,600步。

| 域 | P2-P hidden-only | P2-Q + utility |
|---|---:|---:|
| easy32 | 952/967，20/32 | **955/967，22/32** |
| hard32 | 795/807，22/32 | **793/807，23/32** |

utility有小幅exact收益，但仍显著低于P2-J的28/32、26/32，正式NO-GO。

## 固定标签协议复核

P2-J训练时query-to-GT Hungarian cost包含existence tie-break；P2-M至P2-Q为固定标签将该权重设为0。
冻结P2-J final后比较两套正query集合：

| 域 | weight1/weight0身份重合 | top-N对weight1 | top-N对weight0 | weight0 pointer oracle |
|---|---:|---:|---:|---:|
| easy32 | 875/967，90.49% | 98.66% | 89.45% | 967/967，32/32 |
| hard32 | 722/807，89.47% | 98.27% | 87.98% | 807/807，32/32 |

两套query身份都能通过pointer得到完整GT，但它们是不同的等价排列。M至Q的校准实际上要求head
放弃P2-J已经形成的identity，改学另一套pointer-only identity。因此这些实验只能否决各自的
校准协议，不能证明P2-J query hidden缺少对象证据。

## 决策

- P2-Q正式结论：`pointer_utility_existence_capacity_no_go`。
- 停止继续堆utility或MLP。
- P2-R冻结P2-J final时的existence-aware Hungarian正query集合为Action Cache，随后冻结主干、
  从P2-J原existence head继续训练；标签不再随step变化，也不切换等价身份。这是正确的EM式M-step。

```text
model = chemvector-g3-p2q-pointer-utility-existence-capacity64
checkpoint = outputs/smoke/chemvector-g3-p2q-pointer-utility-existence-capacity64/step-001600.pt
SHA256 = 2cf140988f2cb33acb9df570e390456c848fc457e35ab73ec04b6be7ab591c50
training images = 64
calibration samples seen = 6,400
trainable parameters = 9,877
decision = NO-GO
```
