# G3 P2-M Frozen Pointer Existence Head Capacity64

> 日期：2026-08-04
>
> 状态：65参数head容量NO-GO

## 受控问题

P2-M加载P2-J final，冻结291,329个encoder、decoder、pointer、Count和query参数，只训练
`Linear(64→1)`的65个existence参数。query-to-GT Hungarian只读取冻结pointer mass，
existence cost权重固定为0，因此query标签不再随head输出变化。训练1,600步、6,400 exposures。

## 结果

| 域 | P2-J | P2-M | Count exact |
|---|---:|---:|---:|
| easy32 | 963/967，28/32 | **803/967，0/32** | 32/32 |
| hard32 | 798/807，26/32 | **637/807，0/32** | 32/32 |

即使pointer与标签完全冻结，极值boundary仍约1.33，batch exact接近0。说明P2-J的64维query hidden
对该固定对象标签不能由单个共享线性head严格分开；继续调head学习率、margin或epoch没有依据。

## 决策

- 正式结论：`frozen_pointer_existence_head_capacity_no_go`。
- 独立existence head路线停止。
- 下一模型把NULL作为pointer的第129类：匹配query监督到唯一候选，未匹配query监督到NULL；
  existence直接等于“任意候选pointer相对NULL的优势”，从表示上消除object/pointer身份脱节。

```text
model = chemvector-g3-p2m-frozen-pointer-existence-head-capacity64
checkpoint = outputs/smoke/chemvector-g3-p2m-frozen-pointer-existence-head-capacity64/step-001600.pt
SHA256 = 3387651af91f504d8dd9aa876650b8849afb8784ec085fb71d2fbfe51fb8f9e9
training images = 64
calibration samples seen = 6,400
trainable parameters = 65
decision = NO-GO
```
