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

即使pointer与标签完全冻结，极值boundary仍约1.33，batch exact接近0。

复核：本实验同时使用了冻结线性head和极值boundary，不能单独证明线性head容量不足。P2-N之后
必须补做同一冻结head的weighted-BCE-only隔离；这里只能判定极值目标失败。

## 决策

- 正式结论：`frozen_pointer_existence_head_capacity_no_go`。
- 当前“冻结线性head + 极值boundary”停止。
- P2-N先测试NULL pointer统一语言；无论其结果如何，仍需补做冻结线性head的BCE-only隔离。

```text
model = chemvector-g3-p2m-frozen-pointer-existence-head-capacity64
checkpoint = outputs/smoke/chemvector-g3-p2m-frozen-pointer-existence-head-capacity64/step-001600.pt
SHA256 = 3387651af91f504d8dd9aa876650b8849afb8784ec085fb71d2fbfe51fb8f9e9
training images = 64
calibration samples seen = 6,400
trainable parameters = 65
decision = NO-GO
```
