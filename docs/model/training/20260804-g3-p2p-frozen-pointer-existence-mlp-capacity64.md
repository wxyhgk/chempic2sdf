# G3 P2-P Frozen Pointer Existence MLP Capacity64

> 日期：2026-08-04
>
> 状态：非线性head容量NO-GO

## 受控改动

P2-P与P2-O使用相同P2-J冻结表示、pointer-only固定标签、weighted BCE、1,600 steps和64张图。
唯一改动是把65参数`Linear(64→1)`替换为8,577参数：

```text
LayerNorm(64) -> Linear(64,128) -> GELU -> Linear(128,1)
```

## 结果

| 域 | P2-O线性head | P2-P MLP | Count exact |
|---|---:|---:|---:|
| easy32 | 960/967，26/32 | **952/967，20/32** | 32/32 |
| hard32 | 795/807，24/32 | **795/807，22/32** | 32/32 |

MLP BCE降到约0.125，低于线性head约0.162，但top-N exact反而下降。因此单纯增加query-hidden
分类容量不能解决，BCE降低也不是整组选择改善的可靠代理。

## 决策

- 正式结论：`frozen_pointer_existence_mlp_capacity_no_go`。
- 停止继续扩大只读取64维query hidden的head。
- 下一步应显式给head提供target-independent pointer证据：pointer最大概率/熵、概率加权的原始
  center score、normalized score、rank，以及argmax候选的对应值。pointer、Count和主干继续冻结。

```text
model = chemvector-g3-p2p-frozen-pointer-existence-mlp-capacity64
checkpoint = outputs/smoke/chemvector-g3-p2p-frozen-pointer-existence-mlp-capacity64/step-001600.pt
SHA256 = aac566b7093196b53bb73ca67e604824562967bf306f85e70ad4a942f475b1f4
training images = 64
calibration samples seen = 6,400
trainable parameters = 8,577
decision = NO-GO
```
