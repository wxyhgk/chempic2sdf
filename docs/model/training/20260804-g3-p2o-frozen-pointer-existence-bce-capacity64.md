# G3 P2-O Frozen Pointer Existence BCE Capacity64

> 日期：2026-08-04
>
> 状态：严格隔离完成 / 65参数线性head容量NO-GO

## 受控问题

P2-O加载P2-J final，冻结291,329个非existence参数，pointer-only Hungarian固定query标签，
只训练65参数`Linear(64→1)`。与P2-M相比唯一变化是删除极值boundary，loss只剩weighted BCE：
正query权重1、负query权重0.1。训练1,600步、6,400 exposures。

## 结果

| 域 | P2-J step0 | P2-O final | Count exact |
|---|---:|---:|---:|
| easy32 | 963/967，28/32 | **960/967，26/32** | 32/32 |
| hard32 | 798/807，26/32 | **795/807，24/32** | 32/32 |

BCE从约0.307降到0.162并平台，但整组exact略回退。与P2-M不同，本实验没有boundary梯度。

P2-Q后的复核发现，本实验将P2-J存在感知的Hungarian identity换成了pointer-only identity；两套
正query只重合约90%。因此这里只能否决“切换identity后训练线性head”，不能证明原identity线性不可分。

## 决策

- 正式结论：当前固定标签协议`frozen_pointer_existence_bce_capacity_no_go`。
- 不再调该identity下的线性head学习率、权重、margin或epoch。
- 当前最好完整训练仍是P2-J 4,800步的28/32、26/32；同loss续训P2-K最好观察30/32、29/32，
  但未通过门且不做事后选模。
- P2-P冻结P2-J的pointer/query/Count，唯一把65参数线性head替换为小型非线性MLP，再用相同
  固定标签weighted BCE做容量测试；未过则query hidden本身缺少可分证据。

```text
model = chemvector-g3-p2o-frozen-pointer-existence-bce-capacity64
checkpoint = outputs/smoke/chemvector-g3-p2o-frozen-pointer-existence-bce-capacity64/step-001600.pt
SHA256 = 54458807a790a3cb31f40822cf100ba85fbc5ed8978ba66505482092ad763d0f
training images = 64
calibration samples seen = 6,400
trainable parameters = 65
decision = NO-GO
```
