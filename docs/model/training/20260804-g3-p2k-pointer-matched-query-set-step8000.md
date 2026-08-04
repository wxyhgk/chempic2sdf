# G3 P2-K Pointer-Matched Query Set Step8000

> 日期：2026-08-04
>
> 状态：恢复协议通过 / 严格容量NO-GO / 停止追加epoch

## 受控续训

P2-K从P2-J step4800恢复全部291,394参数和AdamW的102项状态，保持同一64张图、固定顺序、
loss与batch size，只将总训练预算延长到8,000 steps。没有重置优化器、重新初始化、调阈值或
选择中间checkpoint。

```text
resume checkpoint SHA256 = a4e5031ae4994b73c62a9ca97625c720aa06d2f816cf322b109fed20f5f2ad0a
start = 4,800 steps / 19,200 exposures
final = 8,000 steps / 32,000 exposures
```

## Final结果

| 域 | P2-J step4800 | P2-K step8000 | Count exact |
|---|---:|---:|---:|
| easy32 | 963/967，28/32 | **965/967，30/32** | 32/32 |
| hard32 | 798/807，26/32 | **801/807，27/32** | 32/32 |

仍未达到双域全Atom、32/32严格门，正式结论为
`pointer_matched_query_set_step8000_capacity_no_go`。

## 续训波动

| step | easy32 | hard32 |
|---:|---:|---:|
| 5,600 | 963/967，28/32 | 803/807，29/32 |
| 6,400 | 957/967，24/32 | 800/807，26/32 |
| 7,200 | 965/967，30/32 | 803/807，29/32 |
| 8,000 | 965/967，30/32 | 801/807，27/32 |

step7200虽然观察值更高，但协议预注册final step8000，不做事后选模；且任何checkpoint都没有通过
严格门。后半程已明显振荡，不允许继续追加同一loss的epoch。

## 决策

- 保留pointer-mass query matching、全局唯一pointer与Count。
- 最后错误仍是top-N existence边界，而当前object BCE由大量easy no-object平均主导。
- P2-L从与P2-J相同的随机初始化重开4,800步，在现有BCE之外增加整图边界损失：最低正query
  existence必须高于最高负query。不得从P2-K过拟合权重继续，也不扩大数据。

```text
model = chemvector-g3-p2k-pointer-matched-query-set-step8000
checkpoint = outputs/smoke/chemvector-g3-p2k-pointer-matched-query-set-step8000/step-008000.pt
SHA256 = 319cf6ba63e4a118f4f57305d7ac3e25a8c5bbf5aa9bc42ae6b533412c196586
training images = 64
samples seen = 32,000
decision = NO-GO
```
