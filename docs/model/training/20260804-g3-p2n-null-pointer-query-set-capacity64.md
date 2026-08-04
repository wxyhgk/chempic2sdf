# G3 P2-N Null Pointer Query Set Capacity64

> 日期：2026-08-04
>
> 状态：统一语言机制NO-GO

## 受控改动

P2-N保持291,394参数总量，用65参数NULL pointer head替换独立existence head。每个query输出
128个候选峰加1个NULL类；匹配query监督到全局唯一候选，未匹配query监督到NULL。推理existence
为`logsumexp(candidate logits) - null logit`，Count与P2-J相同。

## 结果

| 域 | P2-J | P2-N | Count exact |
|---|---:|---:|---:|
| easy32 | 963/967，28/32 | **894/967，0/32** | 32/32 |
| hard32 | 798/807，26/32 | **756/807，6/32** | 32/32 |

candidate-or-NULL loss从2.54降至约0.17，训练稳定且没有P2-L式数值塌缩，但top-N整组仍显著回退。
统一输出语言在该实现下没有通过容量门，不能扩到5k。

## 对P2-M归因的修正

P2-M同时改变了两件事：冻结线性head、并继续使用极值boundary。它的失败不能单独证明线性head
容量不足，只能证明“冻结head + 极值boundary”失败。P2-N也不能替代该隔离，因为它改变了输出
语言。因此下一实验必须先完成遗漏的最小A/B：P2-J冻结表示、65参数线性head、pointer-only固定
标签、只用weighted BCE，完全不含boundary。

## 决策

- 正式结论：`null_pointer_query_set_capacity_no_go`。
- NULL pointer模型停止，不追加epoch或调NULL权重。
- P2-O做上述BCE-only冻结head隔离；只有它仍失败，才允许增加非线性existence head容量。

```text
model = chemvector-g3-p2n-null-pointer-query-set-capacity64
checkpoint = outputs/smoke/chemvector-g3-p2n-null-pointer-query-set-capacity64/step-004800.pt
SHA256 = 61152c841db8f04eab18ed87ecd158a22315376d1b3ba3a1e643c41243c10de1
training images = 64
samples seen = 19,200
decision = NO-GO
```
