# G3 P2-X Joint Cardinality Decode Capacity64

> 日期：2026-08-04
>
> 状态：容量Gate GO

## 改动

P2-X不训练新模型，只将Atom query选择与candidate pointer分配合并为一次固定基数最大权二部匹配：

```text
edge_score(query, candidate)
  = log sigmoid(existence_logit(query))
  + log softmax(pointer_logits(query))[candidate]

cardinality = argmax(count_logits)
```

匹配同时保证query唯一、candidate唯一和恰好N条真实边。实现没有权重超参，不读取Action Cache、
GT count、像素阈值、距离阈值或键长规则；旧独立top-N接口保留不变。

## 结果

| 域 | 独立top-N | Joint cardinality MAP |
|---|---:|---:|
| easy32 | 966/967，31/32 | **967/967，32/32** |
| hard32 | 807/807，32/32 | **807/807，32/32** |

Count exact双域32/32，八项容量检查全部通过；重复审计逐字段一致，失败样本列表为空。

## 结论

- `joint_cardinality_decode_capacity_go`。
- 此结果证明P2-W已经产生足够的existence与pointer联合证据；之前唯一残差来自串行解码的决策割裂，
  不是候选点、Count或pointer容量不足。
- P2-X没有训练checkpoint，因此不创建模型tag。下一步在完全未参与capacity64训练的样本上固定运行
  joint decoder，测量泛化差距，再决定扩量训练协议。

```text
source checkpoint = outputs/smoke/chemvector-g3-p2w-fixed-identity-pointer-utility-step4800/step-004800.pt
source SHA256 = 229fd1299a68aa544c0bf3ee43acabcdc1386a9df3f57dfb153113c24328b6b9
audit = outputs/audits/chemvector-g3-p2x-joint-cardinality-decode-capacity64/metrics.json
decision = GO
```
