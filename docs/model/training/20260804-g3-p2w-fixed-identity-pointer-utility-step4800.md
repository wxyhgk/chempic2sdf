# G3 P2-W Fixed Identity Pointer Utility Step4800

> 日期：2026-08-04
>
> 状态：独立top-N严格Gate NO-GO / 联合解码方向确认

## 实验

P2-W完整恢复P2-V step1600的模型与optimizer，使用同一P2-R Action Cache续训到预先承诺的
step4800。累计19,200 samples seen，新增200个epoch；只以final checkpoint评判。

| 域 | P2-V step1600 | P2-W step4800 |
|---|---:|---:|
| easy32 | 966/967，31/32 | **966/967，31/32** |
| hard32 | 801/807，28/32 | **807/807，32/32** |

五项协议检查全部通过。hard达到严格容量门，easy仍差1个原子，因此整体NO-GO，按承诺停止追加
相同epoch。

## 唯一残差

唯一失败为`train_000006_coordgen`：38个GT原子中匹配37个。固定正query 21的existence logit
比负query 19低0.375，独立top-N先选错query；pointer随后只能在错误query集合内分配候选。
hard域虽有一张query identity不同，但替代query指向等价候选，因此最终仍32/32。

只读机制验证将“选N个query”和“分配N个唯一候选”合并为固定基数二部匹配，边分数严格为
`log P(exists) + log P(pointer)`，不使用GT或阈值。P2-W在该联合解码下easy/hard均为32/32。
下一步将其实现为核心结构解码器并做重复审计；不再训练相同head。

```text
model = chemvector-g3-p2w-fixed-identity-pointer-utility-step4800
checkpoint = outputs/smoke/chemvector-g3-p2w-fixed-identity-pointer-utility-step4800/step-004800.pt
SHA256 = 229fd1299a68aa544c0bf3ee43acabcdc1386a9df3f57dfb153113c24328b6b9
Action Cache SHA256 = d2134087c5c1d79cb698a3aefa90fcfeb6e3defeba3e602e1e1317a5362e8aca
training images = 64
samples seen = 19,200
trainable parameters = 9,877
decision = NO-GO under independent top-N decoder
```
