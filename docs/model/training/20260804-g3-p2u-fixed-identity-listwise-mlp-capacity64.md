# G3 P2-U Fixed Identity Listwise MLP Capacity64

> 日期：2026-08-04
>
> 状态：非线性head与listwise组合NO-GO

## 实验与结果

P2-U复用P2-R固定Action Cache与P2-J frozen representation，同时使用P2-T的8,577参数MLP和
P2-S的平滑全正负对listwise目标。训练预算仍为64张图、1,600步、6,400 samples seen。

| 域 | P2-T MLP+BCE | P2-U MLP+BCE+listwise |
|---|---:|---:|
| easy32 | 965/967，30/32 | **963/967，28/32** |
| hard32 | 800/807，28/32 | **800/807，26/32** |

Count保持双域32/32，组合目标没有超过BCE-only。

## 决策

- `fixed_identity_listwise_mlp_capacity_no_go`。
- 停止在纯query-hidden head上组合容量、margin与loss权重。
- P2-Q的pointer utility曾因错误identity无法解释；P2-V必须在同一P2-R Action Cache上重新验证
  10个target-independent pointer utility，隔离query hidden是否缺少显式pointer质量证据。

```text
model = chemvector-g3-p2u-fixed-identity-listwise-mlp-capacity64
checkpoint = outputs/smoke/chemvector-g3-p2u-fixed-identity-listwise-mlp-capacity64/step-001600.pt
SHA256 = ca911c04b2cca5008ba2ed9ef691b360d42dcd5374347c01dac65574ae825222
Action Cache SHA256 = d2134087c5c1d79cb698a3aefa90fcfeb6e3defeba3e602e1e1317a5362e8aca
training images = 64
samples seen = 6,400
trainable parameters = 8,577
decision = NO-GO
```
