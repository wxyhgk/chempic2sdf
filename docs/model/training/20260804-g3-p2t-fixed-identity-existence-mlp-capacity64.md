# G3 P2-T Fixed Identity Existence MLP Capacity64

> 日期：2026-08-04
>
> 状态：非线性容量有收益 / 严格Gate NO-GO

## 实验

P2-T与P2-R使用相同P2-J frozen representation、P2-R Action Cache、weighted BCE、64张图和
1,600步。唯一变量是将65参数`Linear(64,1)`替换为8,577参数
`LayerNorm(64) -> Linear(128) -> GELU -> Linear(1)`。

| 域 | P2-R linear | P2-T MLP | Count exact |
|---|---:|---:|---:|
| easy32 | 964/967，29/32 | **965/967，30/32** | 32/32 |
| hard32 | 799/807，27/32 | **800/807，28/32** | 32/32 |

非线性head各提高1张，但仍未通过32/32。

## 学习曲线复核

| step | easy exact | hard exact | easy identity exact | hard identity exact |
|---:|---:|---:|---:|---:|
| 400 | 29/32 | 27/32 | 21/32 | 25/32 |
| 800 | 29/32 | 28/32 | 21/32 | 26/32 |
| 1200 | 30/32 | 28/32 | 21/32 | 26/32 |
| 1600 | 30/32 | 28/32 | 21/32 | 26/32 |

端到端有小幅改善，但固定identity整组排序已经基本平台。因此不直接追加相同BCE epoch。

## 决策

- `fixed_identity_existence_mlp_capacity_no_go`。
- 正确identity下已确认非线性容量比线性head有效，但BCE-only仍不能完成严格排序。
- P2-U组合P2-T的MLP与P2-S的平滑全正负对listwise目标；仍使用P2-J初始化与同一Action Cache，
  不做事后选模。

```text
model = chemvector-g3-p2t-fixed-identity-existence-mlp-capacity64
checkpoint = outputs/smoke/chemvector-g3-p2t-fixed-identity-existence-mlp-capacity64/step-001600.pt
SHA256 = 84ef3b401b1478a36a417f3a07bb61e7c111de7db83184511dab64d6e460a79b
Action Cache SHA256 = d2134087c5c1d79cb698a3aefa90fcfeb6e3defeba3e602e1e1317a5362e8aca
training images = 64
calibration samples seen = 6,400
trainable parameters = 8,577
decision = NO-GO
```
